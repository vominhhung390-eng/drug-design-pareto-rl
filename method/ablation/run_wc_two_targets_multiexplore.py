#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Two-target latent PPO with multi-temperature, multi-scale exploration and
true K-step latent trajectories.

This is a new experiment script and does not modify the previous ablation
runner. Objectives are EGFR and VEGFR2 only. QED/SA/novelty are logged as
quality metrics, not used as optimization objectives.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from finaly import Molecule, ParetoFront  # noqa: E402
from main_pipeline import MO_RL_Integrator  # noqa: E402
from ablation.run_wc_ablation_two_targets import (  # noqa: E402
    ACTIVITY_LOWER,
    ACTIVITY_UPPER,
    RAW_REF_POINT,
    TrajectoryMultiCriticPPOAgent,
    TwoTargetObjectiveCalculator,
    hypervolume_2d,
)
from ablation.weight_controllers import create_controller  # noqa: E402
from ablation.single_critic_ppo_agent import TrajectorySingleCriticPPOAgent  # noqa: E402

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Crippen, Descriptors, QED, rdMolDescriptors

    RDLogger.DisableLog("rdApp.*")
except Exception as exc:  # pragma: no cover
    raise RuntimeError("RDKit is required for this script") from exc

try:
    POLYGON_ROOT = PROJECT_ROOT / "vendor" / "polygon-main"
    if POLYGON_ROOT.exists() and str(POLYGON_ROOT) not in sys.path:
        sys.path.insert(0, str(POLYGON_ROOT))
    from polygon.utils.custom_scoring_fcn import SAScorer  # type: ignore
    from polygon.vae.vae_trainer import VAETrainer  # type: ignore
except Exception:
    SAScorer = None
    VAETrainer = None


REF_POINT = RAW_REF_POINT.copy()
DEFAULT_VAE_MODEL = PROJECT_ROOT / "models" / "polygon_vae_best_valid_novel_stable_020.pt"
DEFAULT_PROTOCOL_CONFIG = PROJECT_ROOT / "config" / "formal_experiments.json"
DEFAULT_CONTROLLER_VARIANT = "ours_full_corrected"
DEFAULT_TEMPERATURE = 1.0
DEFAULT_CHANNELS = [
    # epoch_020 was measured at validity=94.245%, novelty=96.052% at
    # temperature 1.0.  Temperature 1.05 also remains above 93% on both.
    ("A_stable_local", 1.0, 0.5),
    ("B_conservative", 1.0, 1.0),
    ("C_main", 1.0, 1.0),
    ("D_jump", 1.0, 1.5),
    ("E_diverse", 1.05, 1.0),
    ("F_strong_explore", 1.05, 1.5),
]
CHANNEL_PRESETS = {
    "single": [("C_main", 1.0, 1.0)],
    "multi_temperature": [
        ("C_main", 1.0, 1.0),
        ("E_diverse", 1.05, 1.0),
    ],
    "multi_step": [
        ("A_stable_local", 1.0, 0.5),
        ("C_main", 1.0, 1.0),
        ("D_jump", 1.0, 1.5),
    ],
    "multiscale": DEFAULT_CHANNELS,
}
THREE_STAGE_ALLOCATION = {
    "early": [4, 6, 14, 14, 14, 12],
    "middle": [8, 10, 20, 12, 8, 6],
    "late": [14, 16, 22, 8, 4, 0],
}


@dataclass
class RunResult:
    hv_final: float
    pareto_size: int
    egfr_max: float
    vegfr2_max: float
    egfr_mean: float
    vegfr2_mean: float
    best_balanced_score: float
    generated_rows: int
    valid_rows: int
    invalid_rows: int
    valid_rate: float
    unique_valid_smiles: int
    novelty: float
    novel_unique_molecules: int
    qed_mean: float
    sa_mean: float
    runtime_sec: float


def find_target_model(name: str) -> str:
    matches = list((PROJECT_ROOT / "models").rglob(name))
    if not matches:
        raise FileNotFoundError(f"Could not find {name} under {PROJECT_ROOT / 'models'}")
    return str(matches[0])


def canonical(smiles: str) -> Tuple[Optional[str], Optional[Chem.Mol]]:
    if not smiles:
        return None, None
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None, None
    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if not fragments:
        return None, None
    mol = max(fragments, key=lambda item: item.GetNumHeavyAtoms())
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True), mol


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_protocol(path: str) -> Dict:
    protocol_path = Path(path).resolve()
    if not protocol_path.exists():
        raise FileNotFoundError(f"Protocol config not found: {protocol_path}")
    return json.loads(protocol_path.read_text(encoding="utf-8"))


def enforce_registered_protocol(args, protocol: Dict) -> None:
    """Reject silent drift for pre-registered screening/formal budgets."""
    registered_budgets = set(protocol.get("oracle_budgets", []))
    if args.oracle_budget not in registered_budgets:
        return
    checks = {
        "batch": (args.batch, protocol.get("batch_size")),
        "trajectory_step_normalization": (
            args.trajectory_step_normalization,
            protocol.get("trajectory_step_normalization"),
        ),
        "controller_variant": (
            args.controller_variant,
            protocol.get("controller_variant", DEFAULT_CONTROLLER_VARIANT),
        ),
    }
    for name, (actual, expected) in checks.items():
        if expected is not None and actual != expected:
            raise ValueError(
                f"Registered protocol mismatch for {name}: {actual!r} != {expected!r}"
            )
    hv_protocol = protocol.get("activity_hv", {})
    hv_bounds = (
        float(hv_protocol.get("lower_pactivity", ACTIVITY_LOWER)),
        float(hv_protocol.get("upper_pactivity", ACTIVITY_UPPER)),
    )
    if hv_bounds != (ACTIVITY_LOWER, ACTIVITY_UPPER):
        raise ValueError(
            "Registered activity HV bounds do not match the implementation: "
            f"{hv_bounds!r} != {(ACTIVITY_LOWER, ACTIVITY_UPPER)!r}"
        )
    if args.trajectory_length not in protocol.get("trajectory_lengths", []):
        raise ValueError(
            f"trajectory_length={args.trajectory_length} is not registered in the protocol"
        )
    registered_seeds = set(protocol.get("screening_seeds", []))
    registered_seeds.update(protocol.get("formal_seeds", []))
    registered_seeds.update(protocol.get("extended_seeds", []))
    registered_seeds.update(protocol.get("prospective_seeds", []))
    registered_seeds.update(protocol.get("v3_confirmation_seeds", []))
    registered_seeds.update(protocol.get("v4_confirmation_seeds", []))
    registered_seeds.update(protocol.get("v4_ablation_seeds", []))
    if registered_seeds and args.seed not in registered_seeds:
        raise ValueError(f"seed={args.seed} is not registered in the protocol")


def load_train_set(path: Optional[str]) -> set:
    if not path:
        return set()
    p = Path(path)
    if not p.exists():
        return set()
    return {line.strip() for line in p.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()}


def make_sa_scorer(fscores: Optional[str]):
    if SAScorer is not None:
        try:
            return SAScorer(score_modifier=None, fscores=fscores)
        except Exception:
            pass
    try:
        from rdkit.Contrib.SA_Score import sascorer

        return sascorer
    except Exception:
        return None


def sa_score(scorer, smiles: str, mol: Chem.Mol) -> float:
    if scorer is None:
        return float("nan")
    try:
        if hasattr(scorer, "raw_score"):
            return float(scorer.raw_score(smiles))
        if hasattr(scorer, "calculateScore"):
            return float(scorer.calculateScore(mol))
    except Exception:
        return float("nan")
    return float("nan")


def write_csv(path: Path, rows: Iterable[Dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def scaled_allocation(base: Sequence[int], batch_size: int) -> List[int]:
    base = np.asarray(base, dtype=np.float64)
    if base.sum() <= 0:
        base = np.ones_like(base)
    raw = base / base.sum() * batch_size
    alloc = np.floor(raw).astype(int)
    remainder = batch_size - int(alloc.sum())
    order = np.argsort(-(raw - alloc))
    for idx in order[:remainder]:
        alloc[idx] += 1
    return alloc.tolist()


def trajectory_step_scale(
    base_step_scale: float,
    step_multiplier: float,
    trajectory_length: int,
    normalization: str,
) -> float:
    """Calculate the per-decision step scale for a K-step trajectory.

    The default square-root normalization keeps the expected radius of an
    uncorrelated random walk close to K=1, reducing a major K=1/3/5 ablation
    confound while still allowing aligned actions to move farther.
    """
    if trajectory_length < 1:
        raise ValueError("trajectory_length must be at least 1")
    if normalization == "none":
        divisor = 1.0
    elif normalization == "sqrt":
        divisor = float(np.sqrt(trajectory_length))
    elif normalization == "linear":
        divisor = float(trajectory_length)
    else:
        raise ValueError(f"Unknown trajectory step normalization: {normalization}")
    return float(base_step_scale * step_multiplier / divisor)


def linear_ramp(progress: float, start: float, end: float) -> float:
    """Return a clipped 0->1 schedule while preserving legacy instant-on defaults."""
    progress = float(np.clip(progress, 0.0, 1.0))
    start = float(np.clip(start, 0.0, 1.0))
    end = float(np.clip(end, 0.0, 1.0))
    if progress <= start:
        return 0.0
    if end <= start or progress >= end:
        return 1.0
    return (progress - start) / (end - start)


def archive_stagnation_status(
    hv_history: Sequence[float], window: int, minimum_gain: float
) -> Tuple[bool, float]:
    """Return whether archive replay should activate and the recent HV gain."""
    if window <= 0:
        return True, float("nan")
    if len(hv_history) <= window:
        return False, float("nan")
    gain = float(hv_history[-1] - hv_history[-1 - window])
    return gain < float(minimum_gain), gain


def store_terminal_trajectory(
    agent,
    transitions: Sequence[Dict],
    terminal_rewards: np.ndarray,
    preference: np.ndarray,
    auxiliary_reward: float = 0.0,
) -> None:
    """Store a contiguous trajectory with reward only at its terminal step."""
    if not transitions:
        raise ValueError("Cannot store an empty trajectory")
    terminal_rewards = np.asarray(terminal_rewards, dtype=np.float32)
    zero_rewards = np.zeros_like(terminal_rewards)
    for index, transition in enumerate(transitions):
        terminal = index + 1 == len(transitions)
        agent.store_transition_multi(
            transition["policy_state"],
            transition["action"],
            terminal_rewards if terminal else zero_rewards,
            transition["log_prob"],
            transition["values"],
            terminal,
            transition.get("entropy", 0.0),
            preference,
            auxiliary_reward=auxiliary_reward if terminal else 0.0,
        )


def _positive_rank_scale(values: np.ndarray) -> np.ndarray:
    """Map positive values to (0, 1] ranks while leaving zeros at zero."""
    values = np.asarray(values, dtype=np.float64)
    result = np.zeros_like(values)
    positive = values > 1e-12
    if np.any(positive):
        result[positive] = pd.Series(values[positive]).rank(method="average", pct=True).to_numpy()
    return result


def archive_sampling_probabilities(
    scores: np.ndarray,
    strategy: str = "uniform",
    hvc_weight: float = 0.7,
    balance_weight: float = 0.3,
    temperature: float = 0.25,
    uniform_mix: float = 0.10,
) -> np.ndarray:
    """Build auditable elite-parent probabilities for the Pareto archive.

    HVC is computed as the exclusive loss in normalized 2-D hypervolume when
    each archive point is removed.  The optional balance term favors molecules
    whose weaker target is also strong.  A small uniform mixture prevents a
    permanently zero sampling probability for valid frontier regions.
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1, 2)
    count = len(scores)
    if count == 0:
        return np.zeros(0, dtype=np.float64)
    uniform = np.ones(count, dtype=np.float64) / count
    if strategy == "uniform" or count == 1:
        return uniform
    if strategy not in {"hvc", "hvc_balanced"}:
        raise ValueError(f"Unknown archive sampling strategy: {strategy}")

    full_hv = hypervolume_2d(scores)
    exclusive_hvc = np.asarray(
        [
            max(0.0, full_hv - hypervolume_2d(np.delete(scores, index, axis=0)))
            for index in range(count)
        ],
        dtype=np.float64,
    )
    hvc_rank = _positive_rank_scale(exclusive_hvc)
    normalized = np.clip(
        (scores - ACTIVITY_LOWER) / max(ACTIVITY_UPPER - ACTIVITY_LOWER, 1e-8),
        0.0,
        1.0,
    )
    balance_rank = pd.Series(np.min(normalized, axis=1)).rank(
        method="average", pct=True
    ).to_numpy(dtype=np.float64)
    if strategy == "hvc":
        elite_score = hvc_rank
    else:
        elite_score = (
            max(float(hvc_weight), 0.0) * hvc_rank
            + max(float(balance_weight), 0.0) * balance_rank
        )
    if np.allclose(elite_score, elite_score[0]):
        return uniform

    temperature = max(float(temperature), 1e-6)
    logits = (elite_score - np.max(elite_score)) / temperature
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum()
    mix = float(np.clip(uniform_mix, 0.0, 1.0))
    probabilities = (1.0 - mix) * probabilities + mix * uniform
    return probabilities / probabilities.sum()


def generator_elite_scores(
    scores: np.ndarray,
    strategy: str,
    weights: Optional[np.ndarray] = None,
    balance_mix: float = 0.5,
    softmin_temperature: float = 0.1,
) -> np.ndarray:
    """Rank real molecules for generator self-training.

    The legacy strategies reproduce POLYGON's clipped reward view (3.0--6.5).
    ``raw_*`` strategies instead use the pre-registered HV normalization range
    (3.0--10.0), so activity improvements above 6.5 are not discarded.

    ``balance_sync`` is the V5 rule.  It synchronizes the current three-stage
    controller preference with generator self-training.  The weighted term
    retains V4-B's registered 3.0--10.0 continuous-activity signal, while the
    smooth worst-target term is computed from 3.0--6.5 threshold deficits.
    Thus V5 adds balance pressure without discarding improvements above 6.5.
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1, 2)
    if strategy == "balance_sync":
        if not 0.0 <= balance_mix <= 1.0:
            raise ValueError("balance_mix must be in [0, 1]")
        if softmin_temperature <= 0.0:
            raise ValueError("softmin_temperature must be positive")
        if weights is None:
            normalized_weights = np.ones(2, dtype=np.float64) / 2.0
        else:
            normalized_weights = np.clip(
                np.asarray(weights, dtype=np.float64).reshape(2), 0.0, None
            )
            if normalized_weights.sum() <= 0.0:
                normalized_weights = np.ones(2, dtype=np.float64) / 2.0
            else:
                normalized_weights /= normalized_weights.sum()

        raw_desirability = np.clip((scores - 3.0) / (10.0 - 3.0), 0.0, 1.0)
        threshold_desirability = np.clip(
            (scores - 3.0) / (6.5 - 3.0), 0.0, 1.0
        )
        weighted_desirability = raw_desirability @ normalized_weights

        # softmin(d) == 1 - smooth_max(1 - d): explicitly penalize the
        # largest remaining threshold deficit while retaining smooth ranks.
        deficits = 1.0 - threshold_desirability
        scaled = deficits / softmin_temperature
        scaled_max = scaled.max(axis=1, keepdims=True)
        smooth_max_deficit = softmin_temperature * (
            scaled_max[:, 0]
            + np.log(np.exp(scaled - scaled_max).mean(axis=1))
        )
        soft_min = 1.0 - smooth_max_deficit
        return (
            (1.0 - balance_mix) * weighted_desirability
            + balance_mix * soft_min
        )

    upper = 10.0 if strategy.startswith("raw_") else 6.5
    desirability = np.clip((scores - 3.0) / (upper - 3.0), 0.0, 1.0)
    mean_score = desirability.mean(axis=1)
    min_score = desirability.min(axis=1)
    base_strategy = strategy.removeprefix("raw_")
    if base_strategy == "mean":
        return mean_score
    if base_strategy == "min":
        return min_score
    if base_strategy == "mixed":
        return 0.5 * mean_score + 0.5 * min_score
    raise ValueError(f"Unknown generator elite strategy: {strategy}")


def compute_pareto_shaping(
    old_scores: np.ndarray, new_scores: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return raw HVC, rank-scaled HVC, and sparse-front crowding rewards.

    Every candidate is compared with the same pre-batch archive, so rewards do
    not depend on candidate processing order. Crowding is awarded only to
    candidates with positive HVC and is based on nearest-neighbour distance in
    normalized objective space.
    """
    old_scores = np.asarray(old_scores, dtype=np.float64).reshape(-1, 2)
    new_scores = np.asarray(new_scores, dtype=np.float64).reshape(-1, 2)
    if not len(new_scores):
        empty = np.zeros(0, dtype=np.float64)
        return empty, empty, empty
    old_hv = hypervolume_2d(old_scores)
    raw_hvc = np.asarray(
        [
            max(
                0.0,
                hypervolume_2d(np.vstack([old_scores, score]) if len(old_scores) else score.reshape(1, -1))
                - old_hv,
            )
            for score in new_scores
        ],
        dtype=np.float64,
    )
    hvc_rank = _positive_rank_scale(raw_hvc)
    crowd_raw = np.zeros(len(new_scores), dtype=np.float64)
    contributing = raw_hvc > 1e-12
    if np.any(contributing):
        all_points = np.vstack([old_scores, new_scores]) if len(old_scores) else new_scores.copy()
        low = all_points.min(axis=0)
        span = np.maximum(all_points.max(axis=0) - low, 1e-8)
        old_norm = (old_scores - low) / span if len(old_scores) else np.empty((0, 2))
        new_norm = (new_scores - low) / span
        for index in np.flatnonzero(contributing):
            peers = [old_norm]
            other = np.delete(new_norm, index, axis=0)
            if len(other):
                peers.append(other)
            comparison = np.vstack([part for part in peers if len(part)]) if any(len(part) for part in peers) else np.empty((0, 2))
            crowd_raw[index] = (
                float(np.linalg.norm(comparison - new_norm[index], axis=1).min())
                if len(comparison)
                else 1.0
            )
    crowd_rank = _positive_rank_scale(crowd_raw)
    return raw_hvc, hvc_rank, crowd_rank


def stage_name(epoch: int, total_epochs: int) -> str:
    progress = (epoch + 1) / max(total_epochs, 1)
    if progress <= 0.30:
        return "early"
    if progress <= 0.70:
        return "middle"
    return "late"


def build_epoch_channels(
    epoch: int,
    total_epochs: int,
    batch_size: int,
    utility: Optional[np.ndarray] = None,
    channel_definitions: Sequence[Tuple[str, float, float]] = DEFAULT_CHANNELS,
) -> List[Tuple[str, float, float]]:
    stage = stage_name(epoch, total_epochs)
    stage_weights = {
        name: weight
        for (name, _, _), weight in zip(DEFAULT_CHANNELS, THREE_STAGE_ALLOCATION[stage])
    }
    prior = np.asarray(
        [stage_weights.get(name, 1.0) for name, _, _ in channel_definitions],
        dtype=np.float64,
    )
    if utility is not None and len(utility) == len(channel_definitions):
        # Pareto/HV feedback reweights the stage prior while keeping every
        # enabled channel represented.  Subtracting max keeps exp stable.
        centered = np.asarray(utility, dtype=np.float64) - np.max(utility)
        prior = np.maximum(prior, 1.0) * np.exp(centered / 0.35)
    alloc = scaled_allocation(prior, batch_size)
    channels: List[Tuple[str, float, float]] = []
    for (name, temp, step), n in zip(channel_definitions, alloc):
        channels.extend([(name, temp, step)] * n)
    return channels


class MultiExploreIntegrator(MO_RL_Integrator):
    def __init__(self, config: Dict):
        use_v41 = config.get("oracle_system", "original_rf") == "v41"
        super().__init__(
            latent_dim=config.get("latent_dim", 128),
            num_obj=2,
            total_epochs=config.get("total_epochs", 100),
            batch_size=config.get("batch_size", 64),
            vae_model_path=config.get("vae_model_path"),
            egfr_model_path=None if use_v41 else config.get("egfr_model_path"),
            vegfr2_model_path=None if use_v41 else config.get("vegfr2_model_path"),
            config=config,
        )
        self.objective_calculator = TwoTargetObjectiveCalculator(
            config.get("egfr_model_path"),
            config.get("vegfr2_model_path"),
        )
        if use_v41:
            from predictor_v41_oracle import V41TwoTargetObjectiveCalculator

            # Loading ten Chemprop members constructs modules internally and
            # may consume RNG state.  Preserve all experiment RNG streams so
            # paired seeds differ only in the online reward predictor.
            python_rng = random.getstate()
            numpy_rng = np.random.get_state()
            torch_rng = torch.random.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            try:
                self.objective_calculator = V41TwoTargetObjectiveCalculator(
                    PROJECT_ROOT, device=config.get("device", "cuda")
                )
            finally:
                random.setstate(python_rng)
                np.random.set_state(numpy_rng)
                torch.random.set_rng_state(torch_rng)
                if cuda_rng is not None:
                    torch.cuda.set_rng_state_all(cuda_rng)
        # Two explicit context features make temperature and step multiplier
        # observable to both actor and critics.  The action still lives in the
        # original latent dimension.
        agent_class = (
            TrajectoryMultiCriticPPOAgent
            if config.get("critic_mode", "multi") == "multi"
            else TrajectorySingleCriticPPOAgent
        )
        self.agent = agent_class(
            state_dim=self.latent_dim + 2,
            action_dim=self.latent_dim,
            num_obj=2,
            lr=config.get("lr", 3e-4),
            gamma=config.get("gamma", 0.99),
            gae_lambda=config.get("gae_lambda", 0.95),
            ppo_clip=config.get("ppo_clip", 0.2),
            ppo_epochs=config.get("ppo_epochs", 4),
            mini_batch_size=config.get("mini_batch_size", 32),
            entropy_coef=config.get("entropy_coef", 0.01),
            value_loss_coef=config.get("value_loss_coef", 0.5),
            device=config.get("device", None),
        )
        self.controller = create_controller(
            config.get("controller_variant", DEFAULT_CONTROLLER_VARIANT),
            num_obj=2,
            total_epochs=self.total_epochs,
        )
        self.controller.set_ref_point(REF_POINT)
        self.pareto_front = self.controller.pareto_front
        self.invalid_reward = float(config.get("invalid_reward", -1.0))
        self.latent_clip = float(config.get("latent_clip", 4.0))
        self.base_step_scale = float(config.get("base_step_scale", 0.08))
        self.trajectory_length = int(config.get("trajectory_length", 1))
        if self.trajectory_length < 1:
            raise ValueError("trajectory_length must be at least 1")
        self.trajectory_step_normalization = str(
            config.get("trajectory_step_normalization", "sqrt")
        )
        # Validate the mode once during construction rather than failing deep
        # inside the first rollout.
        trajectory_step_scale(
            self.base_step_scale,
            1.0,
            self.trajectory_length,
            self.trajectory_step_normalization,
        )
        self.archive_seed_fraction = float(config.get("archive_seed_fraction", 0.0))
        self.archive_seed_noise = float(config.get("archive_seed_noise", 0.15))
        self.archive_seed_noise_end = float(
            config.get("archive_seed_noise_end", self.archive_seed_noise)
        )
        self.archive_seed_start = float(config.get("archive_seed_start", 0.0))
        self.archive_seed_ramp_end = float(config.get("archive_seed_ramp_end", 0.0))
        self.archive_seed_selection = str(
            config.get("archive_seed_selection", "uniform")
        )
        self.archive_hvc_weight = float(config.get("archive_hvc_weight", 0.7))
        self.archive_balance_weight = float(
            config.get("archive_balance_weight", 0.3)
        )
        self.archive_selection_temperature = float(
            config.get("archive_selection_temperature", 0.25)
        )
        self.archive_uniform_mix = float(config.get("archive_uniform_mix", 0.10))
        self.archive_stagnation_window = int(
            config.get("archive_stagnation_window", 0)
        )
        self.archive_stagnation_delta = float(
            config.get("archive_stagnation_delta", 0.002)
        )
        self.archive_stagnation_noise = float(
            config.get("archive_stagnation_noise", 0.0)
        )
        self.archive_hv_history = [0.0]
        self.preference_floor = float(config.get("preference_floor", 0.10))
        self.preference_ema_alpha = float(config.get("preference_ema_alpha", 0.25))
        self.policy_weights = np.ones(2, dtype=np.float32) / 2.0
        self.weight_mode = str(config.get("weight_mode", "dynamic"))
        self.dirichlet_alpha = float(config.get("dirichlet_alpha", 0.5))
        self.dynamic_weights = bool(config.get("dynamic_weights", True))
        self.adaptive_channels = bool(config.get("adaptive_channels", True))
        self.exploration_mode = str(config.get("exploration_mode", "multiscale"))
        self.channels = CHANNEL_PRESETS[self.exploration_mode]
        self.channel_utility = np.zeros(len(self.channels), dtype=np.float64)
        self.channel_index = {name: i for i, (name, _, _) in enumerate(self.channels)}
        self.hvc_reward_weight = float(config.get("hvc_reward_weight", 0.0))
        self.crowding_reward_weight = float(config.get("crowding_reward_weight", 0.0))
        self.balanced_reward_weight = float(config.get("balanced_reward_weight", 0.0))
        self.sample_preference_mode = str(config.get("sample_preference_mode", "shared"))
        self.sample_preference_blend = float(config.get("sample_preference_blend", 0.75))
        self.sample_preference_start = float(config.get("sample_preference_start", 0.0))
        self.sample_preference_ramp_end = float(
            config.get("sample_preference_ramp_end", 0.0)
        )
        self.pareto_reward_start = float(config.get("pareto_reward_start", 0.0))
        self.pareto_reward_ramp_end = float(config.get("pareto_reward_ramp_end", 0.0))
        self.agent.auxiliary_actor_coef = float(config.get("pareto_actor_coef", 0.0))
        self.actor_mode = str(config.get("actor_mode", "train"))
        if self.actor_mode not in {"train", "frozen", "zero"}:
            raise ValueError(f"Unknown actor_mode: {self.actor_mode}")
        self.generator_finetune_interval = int(
            config.get("generator_finetune_interval", 0)
        )
        self.generator_finetune_epochs = int(
            config.get("generator_finetune_epochs", 0)
        )
        self.generator_finetune_top = int(config.get("generator_finetune_top", 512))
        self.generator_finetune_batch_size = int(
            config.get("generator_finetune_batch_size", 32)
        )
        self.generator_finetune_lr = float(
            config.get("generator_finetune_lr", 3e-4)
        )
        self.generator_elite_strategy = str(
            config.get("generator_elite_strategy", "mean")
        )
        self.generator_balance_mix = float(
            config.get("generator_balance_mix", 0.5)
        )
        self.generator_softmin_temperature = float(
            config.get("generator_softmin_temperature", 0.1)
        )
        self.generator_elite_archive: Dict[str, np.ndarray] = {}
        self.generator_finetune_count = 0
        self.generator_trainer = None
        if self.generator_finetune_interval > 0 and self.generator_finetune_epochs > 0:
            if VAETrainer is None or getattr(self.vae, "_model_type", None) != "polygon":
                raise RuntimeError("Generator self-training requires the Polygon VAE trainer")
            self.generator_trainer = VAETrainer(
                model=self.vae.model,
                model_save=None,
                n_batch=self.generator_finetune_batch_size,
                n_workers=0,
                kl_w_start=1,
                kl_w_end=1,
                lr_start=self.generator_finetune_lr,
                lr_end=self.generator_finetune_lr,
            )

    def _update_generator_elites(self, molecules: Sequence[Molecule]) -> None:
        for molecule in molecules:
            self.generator_elite_archive[molecule.smiles] = np.asarray(
                molecule.scores, dtype=np.float32
            )

    def _finetune_generator_if_due(self, epoch: int) -> bool:
        if self.generator_trainer is None:
            return False
        if (epoch + 1) % self.generator_finetune_interval != 0:
            return False
        if len(self.generator_elite_archive) < 4:
            return False
        smiles = list(self.generator_elite_archive)
        scores = np.vstack([self.generator_elite_archive[item] for item in smiles])
        ranking = generator_elite_scores(
            scores,
            self.generator_elite_strategy,
            weights=self.policy_weights,
            balance_mix=self.generator_balance_mix,
            softmin_temperature=self.generator_softmin_temperature,
        )
        order = np.argsort(-ranking, kind="stable")[: self.generator_finetune_top]
        selected = [smiles[index] for index in order]
        np.random.shuffle(selected)
        split = max(1, int(0.75 * len(selected)))
        train = selected[:split]
        validation = selected[split:] or None
        self.generator_trainer.fit(
            train,
            validation,
            n_epoch=self.generator_finetune_epochs,
            batch_size=min(self.generator_finetune_batch_size, len(train)),
            save_frequency=None,
        )
        self.vae.model.eval()
        self.generator_finetune_count += 1
        return True

    def _normalize_weights(self, weights: np.ndarray) -> np.ndarray:
        """Project onto the simplex with a floor, idempotently.

        The earlier affine floor transform was applied multiple times per
        epoch, which repeatedly pulled already-valid preferences toward 0.5.
        """
        weights = np.asarray(weights, dtype=np.float32)
        weights = np.clip(weights, 0.0, None)
        if weights.sum() <= 0:
            weights = np.ones_like(weights) / len(weights)
        else:
            weights = weights / weights.sum()
        floor = min(self.preference_floor, (1.0 - 1e-6) / len(weights))
        if np.all(weights >= floor - 1e-7):
            return weights
        residual = np.clip(weights - floor, 0.0, None)
        if residual.sum() <= 1e-12:
            return np.ones_like(weights) / len(weights)
        return floor + (1.0 - floor * len(weights)) * residual / residual.sum()

    def _build_sample_preferences(
        self,
        count: int,
        base_weights: np.ndarray,
        epoch: int,
        schedule_strength: float = 1.0,
    ) -> np.ndarray:
        if self.sample_preference_mode == "shared" or count <= 1:
            return np.repeat(base_weights[None, :], count, axis=0)
        # A deterministic simplex grid guarantees simultaneous coverage of
        # both extremes and the central trade-off. Shuffling prevents channel
        # identity from becoming confounded with preference direction.
        grid = np.linspace(self.preference_floor, 1.0 - self.preference_floor, count)
        grid = np.roll(grid, epoch % count)
        np.random.shuffle(grid)
        targets = np.column_stack([grid, 1.0 - grid]).astype(np.float32)
        blend = float(
            np.clip(self.sample_preference_blend, 0.0, 1.0)
            * np.clip(schedule_strength, 0.0, 1.0)
        )
        preferences = (1.0 - blend) * base_weights[None, :] + blend * targets
        return np.vstack([self._normalize_weights(pref) for pref in preferences])

    def run_episode(self, epoch: int) -> Tuple[List[Molecule], float, Dict, List[Dict]]:
        channels = build_epoch_channels(
            epoch,
            self.total_epochs,
            self.batch_size,
            self.channel_utility if self.adaptive_channels else None,
            self.channels,
        )
        np.random.shuffle(channels)
        valid_molecules: List[Molecule] = []
        generated_rows: List[Dict] = []
        pending_valid: List[Dict] = []
        decode_proposals: List[Dict] = []
        invalid_count = 0
        if self.weight_mode == "dirichlet":
            self.policy_weights = np.random.dirichlet(
                np.full(2, self.dirichlet_alpha, dtype=np.float64)
            ).astype(np.float32)
        weights = self._normalize_weights(self.policy_weights)
        progress = (epoch + 1) / max(self.total_epochs, 1)
        pareto_reward_strength = linear_ramp(
            progress, self.pareto_reward_start, self.pareto_reward_ramp_end
        )
        archive_seed_strength = linear_ramp(
            progress, self.archive_seed_start, self.archive_seed_ramp_end
        )
        archive_stagnation_triggered, archive_recent_hv_gain = (
            archive_stagnation_status(
                self.archive_hv_history,
                self.archive_stagnation_window,
                self.archive_stagnation_delta,
            )
        )
        if not archive_stagnation_triggered:
            archive_seed_strength = 0.0
        preference_strength = linear_ramp(
            progress, self.sample_preference_start, self.sample_preference_ramp_end
        )
        effective_archive_fraction = float(
            np.clip(self.archive_seed_fraction, 0.0, 1.0) * archive_seed_strength
        )
        effective_archive_noise = float(
            self.archive_seed_noise
            + archive_seed_strength
            * (self.archive_seed_noise_end - self.archive_seed_noise)
        )
        if archive_stagnation_triggered and self.archive_stagnation_noise > 0:
            effective_archive_noise = max(
                effective_archive_noise, self.archive_stagnation_noise
            )
        effective_preference_blend = float(
            np.clip(self.sample_preference_blend, 0.0, 1.0) * preference_strength
        )
        sample_preferences = self._build_sample_preferences(
            len(channels), weights, epoch, schedule_strength=preference_strength
        )
        z_states = np.random.normal(0, 1, (len(channels), self.latent_dim)).astype(np.float32)
        latent_sources = np.full(len(channels), "global_prior", dtype=object)
        if self.pareto_front.molecules and effective_archive_fraction > 0:
            n_archive = min(
                len(channels),
                int(round(len(channels) * effective_archive_fraction)),
            )
            archive_latents = np.asarray(
                [m.latent_vector for m in self.pareto_front.molecules], dtype=np.float32
            )
            archive_scores = np.asarray(
                [m.scores for m in self.pareto_front.molecules], dtype=np.float64
            )
            parent_probabilities = archive_sampling_probabilities(
                archive_scores,
                strategy=self.archive_seed_selection,
                hvc_weight=self.archive_hvc_weight,
                balance_weight=self.archive_balance_weight,
                temperature=self.archive_selection_temperature,
                uniform_mix=self.archive_uniform_mix,
            )
            chosen = np.random.choice(
                len(archive_latents), size=n_archive, replace=True, p=parent_probabilities
            )
            z_states[:n_archive] = archive_latents[chosen] + np.random.normal(
                0.0, effective_archive_noise, size=(n_archive, self.latent_dim)
            ).astype(np.float32)
            z_states[:n_archive] = np.clip(z_states[:n_archive], -self.latent_clip, self.latent_clip)
            latent_sources[:n_archive] = "pareto_archive"
            permutation = np.random.permutation(len(channels))
            z_states = z_states[permutation]
            latent_sources = latent_sources[permutation]

        for idx, (channel, temperature, step_multiplier) in enumerate(channels):
            z = z_states[idx]
            row = {
                "epoch": epoch + 1,
                "sample_idx": idx,
                "latent_source": latent_sources[idx],
                "channel": channel,
                "temperature": temperature,
                "step_multiplier": step_multiplier,
                "trajectory_length": self.trajectory_length,
                "actor_mode": self.actor_mode,
                "trajectory_steps": 0,
                "effective_step_scale": trajectory_step_scale(
                    self.base_step_scale,
                    step_multiplier,
                    self.trajectory_length,
                    self.trajectory_step_normalization,
                ),
                "path_length": 0.0,
                "net_displacement": 0.0,
                "smiles": "",
                "canonical_smiles": "",
                "is_valid": False,
                "egfr": "",
                "vegfr2": "",
                "balanced": "",
                "min_score": "",
                "qed": "",
                "sa": "",
                "mol_wt": "",
                "logp": "",
                "tpsa": "",
                "heavy_atoms": "",
                "hv_contribution": 0.0,
                "hvc_rank_reward": 0.0,
                "crowding_reward": 0.0,
                "balanced_rank_reward": 0.0,
                "auxiliary_reward": 0.0,
                "pareto_reward_strength": float(pareto_reward_strength),
                "archive_seed_fraction_effective": effective_archive_fraction,
                "archive_seed_noise_effective": effective_archive_noise,
                "archive_stagnation_triggered": archive_stagnation_triggered,
                "archive_recent_hv_gain": archive_recent_hv_gain,
                "preference_blend_effective": effective_preference_blend,
                "pref_egfr": float(sample_preferences[idx, 0]),
                "pref_vegfr2": float(sample_preferences[idx, 1]),
                "error": "",
            }
            try:
                channel_context = np.asarray(
                    [temperature / 1.05, step_multiplier / 1.5], dtype=np.float32
                )
                sample_preference = sample_preferences[idx]
                current_z = z.copy()
                step_scale = float(row["effective_step_scale"])
                transitions = []
                path_length = 0.0
                for trajectory_step in range(self.trajectory_length):
                    policy_state = np.concatenate([current_z, channel_context])
                    if self.actor_mode == "zero":
                        action = np.zeros(self.latent_dim, dtype=np.float32)
                        log_prob = 0.0
                        values = np.zeros(2, dtype=np.float32)
                        entropy = 0.0
                    else:
                        action, log_prob, values, entropy = self.agent.select_action(
                            policy_state, preference=sample_preference
                        )
                    new_z = np.clip(
                        current_z
                        + step_scale * np.asarray(action, dtype=np.float32),
                        -self.latent_clip,
                        self.latent_clip,
                    )
                    path_length += float(np.linalg.norm(new_z - current_z))
                    transitions.append(
                        {
                            "trajectory_step": trajectory_step + 1,
                            "policy_state": policy_state,
                            "action": action,
                            "log_prob": log_prob,
                            "values": values,
                            "entropy": entropy,
                        }
                    )
                    current_z = new_z
                row["trajectory_steps"] = len(transitions)
                row["path_length"] = path_length
                row["net_displacement"] = float(np.linalg.norm(current_z - z))
                decode_proposals.append(
                    {
                        "row": row,
                        "temperature": float(temperature),
                        "new_z": current_z,
                        "transitions": transitions,
                        "preference": sample_preference,
                    }
                )
            except Exception as exc:
                invalid_count += 1
                row["error"] = type(exc).__name__
                generated_rows.append(row)

        # Decode candidates in temperature-homogeneous GPU batches.  Polygon's
        # decoder is autoregressive but vectorized over the batch dimension.
        by_temperature: Dict[float, List[Dict]] = {}
        for item in decode_proposals:
            by_temperature.setdefault(item["temperature"], []).append(item)
        for temperature, items in by_temperature.items():
            latent_batch = np.stack([item["new_z"] for item in items])
            try:
                smiles_batch = self.vae.decode_batch(
                    latent_batch, greedy=False, temperature=temperature
                )
            except Exception as exc:
                smiles_batch = [""] * len(items)
                for item in items:
                    item["row"]["error"] = type(exc).__name__
            for item, smiles in zip(items, smiles_batch):
                row = item["row"]
                row["smiles"] = smiles
                can, mol = canonical(smiles)
                if can is None or mol is None or len(smiles) < 5:
                    invalid_count += 1
                    if not row["error"]:
                        row["error"] = "invalid_smiles"
                    if self.actor_mode == "train":
                        store_terminal_trajectory(
                            self.agent,
                            item["transitions"],
                            np.full(2, self.invalid_reward, dtype=np.float32),
                            item["preference"],
                            auxiliary_reward=0.0,
                        )
                else:
                    pending_valid.append(
                        {
                            **item,
                            "canonical": can,
                            "mol": mol,
                        }
                    )
                generated_rows.append(row)

        # Batch inference avoids invoking both random-forest predictors once
        # per molecule and is substantially faster on large experiment grids.
        if pending_valid:
            score_matrix = self.objective_calculator.calculate_scores_batch(
                [item["canonical"] for item in pending_valid]
            )
            old_scores = np.asarray(self.pareto_front.solutions, dtype=np.float32)
            scored_valid: List[Dict] = []
            for item, scores in zip(pending_valid, score_matrix):
                row = item["row"]
                if np.any(np.isnan(scores)) or np.any(np.isinf(scores)):
                    invalid_count += 1
                    row["error"] = "invalid_scores"
                    if self.actor_mode == "train":
                        store_terminal_trajectory(
                            self.agent,
                            item["transitions"],
                            np.full(2, self.invalid_reward, dtype=np.float32),
                            item["preference"],
                        )
                    continue
                can = item["canonical"]
                mol = item["mol"]
                molecule = Molecule(
                    smiles=can, latent_vector=item["new_z"], scores=scores
                )
                valid_molecules.append(molecule)
                item["scores"] = np.asarray(scores, dtype=np.float32)
                scored_valid.append(item)
                row.update(
                    {
                        "canonical_smiles": can,
                        "is_valid": True,
                        "egfr": float(scores[0]),
                        "vegfr2": float(scores[1]),
                        "balanced": float((scores[0] + scores[1]) / 2),
                        "min_score": float(min(scores[0], scores[1])),
                        "qed": float(QED.qed(mol)),
                        "mol_wt": float(Descriptors.MolWt(mol)),
                        "logp": float(Crippen.MolLogP(mol)),
                        "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
                        "heavy_atoms": int(mol.GetNumHeavyAtoms()),
                    }
                )

            if scored_valid:
                new_scores = np.vstack([item["scores"] for item in scored_valid])
                raw_hvc, hvc_rank, crowd_rank = compute_pareto_shaping(old_scores, new_scores)
                balanced_rank = pd.Series(np.min(new_scores, axis=1)).rank(
                    method="average", pct=True
                ).to_numpy()
                for index, item in enumerate(scored_valid):
                    auxiliary_reward = (
                        self.hvc_reward_weight * float(hvc_rank[index])
                        + self.crowding_reward_weight * float(crowd_rank[index])
                        + self.balanced_reward_weight * float(balanced_rank[index])
                    ) * float(pareto_reward_strength)
                    item["row"].update(
                        {
                            "hv_contribution": float(raw_hvc[index]),
                            "hvc_rank_reward": float(hvc_rank[index]),
                            "crowding_reward": float(crowd_rank[index]),
                            "balanced_rank_reward": float(balanced_rank[index]),
                            "auxiliary_reward": float(auxiliary_reward),
                        }
                    )
                    if self.actor_mode == "train":
                        store_terminal_trajectory(
                            self.agent,
                            item["transitions"],
                            item["scores"],
                            item["preference"],
                            auxiliary_reward=auxiliary_reward,
                        )

        if valid_molecules:
            # Deduplicate the batch by canonical SMILES before archive update.
            unique = {}
            for molecule in valid_molecules:
                unique.setdefault(molecule.smiles, molecule)
            archive_batch = list(unique.values())
            batch_scores = np.vstack([m.scores for m in archive_batch])
        else:
            archive_batch = []
            batch_scores = np.zeros((1, 2), dtype=np.float32)

        # The three-stage controller supplies the next reward/advantage
        # scalarization and the next actor/critic preference condition.  It
        # must see the pre-update archive; otherwise every new point is already
        # present and its HVC is identically zero.
        if self.dynamic_weights:
            raw_next_weights = self._normalize_weights(
                self.controller.get_weights(epoch, batch_scores)
            )
            alpha = np.clip(self.preference_ema_alpha, 0.0, 1.0)
            self.policy_weights = self._normalize_weights(
                (1.0 - alpha) * weights + alpha * raw_next_weights
            )
        else:
            raw_next_weights = np.ones(2, dtype=np.float32) / 2.0
            self.policy_weights = np.ones(2, dtype=np.float32) / 2.0
        if archive_batch:
            self.controller.update_pareto_front(archive_batch)
        self.archive_hv_history.append(
            hypervolume_2d(np.asarray(self.pareto_front.solutions, dtype=np.float64))
            if len(self.pareto_front.solutions)
            else 0.0
        )

        # EMA channel utility couples exploration allocation to Pareto coverage.
        # HVC is primary; validity is a small stabilizer when HVC is sparse.
        epoch_rows = {name: [] for name, _, _ in self.channels}
        for row in generated_rows:
            epoch_rows[row["channel"]].append(row)
        observed = np.zeros_like(self.channel_utility)
        for name, rows in epoch_rows.items():
            idx = self.channel_index[name]
            if rows:
                hvc_mean = float(np.mean([float(r["hv_contribution"]) for r in rows]))
                valid_rate = float(np.mean([bool(r["is_valid"]) for r in rows]))
                observed[idx] = np.log1p(max(hvc_mean, 0.0)) + 0.05 * valid_rate
        if self.adaptive_channels:
            self.channel_utility = 0.8 * self.channel_utility + 0.2 * observed

        loss = 0.0
        if self.actor_mode == "train" and len(self.agent.buffer.states) >= self.agent.mini_batch_size:
            loss = self.agent.update(self.agent.buffer)
            self.agent.buffer.clear()
        self._update_generator_elites(valid_molecules)
        generator_finetuned = self._finetune_generator_if_due(epoch)

        info = {
            "valid_count": len(valid_molecules),
            "invalid_count": invalid_count,
            "policy_transitions": sum(
                int(row.get("trajectory_steps", 0)) for row in generated_rows
            ),
            "trajectory_length": self.trajectory_length,
            "policy_w_egfr": float(weights[0]),
            "policy_w_vegfr2": float(weights[1]),
            "next_w_egfr": float(self.policy_weights[0]),
            "next_w_vegfr2": float(self.policy_weights[1]),
            "controller_raw_w_egfr": float(raw_next_weights[0]),
            "controller_raw_w_vegfr2": float(raw_next_weights[1]),
            "pareto_reward_strength": float(pareto_reward_strength),
            "archive_seed_fraction_effective": effective_archive_fraction,
            "archive_seed_noise_effective": effective_archive_noise,
            "archive_stagnation_triggered": archive_stagnation_triggered,
            "archive_recent_hv_gain": archive_recent_hv_gain,
            "preference_blend_effective": effective_preference_blend,
            "generator_finetuned": generator_finetuned,
            "generator_finetune_count": self.generator_finetune_count,
            "generator_elite_archive_size": len(self.generator_elite_archive),
            "stage": stage_name(epoch, self.total_epochs),
        }
        return valid_molecules, float(loss), info, generated_rows


def add_sa_to_rows(rows: List[Dict], scorer) -> None:
    for row in rows:
        if not row.get("is_valid"):
            continue
        smiles = str(row.get("canonical_smiles") or row.get("smiles") or "")
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            row["sa"] = sa_score(scorer, smiles, mol)


def summarize_quality(rows: List[Dict], train_set: set) -> Dict:
    valid = [r for r in rows if bool(r.get("is_valid")) and r.get("canonical_smiles")]
    unique: Dict[str, Dict] = {}
    for row in valid:
        unique.setdefault(str(row["canonical_smiles"]), row)
    novel = [row for smi, row in unique.items() if smi not in train_set]
    qed_vals = [float(r["qed"]) for r in novel if r.get("qed") != "" and not pd.isna(r["qed"])]
    sa_vals = [float(r["sa"]) for r in novel if r.get("sa") != "" and not pd.isna(r["sa"])]
    return {
        "generated_rows": len(rows),
        "valid_rows": len(valid),
        "invalid_rows": len(rows) - len(valid),
        "validity": len(valid) / max(len(rows), 1),
        "unique_valid_smiles": len(unique),
        "uniqueness": len(unique) / max(len(valid), 1),
        "novel_unique_molecules": len(novel),
        "novelty": len(novel) / max(len(unique), 1),
        "qed_mean": float(np.mean(qed_vals)) if qed_vals else 0.0,
        "sa_mean": float(np.mean(sa_vals)) if sa_vals else 0.0,
    }


def plot_outputs(out_dir: Path) -> None:
    hv = pd.read_csv(out_dir / "hv_history.csv")
    metrics = pd.read_csv(out_dir / "metrics.csv")
    channel = pd.read_csv(out_dir / "channel_metrics.csv")
    pareto = pd.read_csv(out_dir / "pareto_front.csv")
    all_rows = pd.read_csv(out_dir / "all_generated_molecules.csv")
    valid = all_rows[all_rows["is_valid"].astype(str).str.lower().eq("true")].copy()

    plt.figure(figsize=(8, 4.8))
    plt.plot(hv["epoch"], hv["hv"], linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel(f"Normalized hypervolume ({ACTIVITY_LOWER:g}-{ACTIVITY_UPPER:g}, ref=0,0)")
    plt.title("Multi-explore HV trajectory")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_hv_curve.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7, 5.2))
    if len(valid):
        plt.scatter(valid["egfr"], valid["vegfr2"], s=13, alpha=0.25, c="#9CA3AF", label="All valid")
    if len(pareto):
        plt.scatter(pareto["egfr"], pareto["vegfr2"], s=70, c="#DC2626", edgecolors="black", label="Pareto")
    plt.xlabel("EGFR predicted activity")
    plt.ylabel("VEGFR2 predicted activity")
    plt.title("Score space and Pareto front")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig_pareto_front.png", dpi=300)
    plt.close()

    final_channel = channel.groupby("channel", as_index=False).agg(
        valid=("valid_count", "sum"),
        pareto_hits=("pareto_hits", "sum"),
        egfr_mean=("egfr_mean", "mean"),
        vegfr2_mean=("vegfr2_mean", "mean"),
        balanced_max=("balanced_max", "max"),
    )
    plt.figure(figsize=(9, 4.8))
    x = np.arange(len(final_channel))
    plt.bar(x - 0.18, final_channel["valid"], 0.36, label="Valid")
    plt.bar(x + 0.18, final_channel["pareto_hits"], 0.36, label="Pareto hits")
    plt.xticks(x, final_channel["channel"], rotation=25, ha="right")
    plt.ylabel("Count")
    plt.title("Channel contribution")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig_channel_contribution.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 4.8))
    plt.plot(metrics["epoch"], metrics["valid_rate"], label="valid rate", linewidth=2)
    plt.plot(metrics["epoch"], metrics["loss"], label="PPO loss", linewidth=1.3, alpha=0.8)
    plt.xlabel("Epoch")
    plt.title("Training metrics")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig_training_metrics.png", dpi=300)
    plt.close()


def run(args) -> RunResult:
    start = time.time()
    if args.archive_seed_noise_end is None:
        args.archive_seed_noise_end = args.archive_seed_noise
    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    train_set = load_train_set(args.train_cache)
    scorer = make_sa_scorer(args.fscores)

    if args.oracle_budget is not None:
        if args.oracle_budget <= 0:
            raise ValueError("--oracle-budget must be positive")
        if args.oracle_budget % args.batch != 0:
            raise ValueError(
                f"--oracle-budget ({args.oracle_budget}) must be divisible by --batch ({args.batch})"
            )
        args.epochs = args.oracle_budget // args.batch

    protocol = load_protocol(args.protocol_config)
    enforce_registered_protocol(args, protocol)

    config = {
        "vae_model_path": str(Path(args.model).resolve()),
        "egfr_model_path": args.egfr_model or find_target_model("target_EGFR_model.pkl"),
        "vegfr2_model_path": args.vegfr2_model or find_target_model("target_VEGFR2_model.pkl"),
        "total_epochs": args.epochs,
        "batch_size": args.batch,
        "lr": args.lr,
        "ppo_epochs": args.ppo_epochs,
        "mini_batch_size": args.mini_batch_size,
        "entropy_coef": args.entropy_coef,
        "value_loss_coef": args.value_loss_coef,
        "device": args.device,
        "base_step_scale": args.base_step_scale,
        "trajectory_length": args.trajectory_length,
        "trajectory_step_normalization": args.trajectory_step_normalization,
        "archive_seed_fraction": args.archive_seed_fraction,
        "archive_seed_noise": args.archive_seed_noise,
        "archive_seed_noise_end": args.archive_seed_noise_end,
        "archive_seed_start": args.archive_seed_start,
        "archive_seed_ramp_end": args.archive_seed_ramp_end,
        "archive_seed_selection": args.archive_seed_selection,
        "archive_hvc_weight": args.archive_hvc_weight,
        "archive_balance_weight": args.archive_balance_weight,
        "archive_selection_temperature": args.archive_selection_temperature,
        "archive_uniform_mix": args.archive_uniform_mix,
        "archive_stagnation_window": args.archive_stagnation_window,
        "archive_stagnation_delta": args.archive_stagnation_delta,
        "archive_stagnation_noise": args.archive_stagnation_noise,
        "latent_clip": args.latent_clip,
        "invalid_reward": args.invalid_reward,
        "preference_floor": args.preference_floor,
        "preference_ema_alpha": args.preference_ema_alpha,
        "weight_mode": args.weight_mode,
        "dirichlet_alpha": args.dirichlet_alpha,
        "dynamic_weights": args.weight_mode == "dynamic",
        "adaptive_channels": args.channel_mode == "adaptive",
        "exploration_mode": args.exploration_mode,
        "critic_mode": args.critic_mode,
        "controller_variant": args.controller_variant,
        "hvc_reward_weight": args.hvc_reward_weight,
        "crowding_reward_weight": args.crowding_reward_weight,
        "balanced_reward_weight": args.balanced_reward_weight,
        "pareto_actor_coef": args.pareto_actor_coef,
        "actor_mode": args.actor_mode,
        "generator_finetune_interval": args.generator_finetune_interval,
        "generator_finetune_epochs": args.generator_finetune_epochs,
        "generator_finetune_top": args.generator_finetune_top,
        "generator_finetune_batch_size": args.generator_finetune_batch_size,
        "generator_finetune_lr": args.generator_finetune_lr,
        "generator_elite_strategy": args.generator_elite_strategy,
        "generator_balance_mix": args.generator_balance_mix,
        "generator_softmin_temperature": args.generator_softmin_temperature,
        "advantage_normalization": "per_objective_then_scalarized",
        "sample_preference_mode": args.sample_preference_mode,
        "sample_preference_blend": args.sample_preference_blend,
        "sample_preference_start": args.sample_preference_start,
        "sample_preference_ramp_end": args.sample_preference_ramp_end,
        "pareto_reward_start": args.pareto_reward_start,
        "pareto_reward_ramp_end": args.pareto_reward_ramp_end,
        "oracle_budget": args.epochs * args.batch,
        "oracle_system": args.oracle_system,
    }
    asset_paths = {
        "protocol_config": Path(args.protocol_config).resolve(),
        "train_smiles": PROJECT_ROOT / "data" / "train_smiles_only.txt",
        "method_runner": Path(__file__).resolve(),
        "weight_controller": PROJECT_ROOT / "method" / "ablation" / "weight_controllers.py",
        "trajectory_agent": PROJECT_ROOT / "method" / "ablation" / "run_wc_ablation_two_targets.py",
        "multi_critic_agent": PROJECT_ROOT / "method" / "ablation" / "multi_critic_ppo_agent.py",
        "vae_model": Path(config["vae_model_path"]),
        "train_cache": Path(args.train_cache).resolve(),
    }
    if args.oracle_system == "original_rf":
        asset_paths.update({
            "egfr_oracle": Path(config["egfr_model_path"]),
            "vegfr2_oracle": Path(config["vegfr2_model_path"]),
        })
        oracle_contract = "raw RF predictions; no molecular-quality penalty"
    else:
        v41_root = PROJECT_ROOT / "results" / "predictor_v41_20260802"
        asset_paths.update({
            "v41_egfr_reference": PROJECT_ROOT / "results" / "predictor_retraining_v3_20260731" / "data" / "egfr" / "single_protein_assay_ge10" / "development_through_2023.csv",
            "v41_vegfr2_model": v41_root / "deployment" / "vegfr2_extratrees.pkl",
            "v41_vegfr2_metadata": v41_root / "deployment" / "metadata.json",
        })
        for variant in ("dmpnn", "dmpnn_morgan"):
            for index, path in enumerate(sorted((v41_root / "egfr_bindingdb_external_v2" / variant).rglob("best.pt"))):
                asset_paths[f"v41_egfr_{variant}_{index}"] = path
        oracle_contract = (
            "V4.1 EGFR 0.7*D-MPNN + 0.1*Morgan-DMPNN + 0.2*KNN and "
            "VEGFR2 ExtraTrees probabilities; reward=3+7*p; no quality penalty"
        )
    resolved = {
        "protocol": protocol,
        "arguments": vars(args),
        "runtime_config": config,
        "activity_hv": {
            "type": "linear_clipped",
            "lower_pactivity": ACTIVITY_LOWER,
            "upper_pactivity": ACTIVITY_UPPER,
            "reference": [0.0, 0.0],
        },
        "oracle_contract": oracle_contract,
        "canonicalization": "largest-fragment canonical isomeric SMILES",
        "assets": {
            name: {
                "path": str(path),
                "sha256": file_sha256(path),
            }
            for name, path in asset_paths.items()
        },
    }
    (out_dir / "resolved_config.json").write_text(
        json.dumps(resolved, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    integrator = MultiExploreIntegrator(config)

    all_rows: List[Dict] = []
    hv_rows: List[Dict] = []
    metric_rows: List[Dict] = []
    channel_rows: List[Dict] = []
    fields = [
        "epoch", "sample_idx", "latent_source", "channel", "temperature", "step_multiplier",
        "trajectory_length", "actor_mode", "trajectory_steps", "effective_step_scale", "path_length",
        "net_displacement", "smiles", "canonical_smiles", "is_valid",
        "is_pareto_current", "egfr", "vegfr2", "balanced", "min_score",
        "qed", "sa", "mol_wt", "logp", "tpsa", "heavy_atoms",
        "hv_contribution", "hvc_rank_reward", "crowding_reward", "balanced_rank_reward", "auxiliary_reward",
        "pareto_reward_strength", "archive_seed_fraction_effective",
        "archive_seed_noise_effective", "archive_stagnation_triggered",
        "archive_recent_hv_gain", "preference_blend_effective",
        "pref_egfr", "pref_vegfr2", "error",
    ]

    print("Two-target multi-scale trajectory PPO")
    print(f"Output: {out_dir}")
    print(f"Model: {config['vae_model_path']}")
    print(f"Oracle system: {args.oracle_system}")
    print(
        f"Batch={args.batch}, epochs={args.epochs}, oracle_budget={args.epochs * args.batch}, "
        f"mini_batch={args.mini_batch_size}, trajectory_K={args.trajectory_length}, "
        f"step_normalization={args.trajectory_step_normalization}"
    )
    print(
        f"Critic={args.critic_mode}, weights={args.weight_mode}, channels={args.channel_mode}, "
        f"exploration={args.exploration_mode}, preferences={args.sample_preference_mode}, "
        f"HVC={args.hvc_reward_weight}, crowd={args.crowding_reward_weight}, "
        f"balanced={args.balanced_reward_weight}"
    )

    for epoch in range(args.epochs):
        molecules, loss, info, rows = integrator.run_episode(epoch)
        add_sa_to_rows(rows, scorer)
        all_rows.extend(rows)

        pareto_scores = np.array([m.scores for m in integrator.pareto_front.molecules])
        hv = hypervolume_2d(pareto_scores)
        pareto_smiles = {m.smiles for m in integrator.pareto_front.molecules}
        for row in rows:
            row["is_pareto_current"] = row.get("canonical_smiles") in pareto_smiles

        hv_rows.append({"epoch": epoch + 1, "hv": hv, "pareto_size": len(pareto_smiles)})
        metric_rows.append(
            {
                "epoch": epoch + 1,
                "stage": info["stage"],
                "loss": loss,
                "valid_count": info["valid_count"],
                "invalid_count": info["invalid_count"],
                "policy_transitions": info["policy_transitions"],
                "trajectory_length": info["trajectory_length"],
                "valid_rate": info["valid_count"] / max(args.batch, 1),
                "hv": hv,
                "pareto_size": len(pareto_smiles),
                "w_egfr": info["policy_w_egfr"],
                "w_vegfr2": info["policy_w_vegfr2"],
                "next_w_egfr": info["next_w_egfr"],
                "next_w_vegfr2": info["next_w_vegfr2"],
                "controller_raw_w_egfr": info["controller_raw_w_egfr"],
                "controller_raw_w_vegfr2": info["controller_raw_w_vegfr2"],
                "pareto_reward_strength": info["pareto_reward_strength"],
                "archive_seed_fraction_effective": info["archive_seed_fraction_effective"],
                "archive_seed_noise_effective": info["archive_seed_noise_effective"],
                "archive_stagnation_triggered": info["archive_stagnation_triggered"],
                "archive_recent_hv_gain": info["archive_recent_hv_gain"],
                "preference_blend_effective": info["preference_blend_effective"],
                "generator_finetuned": info["generator_finetuned"],
                "generator_finetune_count": info["generator_finetune_count"],
                "generator_elite_archive_size": info["generator_elite_archive_size"],
            }
        )

        epoch_df = pd.DataFrame(rows)
        for channel, group in epoch_df.groupby("channel"):
            valid = group[group["is_valid"].astype(bool)] if len(group) else group
            channel_rows.append(
                {
                    "epoch": epoch + 1,
                    "stage": info["stage"],
                    "channel": channel,
                    "samples": len(group),
                    "valid_count": len(valid),
                    "valid_rate": len(valid) / max(len(group), 1),
                    "pareto_hits": int(group.get("is_pareto_current", pd.Series(dtype=bool)).astype(bool).sum()),
                    "egfr_mean": float(pd.to_numeric(valid.get("egfr"), errors="coerce").mean()) if len(valid) else 0.0,
                    "vegfr2_mean": float(pd.to_numeric(valid.get("vegfr2"), errors="coerce").mean()) if len(valid) else 0.0,
                    "balanced_max": float(pd.to_numeric(valid.get("balanced"), errors="coerce").max()) if len(valid) else 0.0,
                    "hvc_sum": float(pd.to_numeric(group.get("hv_contribution"), errors="coerce").fillna(0).sum()),
                }
            )

        if (epoch + 1) % args.log_interval == 0 or epoch == 0:
            print(
                f"epoch {epoch + 1:03d}/{args.epochs} stage={info['stage']} "
                f"HV={hv:.4f} Pareto={len(pareto_smiles)} "
                f"valid={info['valid_count']}/{args.batch} loss={loss:.4f}",
                flush=True,
            )

        if args.checkpoint_interval > 0 and (
            (epoch + 1) % args.checkpoint_interval == 0 or epoch + 1 == args.epochs
        ):
            write_csv(out_dir / "all_generated_molecules.partial.csv", all_rows, fields)
            write_csv(out_dir / "hv_history.partial.csv", hv_rows, ["epoch", "hv", "pareto_size"])
            write_csv(
                out_dir / "metrics.partial.csv",
                metric_rows,
                ["epoch", "stage", "loss", "valid_count", "invalid_count", "policy_transitions", "trajectory_length", "valid_rate", "hv", "pareto_size", "w_egfr", "w_vegfr2", "next_w_egfr", "next_w_vegfr2", "controller_raw_w_egfr", "controller_raw_w_vegfr2", "pareto_reward_strength", "archive_seed_fraction_effective", "archive_seed_noise_effective", "archive_stagnation_triggered", "archive_recent_hv_gain", "preference_blend_effective", "generator_finetuned", "generator_finetune_count", "generator_elite_archive_size"],
            )
            write_csv(
                out_dir / "channel_metrics.partial.csv",
                channel_rows,
                ["epoch", "stage", "channel", "samples", "valid_count", "valid_rate", "pareto_hits", "egfr_mean", "vegfr2_mean", "balanced_max", "hvc_sum"],
            )
            print(f"checkpoint saved at epoch {epoch + 1}", flush=True)

    pareto_rows = [
        {"smiles": m.smiles, "egfr": float(m.scores[0]), "vegfr2": float(m.scores[1])}
        for m in integrator.pareto_front.molecules
    ]
    for r in pareto_rows:
        r["balanced"] = (r["egfr"] + r["vegfr2"]) / 2
        r["min_score"] = min(r["egfr"], r["vegfr2"])
    top_rows = sorted(
        [r for r in all_rows if bool(r.get("is_valid"))],
        key=lambda x: (float(x["balanced"]), float(x["min_score"])),
        reverse=True,
    )[:50]

    write_csv(out_dir / "all_generated_molecules.csv", all_rows, fields)
    write_csv(out_dir / "hv_history.csv", hv_rows, ["epoch", "hv", "pareto_size"])
    write_csv(out_dir / "metrics.csv", metric_rows, ["epoch", "stage", "loss", "valid_count", "invalid_count", "policy_transitions", "trajectory_length", "valid_rate", "hv", "pareto_size", "w_egfr", "w_vegfr2", "next_w_egfr", "next_w_vegfr2", "controller_raw_w_egfr", "controller_raw_w_vegfr2", "pareto_reward_strength", "archive_seed_fraction_effective", "archive_seed_noise_effective", "archive_stagnation_triggered", "archive_recent_hv_gain", "preference_blend_effective", "generator_finetuned", "generator_finetune_count", "generator_elite_archive_size"])
    write_csv(out_dir / "channel_metrics.csv", channel_rows, ["epoch", "stage", "channel", "samples", "valid_count", "valid_rate", "pareto_hits", "egfr_mean", "vegfr2_mean", "balanced_max", "hvc_sum"])
    write_csv(out_dir / "pareto_front.csv", pareto_rows, ["smiles", "egfr", "vegfr2", "balanced", "min_score"])
    write_csv(out_dir / "top_molecules.csv", top_rows, fields)

    quality = summarize_quality(all_rows, train_set)
    if pareto_rows:
        egfr = np.array([r["egfr"] for r in pareto_rows], dtype=np.float32)
        vegfr2 = np.array([r["vegfr2"] for r in pareto_rows], dtype=np.float32)
        balanced = np.array([r["balanced"] for r in pareto_rows], dtype=np.float32)
    else:
        egfr = vegfr2 = balanced = np.zeros(1, dtype=np.float32)

    result = RunResult(
        hv_final=float(hv_rows[-1]["hv"]) if hv_rows else 0.0,
        pareto_size=len(pareto_rows),
        egfr_max=float(egfr.max()),
        vegfr2_max=float(vegfr2.max()),
        egfr_mean=float(egfr.mean()),
        vegfr2_mean=float(vegfr2.mean()),
        best_balanced_score=float(balanced.max()),
        generated_rows=quality["generated_rows"],
        valid_rows=quality["valid_rows"],
        invalid_rows=quality["invalid_rows"],
        valid_rate=quality["validity"],
        unique_valid_smiles=quality["unique_valid_smiles"],
        novelty=quality["novelty"],
        novel_unique_molecules=quality["novel_unique_molecules"],
        qed_mean=quality["qed_mean"],
        sa_mean=quality["sa_mean"],
        runtime_sec=time.time() - start,
    )
    summary = result.__dict__
    summary["oracle_budget"] = args.epochs * args.batch
    summary["oracle_system"] = args.oracle_system
    summary["trajectory_length"] = args.trajectory_length
    summary["trajectory_step_normalization"] = args.trajectory_step_normalization
    summary["policy_transition_budget"] = args.epochs * args.batch * args.trajectory_length
    summary["controller_variant"] = args.controller_variant
    summary["actor_mode"] = args.actor_mode
    summary["generator_finetune_interval"] = args.generator_finetune_interval
    summary["generator_finetune_epochs"] = args.generator_finetune_epochs
    summary["generator_finetune_top"] = args.generator_finetune_top
    summary["generator_finetune_batch_size"] = args.generator_finetune_batch_size
    summary["generator_finetune_lr"] = args.generator_finetune_lr
    summary["generator_elite_strategy"] = args.generator_elite_strategy
    summary["generator_balance_mix"] = args.generator_balance_mix
    summary["generator_softmin_temperature"] = args.generator_softmin_temperature
    summary["advantage_normalization"] = "per_objective_then_scalarized"
    summary["generator_finetune_count"] = integrator.generator_finetune_count
    summary["hv_definition"] = (
        "v41_probability_hv_via_reward_3_plus_7p"
        if args.oracle_system == "v41"
        else f"linear_clipped_pactivity_{ACTIVITY_LOWER:g}_{ACTIVITY_UPPER:g}"
    )
    summary["hvc_reward_weight"] = args.hvc_reward_weight
    summary["crowding_reward_weight"] = args.crowding_reward_weight
    summary["balanced_reward_weight"] = args.balanced_reward_weight
    summary["pareto_actor_coef"] = args.pareto_actor_coef
    summary["sample_preference_mode"] = args.sample_preference_mode
    summary["sample_preference_blend"] = args.sample_preference_blend
    summary["sample_preference_start"] = args.sample_preference_start
    summary["sample_preference_ramp_end"] = args.sample_preference_ramp_end
    summary["archive_seed_fraction"] = args.archive_seed_fraction
    summary["archive_seed_noise"] = args.archive_seed_noise
    summary["archive_seed_noise_end"] = args.archive_seed_noise_end
    summary["archive_seed_start"] = args.archive_seed_start
    summary["archive_seed_ramp_end"] = args.archive_seed_ramp_end
    summary["archive_seed_selection"] = args.archive_seed_selection
    summary["archive_hvc_weight"] = args.archive_hvc_weight
    summary["archive_balance_weight"] = args.archive_balance_weight
    summary["archive_selection_temperature"] = args.archive_selection_temperature
    summary["archive_uniform_mix"] = args.archive_uniform_mix
    summary["archive_stagnation_window"] = args.archive_stagnation_window
    summary["archive_stagnation_delta"] = args.archive_stagnation_delta
    summary["archive_stagnation_noise"] = args.archive_stagnation_noise
    summary["pareto_reward_start"] = args.pareto_reward_start
    summary["pareto_reward_ramp_end"] = args.pareto_reward_ramp_end
    pd.DataFrame([summary]).to_csv(out_dir / "summary.csv", index=False, encoding="utf-8-sig")
    if integrator.generator_finetune_count > 0:
        torch.save(
            integrator.vae.model.state_dict(),
            out_dir / "generator_finetuned.pt",
        )

    lines = [
        "# Two-target multi-scale trajectory PPO summary",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| generated rows | {result.generated_rows} |",
        f"| oracle calls | {args.epochs * args.batch} |",
        f"| trajectory length K | {args.trajectory_length} |",
        f"| policy transitions | {args.epochs * args.batch * args.trajectory_length} |",
        f"| valid rows | {result.valid_rows} |",
        f"| validity | {result.valid_rate * 100:.3f}% |",
        f"| unique valid SMILES | {result.unique_valid_smiles} |",
        f"| novelty | {result.novelty * 100:.3f}% |",
        f"| novel unique molecules | {result.novel_unique_molecules} |",
        f"| final HV | {result.hv_final:.4f} |",
        f"| HV normalization | pActivity {ACTIVITY_LOWER:g}-{ACTIVITY_UPPER:g} to [0,1] |",
        f"| Pareto size | {result.pareto_size} |",
        f"| EGFR max | {result.egfr_max:.4f} |",
        f"| VEGFR2 max | {result.vegfr2_max:.4f} |",
        f"| EGFR mean | {result.egfr_mean:.4f} |",
        f"| VEGFR2 mean | {result.vegfr2_mean:.4f} |",
        f"| best balanced score | {result.best_balanced_score:.4f} |",
        f"| QED mean | {result.qed_mean:.3f} |",
        f"| SA mean | {result.sa_mean:.3f} |",
        f"| runtime min | {result.runtime_sec / 60:.2f} |",
        "",
        "## Outputs",
        "",
        "- all_generated_molecules.csv",
        "- summary.csv",
        "- resolved_config.json",
        "- hv_history.csv",
        "- metrics.csv",
        "- channel_metrics.csv",
        "- pareto_front.csv",
        "- top_molecules.csv",
        "- fig_hv_curve.png",
        "- fig_pareto_front.png",
        "- fig_channel_contribution.png",
        "- fig_training_metrics.png",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    plot_outputs(out_dir)
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Two-target PPO with multi-scale channels and true K-step latent trajectories."
    )
    parser.add_argument("--model", default=str(DEFAULT_VAE_MODEL))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "logs" / "wc2_multiexplore_b64_100"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--oracle-budget",
        type=int,
        default=None,
        help="Exact number of generated/oracle-evaluated rows; must be divisible by --batch.",
    )
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--mini-batch-size", type=int, default=32)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-loss-coef", type=float, default=0.5)
    parser.add_argument("--base-step-scale", type=float, default=0.08)
    parser.add_argument(
        "--trajectory-length",
        type=int,
        default=1,
        help="Number of sequential latent-policy decisions before one terminal decode/oracle call.",
    )
    parser.add_argument(
        "--trajectory-step-normalization",
        choices=["none", "sqrt", "linear"],
        default="sqrt",
        help="Normalize each latent step across K; sqrt is the fair K=1/3/5 default.",
    )
    parser.add_argument("--archive-seed-fraction", type=float, default=0.0)
    parser.add_argument("--archive-seed-noise", type=float, default=0.15)
    parser.add_argument(
        "--archive-seed-noise-end",
        type=float,
        default=None,
        help="Final archive perturbation scale; defaults to --archive-seed-noise.",
    )
    parser.add_argument("--archive-seed-start", type=float, default=0.0)
    parser.add_argument("--archive-seed-ramp-end", type=float, default=0.0)
    parser.add_argument(
        "--archive-seed-selection",
        choices=["uniform", "hvc", "hvc_balanced"],
        default="uniform",
    )
    parser.add_argument("--archive-hvc-weight", type=float, default=0.7)
    parser.add_argument("--archive-balance-weight", type=float, default=0.3)
    parser.add_argument("--archive-selection-temperature", type=float, default=0.25)
    parser.add_argument("--archive-uniform-mix", type=float, default=0.10)
    parser.add_argument("--archive-stagnation-window", type=int, default=0)
    parser.add_argument("--archive-stagnation-delta", type=float, default=0.002)
    parser.add_argument("--archive-stagnation-noise", type=float, default=0.0)
    parser.add_argument("--latent-clip", type=float, default=4.0)
    parser.add_argument("--invalid-reward", type=float, default=-1.0)
    parser.add_argument("--preference-floor", type=float, default=0.10)
    parser.add_argument("--preference-ema-alpha", type=float, default=0.25)
    parser.add_argument("--hvc-reward-weight", type=float, default=0.0)
    parser.add_argument("--crowding-reward-weight", type=float, default=0.0)
    parser.add_argument("--balanced-reward-weight", type=float, default=0.0)
    parser.add_argument("--pareto-actor-coef", type=float, default=0.0)
    parser.add_argument(
        "--actor-mode", choices=["train", "frozen", "zero"], default="train",
        help="Train PPO normally, hold its initialized actor fixed, or apply zero latent displacement.",
    )
    parser.add_argument("--generator-finetune-interval", type=int, default=0)
    parser.add_argument("--generator-finetune-epochs", type=int, default=0)
    parser.add_argument("--generator-finetune-top", type=int, default=512)
    parser.add_argument("--generator-finetune-batch-size", type=int, default=32)
    parser.add_argument("--generator-finetune-lr", type=float, default=3e-4)
    parser.add_argument(
        "--generator-elite-strategy",
        choices=[
            "mean", "min", "mixed", "raw_mean", "raw_min", "raw_mixed",
            "balance_sync",
        ],
        default="mean",
    )
    parser.add_argument(
        "--generator-balance-mix",
        type=float,
        default=0.5,
        help="V5 mixture weight on soft worst-target desirability.",
    )
    parser.add_argument(
        "--generator-softmin-temperature",
        type=float,
        default=0.1,
        help="V5 smooth maximum temperature for threshold deficits.",
    )
    parser.add_argument(
        "--sample-preference-mode", choices=["shared", "grid"], default="shared"
    )
    parser.add_argument("--sample-preference-blend", type=float, default=0.75)
    parser.add_argument("--sample-preference-start", type=float, default=0.0)
    parser.add_argument("--sample-preference-ramp-end", type=float, default=0.0)
    parser.add_argument("--pareto-reward-start", type=float, default=0.0)
    parser.add_argument("--pareto-reward-ramp-end", type=float, default=0.0)
    parser.add_argument("--weight-mode", choices=["dynamic", "fixed", "dirichlet"], default="dynamic")
    parser.add_argument("--dirichlet-alpha", type=float, default=0.5)
    parser.add_argument("--critic-mode", choices=["single", "multi"], default="multi")
    parser.add_argument(
        "--controller-variant",
        choices=["ours_full", "ours_full_corrected"],
        default=DEFAULT_CONTROLLER_VARIANT,
    )
    parser.add_argument("--protocol-config", default=str(DEFAULT_PROTOCOL_CONFIG))
    parser.add_argument("--channel-mode", choices=["adaptive", "fixed"], default="adaptive")
    parser.add_argument(
        "--exploration-mode",
        choices=sorted(CHANNEL_PRESETS),
        default="multiscale",
        help="Enable single-scale, temperature-only, step-only, or full multiscale channels.",
    )
    parser.add_argument("--egfr-model", default=None)
    parser.add_argument("--vegfr2-model", default=None)
    parser.add_argument(
        "--oracle-system",
        choices=["original_rf", "v41"],
        default="original_rf",
        help="Target predictor pair used online as the generation reward.",
    )
    parser.add_argument("--train-cache", default=str(PROJECT_ROOT / "data" / "train_canonical_cache.txt"))
    parser.add_argument("--fscores", default=str(PROJECT_ROOT / "vendor" / "polygon-main" / "data" / "fpscores.pkl.gz"))
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--checkpoint-interval", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    if not Path(args.model).exists():
        raise FileNotFoundError(args.model)
    if args.trajectory_length < 1:
        raise ValueError("--trajectory-length must be at least 1")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    result = run(args)
    print("\nFinished.")
    print(f"Final HV: {result.hv_final:.4f}")
    print(f"Summary: {Path(args.output).resolve() / 'summary.md'}")


if __name__ == "__main__":
    main()
