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
except Exception:
    SAScorer = None


REF_POINT = np.array([0.0, 0.0], dtype=np.float32)
DEFAULT_VAE_MODEL = PROJECT_ROOT / "models" / "polygon_vae_best_valid_novel_stable_020.pt"
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
    return Chem.MolToSmiles(mol, canonical=True), mol


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
        super().__init__(
            latent_dim=config.get("latent_dim", 128),
            num_obj=2,
            total_epochs=config.get("total_epochs", 100),
            batch_size=config.get("batch_size", 64),
            vae_model_path=config.get("vae_model_path"),
            egfr_model_path=config.get("egfr_model_path"),
            vegfr2_model_path=config.get("vegfr2_model_path"),
            config=config,
        )
        self.objective_calculator = TwoTargetObjectiveCalculator(
            config.get("egfr_model_path"),
            config.get("vegfr2_model_path"),
        )
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
            config.get("controller_variant", "ours_full"),
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
        self.pareto_reward_start = float(config.get("pareto_reward_start", 0.0))
        self.pareto_reward_ramp_end = float(config.get("pareto_reward_ramp_end", 0.0))
        self.agent.auxiliary_actor_coef = float(config.get("pareto_actor_coef", 0.0))

    def _normalize_weights(self, weights: np.ndarray) -> np.ndarray:
        weights = np.asarray(weights, dtype=np.float32)
        weights = np.clip(weights, 0.0, None)
        if weights.sum() <= 0:
            weights = np.ones_like(weights) / len(weights)
        else:
            weights = weights / weights.sum()
        floor = min(self.preference_floor, (1.0 - 1e-6) / len(weights))
        return floor + (1.0 - floor * len(weights)) * weights

    def _build_sample_preferences(
        self, count: int, base_weights: np.ndarray, epoch: int
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
        blend = float(np.clip(self.sample_preference_blend, 0.0, 1.0))
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
        sample_preferences = self._build_sample_preferences(len(channels), weights, epoch)
        progress = (epoch + 1) / max(self.total_epochs, 1)
        if progress <= self.pareto_reward_start:
            pareto_reward_strength = 0.0
        elif self.pareto_reward_ramp_end <= self.pareto_reward_start or progress >= self.pareto_reward_ramp_end:
            pareto_reward_strength = 1.0
        else:
            pareto_reward_strength = (progress - self.pareto_reward_start) / (
                self.pareto_reward_ramp_end - self.pareto_reward_start
            )
        z_states = np.random.normal(0, 1, (len(channels), self.latent_dim)).astype(np.float32)
        latent_sources = np.full(len(channels), "global_prior", dtype=object)
        if self.pareto_front.molecules and self.archive_seed_fraction > 0:
            n_archive = min(
                len(channels),
                int(round(len(channels) * np.clip(self.archive_seed_fraction, 0.0, 1.0))),
            )
            archive_latents = np.asarray(
                [m.latent_vector for m in self.pareto_front.molecules], dtype=np.float32
            )
            chosen = np.random.randint(0, len(archive_latents), size=n_archive)
            z_states[:n_archive] = archive_latents[chosen] + np.random.normal(
                0.0, self.archive_seed_noise, size=(n_archive, self.latent_dim)
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
            self.policy_weights = np.ones(2, dtype=np.float32) / 2.0
        if archive_batch:
            self.controller.update_pareto_front(archive_batch)

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
        if len(self.agent.buffer.states) >= self.agent.mini_batch_size:
            loss = self.agent.update(self.agent.buffer)
            self.agent.buffer.clear()

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
    plt.ylabel("Hypervolume (ref=0,0)")
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
        "sample_preference_mode": args.sample_preference_mode,
        "sample_preference_blend": args.sample_preference_blend,
        "pareto_reward_start": args.pareto_reward_start,
        "pareto_reward_ramp_end": args.pareto_reward_ramp_end,
        "oracle_budget": args.epochs * args.batch,
    }
    integrator = MultiExploreIntegrator(config)

    all_rows: List[Dict] = []
    hv_rows: List[Dict] = []
    metric_rows: List[Dict] = []
    channel_rows: List[Dict] = []
    fields = [
        "epoch", "sample_idx", "latent_source", "channel", "temperature", "step_multiplier",
        "trajectory_length", "trajectory_steps", "effective_step_scale", "path_length",
        "net_displacement", "smiles", "canonical_smiles", "is_valid",
        "is_pareto_current", "egfr", "vegfr2", "balanced", "min_score",
        "qed", "sa", "mol_wt", "logp", "tpsa", "heavy_atoms",
        "hv_contribution", "hvc_rank_reward", "crowding_reward", "balanced_rank_reward", "auxiliary_reward",
        "pareto_reward_strength", "pref_egfr", "pref_vegfr2", "error",
    ]

    print("Two-target multi-scale trajectory PPO")
    print(f"Output: {out_dir}")
    print(f"Model: {config['vae_model_path']}")
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
                ["epoch", "stage", "loss", "valid_count", "invalid_count", "policy_transitions", "trajectory_length", "valid_rate", "hv", "pareto_size", "w_egfr", "w_vegfr2", "next_w_egfr", "next_w_vegfr2"],
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
    write_csv(out_dir / "metrics.csv", metric_rows, ["epoch", "stage", "loss", "valid_count", "invalid_count", "policy_transitions", "trajectory_length", "valid_rate", "hv", "pareto_size", "w_egfr", "w_vegfr2", "next_w_egfr", "next_w_vegfr2"])
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
    summary["trajectory_length"] = args.trajectory_length
    summary["trajectory_step_normalization"] = args.trajectory_step_normalization
    summary["policy_transition_budget"] = args.epochs * args.batch * args.trajectory_length
    summary["controller_variant"] = args.controller_variant
    summary["hvc_reward_weight"] = args.hvc_reward_weight
    summary["crowding_reward_weight"] = args.crowding_reward_weight
    summary["balanced_reward_weight"] = args.balanced_reward_weight
    summary["pareto_actor_coef"] = args.pareto_actor_coef
    summary["sample_preference_mode"] = args.sample_preference_mode
    summary["sample_preference_blend"] = args.sample_preference_blend
    summary["pareto_reward_start"] = args.pareto_reward_start
    summary["pareto_reward_ramp_end"] = args.pareto_reward_ramp_end
    pd.DataFrame([summary]).to_csv(out_dir / "summary.csv", index=False, encoding="utf-8-sig")

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
    parser.add_argument("--latent-clip", type=float, default=4.0)
    parser.add_argument("--invalid-reward", type=float, default=-1.0)
    parser.add_argument("--preference-floor", type=float, default=0.10)
    parser.add_argument("--preference-ema-alpha", type=float, default=0.25)
    parser.add_argument("--hvc-reward-weight", type=float, default=0.0)
    parser.add_argument("--crowding-reward-weight", type=float, default=0.0)
    parser.add_argument("--balanced-reward-weight", type=float, default=0.0)
    parser.add_argument("--pareto-actor-coef", type=float, default=0.0)
    parser.add_argument(
        "--sample-preference-mode", choices=["shared", "grid"], default="shared"
    )
    parser.add_argument("--sample-preference-blend", type=float, default=0.75)
    parser.add_argument("--pareto-reward-start", type=float, default=0.0)
    parser.add_argument("--pareto-reward-ramp-end", type=float, default=0.0)
    parser.add_argument("--weight-mode", choices=["dynamic", "fixed", "dirichlet"], default="dynamic")
    parser.add_argument("--dirichlet-alpha", type=float, default=0.5)
    parser.add_argument("--critic-mode", choices=["single", "multi"], default="multi")
    parser.add_argument(
        "--controller-variant",
        choices=["ours_full", "ours_full_corrected"],
        default="ours_full",
    )
    parser.add_argument("--channel-mode", choices=["adaptive", "fixed"], default="adaptive")
    parser.add_argument(
        "--exploration-mode",
        choices=sorted(CHANNEL_PRESETS),
        default="multiscale",
        help="Enable single-scale, temperature-only, step-only, or full multiscale channels.",
    )
    parser.add_argument("--egfr-model", default=None)
    parser.add_argument("--vegfr2-model", default=None)
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
