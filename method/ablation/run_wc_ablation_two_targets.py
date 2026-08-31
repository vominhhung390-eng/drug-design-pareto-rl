#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Experiment 1: dynamic weight strategy ablation, two-target version.

This script keeps the original three-objective ablation untouched and runs a
clean two-objective setup:
    objectives = [EGFR activity, VEGFR2 activity]

QED is audited separately and never modifies the shared EGFR/VEGFR2 oracle
values used for optimization, Pareto ranking, or reporting.
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from main_pipeline import MO_RL_Integrator, ObjectiveCalculator  # noqa: E402
from finaly import Molecule, ParetoFront  # noqa: E402
from ablation.multi_critic_ppo_agent import MultiCriticPPOAgent  # noqa: E402
from ablation.weight_controllers import (  # noqa: E402
    ALL_VARIANTS,
    VARIANT_COLORS,
    VARIANT_LABELS,
    create_controller,
)

try:
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
except Exception:
    pass


OBJECTIVE_NAMES = ["EGFR", "VEGFR2"]
ACTIVITY_LOWER = 3.0
# Evaluation bound, deliberately distinct from the optimizer's 6.5
# desirability saturation.  pActivity 10 corresponds to 0.1 nM and remains
# inside the locked RF models' leaf-value support (EGFR 1.26-11.22; VEGFR2
# 3.92-10.62), preventing routine generated molecules from saturating HV.
ACTIVITY_UPPER = 10.0
RAW_REF_POINT = np.array([ACTIVITY_LOWER, ACTIVITY_LOWER], dtype=np.float32)
HV_REF_POINT = np.array([0.0, 0.0], dtype=np.float32)


@dataclass
class VariantResult:
    variant: str
    hv_final: float
    pareto_size: int
    egfr_max: float
    vegfr2_max: float
    egfr_mean: float
    vegfr2_mean: float
    valid_total: int
    invalid_total: int
    runtime_sec: float
    output_dir: str


class TwoTargetObjectiveCalculator(ObjectiveCalculator):
    """Return the unmodified shared EGFR and VEGFR2 RF predictions.

    Molecular quality belongs to separate QED/SA/alert metrics.  Applying the
    legacy SMILES-length penalty to activity here would make the primary method
    use a different oracle from the five external baselines.
    """

    def calculate_scores(self, smiles: str) -> np.ndarray:
        fingerprint = self._smiles_to_fingerprint(smiles).reshape(1, -1)
        egfr = (
            float(self.egfr_model.predict(fingerprint)[0])
            if self.egfr_model is not None else 0.0
        )
        vegfr2 = (
            float(self.vegfr2_model.predict(fingerprint)[0])
            if self.vegfr2_model is not None else 0.0
        )
        return np.asarray([egfr, vegfr2], dtype=np.float32)

    def calculate_scores_batch(self, smiles_list: Sequence[str]) -> np.ndarray:
        """Vectorized target prediction for a decoded molecule batch."""
        if not smiles_list:
            return np.zeros((0, 2), dtype=np.float32)
        fingerprints = np.vstack([
            self._smiles_to_fingerprint(smiles) for smiles in smiles_list
        ])
        egfr = (
            np.asarray(self.egfr_model.predict(fingerprints), dtype=np.float32)
            if self.egfr_model is not None else np.zeros(len(smiles_list), dtype=np.float32)
        )
        vegfr2 = (
            np.asarray(self.vegfr2_model.predict(fingerprints), dtype=np.float32)
            if self.vegfr2_model is not None else np.zeros(len(smiles_list), dtype=np.float32)
        )
        return np.column_stack([egfr, vegfr2]).astype(np.float32)


class TrajectoryMultiCriticPPOAgent(MultiCriticPPOAgent):
    """
    Preference-conditioned PPO for terminal-reward latent trajectories.

    K=1 is the original one-step method.  For K>1, zero intermediate rewards
    and the terminal molecular score are propagated through each objective's
    critic with GAE.  Flat buffer entries remain safe because ``done`` marks
    every trajectory boundary.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_obj: int = 2,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        ppo_clip: float = 0.2,
        ppo_epochs: int = 4,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        mini_batch_size: int = 16,
        max_grad_norm: float = 0.5,
        device: Optional[str] = None,
    ):
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            num_obj=num_obj,
            lr=lr,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ppo_clip=ppo_clip,
            ppo_epochs=ppo_epochs,
            entropy_coef=entropy_coef,
            mini_batch_size=mini_batch_size,
            max_grad_norm=max_grad_norm,
        )
        self.value_loss_coef = value_loss_coef
        self.auxiliary_actor_coef = 0.0
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.policy.to(self.device)

    @staticmethod
    def _standardize(x: torch.Tensor, dim: Optional[int] = None) -> torch.Tensor:
        if x.numel() <= 1:
            return x - x.mean()
        if dim is None:
            std = x.std(unbiased=False)
            return (x - x.mean()) / (std + 1e-8)
        std = x.std(dim=dim, unbiased=False, keepdim=True)
        mean = x.mean(dim=dim, keepdim=True)
        return (x - mean) / (std + 1e-8)

    def select_action(self, state: np.ndarray, preference: np.ndarray = None) -> tuple:
        if preference is None:
            preference = np.ones(self.num_obj, dtype=np.float32) / self.num_obj

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            pref_t = torch.FloatTensor(preference).unsqueeze(0).to(self.device)
            action_mean, action_std, critic_values = self.policy(state_t, pref_t)
            dist = torch.distributions.Normal(action_mean, action_std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1, keepdim=True)
            entropy = dist.entropy().sum(-1, keepdim=True)
            values = torch.cat([v for v in critic_values], dim=-1)

        return (
            action.squeeze(0).detach().cpu().numpy(),
            float(log_prob.squeeze(0).detach().cpu().item()),
            values.squeeze(0).detach().cpu().numpy(),
            float(entropy.squeeze(0).detach().cpu().item()),
        )

    def _compute_trajectory_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        advantages = torch.zeros_like(rewards)
        last_advantage = torch.zeros(
            rewards.shape[1], dtype=rewards.dtype, device=rewards.device
        )
        for step in range(len(rewards) - 1, -1, -1):
            nonterminal = 1.0 - dones[step]
            next_value = (
                values[step + 1]
                if step + 1 < len(values)
                else torch.zeros_like(values[step])
            )
            delta = (
                rewards[step]
                + self.gamma * next_value * nonterminal
                - values[step]
            )
            last_advantage = (
                delta
                + self.gamma * self.gae_lambda * nonterminal * last_advantage
            )
            advantages[step] = last_advantage
        return advantages

    def _discount_auxiliary_rewards(
        self, rewards: torch.Tensor, dones: torch.Tensor
    ) -> torch.Tensor:
        advantages = torch.zeros_like(rewards)
        running = torch.zeros((), dtype=rewards.dtype, device=rewards.device)
        for step in range(len(rewards) - 1, -1, -1):
            nonterminal = 1.0 - dones[step]
            running = (
                rewards[step]
                + self.gamma * self.gae_lambda * nonterminal * running
            )
            advantages[step] = running
        return advantages

    def update(self, memory) -> float:
        if len(memory.states) < self.mini_batch_size:
            return 0.0

        states = torch.FloatTensor(np.array(memory.states)).to(self.device)
        actions = torch.FloatTensor(np.array(memory.actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(memory.rewards)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(memory.log_probs)).to(self.device)
        old_values = torch.FloatTensor(np.array(memory.values)).to(self.device)
        dones = torch.as_tensor(
            np.asarray(memory.dones), dtype=torch.float32, device=self.device
        )
        preferences = torch.FloatTensor(np.array(memory.preferences)).to(self.device)
        auxiliary_rewards = torch.as_tensor(
            np.asarray(getattr(memory, "auxiliary_rewards", np.zeros(len(memory.states)))),
            dtype=torch.float32,
            device=self.device,
        )

        if old_log_probs.ndim == 1:
            old_log_probs = old_log_probs.unsqueeze(-1)
        if rewards.ndim == 1:
            rewards = rewards.unsqueeze(-1)
        if old_values.ndim == 1:
            old_values = old_values.unsqueeze(-1)

        advantages = self._compute_trajectory_gae(
            rewards, old_values.detach(), dones
        )
        advantages_norm = self._standardize(advantages, dim=0)
        combined_advantages = (advantages_norm * preferences).sum(dim=-1)
        if self.auxiliary_actor_coef != 0.0 and auxiliary_rewards.numel():
            auxiliary_advantages = self._discount_auxiliary_rewards(
                auxiliary_rewards, dones
            )
            combined_advantages = (
                combined_advantages
                + self.auxiliary_actor_coef
                * self._standardize(auxiliary_advantages)
            )
        combined_advantages = self._standardize(combined_advantages)
        returns = advantages + old_values.detach()

        total_loss = 0.0
        n_updates = 0
        batch_size = states.shape[0]

        for _ in range(self.ppo_epochs):
            indices = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, self.mini_batch_size):
                mb_idx = indices[start : start + self.mini_batch_size]
                if mb_idx.numel() == 0:
                    continue

                mb_states = states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_prefs = preferences[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_advantages = combined_advantages[mb_idx].unsqueeze(-1)
                mb_returns = returns[mb_idx]

                action_mean, action_std, critic_values = self.policy(mb_states, mb_prefs)
                dist = torch.distributions.Normal(action_mean, action_std)
                log_probs = dist.log_prob(mb_actions).sum(-1, keepdim=True)
                entropy = dist.entropy().sum(-1, keepdim=True)
                values = torch.cat([v for v in critic_values], dim=-1)

                ratios = torch.exp(log_probs - mb_old_log_probs)
                surr1 = ratios * mb_advantages
                surr2 = torch.clamp(
                    ratios, 1.0 - self.ppo_clip, 1.0 + self.ppo_clip
                ) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_losses = []
                for obj_idx in range(self.num_obj):
                    critic_losses.append(
                        torch.nn.functional.mse_loss(
                            values[:, obj_idx], mb_returns[:, obj_idx]
                        )
                    )
                critic_loss = torch.stack(critic_losses).mean()

                entropy_loss = -entropy.mean()
                loss = (
                    actor_loss
                    + self.value_loss_coef * critic_loss
                    + self.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss += float(loss.item())
                n_updates += 1

        return total_loss / max(n_updates, 1)


# Compatibility for the older one-step ablation runner.  K=1 behavior is a
# special case of the trajectory implementation.
OneStepMultiCriticPPOAgent = TrajectoryMultiCriticPPOAgent


class TwoTargetWCIntegrator(MO_RL_Integrator):
    """Two-target runner with dynamic controller weights driving PPO preference."""

    def __init__(self, config: Dict):
        super().__init__(
            latent_dim=config.get("latent_dim", 128),
            num_obj=2,
            total_epochs=config.get("total_epochs", 100),
            batch_size=config.get("batch_size", 32),
            vae_model_path=config.get("vae_model_path"),
            egfr_model_path=config.get("egfr_model_path"),
            vegfr2_model_path=config.get("vegfr2_model_path"),
            config=config,
        )

        self.objective_calculator = TwoTargetObjectiveCalculator(
            config.get("egfr_model_path"),
            config.get("vegfr2_model_path"),
        )

        self.agent = OneStepMultiCriticPPOAgent(
            state_dim=self.latent_dim,
            action_dim=self.latent_dim,
            num_obj=2,
            lr=config.get("lr", 3e-4),
            gamma=config.get("gamma", 0.99),
            gae_lambda=config.get("gae_lambda", 0.95),
            ppo_clip=config.get("ppo_clip", 0.2),
            ppo_epochs=config.get("ppo_epochs", 4),
            mini_batch_size=config.get("mini_batch_size", 16),
            entropy_coef=config.get("entropy_coef", 0.01),
            value_loss_coef=config.get("value_loss_coef", 0.5),
            device=config.get("device", None),
        )

        self.pareto_front = ParetoFront(num_obj=2)
        self.invalid_reward = float(config.get("invalid_reward", -1.0))
        self.step_scale = float(config.get("step_scale", 0.08))
        self.latent_clip = float(config.get("latent_clip", 4.0))
        self.policy_weights = np.ones(2, dtype=np.float32) / 2.0

    @staticmethod
    def _normalize_weights(weights: np.ndarray) -> np.ndarray:
        weights = np.asarray(weights, dtype=np.float32)
        weights = np.clip(weights, 1e-6, None)
        return weights / weights.sum()

    def set_controller(self, controller):
        self.controller = controller
        self.policy_weights = np.ones(2, dtype=np.float32) / 2.0

    def run_episode(self, batch_size: int, epoch: int) -> Tuple[List[Molecule], float, Dict]:
        valid_molecules: List[Molecule] = []
        invalid_count = 0
        policy_weights = self._normalize_weights(self.policy_weights)

        z_states = np.random.normal(0, 1, (batch_size, self.latent_dim)).astype(
            np.float32
        )

        for z in z_states:
            try:
                action, log_prob, values, _ = self.agent.select_action(
                    z, preference=policy_weights
                )
                new_z = np.clip(
                    z + self.step_scale * np.asarray(action, dtype=np.float32),
                    -self.latent_clip,
                    self.latent_clip,
                )
                smiles = self.vae.decode(
                    new_z,
                    greedy=False,
                    temperature=float(self.config.get("temperature", 0.7)),
                )

                if not smiles or len(smiles) < 5:
                    invalid_count += 1
                    self.agent.store_transition_multi(
                        z,
                        action,
                        np.full(2, self.invalid_reward, dtype=np.float32),
                        log_prob,
                        values,
                        True,
                        0.0,
                        policy_weights,
                    )
                    continue

                scores = self.objective_calculator.calculate_scores(smiles)
                molecule = Molecule(smiles=smiles, latent_vector=new_z, scores=scores)
                valid_molecules.append(molecule)

                self.agent.store_transition_multi(
                    z,
                    action,
                    scores,
                    log_prob,
                    values,
                    True,
                    0.0,
                    policy_weights,
                )
            except Exception:
                invalid_count += 1

        if valid_molecules:
            self.pareto_front.update(
                [m.scores for m in valid_molecules],
                valid_molecules,
            )
            batch_scores = np.vstack([m.scores for m in valid_molecules])
        else:
            batch_scores = np.zeros((1, 2), dtype=np.float32)

        next_weights = self.controller.get_weights(epoch, batch_scores)
        self.policy_weights = self._normalize_weights(next_weights)

        loss = 0.0
        if len(self.agent.buffer.states) >= self.agent.mini_batch_size:
            loss = self.agent.update(self.agent.buffer)
            self.agent.buffer.clear()

        info = {
            "valid_count": len(valid_molecules),
            "invalid_count": invalid_count,
            "policy_w_egfr": float(policy_weights[0]),
            "policy_w_vegfr2": float(policy_weights[1]),
            "next_w_egfr": float(self.policy_weights[0]),
            "next_w_vegfr2": float(self.policy_weights[1]),
        }
        return valid_molecules, float(loss), info


def dominates(a: Sequence[float], b: Sequence[float]) -> bool:
    return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))


def pareto_points_2d(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.empty((0, 2), dtype=np.float32)
    front = []
    for i, point in enumerate(points):
        if not any(i != j and dominates(other, point) for j, other in enumerate(points)):
            front.append(point)
    return np.asarray(front, dtype=np.float32)


def normalize_activity_scores(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    return np.clip(
        (points - ACTIVITY_LOWER) / (ACTIVITY_UPPER - ACTIVITY_LOWER),
        0.0,
        1.0,
    )


def hypervolume_2d(points: np.ndarray, ref_point: np.ndarray = HV_REF_POINT) -> float:
    """Pre-registered [0, 1] activity-space hypervolume."""
    front = pareto_points_2d(normalize_activity_scores(points))
    if len(front) == 0:
        return 0.0
    front = front[np.argsort(front[:, 0])]
    hv = 0.0
    prev_x = float(ref_point[0])
    for x, y in front:
        x = max(float(x), float(ref_point[0]))
        y = max(float(y), float(ref_point[1]))
        if x > prev_x:
            hv += (x - prev_x) * y
            prev_x = x
    return float(hv)


def default_config(args, variant: str) -> Dict:
    model_path = Path(args.model).resolve()
    project_root = ROOT.parent
    return {
        "vae_model_path": str(model_path),
        "egfr_model_path": str(
            project_root / "models" / "蛋白靶点预测器" / "target_EGFR_model.pkl"
        ),
        "vegfr2_model_path": str(
            project_root / "models" / "蛋白靶点预测器" / "target_VEGFR2_model.pkl"
        ),
        "total_epochs": args.epochs,
        "batch_size": args.batch,
        "lr": args.lr,
        "ppo_epochs": args.ppo_epochs,
        "mini_batch_size": args.mini_batch_size,
        "entropy_coef": args.entropy_coef,
        "value_loss_coef": args.value_loss_coef,
        "device": args.device,
        "step_scale": args.step_scale,
        "latent_clip": args.latent_clip,
        "invalid_reward": args.invalid_reward,
        "temperature": args.temperature,
        "pareto_max_size": args.pareto_max_size,
        "variant": variant,
    }


def write_csv(path: Path, rows: Iterable[Dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_variant(args_dict: Dict, variant: str) -> VariantResult:
    class Obj:
        pass

    args = Obj()
    for key, value in args_dict.items():
        setattr(args, key, value)

    start_time = time.time()
    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    config = default_config(args, variant)
    integrator = TwoTargetWCIntegrator(config)
    integrator.set_controller(create_controller(variant, num_obj=2, total_epochs=args.epochs))

    hv_rows: List[Dict] = []
    weight_rows: List[Dict] = []
    metric_rows: List[Dict] = []

    print(f"\n=== Two-target WC ablation: {variant} ===", flush=True)
    for epoch in range(args.epochs):
        molecules, loss, info = integrator.run_episode(args.batch, epoch)

        pareto_scores = np.array([m.scores for m in integrator.pareto_front.molecules])
        hv = hypervolume_2d(pareto_scores)
        pareto_size = len(integrator.pareto_front.molecules)

        hv_rows.append({"epoch": epoch + 1, "hv": hv, "pareto_size": pareto_size})
        weight_rows.append(
            {
                "epoch": epoch + 1,
                "w_egfr": info["next_w_egfr"],
                "w_vegfr2": info["next_w_vegfr2"],
            }
        )
        metric_rows.append(
            {
                "epoch": epoch + 1,
                "loss": loss,
                "valid_count": info["valid_count"],
                "invalid_count": info["invalid_count"],
                "valid_rate": info["valid_count"] / max(args.batch, 1),
                "policy_w_egfr": info["policy_w_egfr"],
                "policy_w_vegfr2": info["policy_w_vegfr2"],
                "next_w_egfr": info["next_w_egfr"],
                "next_w_vegfr2": info["next_w_vegfr2"],
                "hv": hv,
                "pareto_size": pareto_size,
            }
        )

        if (epoch + 1) % args.log_interval == 0 or epoch == 0:
            print(
                f"[{variant}] epoch {epoch + 1:03d}/{args.epochs} "
                f"HV={hv:.4f} Pareto={pareto_size} "
                f"valid={info['valid_count']}/{args.batch} "
                f"w=({info['next_w_egfr']:.3f},{info['next_w_vegfr2']:.3f}) "
                f"loss={loss:.4f}",
                flush=True,
            )

    pareto_rows = [
        {
            "smiles": m.smiles,
            "egfr": float(m.scores[0]),
            "vegfr2": float(m.scores[1]),
        }
        for m in integrator.pareto_front.molecules
    ]

    write_csv(out_dir / f"hv_history_wc2_{variant}.csv", hv_rows, ["epoch", "hv", "pareto_size"])
    write_csv(
        out_dir / f"weight_history_wc2_{variant}.csv",
        weight_rows,
        ["epoch", "w_egfr", "w_vegfr2"],
    )
    write_csv(
        out_dir / f"metrics_wc2_{variant}.csv",
        metric_rows,
        [
            "epoch",
            "loss",
            "valid_count",
            "invalid_count",
            "valid_rate",
            "policy_w_egfr",
            "policy_w_vegfr2",
            "next_w_egfr",
            "next_w_vegfr2",
            "hv",
            "pareto_size",
        ],
    )
    write_csv(out_dir / f"pareto_front_wc2_{variant}.csv", pareto_rows, ["smiles", "egfr", "vegfr2"])

    if pareto_rows:
        egfr = np.array([r["egfr"] for r in pareto_rows], dtype=np.float32)
        vegfr2 = np.array([r["vegfr2"] for r in pareto_rows], dtype=np.float32)
        egfr_max = float(egfr.max())
        vegfr2_max = float(vegfr2.max())
        egfr_mean = float(egfr.mean())
        vegfr2_mean = float(vegfr2.mean())
    else:
        egfr_max = vegfr2_max = egfr_mean = vegfr2_mean = 0.0

    return VariantResult(
        variant=variant,
        hv_final=float(hv_rows[-1]["hv"]) if hv_rows else 0.0,
        pareto_size=len(pareto_rows),
        egfr_max=egfr_max,
        vegfr2_max=vegfr2_max,
        egfr_mean=egfr_mean,
        vegfr2_mean=vegfr2_mean,
        valid_total=int(sum(r["valid_count"] for r in metric_rows)),
        invalid_total=int(sum(r["invalid_count"] for r in metric_rows)),
        runtime_sec=time.time() - start_time,
        output_dir=str(out_dir),
    )


def plot_hv(out_dir: Path, variants: Sequence[str]) -> None:
    plt.figure(figsize=(9, 5))
    for variant in variants:
        path = out_dir / f"hv_history_wc2_{variant}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        plt.plot(
            df["epoch"],
            df["hv"],
            label=VARIANT_LABELS.get(variant, variant),
            color=VARIANT_COLORS.get(variant),
            linewidth=2,
        )
    plt.xlabel("Epoch")
    plt.ylabel("2D Hypervolume (EGFR/VEGFR2)")
    plt.title("Two-target Dynamic Weight Ablation: HV")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "wc2_hv_curves.png", dpi=300)
    plt.close()


def plot_weights(out_dir: Path, variants: Sequence[str]) -> None:
    fig, axes = plt.subplots(len(variants), 1, figsize=(9, max(3, 2.4 * len(variants))), sharex=True)
    if len(variants) == 1:
        axes = [axes]
    for ax, variant in zip(axes, variants):
        path = out_dir / f"weight_history_wc2_{variant}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        ax.plot(df["epoch"], df["w_egfr"], label="EGFR", linewidth=1.8)
        ax.plot(df["epoch"], df["w_vegfr2"], label="VEGFR2", linewidth=1.8)
        ax.set_ylabel(VARIANT_LABELS.get(variant, variant))
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("Epoch")
    fig.suptitle("Two-target Dynamic Weights")
    plt.tight_layout()
    plt.savefig(out_dir / "wc2_weight_trajectories.png", dpi=300)
    plt.close()


def plot_pareto(out_dir: Path, variants: Sequence[str]) -> None:
    n = len(variants)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    for ax, variant in zip(axes.ravel(), variants):
        path = out_dir / f"pareto_front_wc2_{variant}.csv"
        ax.set_title(VARIANT_LABELS.get(variant, variant))
        ax.set_xlabel("EGFR")
        ax.set_ylabel("VEGFR2")
        ax.grid(True, alpha=0.25)
        if path.exists():
            df = pd.read_csv(path)
            if len(df):
                ax.scatter(
                    df["egfr"],
                    df["vegfr2"],
                    s=24,
                    alpha=0.75,
                    color=VARIANT_COLORS.get(variant),
                )
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Two-target Pareto Fronts")
    plt.tight_layout()
    plt.savefig(out_dir / "wc2_pareto_2d.png", dpi=300)
    plt.close()


def plot_metrics(out_dir: Path, variants: Sequence[str]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    for variant in variants:
        path = out_dir / f"metrics_wc2_{variant}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        label = VARIANT_LABELS.get(variant, variant)
        color = VARIANT_COLORS.get(variant)
        axes[0].plot(df["epoch"], df["loss"], label=label, color=color, alpha=0.85)
        axes[1].plot(df["epoch"], df["valid_rate"], label=label, color=color, alpha=0.85)
    axes[0].set_ylabel("PPO loss")
    axes[1].set_ylabel("Valid rate")
    axes[1].set_xlabel("Epoch")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend()
    fig.suptitle("Two-target Training Metrics")
    plt.tight_layout()
    plt.savefig(out_dir / "wc2_training_metrics.png", dpi=300)
    plt.close()


def write_summary(out_dir: Path, results: Sequence[VariantResult], args) -> None:
    rows = [r.__dict__ for r in results]
    summary_csv = out_dir / "wc2_ablation_summary.csv"
    pd.DataFrame(rows).sort_values("hv_final", ascending=False).to_csv(
        summary_csv, index=False, encoding="utf-8-sig"
    )

    lines = [
        "# 实验一：动态权重策略消融（双目标 EGFR / VEGFR2）",
        "",
        "## 实验设置",
        "",
        f"- VAE: `{Path(args.model).resolve()}`",
        f"- Epochs: {args.epochs}",
        f"- Batch size: {args.batch}",
        f"- Decode temperature: {args.temperature}",
        "- 优化目标：EGFR 活性、VEGFR2 活性",
        "- 不作为目标：QED（不进入奖励、动态权重、Pareto、HV）",
        "- PPO 更新：每个生成分子按 one-step transition 处理，避免将 batch 误当连续轨迹。",
        "",
        "## 结果汇总",
        "",
        "| Variant | Final HV | Pareto size | EGFR max | VEGFR2 max | Valid total | Runtime min |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(results, key=lambda x: x.hv_final, reverse=True):
        lines.append(
            f"| {r.variant} | {r.hv_final:.4f} | {r.pareto_size} | "
            f"{r.egfr_max:.4f} | {r.vegfr2_max:.4f} | {r.valid_total} | "
            f"{r.runtime_sec / 60:.2f} |"
        )
    lines.extend(
        [
            "",
            "## 输出文件",
            "",
            "- `hv_history_wc2_<variant>.csv`: 每轮 2D HV 与 Pareto 数量",
            "- `weight_history_wc2_<variant>.csv`: EGFR / VEGFR2 动态权重轨迹",
            "- `metrics_wc2_<variant>.csv`: loss、合法生成数、有效率、权重、HV",
            "- `pareto_front_wc2_<variant>.csv`: 最终 Pareto 分子与两靶点得分",
            "- `wc2_hv_curves.png`: HV 对比曲线",
            "- `wc2_weight_trajectories.png`: 权重变化图",
            "- `wc2_pareto_2d.png`: EGFR-VEGFR2 Pareto 散点图",
            "- `wc2_training_metrics.png`: loss 与有效率曲线",
        ]
    )
    (out_dir / "wc2_ablation_summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    default_model = (
        ROOT.parent
        / "logs"
        / "polygon_train_smiles_only_80"
        / "polygon_train_smiles_only_80.pt"
    )
    parser = argparse.ArgumentParser(
        description="Two-target dynamic weight strategy ablation for EGFR/VEGFR2."
    )
    parser.add_argument("--model", default=str(default_model), help="Trained VAE checkpoint")
    parser.add_argument(
        "--output",
        default=str(ROOT.parent / "logs" / "wc_ablation_two_targets"),
        help="Output directory",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--variants", nargs="+", default=ALL_VARIANTS)
    parser.add_argument("--parallel", type=int, default=1, help="Number of variants to run in parallel")
    parser.add_argument("--device", default=None, help="cuda/cpu; default lets torch decide")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--mini-batch-size", type=int, default=16)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-loss-coef", type=float, default=0.5)
    parser.add_argument("--step-scale", type=float, default=0.08)
    parser.add_argument("--latent-clip", type=float, default=4.0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--invalid-reward", type=float, default=-1.0)
    parser.add_argument("--pareto-max-size", type=int, default=2000)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate_args(args) -> None:
    missing = []
    if not Path(args.model).exists():
        missing.append(f"VAE checkpoint not found: {args.model}")
    for variant in args.variants:
        if variant not in ALL_VARIANTS:
            missing.append(f"Unknown variant: {variant}; available={ALL_VARIANTS}")
    if missing:
        raise FileNotFoundError("\n".join(missing))


def main():
    args = parse_args()
    validate_args(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Two-target dynamic weight ablation")
    print(f"Output: {out_dir}")
    print(f"Variants: {args.variants}")
    print(f"Device: {args.device or ('cuda' if torch.cuda.is_available() else 'cpu')}")

    args_dict = vars(args).copy()
    results: List[VariantResult] = []

    if args.parallel > 1 and len(args.variants) > 1:
        workers = min(args.parallel, len(args.variants))
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
            futures = {
                executor.submit(run_variant, args_dict, variant): variant
                for variant in args.variants
            }
            for future in as_completed(futures):
                variant = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    print(
                        f"[done] {variant}: HV={result.hv_final:.4f}, "
                        f"Pareto={result.pareto_size}",
                        flush=True,
                    )
                except Exception as exc:
                    print(f"[failed] {variant}: {exc}", flush=True)
                    raise
    else:
        for variant in args.variants:
            results.append(run_variant(args_dict, variant))

    plot_hv(out_dir, args.variants)
    plot_weights(out_dir, args.variants)
    plot_pareto(out_dir, args.variants)
    plot_metrics(out_dir, args.variants)
    write_summary(out_dir, results, args)

    print("\nFinished two-target ablation.")
    print(f"Summary: {out_dir / 'wc2_ablation_summary.md'}")


if __name__ == "__main__":
    main()
