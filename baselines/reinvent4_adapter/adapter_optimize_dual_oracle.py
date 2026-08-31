"""REINVENT4 DAP optimization using the shared exact-budget dual oracle."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = REPO_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from baselines.common.oracle_ledger import OracleLedger  # noqa: E402
from reinvent.runmodes import create_adapter  # noqa: E402
from reinvent.runmodes.RL.reward import dap_strategy  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sigma", type=float, default=128.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if args.budget % args.batch_size != 0:
        raise ValueError("REINVENT4 budget must be divisible by batch size")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    prior, _, prior_type = create_adapter(str(args.prior.resolve()), "inference", device)
    agent, _, agent_type = create_adapter(str(args.prior.resolve()), "training", device)
    if prior_type != "Reinvent" or agent_type != "Reinvent":
        raise RuntimeError(f"Expected Reinvent prior, got {prior_type}/{agent_type}")
    for parameter in prior.get_network_parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(agent.get_network_parameters(), lr=args.learning_rate)
    ledger = OracleLedger(args.budget, output_dir / "generated.csv")
    metrics = []
    started = time.time()

    for iteration in range(args.budget // args.batch_size):
        agent.set_mode("inference")
        sampled = agent.sample(args.batch_size)
        sequences = sampled.items1
        smiles = sampled.items2
        _, objectives = ledger.score(smiles, phase="optimization", iteration=iteration)
        # The geometric mean is REINVENT's standard MPO aggregation while both
        # raw objectives remain available in the ledger and evaluator.
        scalar_scores = np.sqrt(objectives[:, 0] * objectives[:, 1])
        scores = torch.as_tensor(scalar_scores, device=device, dtype=torch.float32)

        agent.set_mode("training")
        agent_nll = agent.likelihood(sequences)
        with torch.no_grad():
            prior_nll = prior.likelihood(sequences)
        agent_ll = -agent_nll
        prior_ll = -prior_nll
        per_sample_loss, augmented_ll = dap_strategy(
            agent_ll, scores, prior_ll, args.sigma
        )
        loss = per_sample_loss.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.get_network_parameters(), 5.0)
        optimizer.step()

        row = {
            "iteration": iteration,
            "oracle_used": ledger.used,
            "loss": float(loss.detach().cpu()),
            "mean_reward": float(scalar_scores.mean()),
            "max_reward": float(scalar_scores.max()),
            "seconds": time.time() - started,
        }
        metrics.append(row)
        print(json.dumps(row), flush=True)
        (output_dir / "iterations.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    agent.save_to_file(str(output_dir / "reinvent4_optimized.model"))
    ledger.write_metadata(
        output_dir / "metadata.json",
        method="REINVENT4",
        seed=args.seed,
        prior=str(args.prior.resolve()),
        from_scratch_prior=True,
        algorithm="DAP",
        aggregation="geometric_mean",
        sigma=args.sigma,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
