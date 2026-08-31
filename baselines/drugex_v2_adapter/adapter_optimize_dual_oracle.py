"""DrugEx v2 Pareto-RL optimization under the shared exact oracle budget."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = REPO_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

import models  # noqa: E402
import utils  # noqa: E402
from baselines.common.oracle_ledger import OracleLedger  # noqa: E402
from utils.nsgaii import similarity_sort  # noqa: E402


def pareto_similarity_reward(smiles: list[str], objectives: np.ndarray) -> np.ndarray:
    """DrugEx v2's PR scheme with guards for all/none-desired batches."""
    mols = [Chem.MolFromSmiles(item) for item in smiles]
    fps = utils.Env.calc_fps(mols)
    ranks = similarity_sort(objectives, fps, is_gpu=True)
    desired = int(np.all(objectives >= 0.99, axis=1).sum())
    undesired = len(smiles) - desired
    if desired == 0:
        ordered = np.arange(undesired, dtype=np.float32) / max(1, undesired) / 2
    elif undesired == 0:
        ordered = np.arange(desired, dtype=np.float32) / max(1, desired) / 2 + 0.5
    else:
        ordered = np.concatenate(
            [
                np.arange(undesired, dtype=np.float32) / undesired / 2,
                np.arange(desired, dtype=np.float32) / desired / 2 + 0.5,
            ]
        )
    rewards = np.zeros((len(smiles), 1), dtype=np.float32)
    rewards[np.asarray(ranks), 0] = ordered
    return rewards


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior", type=Path, default=None)
    parser.add_argument("--vocabulary", type=Path, default=PROJECT_ROOT / "results/baselines/drugex_v2/data/common_voc.txt")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--replay",
        type=int,
        default=10,
        help="Number of evolve batches per policy-gradient update (official Evolve default: 10).",
    )
    parser.add_argument("--epsilon", type=float, default=1e-3)
    args = parser.parse_args()

    update_size = args.batch_size * args.replay
    if args.budget % update_size != 0:
        raise ValueError("DrugEx v2 budget must be divisible by batch_size * replay")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    vocabulary = utils.Voc(init_from_file=str(args.vocabulary.resolve()))
    agent = models.Generator(vocabulary, is_lstm=True)
    prior = models.Generator(vocabulary, is_lstm=True)
    crossover = models.Generator(vocabulary, is_lstm=True)
    if args.prior is not None:
        state = torch.load(args.prior.resolve(), map_location=utils.dev, weights_only=True)
        agent.load_state_dict(state)
        prior.load_state_dict(state)
        crossover.load_state_dict(state)
    else:
        # Smoke-test mode only; formal runs require the from-scratch trained prior.
        prior.load_state_dict(agent.state_dict())
        crossover.load_state_dict(agent.state_dict())
    prior.eval()
    crossover.eval()
    for model in (prior, crossover):
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    ledger = OracleLedger(args.budget, output_dir / "generated.csv")
    metrics = []
    started = time.time()
    for iteration in range(args.budget // update_size):
        agent.eval()
        with torch.no_grad():
            replay_sequences = [
                agent.evolve1(
                    args.batch_size,
                    epsilon=args.epsilon,
                    crover=crossover,
                    mutate=prior,
                )
                for _ in range(args.replay)
            ]
            sequences = torch.cat(replay_sequences, dim=0)
        smiles = [agent.voc.decode(sequence) for sequence in sequences]
        _, objectives = ledger.score(smiles, phase="optimization", iteration=iteration)

        # Match the released Evolve.policy_gradient implementation, which
        # deduplicates decoded SMILES before calculating PR rewards and PGLoss.
        unique_indices_np = np.asarray(
            utils.unique(np.asarray([[item] for item in smiles])), dtype=np.int64
        )
        unique_indices = torch.as_tensor(unique_indices_np, device=sequences.device)
        unique_sequences = sequences[unique_indices]
        unique_smiles = [smiles[int(index)] for index in unique_indices_np]
        unique_objectives = objectives[unique_indices_np]
        rewards = pareto_similarity_reward(unique_smiles, unique_objectives)
        dataset = TensorDataset(unique_sequences, torch.as_tensor(rewards, device=utils.dev))
        loader = DataLoader(dataset, batch_size=128, shuffle=True)
        agent.train()
        agent.PGLoss(loader)

        row = {
            "iteration": iteration,
            "oracle_used": ledger.used,
            "unique_training_sequences": len(unique_sequences),
            "mean_reward": float(rewards.mean()),
            "max_reward": float(rewards.max()),
            "seconds": time.time() - started,
        }
        metrics.append(row)
        print(json.dumps(row), flush=True)
        (output_dir / "iterations.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    torch.save(agent.state_dict(), output_dir / "drugex_v2_optimized.pkg")
    ledger.write_metadata(
        output_dir / "metadata.json",
        method="DrugEx-v2",
        seed=args.seed,
        prior=str(args.prior.resolve()) if args.prior else None,
        from_scratch_prior=args.prior is not None,
        algorithm="Evolve/Pareto-ranking",
        reward_scheme="PR",
        epsilon=args.epsilon,
        batch_size=args.batch_size,
        replay=args.replay,
        policy_gradient_update_size=update_size,
    )


if __name__ == "__main__":
    main()
