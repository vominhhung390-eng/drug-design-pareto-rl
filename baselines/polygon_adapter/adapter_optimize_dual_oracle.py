"""Run POLYGON's sample-rank-finetune loop under an exact dual-oracle budget."""

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

from adapter_sample_checkpoint import infer_config  # noqa: E402
from baselines.common.oracle_ledger import OracleLedger  # noqa: E402
from polygon.vae.vae_model import VAE  # noqa: E402
from polygon.vae.vae_trainer import VAETrainer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "models/polygon_vae_best_valid_novel_stable_020.pt",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--keep-top", type=int, default=64)
    # Match POLYGON's official ``optimize_n_epochs`` default.  The upstream
    # KL annealer is also defined for two or more epochs (one epoch divides by
    # zero because kl_start defaults to one).
    parser.add_argument("--finetune-epochs", type=int, default=2)
    parser.add_argument("--finetune-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-length", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if args.budget % args.batch_size != 0:
        raise ValueError("POLYGON budget must be divisible by batch size for comparable checkpoints")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    state = torch.load(args.checkpoint.resolve(), map_location="cpu", weights_only=True)
    inferred = infer_config(state)
    model = VAE(**inferred)
    model.load_state_dict(state)
    model.to(device).eval()
    trainer = VAETrainer(
        model=model,
        model_save=None,
        device=device,
        log_dir=str(output_dir),
        kl_w_start=1,
        kl_w_end=1,
        lr_start=args.learning_rate,
        lr_end=args.learning_rate,
    )
    ledger = OracleLedger(args.budget, output_dir / "generated.csv")
    archive: dict[str, float] = {}
    iteration_metrics = []
    started = time.time()

    for iteration in range(args.budget // args.batch_size):
        model.eval()
        with torch.inference_mode():
            raw_smiles = model.sample(
                args.batch_size,
                max_len=args.max_length,
                temp=args.temperature,
                multinomial=True,
            )
        results, desirabilities = ledger.score(
            raw_smiles, phase="optimization", iteration=iteration
        )
        for result, scores in zip(results, desirabilities):
            if result.valid:
                archive[result.canonical_smiles] = max(
                    archive.get(result.canonical_smiles, -np.inf), float(scores.mean())
                )
        ranked = sorted(archive, key=lambda item: (archive[item], item), reverse=True)
        top = ranked[: args.keep_top]
        if args.finetune_epochs > 0 and len(top) >= 4:
            np.random.shuffle(top)
            split = max(1, int(0.75 * len(top)))
            train = top[:split]
            validation = top[split:] or None
            trainer.fit(
                train,
                validation,
                n_epoch=args.finetune_epochs,
                batch_size=min(args.finetune_batch_size, len(train)),
                save_frequency=None,
            )
        row = {
            "iteration": iteration,
            "oracle_used": ledger.used,
            "valid_in_batch": sum(result.valid for result in results),
            "archive_size": len(archive),
            "best_balanced_desirability": max(archive.values(), default=0.0),
            "seconds": time.time() - started,
        }
        iteration_metrics.append(row)
        print(json.dumps(row), flush=True)
        (output_dir / "iterations.json").write_text(
            json.dumps(iteration_metrics, indent=2), encoding="utf-8"
        )

    torch.save(model.state_dict(), output_dir / "polygon_optimized.pt")
    ledger.write_metadata(
        output_dir / "metadata.json",
        method="POLYGON-original",
        seed=args.seed,
        checkpoint=str(args.checkpoint.resolve()),
        reused_generator_checkpoint=True,
        inferred_model_config=inferred,
        batch_size=args.batch_size,
        keep_top=args.keep_top,
        finetune_epochs=args.finetune_epochs,
        temperature=args.temperature,
        max_length=args.max_length,
    )


if __name__ == "__main__":
    main()
