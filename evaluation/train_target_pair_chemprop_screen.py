#!/usr/bin/env python
"""Screen Chemprop regressors on frozen rolling-time folds for any target pair."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "evaluation" / "run_chemprop_utf8.py"


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    return {
        "n": int(len(y)),
        "rmse": float(mean_squared_error(y, pred) ** 0.5),
        "mae": float(mean_absolute_error(y, pred)),
        "r2": float(r2_score(y, pred)),
        "spearman": float(spearmanr(y, pred).statistic),
        "bias": float(np.mean(pred - y)),
    }


def train_one(
    run: Path,
    target: str,
    profile: str,
    fold: str,
    variant: str,
    epochs: int,
) -> dict[str, object]:
    data_dir = run / "data" / target.lower() / profile / fold
    train_csv = data_dir / "train.csv"
    validation_csv = data_dir / "validation.csv"
    output = run / "chemprop_screen" / target.lower() / profile / fold / variant
    output.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-X",
        "utf8",
        str(WRAPPER),
        "train",
        "-i",
        str(train_csv),
        str(validation_csv),
        str(validation_csv),
        "-o",
        str(output),
        "--smiles-columns",
        "smiles",
        "--target-columns",
        "pactivity",
        "--task-type",
        "regression",
        "--metrics",
        "rmse",
        "r2",
        "--accelerator",
        "gpu",
        "--devices",
        "1",
        "--num-workers",
        "0",
        "--batch-size",
        "256",
        "--epochs",
        str(epochs),
        "--patience",
        "10",
        "--warmup-epochs",
        "2",
        "--data-seed",
        "42",
        "--pytorch-seed",
        "42",
        "--message-hidden-dim",
        "300",
        "--depth",
        "3",
        "--ffn-hidden-dim",
        "300",
        "--ffn-num-layers",
        "1",
    ]
    if variant == "dmpnn_morgan":
        command += ["--molecule-featurizers", "morgan_binary"]
    elif variant == "chemeleon":
        command += ["--from-foundation", "CHEMELEON"]
    elif variant != "dmpnn":
        raise ValueError(f"Unknown Chemprop variant: {variant}")

    prediction_file = output / "model_0" / "test_predictions.csv"
    started = time.perf_counter()
    if not prediction_file.exists():
        environment = os.environ.copy()
        environment.update(
            {
                "PYTHONUTF8": "1",
                "PYTHONIOENCODING": "utf-8",
                "RICH_FORCE_TERMINAL": "false",
            }
        )
        log_file = output / "launcher.log"
        with log_file.open("w", encoding="utf-8") as log:
            process = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if process.returncode != 0:
            tail = log_file.read_text(encoding="utf-8", errors="replace")[-8000:]
            raise RuntimeError(f"Chemprop failed ({target}/{fold}/{variant})\n{tail}")

    truth = pd.read_csv(validation_csv)
    prediction = pd.read_csv(prediction_file)
    if len(truth) != len(prediction):
        raise RuntimeError(f"Prediction row mismatch: {len(truth)} != {len(prediction)}")
    row: dict[str, object] = {
        "target": target,
        "profile": profile,
        "fold": fold,
        "model": variant,
        "fit_seconds": time.perf_counter() - started,
        **metrics(
            truth["pactivity"].to_numpy(float),
            prediction["pactivity"].to_numpy(float),
        ),
    }
    prediction.assign(observed_pactivity=truth["pactivity"].to_numpy()).to_csv(
        output / "validation_predictions_with_truth.csv", index=False, encoding="utf-8-sig"
    )
    (output / "metrics.json").write_text(
        json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"{target} {profile} {fold} {variant}: "
        f"rho={row['spearman']:.3f} RMSE={row['rmse']:.3f} R2={row['r2']:.3f}",
        flush=True,
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--profile", default="single_protein_assay_ge10")
    parser.add_argument("--variants", nargs="+", default=["dmpnn", "dmpnn_morgan"])
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    run = ROOT / config["output_dir"]
    rows: list[dict[str, object]] = []
    for target in args.targets:
        for fold in ("fold_a", "fold_b"):
            for variant in args.variants:
                rows.append(
                    train_one(run, target, args.profile, fold, variant, args.epochs)
                )

    frame = pd.DataFrame(rows)
    output = run / "chemprop_screen"
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "all_metrics.csv", index=False, encoding="utf-8-sig")
    summary = (
        frame.groupby(["target", "profile", "model"], as_index=False)
        .agg(
            mean_spearman=("spearman", "mean"),
            worst_spearman=("spearman", "min"),
            mean_rmse=("rmse", "mean"),
            worst_rmse=("rmse", "max"),
            mean_r2=("r2", "mean"),
        )
        .sort_values(["target", "mean_spearman", "mean_rmse"], ascending=[True, False, True])
    )
    summary.to_csv(output / "summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
