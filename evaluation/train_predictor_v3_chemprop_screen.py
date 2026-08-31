#!/usr/bin/env python
"""Train and evaluate Chemprop candidates on frozen predictor V3 time folds."""
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
RUN = ROOT / "results" / "predictor_retraining_v3_20260731"
WRAPPER = ROOT / "evaluation" / "run_chemprop_utf8.py"


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "n": int(len(y)),
        "rmse": float(mean_squared_error(y, pred) ** 0.5),
        "mae": float(mean_absolute_error(y, pred)),
        "r2": float(r2_score(y, pred)),
        "spearman": float(spearmanr(y, pred).statistic),
    }


def train_one(target: str, profile: str, fold: str, variant: str) -> dict:
    data_dir = RUN / "data" / target.lower() / profile / fold
    train_csv = data_dir / "train.csv"
    val_csv = data_dir / "validation.csv"
    out = RUN / "chemprop_screen" / target.lower() / profile / fold / variant
    out.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-X", "utf8", str(WRAPPER), "train",
        "-i", str(train_csv), str(val_csv), str(val_csv),
        "-o", str(out),
        "--smiles-columns", "smiles",
        "--target-columns", "pactivity",
        "--task-type", "regression",
        "--metrics", "rmse", "r2",
        "--accelerator", "gpu",
        "--devices", "1",
        "--num-workers", "0",
        "--batch-size", "256",
        "--epochs", "50",
        "--patience", "10",
        "--warmup-epochs", "2",
        "--data-seed", "42",
        "--pytorch-seed", "42",
        "--message-hidden-dim", "300",
        "--depth", "3",
        "--ffn-hidden-dim", "300",
        "--ffn-num-layers", "1",
    ]
    if variant == "dmpnn_morgan":
        cmd += ["--molecule-featurizers", "morgan_binary"]
    elif variant == "chemeleon":
        cmd += ["--from-foundation", "CHEMELEON"]

    prediction_file = out / "model_0" / "test_predictions.csv"
    start = time.time()
    if not prediction_file.exists():
        env = os.environ.copy()
        env.update({"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "RICH_FORCE_TERMINAL": "false"})
        log_file = out / "launcher.log"
        with log_file.open("w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            tail = log_file.read_text(encoding="utf-8", errors="replace")[-6000:]
            raise RuntimeError(f"Chemprop failed ({target}/{fold}/{variant})\n{tail}")

    truth = pd.read_csv(val_csv)
    prediction = pd.read_csv(prediction_file)
    if len(truth) != len(prediction):
        raise RuntimeError(f"Prediction row mismatch: {len(truth)} != {len(prediction)}")
    row = {
        "target": target,
        "profile": profile,
        "fold": fold,
        "model": variant,
        "fit_seconds": time.time() - start,
        **metrics(truth["pactivity"].to_numpy(float), prediction["pactivity"].to_numpy(float)),
    }
    prediction.assign(observed_pactivity=truth["pactivity"].to_numpy()).to_csv(
        out / "validation_predictions_with_truth.csv", index=False
    )
    (out / "metrics.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"{target} {profile} {fold} {variant}: "
        f"rho={row['spearman']:.3f} rmse={row['rmse']:.3f} r2={row['r2']:.3f}",
        flush=True,
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", nargs="+", default=["EGFR"])
    parser.add_argument("--profile", default="single_protein_assay_ge10")
    parser.add_argument("--variants", nargs="+", default=["dmpnn", "dmpnn_morgan", "chemeleon"])
    args = parser.parse_args()

    rows = []
    for target in args.targets:
        for fold in ("fold_a", "fold_b"):
            for variant in args.variants:
                rows.append(train_one(target, args.profile, fold, variant))

    frame = pd.DataFrame(rows)
    out = RUN / "chemprop_screen"
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "all_metrics.csv", index=False)
    summary = (
        frame.groupby(["target", "profile", "model"], as_index=False)
        .agg(
            mean_spearman=("spearman", "mean"),
            worst_spearman=("spearman", "min"),
            mean_rmse=("rmse", "mean"),
            worst_rmse=("rmse", "max"),
            mean_r2=("r2", "mean"),
        )
        .sort_values(["mean_spearman", "mean_rmse"], ascending=[False, True])
    )
    summary.to_csv(out / "summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
