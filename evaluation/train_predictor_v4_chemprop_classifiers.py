#!/usr/bin/env python
"""Train Chemprop classifiers for predeclared V4 high-confidence endpoints."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config" / "predictor_v4_90plus.json").read_text(encoding="utf-8"))
V3 = ROOT / CONFIG["source_run"] / "data"
OUT = ROOT / "results" / CONFIG["run_id"] / "chemprop_classification"
WRAPPER = ROOT / "evaluation" / "run_chemprop_utf8.py"


def prepare(target, fold, task_name):
    task = CONFIG["tasks"][task_name]
    profile = "single_protein_assay_ge10"
    source = V3 / target.lower() / profile / fold
    output = OUT / "data" / target.lower() / task_name / fold
    output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation"):
        frame = pd.read_csv(source / f"{split}.csv")
        mask = (frame.pactivity <= task["inactive_max"]) | (frame.pactivity >= task["active_min"])
        selected = frame[mask].copy()
        selected["active_label"] = (selected.pactivity >= task["active_min"]).astype(int)
        selected[["smiles", "active_label"]].to_csv(output / f"{split}.csv", index=False)
    return output


def train(target, fold, task_name, variant):
    data = prepare(target, fold, task_name)
    out = OUT / target.lower() / task_name / fold / variant
    out.mkdir(parents=True, exist_ok=True)
    pred_file = out / "model_0" / "test_predictions.csv"
    cmd = [sys.executable, "-X", "utf8", str(WRAPPER), "train", "-i",
           str(data / "train.csv"), str(data / "validation.csv"), str(data / "validation.csv"),
           "-o", str(out), "--smiles-columns", "smiles", "--target-columns", "active_label",
           "--task-type", "classification", "--metrics", "roc", "prc", "accuracy", "f1",
           "--class-balance", "--accelerator", "gpu", "--devices", "1", "--num-workers", "0",
           "--batch-size", "256", "--epochs", "50", "--patience", "10", "--warmup-epochs", "2",
           "--data-seed", "42", "--pytorch-seed", "42", "--message-hidden-dim", "300", "--depth", "3",
           "--ffn-hidden-dim", "300", "--ffn-num-layers", "1"]
    if variant == "dmpnn_morgan":
        cmd += ["--molecule-featurizers", "morgan_binary"]
    elif variant == "chemeleon":
        cmd += ["--from-foundation", "CHEMELEON"]
    if not pred_file.exists():
        env = os.environ.copy(); env.update({"PYTHONUTF8":"1","PYTHONIOENCODING":"utf-8","RICH_FORCE_TERMINAL":"false"})
        with (out / "launcher.log").open("w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        if proc.returncode:
            raise RuntimeError((out / "launcher.log").read_text(encoding="utf-8", errors="replace")[-5000:])
    truth = pd.read_csv(data / "validation.csv")
    pred = pd.read_csv(pred_file)
    probability = pred["active_label"].to_numpy(float)
    y = truth.active_label.to_numpy(int)
    row = {"target":target,"task":task_name,"fold":fold,"model":variant,"n":len(y),
           "n_active":int(y.sum()),"n_inactive":int(len(y)-y.sum()),
           "auroc":float(roc_auc_score(y,probability)),"auprc":float(average_precision_score(y,probability)),
           "balanced_accuracy":float(balanced_accuracy_score(y,probability>=.5)),
           "brier":float(brier_score_loss(y,probability))}
    truth.assign(prediction=probability).to_csv(out / "validation_predictions.csv", index=False)
    print(target,fold,variant,f"AUROC={row['auroc']:.3f}",flush=True)
    return row


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows=[]
    for fold in ("fold_a","fold_b"):
        for variant in ("dmpnn","dmpnn_morgan","chemeleon"):
            rows.append(train("EGFR",fold,"confidence_margin_1_0",variant))
    frame=pd.DataFrame(rows); frame.to_csv(OUT/"fold_metrics.csv",index=False)
    summary=(frame.groupby(["target","task","model"],as_index=False)
             .agg(mean_auroc=("auroc","mean"),worst_auroc=("auroc","min"),mean_auprc=("auprc","mean"),
                  mean_balanced_accuracy=("balanced_accuracy","mean"),mean_brier=("brier","mean")))
    summary.to_csv(OUT/"summary.csv",index=False); print(summary.to_string(index=False),flush=True)


if __name__ == "__main__": main()
