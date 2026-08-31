#!/usr/bin/env python
"""Frozen V4 classifier screen on strict rolling-time validation folds."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, roc_auc_score
from xgboost import XGBClassifier

from benchmark_predictor_round2 import featurize


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config" / "predictor_v4_90plus.json").read_text(encoding="utf-8"))
SOURCE = ROOT / CONFIG["source_run"] / "data"
OUT = ROOT / "results" / CONFIG["run_id"] / "classification_screen"
TARGET_PROFILES = {"EGFR": "single_protein_assay_ge10", "VEGFR2": "single_protein_assay_ge5"}


def select_task(frame: pd.DataFrame, task: dict) -> tuple[pd.DataFrame, np.ndarray]:
    mask = (frame.pactivity <= task["inactive_max"]) | (frame.pactivity >= task["active_min"])
    selected = frame[mask].reset_index(drop=True)
    labels = (selected.pactivity >= task["active_min"]).astype(int).to_numpy()
    return selected, labels


def model_predictions(x_train, y_train, x_val, seed=42):
    ratio = max(1, int((y_train == 0).sum())) / max(1, int((y_train == 1).sum()))
    models = {
        "extratrees": ExtraTreesClassifier(n_estimators=1000, max_features="sqrt", min_samples_leaf=2,
                                            class_weight="balanced", n_jobs=-1, random_state=seed),
        "randomforest": RandomForestClassifier(n_estimators=800, max_features="sqrt", min_samples_leaf=2,
                                                class_weight="balanced", n_jobs=-1, random_state=seed),
        "xgb": XGBClassifier(n_estimators=900, learning_rate=0.03, max_depth=7, min_child_weight=6,
                             subsample=0.85, colsample_bytree=0.65, reg_lambda=8.0, reg_alpha=0.1,
                             objective="binary:logistic", eval_metric="auc", tree_method="hist", device="cuda",
                             scale_pos_weight=ratio, n_jobs=-1, random_state=seed),
    }
    predictions = {}
    timings = {}
    for name, model in models.items():
        start = time.perf_counter()
        model.fit(x_train, y_train)
        predictions[name] = model.predict_proba(x_val)[:, 1]
        timings[name] = time.perf_counter() - start
    return predictions, timings


def knn_probability(train_smiles, y_train, val_smiles, radius=2, k=20, power=3):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=2048)
    train_fp = [generator.GetFingerprint(Chem.MolFromSmiles(s)) for s in train_smiles]
    output = []
    max_similarity = []
    for smiles in val_smiles:
        fp = generator.GetFingerprint(Chem.MolFromSmiles(smiles))
        similarities = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, train_fp), dtype=float)
        idx = np.argsort(similarities)[::-1][:k]
        weights = np.maximum(similarities[idx], 1e-6) ** power
        output.append(float(np.average(y_train[idx], weights=weights)))
        max_similarity.append(float(similarities[idx[0]]))
    return np.asarray(output), np.asarray(max_similarity)


def metrics(y, probability):
    return {
        "n": int(len(y)), "n_active": int(y.sum()), "n_inactive": int(len(y) - y.sum()),
        "auroc": float(roc_auc_score(y, probability)),
        "auprc": float(average_precision_score(y, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(y, probability >= 0.5)),
        "brier": float(brier_score_loss(y, probability)),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for target, profile in TARGET_PROFILES.items():
        for task_name, task in CONFIG["tasks"].items():
            for fold in ("fold_a", "fold_b"):
                fold_dir = SOURCE / target.lower() / profile / fold
                raw_train = pd.read_csv(fold_dir / "train.csv")
                raw_val = pd.read_csv(fold_dir / "validation.csv")
                train, y_train = select_task(raw_train, task)
                val, y_val = select_task(raw_val, task)
                x_train, x_val = featurize(train.smiles), featurize(val.smiles)
                predictions, timings = model_predictions(x_train, y_train, x_val)
                started = time.perf_counter()
                predictions["similarity_knn"] , max_sim = knn_probability(train.smiles, y_train, val.smiles)
                timings["similarity_knn"] = time.perf_counter() - started
                predictions["et_knn_50"] = 0.5 * predictions["extratrees"] + 0.5 * predictions["similarity_knn"]
                predictions["et_xgb_50"] = 0.5 * predictions["extratrees"] + 0.5 * predictions["xgb"]
                timings.update({"et_knn_50": 0.0, "et_xgb_50": 0.0})
                prediction_frame = val[["smiles", "pactivity", "scaffold_seen_in_train"]].copy()
                prediction_frame["label"] = y_val
                prediction_frame["max_train_similarity"] = max_sim
                for name, probability in predictions.items():
                    result = {"target": target, "profile": profile, "task": task_name, "fold": fold,
                              "model": name, "fit_seconds": timings[name], **metrics(y_val, probability)}
                    rows.append(result)
                    prediction_frame[name] = probability
                    print(target, task_name, fold, name, f"AUROC={result['auroc']:.3f}",
                          f"BAcc={result['balanced_accuracy']:.3f}", flush=True)
                prediction_frame.to_csv(OUT / f"{target.lower()}_{task_name}_{fold}_predictions.csv", index=False)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "fold_metrics.csv", index=False)
    summary = (frame.groupby(["target", "profile", "task", "model"], as_index=False)
               .agg(mean_auroc=("auroc", "mean"), worst_auroc=("auroc", "min"),
                    mean_auprc=("auprc", "mean"), worst_auprc=("auprc", "min"),
                    mean_balanced_accuracy=("balanced_accuracy", "mean"), mean_brier=("brier", "mean"),
                    min_validation_n=("n", "min"), min_class_n=("n_inactive", "min"))
               .sort_values(["target", "task", "worst_auroc", "mean_auroc"], ascending=[True, True, False, False]))
    summary.to_csv(OUT / "summary.csv", index=False)
    print(summary.groupby(["target", "task"]).head(5).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
