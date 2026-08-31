#!/usr/bin/env python
"""Test frozen broader training pools and confidence margins on fixed validation cohorts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, roc_auc_score

from benchmark_predictor_round2 import featurize


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config" / "predictor_v4_90plus.json").read_text(encoding="utf-8"))
V3 = ROOT / CONFIG["source_run"]
OUT = ROOT / "results" / CONFIG["run_id"] / "training_pool_screen"
VALIDATION_PROFILES = {"EGFR": "single_protein_assay_ge10", "VEGFR2": "single_protein_assay_ge5"}


def select(frame, task):
    mask = (frame.pactivity <= task["inactive_max"]) | (frame.pactivity >= task["active_min"])
    data = frame[mask].reset_index(drop=True)
    return data, (data.pactivity >= task["active_min"]).astype(int).to_numpy()


def bindingdb_pool(cutoff, excluded):
    path = V3 / "bindingdb_augmented" / "bindingdb_egfr_exact_ic50.csv"
    data = pd.read_csv(path)
    data = data[(data.document_year <= cutoff) & ~data.smiles.isin(excluded)]
    grouped = data.groupby("smiles", as_index=False).agg(
        pactivity=("pactivity", "median"), low=("pactivity", "min"), high=("pactivity", "max"),
        first_document_year=("document_year", "min"))
    return grouped[(grouped.high - grouped.low) <= 1.0][["smiles", "pactivity", "first_document_year"]]


def build_pool(target, fold, pool, validation):
    base = V3 / "data" / target.lower()
    if pool == "strict":
        return pd.read_csv(base / VALIDATION_PROFILES[target] / fold / "train.csv")
    broad = pd.read_csv(base / "single_protein_wt_or_unspecified" / fold / "train.csv")
    if pool == "broad_single_protein":
        return broad
    cutoff = 2019 if fold == "fold_a" else 2021
    binding = bindingdb_pool(cutoff, set(validation.smiles))
    union = pd.concat([broad[["smiles", "pactivity", "first_document_year"]], binding], ignore_index=True)
    return union.groupby("smiles", as_index=False).agg(pactivity=("pactivity", "median"),
                                                        first_document_year=("first_document_year", "min"))


def knn_all(train, y, validation, ks=(5, 10, 20, 40), radius=2, power=3):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=2048)
    train_fp = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in train.smiles]
    probabilities = {k: [] for k in ks}
    similarities = []
    for smiles in validation.smiles:
        fp = gen.GetFingerprint(Chem.MolFromSmiles(smiles))
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, train_fp))
        order = np.argsort(sims)[::-1]
        similarities.append(float(sims[order[0]]))
        for k in ks:
            idx = order[:k]
            probabilities[k].append(float(np.average(y[idx], weights=np.maximum(sims[idx], 1e-6) ** power)))
    return {k: np.asarray(values) for k, values in probabilities.items()}, np.asarray(similarities)


def metrics(y, prob):
    return {"n": len(y), "n_active": int(y.sum()), "n_inactive": int(len(y)-y.sum()),
            "auroc": float(roc_auc_score(y, prob)), "auprc": float(average_precision_score(y, prob)),
            "balanced_accuracy": float(balanced_accuracy_score(y, prob >= 0.5)),
            "brier": float(brier_score_loss(y, prob))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    pools = {"EGFR": ["strict", "broad_single_protein", "bindingdb_augmented"],
             "VEGFR2": ["strict", "broad_single_protein"]}
    task_names = ["confidence_margin_0_5", "confidence_margin_0_75", "confidence_margin_1_0"]
    for target in ("EGFR", "VEGFR2"):
        for fold in ("fold_a", "fold_b"):
            raw_val = pd.read_csv(V3 / "data" / target.lower() / VALIDATION_PROFILES[target] / fold / "validation.csv")
            for task_name in task_names:
                task = CONFIG["tasks"][task_name]
                val, y_val = select(raw_val, task)
                for pool in pools[target]:
                    raw_train = build_pool(target, fold, pool, raw_val)
                    train, y_train = select(raw_train, task)
                    forest = ExtraTreesClassifier(n_estimators=1200, max_features="sqrt", min_samples_leaf=2,
                                                  class_weight="balanced", n_jobs=-1, random_state=42)
                    forest.fit(featurize(train.smiles), y_train)
                    et_prob = forest.predict_proba(featurize(val.smiles))[:, 1]
                    knn_predictions, max_sim = knn_all(train, y_train, val)
                    for k, knn_prob in knn_predictions.items():
                        for name, prob in {f"knn{k}": knn_prob, f"et_knn{k}_25": 0.25*et_prob+0.75*knn_prob}.items():
                            row = {"target": target, "task": task_name, "fold": fold, "training_pool": pool,
                                   "model": name, "training_n": len(train), **metrics(y_val, prob)}
                            rows.append(row)
                    rows.append({"target": target, "task": task_name, "fold": fold, "training_pool": pool,
                                 "model": "extratrees", "training_n": len(train), **metrics(y_val, et_prob)})
                    print(target, fold, task_name, pool, "done", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "fold_metrics.csv", index=False)
    summary = (frame.groupby(["target","task","training_pool","model"], as_index=False)
               .agg(mean_auroc=("auroc","mean"), worst_auroc=("auroc","min"), mean_auprc=("auprc","mean"),
                    mean_balanced_accuracy=("balanced_accuracy","mean"), mean_brier=("brier","mean"),
                    min_validation_n=("n","min"), min_class_n=("n_inactive","min"), min_training_n=("training_n","min"))
               .sort_values(["target","task","worst_auroc","mean_auroc"], ascending=[True,True,False,False]))
    summary.to_csv(OUT / "summary.csv", index=False)
    print(summary.groupby(["target","task"]).head(8).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
