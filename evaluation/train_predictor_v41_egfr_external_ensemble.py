#!/usr/bin/env python
"""Retrain the frozen EGFR V4.1 ensemble and test on BindingDB 2024+.

The external test set is never used for early stopping, model selection, or
ensemble-weight selection.  All hyperparameters below were frozen using the
two historical ChEMBL rolling-time folds.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / "results" / "predictor_retraining_v3_20260731"
OUT = ROOT / "results" / "predictor_v41_20260802" / "egfr_bindingdb_external_v2"
WRAPPER = ROOT / "evaluation" / "run_chemprop_utf8.py"


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "n": int(len(y)),
        "n_active": int(y.sum()),
        "n_inactive": int(len(y) - y.sum()),
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "balanced_accuracy": float(balanced_accuracy_score(y, p >= 0.5)),
        "brier": float(brier_score_loss(y, p)),
    }


def prepare_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_dir = OUT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    chembl_all = pd.read_csv(
        V3 / "data" / "egfr" / "single_protein_assay_ge10" / "development_through_2023.csv"
    )
    all_chembl_smiles = set(chembl_all.smiles)
    chembl = chembl_all.copy()
    chembl = chembl[(chembl.pactivity <= 5.5) | (chembl.pactivity >= 7.5)].copy()
    chembl["active_label"] = (chembl.pactivity >= 7.5).astype(int)

    # A fixed, stratified internal split is used only for early stopping.  The
    # held-out BindingDB rows below are not exposed to Chemprop during fitting.
    train_idx, val_idx = train_test_split(
        np.arange(len(chembl)),
        test_size=0.10,
        random_state=240731,
        stratify=chembl.active_label,
    )
    train = chembl.iloc[train_idx].reset_index(drop=True)
    validation = chembl.iloc[val_idx].reset_index(drop=True)

    bindingdb = pd.read_csv(
        V3 / "bindingdb_augmented" / "bindingdb_egfr_exact_ic50.csv"
    )
    grouped = (
        bindingdb[bindingdb.document_year >= 2024]
        .groupby("smiles", as_index=False)
        .agg(
            pactivity=("pactivity", "median"),
            pactivity_min=("pactivity", "min"),
            pactivity_max=("pactivity", "max"),
            first_document_year=("document_year", "min"),
            n_measurements=("pactivity", "size"),
        )
    )
    grouped = grouped[
        ((grouped.pactivity_max - grouped.pactivity_min) <= 1.0)
        & (~grouped.smiles.isin(all_chembl_smiles))
    ]
    test = grouped[(grouped.pactivity <= 5.5) | (grouped.pactivity >= 7.5)].copy()
    test["active_label"] = (test.pactivity >= 7.5).astype(int)
    test = test.reset_index(drop=True)

    train[["smiles", "active_label"]].to_csv(data_dir / "train.csv", index=False)
    validation[["smiles", "active_label"]].to_csv(data_dir / "validation.csv", index=False)
    test[["smiles", "active_label"]].to_csv(data_dir / "external_test.csv", index=False)
    return train, validation, test


def train_chemprop(variant: str) -> tuple[np.ndarray, np.ndarray]:
    data_dir = OUT / "data"
    model_dir = OUT / variant
    model_dir.mkdir(parents=True, exist_ok=True)
    expected = model_dir / "model_4" / "test_predictions.csv"
    command = [
        sys.executable,
        "-X",
        "utf8",
        str(WRAPPER),
        "train",
        "-i",
        str(data_dir / "train.csv"),
        str(data_dir / "validation.csv"),
        str(data_dir / "external_test.csv"),
        "-o",
        str(model_dir),
        "--smiles-columns",
        "smiles",
        "--target-columns",
        "active_label",
        "--task-type",
        "classification",
        "--metrics",
        "roc",
        "prc",
        "accuracy",
        "f1",
        "--class-balance",
        "--accelerator",
        "gpu",
        "--devices",
        "1",
        "--num-workers",
        "0",
        "--batch-size",
        "256",
        "--epochs",
        "60",
        "--patience",
        "12",
        "--warmup-epochs",
        "2",
        "--ensemble-size",
        "5",
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
    if not expected.exists():
        env = os.environ.copy()
        env.update(
            {
                "PYTHONUTF8": "1",
                "PYTHONIOENCODING": "utf-8",
                "RICH_FORCE_TERMINAL": "false",
            }
        )
        with (model_dir / "launcher.log").open("w", encoding="utf-8") as log:
            process = subprocess.run(
                command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
            )
        if process.returncode:
            tail = (model_dir / "launcher.log").read_text(
                encoding="utf-8", errors="replace"
            )[-8000:]
            raise RuntimeError(tail)

    members = []
    for index in range(5):
        members.append(
            pd.read_csv(model_dir / f"model_{index}" / "test_predictions.csv")
            .active_label.to_numpy(float)
        )
    matrix = np.vstack(members)
    return matrix.mean(axis=0), matrix.std(axis=0)


def similarity_probability(
    train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    # Frozen historical-fold winner: radius 1, bit Morgan, k=20, power=6.
    reference = pd.concat([train, validation], ignore_index=True)
    labels = reference.active_label.to_numpy(int)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=1, fpSize=2048)
    reference_fp = [
        generator.GetFingerprint(Chem.MolFromSmiles(smiles))
        for smiles in reference.smiles
    ]
    probabilities = []
    max_similarities = []
    for smiles in test.smiles:
        fingerprint = generator.GetFingerprint(Chem.MolFromSmiles(smiles))
        similarities = np.asarray(
            DataStructs.BulkTanimotoSimilarity(fingerprint, reference_fp), dtype=float
        )
        indices = np.argsort(similarities)[::-1][:20]
        weights = np.maximum(similarities[indices], 1e-6) ** 6
        probabilities.append(float(np.average(labels[indices], weights=weights)))
        max_similarities.append(float(similarities[indices[0]]))
    return np.asarray(probabilities), np.asarray(max_similarities)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    train, validation, test = prepare_data()
    print(
        f"train={len(train)} validation={len(validation)} external={len(test)}",
        flush=True,
    )
    dmpnn, dmpnn_std = train_chemprop("dmpnn")
    print("D-MPNN ensemble complete", flush=True)
    morgan, morgan_std = train_chemprop("dmpnn_morgan")
    print("Morgan-DMPNN ensemble complete", flush=True)
    knn, max_similarity = similarity_probability(train, validation, test)
    probability = 0.7 * dmpnn + 0.1 * morgan + 0.2 * knn
    y = test.active_label.to_numpy(int)

    component_metrics = {
        "dmpnn_5seed": metrics(y, dmpnn),
        "morgan_dmpnn_5seed": metrics(y, morgan),
        "similarity_r1_k20_p6": metrics(y, knn),
        "frozen_70_10_20_ensemble": metrics(y, probability),
    }
    test.assign(
        prediction=probability,
        dmpnn=dmpnn,
        dmpnn_std=dmpnn_std,
        dmpnn_morgan=morgan,
        dmpnn_morgan_std=morgan_std,
        similarity_knn=knn,
        max_train_similarity=max_similarity,
    ).to_csv(OUT / "predictions.csv", index=False)
    result = {
        "target": "EGFR",
        "external_source": "BindingDB",
        "external_year": "2024+",
        "training_source": "ChEMBL through 2023",
        "decision_thresholds": {"inactive_max": 5.5, "active_min": 7.5},
        "no_external_tuning": True,
        "components": component_metrics,
    }
    (OUT / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
