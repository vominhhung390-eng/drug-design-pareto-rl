#!/usr/bin/env python
"""Five-fold scaffold-grouped comparison of two fixed RF specifications."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold


THRESHOLD = 6.5
TARGETS = {
    "PARP1": "single_protein_assay_ge5",
    "BRD4": "single_protein_assay_ge10",
}
PRESETS = {
    "current_sqrt_leaf2": {"max_features": "sqrt", "min_samples_leaf": 2},
    "legacy_aligned": {"max_features": 1.0, "min_samples_leaf": 1},
}


def fingerprints(smiles: pd.Series) -> np.ndarray:
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048, includeChirality=True
    )
    matrix = np.empty((len(smiles), 2048), dtype=np.uint8)
    for index, value in enumerate(smiles.astype(str)):
        mol = Chem.MolFromSmiles(value)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {value!r}")
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(mol), matrix[index])
    return matrix


def evaluate(observed: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    active = observed >= THRESHOLD
    predicted_active = prediction >= THRESHOLD
    result: dict[str, float | int] = {
        "n": int(len(observed)),
        "positive_rate": float(active.mean()),
        "predicted_positive_rate": float(predicted_active.mean()),
        "r2": float(r2_score(observed, prediction)),
        "rmse": float(math.sqrt(mean_squared_error(observed, prediction))),
        "mae": float(mean_absolute_error(observed, prediction)),
        "spearman": float(spearmanr(observed, prediction).statistic),
        "bias": float(np.mean(prediction - observed)),
    }
    if len(np.unique(active)) == 2:
        result.update(
            {
                "auroc": float(roc_auc_score(active, prediction)),
                "auprc": float(average_precision_score(active, prediction)),
                "balanced_accuracy": float(
                    balanced_accuracy_score(active, predicted_active)
                ),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    RDLogger.DisableLog("rdApp.error")
    RDLogger.DisableLog("rdApp.warning")

    rows: list[dict[str, object]] = []
    for target, profile in TARGETS.items():
        path = args.data_root / target.lower() / profile / "development_through_2023.csv"
        frame = pd.read_csv(path, encoding="utf-8-sig")
        frame = frame[frame["scaffold"].fillna("").astype(str).str.len() > 0].reset_index(drop=True)
        matrix = fingerprints(frame["smiles"])
        observed = frame["pactivity"].to_numpy(float)
        groups = frame["scaffold"].astype(str).to_numpy()
        try:
            splitter = GroupKFold(n_splits=args.folds, shuffle=True, random_state=0)
        except TypeError:
            splitter = GroupKFold(n_splits=args.folds)
        splits = list(splitter.split(matrix, observed, groups))
        for preset, extra in PRESETS.items():
            for fold, (train_index, test_index) in enumerate(splits, start=1):
                started = time.perf_counter()
                model = RandomForestRegressor(
                    n_estimators=1000,
                    max_features=extra["max_features"],
                    min_samples_leaf=extra["min_samples_leaf"],
                    random_state=0,
                    n_jobs=-1,
                )
                model.fit(matrix[train_index], observed[train_index])
                prediction = model.predict(matrix[test_index])
                row = {
                    "target": target,
                    "profile": profile,
                    "preset": preset,
                    "fold": fold,
                    "train_n": int(len(train_index)),
                    "unique_train_scaffolds": int(len(np.unique(groups[train_index]))),
                    "unique_test_scaffolds": int(len(np.unique(groups[test_index]))),
                    "fit_and_predict_seconds": time.perf_counter() - started,
                    **evaluate(observed[test_index], prediction),
                }
                rows.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)

    per_fold = pd.DataFrame(rows)
    per_fold.to_csv(args.output / "scaffold_fold_metrics.csv", index=False, encoding="utf-8-sig")
    summary = (
        per_fold.groupby(["target", "profile", "preset"], as_index=False)
        .agg(
            mean_spearman=("spearman", "mean"),
            worst_spearman=("spearman", "min"),
            mean_rmse=("rmse", "mean"),
            worst_rmse=("rmse", "max"),
            mean_auroc=("auroc", "mean"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            mean_bias=("bias", "mean"),
        )
    )
    summary.to_csv(args.output / "scaffold_summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
