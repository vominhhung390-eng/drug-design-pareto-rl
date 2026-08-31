#!/usr/bin/env python
"""Select an EGFR BindingDB-native forest on pre-2024 time folds."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)

from benchmark_predictor_round2 import featurize


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "results"
    / "predictor_retraining_v3_20260731"
    / "bindingdb_augmented"
    / "bindingdb_egfr_exact_ic50.csv"
)
OUT = ROOT / "results" / "predictor_v41_20260802" / "egfr_bindingdb_native_grid"


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


def load_data() -> pd.DataFrame:
    raw = pd.read_csv(SOURCE)
    data = raw.groupby("smiles", as_index=False).agg(
        pactivity=("pactivity", "median"),
        low=("pactivity", "min"),
        high=("pactivity", "max"),
        first_year=("document_year", "min"),
        last_year=("document_year", "max"),
        n_measurements=("pactivity", "size"),
    )
    data = data[(data.high - data.low) <= 1.0]
    data = data[(data.pactivity <= 5.5) | (data.pactivity >= 7.5)].copy()
    data["label"] = (data.pactivity >= 7.5).astype(int)
    return data.reset_index(drop=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_data()
    folds = {
        "fold_a": (data.first_year <= 2019, data.first_year.between(2020, 2021)),
        "fold_b": (data.first_year <= 2021, data.first_year.between(2022, 2023)),
    }
    specs = [
        (max_features, min_leaf, criterion)
        for max_features in ("sqrt", 0.05, 0.1, 0.2, 0.4)
        for min_leaf in (1, 2, 4, 8)
        for criterion in ("gini", "entropy")
    ]
    rows = []
    for fold, (train_mask, validation_mask) in folds.items():
        train = data[train_mask].reset_index(drop=True)
        validation = data[validation_mask].reset_index(drop=True)
        x_train, x_validation = featurize(train.smiles), featurize(validation.smiles)
        y_train = train.label.to_numpy(int)
        y_validation = validation.label.to_numpy(int)
        for max_features, min_leaf, criterion in specs:
            name = f"mf{max_features}_leaf{min_leaf}_{criterion}"
            model = ExtraTreesClassifier(
                n_estimators=400,
                max_features=max_features,
                min_samples_leaf=min_leaf,
                criterion=criterion,
                class_weight="balanced",
                n_jobs=-1,
                random_state=42,
            )
            model.fit(x_train, y_train)
            probability = model.predict_proba(x_validation)[:, 1]
            rows.append(
                {
                    "fold": fold,
                    "model": name,
                    "max_features": max_features,
                    "min_samples_leaf": min_leaf,
                    "criterion": criterion,
                    **metrics(y_validation, probability),
                }
            )
        print(f"{fold} complete", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "fold_metrics.csv", index=False)
    summary = (
        frame.groupby(
            ["model", "max_features", "min_samples_leaf", "criterion"],
            as_index=False,
        )
        .agg(
            mean_auroc=("auroc", "mean"),
            worst_auroc=("auroc", "min"),
            mean_auprc=("auprc", "mean"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            mean_brier=("brier", "mean"),
        )
        .sort_values(["worst_auroc", "mean_auroc"], ascending=False)
    )
    summary.to_csv(OUT / "summary.csv", index=False)
    best = summary.iloc[0]

    train = data[data.first_year <= 2023].reset_index(drop=True)
    test = data[data.first_year >= 2024].reset_index(drop=True)
    model = ExtraTreesClassifier(
        n_estimators=2000,
        max_features=best.max_features,
        min_samples_leaf=int(best.min_samples_leaf),
        criterion=best.criterion,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(featurize(train.smiles), train.label)
    probability = model.predict_proba(featurize(test.smiles))[:, 1]
    test.assign(prediction=probability).to_csv(
        OUT / "external_2024plus_predictions.csv", index=False
    )
    result = {
        "target": "EGFR",
        "source": "BindingDB native exact IC50",
        "selection": "two pre-2024 rolling-time folds",
        "frozen_model": {
            "max_features": best.max_features,
            "min_samples_leaf": int(best.min_samples_leaf),
            "criterion": best.criterion,
            "historical_mean_auroc": float(best.mean_auroc),
            "historical_worst_auroc": float(best.worst_auroc),
        },
        "external_2024plus": metrics(test.label.to_numpy(int), probability),
    }
    (OUT / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(summary.head(10).to_string(index=False), flush=True)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
