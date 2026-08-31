#!/usr/bin/env python
"""Stratified bootstrap confidence intervals for frozen V4.1 external tests."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results" / "predictor_v41_20260802"
OUT = RUN / "bootstrap_confidence_intervals.json"


def metric_values(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    return {
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "balanced_accuracy": float(balanced_accuracy_score(y, p >= threshold)),
        "brier": float(brier_score_loss(y, p)),
    }


def stratified_bootstrap(
    y: np.ndarray,
    p: np.ndarray,
    threshold: float,
    n_bootstrap: int = 5000,
    seed: int = 240802,
) -> dict:
    rng = np.random.default_rng(seed)
    negative = np.flatnonzero(y == 0)
    positive = np.flatnonzero(y == 1)
    samples = {name: [] for name in ("auroc", "auprc", "balanced_accuracy", "brier")}
    for _ in range(n_bootstrap):
        indices = np.concatenate(
            [
                rng.choice(negative, size=len(negative), replace=True),
                rng.choice(positive, size=len(positive), replace=True),
            ]
        )
        values = metric_values(y[indices], p[indices], threshold)
        for name, value in values.items():
            samples[name].append(value)
    point = metric_values(y, p, threshold)
    return {
        "n": int(len(y)),
        "n_active": int(y.sum()),
        "n_inactive": int(len(y) - y.sum()),
        "threshold": float(threshold),
        "n_bootstrap": n_bootstrap,
        "metrics": {
            name: {
                "estimate": float(point[name]),
                "ci95_low": float(np.percentile(values, 2.5)),
                "ci95_high": float(np.percentile(values, 97.5)),
            }
            for name, values in samples.items()
        },
    }


def main() -> None:
    egfr = pd.read_csv(RUN / "egfr_bindingdb_external_v2" / "predictions.csv")
    veg_latest = pd.read_csv(
        RUN / "vegfr2_forest_grid" / "latest_2024plus_predictions.csv"
    )
    veg_2025 = pd.read_csv(
        RUN / "vegfr2_forest_grid" / "holdout_2025_predictions.csv"
    )
    veg_probability_column = (
        "probability" if "probability" in veg_latest.columns else "prediction"
    )
    veg_label_column = "label" if "label" in veg_latest.columns else "active_label"
    result = {
        "method": "class-stratified nonparametric bootstrap percentile interval",
        "egfr_bindingdb_2024plus": stratified_bootstrap(
            egfr.active_label.to_numpy(int),
            egfr.prediction.to_numpy(float),
            threshold=0.595,
        ),
        "vegfr2_chembl_2024plus": stratified_bootstrap(
            veg_latest[veg_label_column].to_numpy(int),
            veg_latest[veg_probability_column].to_numpy(float),
            threshold=0.5,
            seed=240803,
        ),
        "vegfr2_chembl_2025_exploratory": stratified_bootstrap(
            veg_2025[veg_label_column].to_numpy(int),
            veg_2025[veg_probability_column].to_numpy(float),
            threshold=0.5,
            seed=240804,
        ),
    }
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
