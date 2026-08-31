#!/usr/bin/env python
"""Freeze a VEGFR2 decision threshold using historical folds only."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import balanced_accuracy_score

from benchmark_predictor_round2 import featurize


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "predictor_retraining_v3_20260731" / "data" / "vegfr2"
OUT = ROOT / "results" / "predictor_v41_20260802" / "vegfr2_forest_grid"


def select(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    selected = frame[
        (frame.pactivity <= 5.75) | (frame.pactivity >= 7.25)
    ].reset_index(drop=True)
    return selected, (selected.pactivity >= 7.25).astype(int).to_numpy()


def main() -> None:
    fold_frames = []
    for fold in ("fold_a", "fold_b"):
        train, y_train = select(
            pd.read_csv(
                DATA / "single_protein_wt_or_unspecified" / fold / "train.csv"
            )
        )
        validation, y_validation = select(
            pd.read_csv(DATA / "single_protein_assay_ge5" / fold / "validation.csv")
        )
        model = ExtraTreesClassifier(
            n_estimators=2000,
            max_features="sqrt",
            min_samples_leaf=1,
            criterion="entropy",
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )
        model.fit(featurize(train.smiles), y_train)
        probability = model.predict_proba(featurize(validation.smiles))[:, 1]
        frame = validation[["smiles", "pactivity"]].copy()
        frame["fold"] = fold
        frame["label"] = y_validation
        frame["probability"] = probability
        fold_frames.append(frame)

    historical = pd.concat(fold_frames, ignore_index=True)
    thresholds = np.linspace(0.05, 0.95, 901)
    rows = []
    for threshold in thresholds:
        fold_scores = []
        for fold, frame in historical.groupby("fold"):
            fold_scores.append(
                balanced_accuracy_score(
                    frame.label, frame.probability >= threshold
                )
            )
        rows.append(
            {
                "threshold": float(threshold),
                "mean_balanced_accuracy": float(np.mean(fold_scores)),
                "worst_balanced_accuracy": float(np.min(fold_scores)),
                "fold_a_balanced_accuracy": float(fold_scores[0]),
                "fold_b_balanced_accuracy": float(fold_scores[1]),
            }
        )
    grid = pd.DataFrame(rows).sort_values(
        ["worst_balanced_accuracy", "mean_balanced_accuracy"], ascending=False
    )
    # Deterministic tie-break: prefer the threshold nearest 0.5.
    best_score = grid.iloc[0][
        ["worst_balanced_accuracy", "mean_balanced_accuracy"]
    ].to_numpy()
    tied = grid[
        np.isclose(grid.worst_balanced_accuracy, best_score[0])
        & np.isclose(grid.mean_balanced_accuracy, best_score[1])
    ].copy()
    tied["distance_from_half"] = (tied.threshold - 0.5).abs()
    best = tied.sort_values("distance_from_half").iloc[0]
    threshold = float(best.threshold)

    historical["predicted_label"] = (
        historical.probability >= threshold
    ).astype(int)
    historical.to_csv(OUT / "historical_calibration_predictions.csv", index=False)
    grid.sort_values("threshold").to_csv(OUT / "threshold_grid.csv", index=False)

    latest_results = {}
    for name in ("latest_2024plus", "holdout_2025"):
        path = OUT / f"{name}_predictions.csv"
        frame = pd.read_csv(path)
        probability_column = (
            "probability" if "probability" in frame.columns else "prediction"
        )
        label_column = "label" if "label" in frame.columns else "active_label"
        frame["predicted_label_frozen_threshold"] = (
            frame[probability_column] >= threshold
        ).astype(int)
        score = balanced_accuracy_score(
            frame[label_column], frame.predicted_label_frozen_threshold
        )
        frame.to_csv(OUT / f"{name}_calibrated_predictions.csv", index=False)
        latest_results[name] = {
            "n": int(len(frame)), "balanced_accuracy": float(score)
        }

    result = {
        "target": "VEGFR2",
        "model": "ExtraTrees sqrt leaf1 entropy 2000 trees",
        "threshold_selected_on": "historical rolling-time folds only",
        "frozen_threshold": threshold,
        "historical": {
            key: float(best[key])
            for key in (
                "mean_balanced_accuracy",
                "worst_balanced_accuracy",
                "fold_a_balanced_accuracy",
                "fold_b_balanced_accuracy",
            )
        },
        "latest_application": latest_results,
    }
    (OUT / "calibration_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
