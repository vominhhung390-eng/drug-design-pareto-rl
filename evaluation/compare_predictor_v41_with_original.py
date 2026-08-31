#!/usr/bin/env python
"""Score the original locked RF predictors on the frozen V4.1 holdouts."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score

from validate_target_predictors import fingerprints


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results" / "predictor_v41_20260802"
OUT = RUN / "original_v0_same_holdout_comparison.json"


def original_prediction(target: str, smiles: pd.Series) -> np.ndarray:
    path = ROOT / "models" / "oracles" / f"target_{target}_model.pkl"
    with path.open("rb") as handle:
        model = pickle.load(handle)
    x, _ = fingerprints(smiles, use_chirality=True)
    return model.predict(x).astype(float)


def ranking_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
    }


def bootstrap_metrics(
    y: np.ndarray,
    score: np.ndarray,
    predicted_label: np.ndarray,
    seed: int,
    n_bootstrap: int = 5000,
) -> dict:
    rng = np.random.default_rng(seed)
    negative = np.flatnonzero(y == 0)
    positive = np.flatnonzero(y == 1)
    values = {"auroc": [], "auprc": [], "balanced_accuracy": []}
    for _ in range(n_bootstrap):
        indices = np.concatenate(
            [
                rng.choice(negative, size=len(negative), replace=True),
                rng.choice(positive, size=len(positive), replace=True),
            ]
        )
        values["auroc"].append(roc_auc_score(y[indices], score[indices]))
        values["auprc"].append(average_precision_score(y[indices], score[indices]))
        values["balanced_accuracy"].append(
            balanced_accuracy_score(y[indices], predicted_label[indices])
        )
    return {
        name: {
            "ci95_low": float(np.percentile(samples, 2.5)),
            "ci95_high": float(np.percentile(samples, 97.5)),
        }
        for name, samples in values.items()
    }


def main() -> None:
    egfr = pd.read_csv(RUN / "egfr_bindingdb_external_v2" / "predictions.csv")
    veg = pd.read_csv(RUN / "vegfr2_forest_grid" / "latest_2024plus_predictions.csv")

    veg_label_column = "label" if "label" in veg.columns else "active_label"
    veg_probability_column = "probability" if "probability" in veg.columns else "prediction"

    egfr_y = egfr.active_label.to_numpy(int)
    veg_y = veg[veg_label_column].to_numpy(int)
    egfr_original = original_prediction("EGFR", egfr.smiles)
    veg_original = original_prediction("VEGFR2", veg.smiles)
    egfr_current = egfr.prediction.to_numpy(float)
    veg_current = veg[veg_probability_column].to_numpy(float)

    result = {
        "comparison_rule": (
            "same frozen molecules and labels; AUROC/AUPRC are directly comparable. "
            "Original balanced accuracy uses its pActivity midpoint threshold 6.5."
        ),
        "original_models": {
            "type": "RandomForestRegressor, 1000 trees, operational chiral ECFP4",
            "egfr_training_samples_retained": 1096,
            "vegfr2_training_samples_retained": 723,
        },
        "EGFR": {
            "holdout": "BindingDB 2024+, ChEMBL-overlap SMILES excluded",
            "n": int(len(egfr)),
            "original": {
                **ranking_metrics(egfr_y, egfr_original),
                "balanced_accuracy_at_pactivity_6_5": float(
                    balanced_accuracy_score(egfr_y, egfr_original >= 6.5)
                ),
                "bootstrap_ci95": bootstrap_metrics(
                    egfr_y, egfr_original, egfr_original >= 6.5, seed=240805
                ),
            },
            "v4_1": {
                **ranking_metrics(egfr_y, egfr_current),
                "balanced_accuracy_at_frozen_threshold": float(
                    balanced_accuracy_score(egfr_y, egfr_current >= 0.595)
                ),
            },
        },
        "VEGFR2": {
            "holdout": "ChEMBL 2024+ high-confidence time holdout",
            "n": int(len(veg)),
            "original": {
                **ranking_metrics(veg_y, veg_original),
                "balanced_accuracy_at_pactivity_6_5": float(
                    balanced_accuracy_score(veg_y, veg_original >= 6.5)
                ),
                "bootstrap_ci95": bootstrap_metrics(
                    veg_y, veg_original, veg_original >= 6.5, seed=240806
                ),
            },
            "v4_1": {
                **ranking_metrics(veg_y, veg_current),
                "balanced_accuracy_at_threshold_0_5": float(
                    balanced_accuracy_score(veg_y, veg_current >= 0.5)
                ),
            },
        },
    }
    for target in ("EGFR", "VEGFR2"):
        result[target]["delta"] = {
            "auroc_absolute": (
                result[target]["v4_1"]["auroc"]
                - result[target]["original"]["auroc"]
            ),
            "auprc_absolute": (
                result[target]["v4_1"]["auprc"]
                - result[target]["original"]["auprc"]
            ),
        }
    egfr.assign(
        original_prediction=egfr_original,
        original_predicted_label=(egfr_original >= 6.5).astype(int),
    ).to_csv(RUN / "original_v0_egfr_same_holdout_predictions.csv", index=False)
    veg.assign(
        original_prediction=veg_original,
        original_predicted_label=(veg_original >= 6.5).astype(int),
    ).to_csv(RUN / "original_v0_vegfr2_same_holdout_predictions.csv", index=False)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
