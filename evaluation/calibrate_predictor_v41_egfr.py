#!/usr/bin/env python
"""Select the EGFR V4.1 classification threshold on historical folds only."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.metrics import balanced_accuracy_score


ROOT = Path(__file__).resolve().parents[1]
DATA = (
    ROOT
    / "results"
    / "predictor_retraining_v3_20260731"
    / "data"
    / "egfr"
    / "single_protein_assay_ge10"
)
SEEDS = ROOT / "results" / "predictor_v41_20260802" / "egfr_seed_ensemble"
EXTERNAL = (
    ROOT / "results" / "predictor_v41_20260802" / "egfr_bindingdb_external_v2"
)
OUT = ROOT / "results" / "predictor_v41_20260802" / "egfr_calibration"


def similarity(train: pd.DataFrame, validation: pd.DataFrame) -> np.ndarray:
    y_train = (train.pactivity >= 7.5).astype(int).to_numpy()
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=1, fpSize=2048)
    train_fp = [
        generator.GetFingerprint(Chem.MolFromSmiles(smiles))
        for smiles in train.smiles
    ]
    output = []
    for smiles in validation.smiles:
        fp = generator.GetFingerprint(Chem.MolFromSmiles(smiles))
        similarities = np.asarray(
            DataStructs.BulkTanimotoSimilarity(fp, train_fp), dtype=float
        )
        indices = np.argsort(similarities)[::-1][:20]
        output.append(
            float(
                np.average(
                    y_train[indices],
                    weights=np.maximum(similarities[indices], 1e-6) ** 6,
                )
            )
        )
    return np.asarray(output)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fold_frames = []
    for fold in ("fold_a", "fold_b"):
        train = pd.read_csv(DATA / fold / "train.csv")
        validation = pd.read_csv(DATA / fold / "validation.csv")
        train = train[(train.pactivity <= 5.5) | (train.pactivity >= 7.5)].reset_index(
            drop=True
        )
        validation = validation[
            (validation.pactivity <= 5.5) | (validation.pactivity >= 7.5)
        ].reset_index(drop=True)
        y = (validation.pactivity >= 7.5).astype(int).to_numpy()
        dmpnn = pd.read_csv(SEEDS / fold / "dmpnn" / "ensemble_predictions.csv")[
            "prediction"
        ].to_numpy(float)
        morgan = pd.read_csv(
            SEEDS / fold / "dmpnn_morgan" / "ensemble_predictions.csv"
        )["prediction"].to_numpy(float)
        knn = similarity(train, validation)
        probability = 0.7 * dmpnn + 0.1 * morgan + 0.2 * knn
        fold_frames.append(
            pd.DataFrame(
                {
                    "fold": fold,
                    "smiles": validation.smiles,
                    "label": y,
                    "probability": probability,
                }
            )
        )

    historical = pd.concat(fold_frames, ignore_index=True)
    rows = []
    for threshold in np.linspace(0.05, 0.95, 901):
        scores = [
            balanced_accuracy_score(frame.label, frame.probability >= threshold)
            for _, frame in historical.groupby("fold")
        ]
        rows.append(
            {
                "threshold": float(threshold),
                "mean_balanced_accuracy": float(np.mean(scores)),
                "worst_balanced_accuracy": float(np.min(scores)),
                "fold_a_balanced_accuracy": float(scores[0]),
                "fold_b_balanced_accuracy": float(scores[1]),
            }
        )
    grid = pd.DataFrame(rows)
    ranked = grid.sort_values(
        ["worst_balanced_accuracy", "mean_balanced_accuracy"], ascending=False
    )
    top = ranked.iloc[0]
    tied = ranked[
        np.isclose(ranked.worst_balanced_accuracy, top.worst_balanced_accuracy)
        & np.isclose(ranked.mean_balanced_accuracy, top.mean_balanced_accuracy)
    ].copy()
    tied["distance_from_half"] = (tied.threshold - 0.5).abs()
    best = tied.sort_values("distance_from_half").iloc[0]
    threshold = float(best.threshold)

    external = pd.read_csv(EXTERNAL / "predictions.csv")
    default_bacc = balanced_accuracy_score(
        external.active_label, external.prediction >= 0.5
    )
    frozen_bacc = balanced_accuracy_score(
        external.active_label, external.prediction >= threshold
    )
    external["predicted_label_historical_threshold"] = (
        external.prediction >= threshold
    ).astype(int)
    external.to_csv(OUT / "external_predictions.csv", index=False)
    historical.to_csv(OUT / "historical_predictions.csv", index=False)
    grid.to_csv(OUT / "threshold_grid.csv", index=False)
    result = {
        "target": "EGFR",
        "threshold_selected_on": "two historical ChEMBL rolling-time folds only",
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
        "bindingdb_external": {
            "n": int(len(external)),
            "balanced_accuracy_default_0_5": float(default_bacc),
            "balanced_accuracy_frozen_threshold": float(frozen_bacc),
        },
    }
    (OUT / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
