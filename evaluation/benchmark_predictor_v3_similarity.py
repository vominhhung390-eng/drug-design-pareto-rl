#!/usr/bin/env python
"""Similarity-neighbour and fixed ensemble screen for EGFR V3."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results" / "predictor_retraining_v3_20260731"
PROFILE = "single_protein_assay_ge10"
OUT = RUN / "similarity_benchmark"


def score(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "spearman": float(spearmanr(y, pred).statistic),
        "rmse": float(mean_squared_error(y, pred) ** 0.5),
        "mae": float(mean_absolute_error(y, pred)),
        "r2": float(r2_score(y, pred)),
    }


def neighbour_predictions(train: pd.DataFrame, valid: pd.DataFrame, radius: int, count: bool) -> tuple[dict, np.ndarray]:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=2048)
    getter = generator.GetCountFingerprint if count else generator.GetFingerprint
    train_fp = [getter(Chem.MolFromSmiles(s)) for s in train["smiles"]]
    valid_fp = [getter(Chem.MolFromSmiles(s)) for s in valid["smiles"]]
    y_train = train["pactivity"].to_numpy(float)
    specs = [(k, power) for k in (1, 3, 5, 10, 20, 50) for power in (1, 3)]
    predictions = {spec: [] for spec in specs}
    max_similarity = []
    for fp in valid_fp:
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, train_fp), dtype=float)
        order = np.argsort(sims)[::-1]
        max_similarity.append(float(sims[order[0]]))
        for k, power in specs:
            idx = order[:k]
            weights = np.maximum(sims[idx], 1e-6) ** power
            predictions[(k, power)].append(float(np.average(y_train[idx], weights=weights)))
    return {key: np.asarray(value) for key, value in predictions.items()}, np.asarray(max_similarity)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    prediction_frames = []
    for fold in ("fold_a", "fold_b"):
        data = RUN / "data" / "egfr" / PROFILE / fold
        train = pd.read_csv(data / "train.csv")
        valid = pd.read_csv(data / "validation.csv")
        y = valid["pactivity"].to_numpy(float)
        et_file = RUN / "classical_benchmark" / f"egfr_{PROFILE}_{fold}_extratrees_recent_hl6_regression.csv"
        et_pred = pd.read_csv(et_file)["prediction"].to_numpy(float)

        for radius in (2, 3, 4):
            for count in (False, True):
                predictions, max_sim = neighbour_predictions(train, valid, radius, count)
                fp_name = f"morgan_r{radius}_{'count' if count else 'bit'}"
                for (k, power), pred in predictions.items():
                    model = f"{fp_name}_knn{k}_p{power}"
                    result = {"fold": fold, "model": model, "n": len(y), **score(y, pred)}
                    rows.append(result)
                    prediction_frames.append(pd.DataFrame({
                        "fold": fold, "model": model, "smiles": valid["smiles"],
                        "observed": y, "prediction": pred, "max_train_similarity": max_sim,
                    }))
                    for et_weight in (0.5, 0.75):
                        ensemble = et_weight * et_pred + (1.0 - et_weight) * pred
                        ensemble_name = f"et{int(et_weight*100)}_{model}"
                        rows.append({"fold": fold, "model": ensemble_name, "n": len(y), **score(y, ensemble)})
                        prediction_frames.append(pd.DataFrame({
                            "fold": fold, "model": ensemble_name, "smiles": valid["smiles"],
                            "observed": y, "prediction": ensemble, "max_train_similarity": max_sim,
                        }))

        rows.append({"fold": fold, "model": "extratrees_recent_hl6", "n": len(y), **score(y, et_pred)})
        print(f"{fold} complete", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "fold_metrics.csv", index=False)
    summary = (
        frame.groupby("model", as_index=False)
        .agg(mean_spearman=("spearman", "mean"), worst_spearman=("spearman", "min"),
             mean_rmse=("rmse", "mean"), worst_rmse=("rmse", "max"))
        .sort_values(["mean_spearman", "worst_spearman"], ascending=False)
    )
    summary.to_csv(OUT / "summary.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(OUT / "all_predictions.csv", index=False)
    (OUT / "screen_definition.json").write_text(json.dumps({
        "radii": [2, 3, 4], "fingerprint_types": ["bit", "count"],
        "neighbors": [1, 3, 5, 10, 20, 50], "similarity_powers": [1, 3],
        "fixed_extratrees_weights": [0.5, 0.75],
        "selection_rule": "highest mean Spearman, then highest worst-fold Spearman",
    }, indent=2), encoding="utf-8")
    print(summary.head(15).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
