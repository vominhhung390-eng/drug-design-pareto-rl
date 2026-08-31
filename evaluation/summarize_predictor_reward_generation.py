#!/usr/bin/env python
"""Cross-evaluate V4-B generations driven by the original RF and V4.1 oracles."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results" / "own_method_v4" / "predictor_reward_generation_20260802"
OLD = ROOT / "results" / "own_method_v4" / "common_seeds_42_51_10240"
CACHE = (
    ROOT
    / "results"
    / "own_method_v4"
    / "predictor_crosscheck_20260802"
    / "unique_predictions.csv"
)
SEEDS = tuple(range(42, 52))
RDLogger.DisableLog("rdApp.*")


def unit_hypervolume(points: np.ndarray) -> float:
    """Exact two-objective maximization HV in [0,1]^2, O(n log n)."""
    points = np.clip(np.asarray(points, dtype=float), 0.0, 1.0)
    if not len(points):
        return 0.0
    order = np.lexsort((-points[:, 1], -points[:, 0]))
    best_y = -np.inf
    front = []
    for index in order:
        y_value = points[index, 1]
        if y_value > best_y:
            front.append(points[index])
            best_y = y_value
    front = np.asarray(front, dtype=float)
    front = front[np.argsort(front[:, 0])]
    result = 0.0
    best_y = 0.0
    for x_value, y_value in front[::-1]:
        if y_value > best_y:
            result += x_value * (y_value - best_y)
            best_y = y_value
    return float(result)


def rf_hv(points: np.ndarray) -> float:
    return unit_hypervolume((np.asarray(points, dtype=float) - 3.0) / 7.0)


def fingerprints(smiles: pd.Series) -> np.ndarray:
    output = np.empty((len(smiles), 2048), dtype=np.float32)
    for index, value in enumerate(smiles):
        mol = Chem.MolFromSmiles(value)
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=2, nBits=2048, useChirality=True
        )
        DataStructs.ConvertToNumpyArray(fingerprint, output[index])
    return output


def load_rf_models():
    models = []
    for target in ("EGFR", "VEGFR2"):
        with (ROOT / "models" / "oracles" / f"target_{target}_model.pkl").open("rb") as handle:
            models.append(pickle.load(handle))
    return models


def valid_unique(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    valid = frame.is_valid
    if valid.dtype != bool:
        valid = valid.astype(str).str.lower().eq("true")
    frame = frame[valid & frame.canonical_smiles.notna()].copy()
    return frame.drop_duplicates("canonical_smiles").reset_index(drop=True)


def paired_summary(original: np.ndarray, v41: np.ndarray, seed: int = 20260802) -> dict:
    delta = np.asarray(v41, dtype=float) - np.asarray(original, dtype=float)
    rng = np.random.default_rng(seed)
    boot = np.empty(100000, dtype=float)
    for index in range(len(boot)):
        draw = rng.integers(0, len(delta), len(delta))
        boot[index] = delta[draw].mean()
    try:
        wilcoxon_p = float(stats.wilcoxon(delta).pvalue)
    except ValueError:
        wilcoxon_p = 1.0
    return {
        "n_pairs": int(len(delta)),
        "original_generation_mean": float(np.mean(original)),
        "v41_generation_mean": float(np.mean(v41)),
        "paired_delta_v41_minus_original": float(delta.mean()),
        "paired_delta_sd": float(delta.std(ddof=1)),
        "paired_bootstrap_95_ci": [
            float(np.quantile(boot, 0.025)),
            float(np.quantile(boot, 0.975)),
        ],
        "paired_ttest_p": float(stats.ttest_rel(v41, original).pvalue),
        "wilcoxon_p": wilcoxon_p,
        "v41_wins": int((delta > 0).sum()),
        "ties": int((delta == 0).sum()),
        "original_wins": int((delta < 0).sum()),
    }


def main() -> None:
    cache = pd.read_csv(CACHE)[
        ["smiles", "v41_egfr_probability", "v41_vegfr2_probability"]
    ].drop_duplicates("smiles")
    cache = cache.set_index("smiles")
    egfr_rf, vegfr2_rf = load_rf_models()
    rows = []
    reproduction_deltas = []

    for generation_oracle in ("original_rf", "v41"):
        for seed in SEEDS:
            directory = RUN / generation_oracle / f"seed{seed}"
            summary = pd.read_csv(directory / "summary.csv").iloc[0]
            frame = valid_unique(directory / "all_generated_molecules.csv")
            smiles = frame.canonical_smiles.astype(str)

            if generation_oracle == "original_rf":
                rf_points = frame[["egfr", "vegfr2"]].to_numpy(float)
                missing = [value for value in smiles if value not in cache.index]
                if missing:
                    pd.DataFrame({"smiles": sorted(set(missing))}).to_csv(
                        RUN / "missing_original_generation_v41_scores.csv", index=False
                    )
                    raise RuntimeError(
                        f"{len(set(missing))} original-generation molecules lack frozen V4.1 scores"
                    )
                v41_points = cache.loc[smiles][
                    ["v41_egfr_probability", "v41_vegfr2_probability"]
                ].to_numpy(float)
                old_summary = json.loads(
                    (
                        OLD
                        / f"v4_b_raw_mean_seed{seed}"
                        / "evaluation"
                        / "evaluation_summary.json"
                    ).read_text(encoding="utf-8")
                )
                reproduction_deltas.append(
                    abs(float(summary.hv_final) - float(old_summary["hypervolume"]))
                )
            else:
                v41_points = np.clip(
                    (frame[["egfr", "vegfr2"]].to_numpy(float) - 3.0) / 7.0,
                    0.0,
                    1.0,
                )
                features = fingerprints(smiles)
                rf_points = np.column_stack(
                    [egfr_rf.predict(features), vegfr2_rf.predict(features)]
                )

            rows.append(
                {
                    "generation_oracle": generation_oracle,
                    "seed": seed,
                    "valid_unique": int(len(frame)),
                    "validity": float(summary.valid_rate),
                    "self_reported_hv": float(summary.hv_final),
                    "rf_evaluator_hv": rf_hv(rf_points),
                    "v41_evaluator_hv": unit_hypervolume(v41_points),
                }
            )

    detail = pd.DataFrame(rows)
    detail.to_csv(RUN / "cross_evaluated_hv_per_seed.csv", index=False)
    aggregates = []
    for oracle, group in detail.groupby("generation_oracle"):
        for metric in (
            "valid_unique",
            "validity",
            "self_reported_hv",
            "rf_evaluator_hv",
            "v41_evaluator_hv",
        ):
            values = group[metric].to_numpy(float)
            aggregates.append(
                {
                    "generation_oracle": oracle,
                    "metric": metric,
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)),
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
            )
    pd.DataFrame(aggregates).to_csv(RUN / "cross_evaluated_hv_aggregate.csv", index=False)

    pivot_rf = detail.pivot(index="seed", columns="generation_oracle", values="rf_evaluator_hv")
    pivot_v41 = detail.pivot(index="seed", columns="generation_oracle", values="v41_evaluator_hv")
    result = {
        "protocol": "paired predictor-as-reward generation, seeds 42-51, budget 10240",
        "original_formal_reproduction_max_abs_hv_delta": float(max(reproduction_deltas)),
        "rf_evaluator": paired_summary(
            pivot_rf.original_rf.to_numpy(), pivot_rf.v41.to_numpy()
        ),
        "v41_evaluator": paired_summary(
            pivot_v41.original_rf.to_numpy(), pivot_v41.v41.to_numpy()
        ),
    }
    (RUN / "paired_hv_tests.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
