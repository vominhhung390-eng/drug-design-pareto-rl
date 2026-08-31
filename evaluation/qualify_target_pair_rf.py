#!/usr/bin/env python
"""Qualify and freeze a POLYGON-compatible dual-target RF oracle.

Model selection must be completed before this script is run.  The script
replays the selected specification on the two historical rolling folds, then
fits through 2023 and opens the locked 2024+ set exactly once.  The serialized
models are plain sklearn regressors so all baseline adapters can share the
same oracle implementation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results" / "predictor_parp1_brd4_20260804"
OUTPUT = RUN / "rf1000_sqrt_leaf2_locked_qualification"
MODEL_OUTPUT = RUN / "frozen_oracle_candidate_sqrt_leaf2"
THRESHOLD = 6.5
SEED = 0
MODEL_PARAMS: dict[str, object] = {
    "n_estimators": 1000,
    "max_features": "sqrt",
    "min_samples_leaf": 2,
    "random_state": SEED,
    "n_jobs": -1,
}
MODEL_PRESET = "rolling_selected_sqrt_leaf2"
TARGETS = {
    "PARP1": "single_protein_assay_ge5",
    "BRD4": "single_protein_assay_ge10",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprints(smiles: pd.Series) -> np.ndarray:
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048, includeChirality=True
    )
    matrix = np.empty((len(smiles), 2048), dtype=np.uint8)
    for index, value in enumerate(smiles):
        mol = Chem.MolFromSmiles(value)
        if mol is None:
            raise ValueError(f"Invalid SMILES at row {index}: {value!r}")
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(mol), matrix[index])
    return matrix


def fit(train: pd.DataFrame) -> RandomForestRegressor:
    model = RandomForestRegressor(**MODEL_PARAMS)
    model.fit(fingerprints(train["smiles"]), train["pactivity"].to_numpy(float))
    return model


def metrics(frame: pd.DataFrame, prediction: np.ndarray) -> dict[str, float | int]:
    observed = frame["pactivity"].to_numpy(float)
    active = observed >= THRESHOLD
    result: dict[str, float | int] = {
        "n": int(len(frame)),
        "positive_rate": float(active.mean()),
        "r2": float(r2_score(observed, prediction)),
        "rmse": float(math.sqrt(mean_squared_error(observed, prediction))),
        "mae": float(mean_absolute_error(observed, prediction)),
        "pearson": float(pearsonr(observed, prediction).statistic),
        "spearman": float(spearmanr(observed, prediction).statistic),
        "bias": float(np.mean(prediction - observed)),
    }
    if len(np.unique(active)) == 2:
        result["auroc_at_pactivity_6_5"] = float(roc_auc_score(active, prediction))
        result["auprc_at_pactivity_6_5"] = float(
            average_precision_score(active, prediction)
        )
    else:
        result["auroc_at_pactivity_6_5"] = math.nan
        result["auprc_at_pactivity_6_5"] = math.nan
    return result


def evaluate(
    target: str,
    profile: str,
    cohort: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    save_model: bool,
) -> tuple[dict[str, object], RandomForestRegressor]:
    started = time.perf_counter()
    model = fit(train)
    fit_seconds = time.perf_counter() - started
    prediction = model.predict(fingerprints(test["smiles"]))
    output = test.copy()
    output["prediction"] = prediction
    output.to_csv(
        OUTPUT / f"{target.lower()}_{cohort}_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    row: dict[str, object] = {
        "target": target,
        "profile": profile,
        "cohort": cohort,
        "train_n": int(len(train)),
        "fit_seconds": fit_seconds,
        **metrics(test, prediction),
    }
    if save_model:
        with (MODEL_OUTPUT / f"target_{target}_model.pkl").open("wb") as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return row, model


def main() -> None:
    global MODEL_OUTPUT, MODEL_PARAMS, MODEL_PRESET, OUTPUT
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--deploy",
        action="store_true",
        help="Also copy qualified models to models/oracles/parp1_brd4_20260804.",
    )
    parser.add_argument(
        "--legacy-aligned",
        action="store_true",
        help=(
            "Use the exact RandomForestRegressor hyperparameters recovered from "
            "the prior EGFR/VEGFR2 oracle (1000 trees, max_features=1.0, "
            "min_samples_leaf=1). Results are written to a separate 20260806 run."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help=(
            "Optional independent output directory for a legacy-aligned rerun. "
            "The directory will contain legacy_aligned_validation and "
            "legacy_aligned_models subdirectories."
        ),
    )
    args = parser.parse_args()
    if args.output_root is not None and not args.legacy_aligned:
        parser.error("--output-root currently requires --legacy-aligned")
    if args.legacy_aligned:
        retrain_root = (
            args.output_root.resolve()
            if args.output_root is not None
            else ROOT / "results" / "predictor_parp1_brd4_retrain_20260806"
        )
        OUTPUT = retrain_root / "legacy_aligned_validation"
        MODEL_OUTPUT = retrain_root / "legacy_aligned_models"
        MODEL_PARAMS = {
            "n_estimators": 1000,
            "max_features": 1.0,
            "min_samples_leaf": 1,
            "random_state": SEED,
            "n_jobs": -1,
        }
        MODEL_PRESET = "legacy_egfr_vegfr2_aligned"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    MODEL_OUTPUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    final_models: dict[str, RandomForestRegressor] = {}
    hashes: dict[str, str] = {}

    for target, profile in TARGETS.items():
        base = RUN / "data" / target.lower() / profile
        for fold in ("fold_a", "fold_b"):
            train_path = base / fold / "train.csv"
            test_path = base / fold / "validation.csv"
            train = pd.read_csv(train_path)
            test = pd.read_csv(test_path)
            row, _ = evaluate(target, profile, fold, train, test, save_model=False)
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)

        train_path = base / "development_through_2023.csv"
        test_path = base / "exploratory_2024plus.csv"
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        row, model = evaluate(
            target, profile, "locked_2024plus", train, test, save_model=True
        )
        rows.append(row)
        final_models[target] = model
        hashes[f"{target}_development_through_2023.csv"] = sha256(train_path)
        hashes[f"{target}_locked_2024plus.csv"] = sha256(test_path)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT / "metrics.csv", index=False, encoding="utf-8-sig")
    rolling = frame[frame["cohort"].isin(["fold_a", "fold_b"])]
    rolling_summary = (
        rolling.groupby(["target", "profile"], as_index=False)
        .agg(
            mean_temporal_spearman=("spearman", "mean"),
            worst_temporal_spearman=("spearman", "min"),
            mean_temporal_rmse=("rmse", "mean"),
            worst_temporal_rmse=("rmse", "max"),
            mean_temporal_auroc=("auroc_at_pactivity_6_5", "mean"),
            minimum_temporal_n=("n", "min"),
        )
    )
    rolling_summary.to_csv(
        OUTPUT / "rolling_summary.csv", index=False, encoding="utf-8-sig"
    )

    metadata = {
        "schema_version": "dual-target-rf-v1",
        "target_pair": ["PARP1", "BRD4"],
        "target_ids": {"PARP1": "CHEMBL3105", "BRD4": "CHEMBL1163125"},
        "chembl_release": "ChEMBL_37",
        "endpoint": "human binding IC50, exact relation, nM, single protein",
        "profiles": TARGETS,
        "training_cutoff": 2023,
        "locked_test_start": 2024,
        "activity_threshold_pIC50": THRESHOLD,
        "model": {
            "class": "sklearn.ensemble.RandomForestRegressor",
            "sklearn_version": sklearn.__version__,
            "preset": MODEL_PRESET,
            "n_estimators": MODEL_PARAMS["n_estimators"],
            "max_features": MODEL_PARAMS["max_features"],
            "min_samples_leaf": MODEL_PARAMS["min_samples_leaf"],
            "random_state": MODEL_PARAMS["random_state"],
            "other_hyperparameters": "sklearn defaults",
        },
        "features": {
            "type": "Morgan bit vector (ECFP4)",
            "radius": 2,
            "n_bits": 2048,
            "include_chirality": True,
        },
        "selection_rule": (
            "The target pair and assay-size profiles were fixed previously. For the "
            "legacy-aligned preset, model hyperparameters were copied exactly from "
            "the prior EGFR/VEGFR2 oracle and were not selected to maximize PARP1/BRD4 "
            "hit rates. The 2024+ cohort is a repeat audit because it was opened in "
            "the original 20260804 qualification and is not claimed as a new locked test."
            if args.legacy_aligned
            else
            "Target pair and assay-size profiles selected only on rolling historical "
            "folds; 2024+ compounds remained locked until the original qualification run."
        ),
        "intended_use": "shared relative pActivity ranking oracle for all methods",
        "absolute_activity_claims_allowed": False,
        "data_sha256": hashes,
        "metrics_file": str((OUTPUT / "metrics.csv").relative_to(ROOT)),
        "models": {
            target: str(
                (MODEL_OUTPUT / f"target_{target}_model.pkl").relative_to(ROOT)
            )
            for target in TARGETS
        },
    }
    (MODEL_OUTPUT / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if args.deploy:
        deploy_dir = ROOT / "models" / "oracles" / "parp1_brd4_20260804"
        deploy_dir.mkdir(parents=True, exist_ok=True)
        for target in TARGETS:
            source = MODEL_OUTPUT / f"target_{target}_model.pkl"
            destination = deploy_dir / source.name
            destination.write_bytes(source.read_bytes())
        (deploy_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Deployed to {deploy_dir}", flush=True)

    print("\nRolling summary", flush=True)
    print(rolling_summary.to_string(index=False), flush=True)
    print("\nLocked 2024+", flush=True)
    print(
        frame[frame["cohort"].eq("locked_2024plus")].to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
