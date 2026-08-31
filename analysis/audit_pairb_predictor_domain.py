#!/usr/bin/env python
"""Audit exact training-domain similarity and RF dispersion for PARP1--BRD4.

The deployed oracle is not changed.  This script evaluates its locked 2024+
cohort and the uniformly selected docking candidates against the exact
development-through-2023 structures used to fit each target model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score


TARGETS = {
    "PARP1": "single_protein_assay_ge5",
    "BRD4": "single_protein_assay_ge10",
}
THRESHOLD = 6.5
DOMAIN_THRESHOLDS = (0.40, 0.50, 0.60)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generator():
    return rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048, includeChirality=True
    )


def bit_fingerprints(smiles: pd.Series) -> list:
    gen = generator()
    fingerprints = []
    for index, value in enumerate(smiles.astype(str)):
        mol = Chem.MolFromSmiles(value)
        if mol is None:
            raise ValueError(f"Invalid SMILES at row {index}: {value!r}")
        fingerprints.append(gen.GetFingerprint(mol))
    return fingerprints


def dense_fingerprints(smiles: pd.Series) -> np.ndarray:
    fps = bit_fingerprints(smiles)
    matrix = np.empty((len(fps), 2048), dtype=np.uint8)
    for index, fingerprint in enumerate(fps):
        DataStructs.ConvertToNumpyArray(fingerprint, matrix[index])
    return matrix


def maximum_similarity(query_smiles: pd.Series, reference_smiles: pd.Series) -> np.ndarray:
    reference = bit_fingerprints(reference_smiles)
    values = []
    for fingerprint in bit_fingerprints(query_smiles):
        values.append(max(DataStructs.BulkTanimotoSimilarity(fingerprint, reference)))
    return np.asarray(values, dtype=float)


def rf_predictions(model, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    members = np.vstack([tree.predict(matrix) for tree in model.estimators_])
    return members.mean(axis=0), members.std(axis=0, ddof=1)


def metrics(observed: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    observed = np.asarray(observed, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    active = observed >= THRESHOLD
    result: dict[str, float | int] = {
        "n": int(len(observed)),
        "r2": float(r2_score(observed, prediction)) if len(observed) > 1 else math.nan,
        "rmse": float(math.sqrt(mean_squared_error(observed, prediction))) if len(observed) else math.nan,
        "mae": float(mean_absolute_error(observed, prediction)) if len(observed) else math.nan,
        "spearman": float(stats.spearmanr(observed, prediction).statistic) if len(observed) > 1 else math.nan,
        "positive_rate": float(active.mean()) if len(observed) else math.nan,
        "auroc_at_pactivity_6_5": (
            float(roc_auc_score(active, prediction)) if len(np.unique(active)) == 2 else math.nan
        ),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project = args.project.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    RDLogger.DisableLog("rdApp.error")
    RDLogger.DisableLog("rdApp.warning")

    run = project / "results" / "predictor_parp1_brd4_20260804"
    qualification = run / "rf1000_sqrt_leaf2_locked_qualification"
    model_root = project / "models" / "oracles" / "parp1_brd4_20260804"
    candidates_path = (
        project / "docking" / "parp1_brd4_unified_7method_top5" / "selected_compounds.csv"
    )
    candidates = pd.read_csv(candidates_path, encoding="utf-8-sig")
    locked_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    candidate_output = candidates.copy()
    hashes: dict[str, str] = {"selected_compounds.csv": sha256(candidates_path)}

    for target, profile in TARGETS.items():
        lower = target.lower()
        development_path = run / "data" / lower / profile / "development_through_2023.csv"
        locked_path = qualification / f"{lower}_locked_2024plus_predictions.csv"
        model_path = model_root / f"target_{target}_model.pkl"
        development = pd.read_csv(development_path, encoding="utf-8-sig")
        locked = pd.read_csv(locked_path, encoding="utf-8-sig")
        with model_path.open("rb") as handle:
            model = pickle.load(handle)

        locked_matrix = dense_fingerprints(locked["smiles"])
        prediction, dispersion = rf_predictions(model, locked_matrix)
        if not np.allclose(prediction, locked["prediction"].to_numpy(float), atol=1e-10):
            raise ValueError(f"{target}: deployed model does not reproduce locked predictions")
        locked = locked.copy()
        locked["target"] = target
        locked["max_tanimoto_to_exact_training"] = maximum_similarity(
            locked["smiles"], development["smiles"]
        )
        locked["rf_tree_prediction_sd"] = dispersion
        locked_frames.append(locked)

        full = metrics(locked["pactivity"].to_numpy(float), prediction)
        summary_rows.append(
            {
                "target": target,
                "cohort": "locked_2024plus_all",
                "domain_threshold": math.nan,
                "domain_rate": 1.0,
                "median_max_tanimoto": float(locked["max_tanimoto_to_exact_training"].median()),
                "median_rf_tree_sd": float(locked["rf_tree_prediction_sd"].median()),
                **full,
            }
        )
        for threshold in DOMAIN_THRESHOLDS:
            subset = locked[locked["max_tanimoto_to_exact_training"] >= threshold]
            scored = metrics(subset["pactivity"].to_numpy(float), subset["prediction"].to_numpy(float))
            summary_rows.append(
                {
                    "target": target,
                    "cohort": "locked_2024plus_in_domain",
                    "domain_threshold": threshold,
                    "domain_rate": float(len(subset) / len(locked)),
                    "median_max_tanimoto": (
                        float(subset["max_tanimoto_to_exact_training"].median()) if len(subset) else math.nan
                    ),
                    "median_rf_tree_sd": (
                        float(subset["rf_tree_prediction_sd"].median()) if len(subset) else math.nan
                    ),
                    **scored,
                }
            )

        candidate_matrix = dense_fingerprints(candidates["canonical_smiles"])
        candidate_prediction, candidate_dispersion = rf_predictions(model, candidate_matrix)
        candidate_output[f"{lower}_audit_prediction"] = candidate_prediction
        candidate_output[f"{lower}_rf_tree_prediction_sd"] = candidate_dispersion
        candidate_output[f"{lower}_max_tanimoto_to_exact_training"] = maximum_similarity(
            candidates["canonical_smiles"], development["smiles"]
        )
        candidate_output[f"{lower}_within_domain_0_60"] = (
            candidate_output[f"{lower}_max_tanimoto_to_exact_training"] >= 0.60
        )
        hashes[f"{target}_development_through_2023.csv"] = sha256(development_path)
        hashes[f"{target}_locked_2024plus_predictions.csv"] = sha256(locked_path)
        hashes[f"target_{target}_model.pkl"] = sha256(model_path)

    locked_output = pd.concat(locked_frames, ignore_index=True)
    locked_output.to_csv(output / "locked_2024plus_domain_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summary_rows).to_csv(
        output / "locked_2024plus_domain_summary.csv", index=False, encoding="utf-8-sig"
    )
    candidate_output["within_joint_domain_0_60"] = (
        candidate_output["parp1_within_domain_0_60"]
        & candidate_output["brd4_within_domain_0_60"]
    )
    candidate_output.to_csv(
        output / "selected_candidates_predictor_domain.csv", index=False, encoding="utf-8-sig"
    )
    candidate_summary = (
        candidate_output.groupby("method", as_index=False)
        .agg(
            n=("compound_id", "count"),
            parp1_median_similarity=("parp1_max_tanimoto_to_exact_training", "median"),
            brd4_median_similarity=("brd4_max_tanimoto_to_exact_training", "median"),
            parp1_in_domain_rate=("parp1_within_domain_0_60", "mean"),
            brd4_in_domain_rate=("brd4_within_domain_0_60", "mean"),
            joint_in_domain_rate=("within_joint_domain_0_60", "mean"),
            parp1_median_tree_sd=("parp1_rf_tree_prediction_sd", "median"),
            brd4_median_tree_sd=("brd4_rf_tree_prediction_sd", "median"),
        )
    )
    candidate_summary.to_csv(
        output / "selected_candidates_domain_summary.csv", index=False, encoding="utf-8-sig"
    )
    manifest = {
        "status": "complete",
        "oracle_changed": False,
        "purpose": "exact-training-set applicability-domain and RF-dispersion audit",
        "domain_definition": "maximum chiral ECFP4 Tanimoto to exact development-through-2023 training structures",
        "primary_domain_threshold": 0.60,
        "uncertainty_boundary": "RF tree dispersion is a diagnostic disagreement score, not a calibrated prediction interval",
        "hashes_sha256": hashes,
    }
    (output / "predictor_domain_audit_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(pd.DataFrame(summary_rows).to_string(index=False), flush=True)
    print("\nCandidate domain summary", flush=True)
    print(candidate_summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
