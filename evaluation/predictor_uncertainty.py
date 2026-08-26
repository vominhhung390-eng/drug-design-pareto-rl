#!/usr/bin/env python
"""Audit RF tree dispersion as a training-set-free predictor reliability proxy.

Tree dispersion is not a calibrated prediction interval and is not a fingerprint
applicability domain.  For comparisons across generators, use one independent
calibration set (or a saved cutoff JSON) for every method.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

from multiobjective_metrics import hypervolume_2d, pareto_front

RDLogger.DisableLog("rdApp.error")


def fingerprints(smiles: list[str]) -> tuple[np.ndarray, list[int]]:
    gen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    values, indices = [], []
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is not None:
            values.append(np.asarray(gen.GetFingerprintAsNumPy(mol), dtype=np.float32))
            indices.append(i)
    if not values:
        return np.empty((0, 2048), dtype=np.float32), []
    return np.asarray(values, dtype=np.float32), indices


def ensemble_stats(model, x: np.ndarray, batch_size: int) -> dict[str, np.ndarray]:
    mean, std, q05, q95 = [], [], [], []
    for start in range(0, len(x), batch_size):
        xb = x[start : start + batch_size]
        predictions = np.asarray([tree.predict(xb) for tree in model.estimators_])
        mean.append(predictions.mean(axis=0))
        std.append(predictions.std(axis=0))
        q05.append(np.quantile(predictions, 0.05, axis=0))
        q95.append(np.quantile(predictions, 0.95, axis=0))
    return {
        name: np.concatenate(parts) if parts else np.empty(0)
        for name, parts in {"mean": mean, "std": std, "q05": q05, "q95": q95}.items()
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_metadata(path: Path, model) -> dict:
    root_counts = [float(tree.tree_.weighted_n_node_samples[0]) for tree in model.estimators_]
    root_count = int(round(root_counts[0])) if root_counts else None
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "class": type(model).__name__,
        "sklearn_runtime": sklearn.__version__,
        "n_estimators": len(model.estimators_),
        "n_features_in": int(model.n_features_in_),
        "training_samples_retained_by_model": int(getattr(model, "_n_samples", root_count)),
        "root_weighted_sample_count_consistent": bool(
            root_counts and np.allclose(root_counts, root_counts[0])
        ),
        "random_state": model.random_state,
        "bootstrap": bool(model.bootstrap),
        "max_features": model.max_features,
    }


def sample_frame(frame: pd.DataFrame, maximum: int, seed: int) -> pd.DataFrame:
    if maximum and len(frame) > maximum:
        return frame.sample(n=maximum, random_state=seed).reset_index(drop=True)
    return frame.reset_index(drop=True)


def calibration_cutoffs(
    path: Path,
    smiles_column: str,
    maximum: int,
    seed: int,
    quantile: float,
    models: dict,
    batch_size: int,
) -> tuple[dict[str, float], dict]:
    frame = sample_frame(pd.read_csv(path), maximum, seed)
    if smiles_column not in frame.columns:
        raise ValueError(f"Calibration SMILES column {smiles_column!r} not found in {path}")
    x, _ = fingerprints(frame[smiles_column].astype(str).tolist())
    if len(x) == 0:
        raise ValueError("No valid calibration molecules")
    stats = {target: ensemble_stats(model, x, batch_size) for target, model in models.items()}
    cutoffs = {
        target: float(np.quantile(target_stats["std"], quantile))
        for target, target_stats in stats.items()
    }
    details = {
        "cutoff_source": "independent_calibration_set",
        "calibration_path": str(path.resolve()),
        "calibration_rows_read": len(frame),
        "calibration_valid_molecules": len(x),
        "calibration_sampling_seed": seed,
        "dispersion_quantile": quantile,
        "cutoffs": cutoffs,
        "interpretation": (
            "Fixed RF tree-dispersion cutoffs calibrated on an independent molecular set; "
            "not a calibrated error interval and not a predictor-training applicability domain."
        ),
    }
    return cutoffs, details


def points(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    if not set(columns).issubset(frame.columns):
        return np.empty((0, 2))
    values = frame[columns].dropna().to_numpy(float)
    return pareto_front(values) if len(values) else np.empty((0, 2))


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--smiles-column", default="smiles")
    p.add_argument(
        "--egfr-model",
        type=Path,
        default=project_root / "models" / "oracles" / "target_EGFR_model.pkl",
    )
    p.add_argument(
        "--vegfr2-model",
        type=Path,
        default=project_root / "models" / "oracles" / "target_VEGFR2_model.pkl",
    )
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--dispersion-quantile", type=float, default=0.75)
    p.add_argument("--cutoff-json", type=Path)
    p.add_argument("--calibration-input", type=Path)
    p.add_argument("--calibration-smiles-column", default="smiles")
    p.add_argument("--calibration-max-molecules", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.cutoff_json and args.calibration_input:
        p.error("Use either --cutoff-json or --calibration-input, not both")
    if not 0.0 < args.dispersion_quantile < 1.0:
        p.error("--dispersion-quantile must be between 0 and 1")

    frame = pd.read_csv(args.input)
    if args.smiles_column not in frame.columns:
        raise ValueError(f"SMILES column {args.smiles_column!r} not found in {args.input}")
    x, indices = fingerprints(frame[args.smiles_column].astype(str).tolist())
    if len(x) == 0:
        raise ValueError("No valid input molecules")
    frame = frame.iloc[indices].copy().reset_index(drop=True)

    with args.egfr_model.open("rb") as handle:
        egfr_model = pickle.load(handle)
    with args.vegfr2_model.open("rb") as handle:
        vegfr2_model = pickle.load(handle)
    models = {"egfr": egfr_model, "vegfr2": vegfr2_model}
    metadata = {
        target: model_metadata(path, models[target])
        for target, path in {"egfr": args.egfr_model, "vegfr2": args.vegfr2_model}.items()
    }

    if args.cutoff_json:
        calibration = json.loads(args.cutoff_json.read_text(encoding="utf-8"))
        cutoffs = {target: float(calibration["cutoffs"][target]) for target in models}
        calibration = {
            **calibration,
            "cutoff_source": "saved_fixed_cutoffs",
            "cutoff_json": str(args.cutoff_json.resolve()),
        }
    elif args.calibration_input:
        cutoffs, calibration = calibration_cutoffs(
            args.calibration_input,
            args.calibration_smiles_column,
            args.calibration_max_molecules,
            args.seed,
            args.dispersion_quantile,
            models,
            args.batch_size,
        )
    else:
        calibration = {
            "cutoff_source": "input_local_quantile_not_cross_method_comparable",
            "dispersion_quantile": args.dispersion_quantile,
            "interpretation": (
                "Input-local rank split only; not comparable across generators, not a calibrated "
                "error interval, and not a predictor-training applicability domain."
            ),
        }
        cutoffs = {}

    stats = {
        target: ensemble_stats(model, x, args.batch_size) for target, model in models.items()
    }
    if not cutoffs:
        cutoffs = {
            target: float(np.quantile(target_stats["std"], args.dispersion_quantile))
            for target, target_stats in stats.items()
        }
    calibration["cutoffs"] = cutoffs

    for target, target_stats in stats.items():
        for name, values in target_stats.items():
            frame[f"{target}_rf_{name}"] = values
    frame["rf_q05_min"] = frame[["egfr_rf_q05", "vegfr2_rf_q05"]].min(axis=1)
    frame["both_below_fixed_tree_dispersion_cutoff"] = (
        (frame["egfr_rf_std"] <= cutoffs["egfr"])
        & (frame["vegfr2_rf_std"] <= cutoffs["vegfr2"])
    )

    raw_columns = (
        ["egfr", "vegfr2"]
        if {"egfr", "vegfr2"}.issubset(frame.columns)
        else ["egfr_rf_mean", "vegfr2_rf_mean"]
    )
    all_front = points(frame, raw_columns)
    below = frame[frame["both_below_fixed_tree_dispersion_cutoff"]]
    below_front = points(below, raw_columns)
    q05_front = points(frame, ["egfr_rf_q05", "vegfr2_rf_q05"])

    score_agreement = {}
    for target in models:
        if target in frame.columns:
            score_agreement[f"{target}_stored_vs_rf_mean_max_abs_diff"] = float(
                np.max(np.abs(frame[target].to_numpy(float) - frame[f"{target}_rf_mean"].to_numpy(float)))
            )

    summary = {
        "metric_scope": "RF_tree_dispersion_proxy_not_predictor_training_AD",
        "molecules": len(frame),
        "cutoff_source": calibration["cutoff_source"],
        "egfr_tree_std_mean": float(frame["egfr_rf_std"].mean()),
        "egfr_tree_std_median": float(frame["egfr_rf_std"].median()),
        "egfr_tree_std_p90": float(frame["egfr_rf_std"].quantile(0.90)),
        "vegfr2_tree_std_mean": float(frame["vegfr2_rf_std"].mean()),
        "vegfr2_tree_std_median": float(frame["vegfr2_rf_std"].median()),
        "vegfr2_tree_std_p90": float(frame["vegfr2_rf_std"].quantile(0.90)),
        "egfr_tree_std_cutoff": cutoffs["egfr"],
        "vegfr2_tree_std_cutoff": cutoffs["vegfr2"],
        "both_below_cutoff_rate": float(frame["both_below_fixed_tree_dispersion_cutoff"].mean()),
        "raw_score_hypervolume": hypervolume_2d(all_front),
        "below_cutoff_hypervolume": hypervolume_2d(below_front),
        "below_cutoff_pareto_size": len(below_front),
        "tree_q05_hypervolume": hypervolume_2d(q05_front),
        "tree_q05_pareto_size": len(q05_front),
        **score_agreement,
        "interpretation": (
            "Tree-to-tree variation is a heuristic reliability signal. It is not a calibrated "
            "prediction interval and, because the original RF training compounds are unavailable, "
            "it is not a predictor-training applicability-domain test."
        ),
    }

    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / "rf_tree_dispersion_molecules.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([summary]).to_csv(
        args.output / "rf_tree_dispersion_summary.csv", index=False, encoding="utf-8-sig"
    )
    (args.output / "rf_tree_dispersion_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (args.output / "rf_tree_dispersion_cutoffs.json").write_text(
        json.dumps(calibration, indent=2), encoding="utf-8"
    )
    (args.output / "rf_model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
