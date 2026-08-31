#!/usr/bin/env python
"""Rescore completed PARP1/BRD4 formal runs with a candidate oracle.

This script never overwrites formal generation outputs. It writes a separate
audit containing seed-level metrics, aggregate metrics, pooled hit counts, and
the highest balanced-activity candidates under the supplied model pair.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator


SEEDS = tuple(range(42, 52))
THRESHOLDS = (6.0, 6.5, 7.0)
METHODS = (
    ("Ours (V4-B)", "own_method"),
    ("POLYGON", "baselines/polygon_original"),
    ("REINVENT4", "baselines/reinvent4"),
    ("DrugEx v2", "baselines/drugex_v2"),
    ("MO-LSO", "baselines/mo_lso"),
    ("GraphPareto-NSGA-II", "baselines/graphpareto_nsga2"),
)


def parse_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def fingerprint_chunks(smiles: list[str], chunk_size: int = 4096):
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048, includeChirality=True
    )
    for start in range(0, len(smiles), chunk_size):
        values = smiles[start : start + chunk_size]
        matrix = np.empty((len(values), 2048), dtype=np.uint8)
        for index, value in enumerate(values):
            mol = Chem.MolFromSmiles(value)
            if mol is None:
                raise ValueError(f"Invalid standardized SMILES: {value!r}")
            DataStructs.ConvertToNumpyArray(generator.GetFingerprint(mol), matrix[index])
        yield start, matrix


def predict_pair(
    smiles: list[str], parp1_model: object, brd4_model: object
) -> tuple[np.ndarray, np.ndarray]:
    parp1 = np.empty(len(smiles), dtype=float)
    brd4 = np.empty(len(smiles), dtype=float)
    for start, matrix in fingerprint_chunks(smiles):
        stop = start + len(matrix)
        parp1[start:stop] = parp1_model.predict(matrix)
        brd4[start:stop] = brd4_model.predict(matrix)
    return parp1, brd4


def summarize_subset(prefix: str, parp1: np.ndarray, brd4: np.ndarray) -> dict[str, object]:
    result: dict[str, object] = {
        f"{prefix}_n": int(len(parp1)),
        f"{prefix}_parp1_mean": float(np.mean(parp1)) if len(parp1) else math.nan,
        f"{prefix}_brd4_mean": float(np.mean(brd4)) if len(brd4) else math.nan,
        f"{prefix}_best_min": float(np.minimum(parp1, brd4).max()) if len(parp1) else math.nan,
        f"{prefix}_parp1_max": float(parp1.max()) if len(parp1) else math.nan,
        f"{prefix}_brd4_max": float(brd4.max()) if len(brd4) else math.nan,
    }
    for threshold in THRESHOLDS:
        label = str(threshold).replace(".", "_")
        hits = (parp1 >= threshold) & (brd4 >= threshold)
        result[f"{prefix}_dual_at_{label}"] = float(hits.mean()) if len(hits) else 0.0
        result[f"{prefix}_dual_at_{label}_count"] = int(hits.sum())
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_root", type=Path)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--top-per-method", type=int, default=100)
    args = parser.parse_args()

    RDLogger.DisableLog("rdApp.error")
    RDLogger.DisableLog("rdApp.warning")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.model_dir / "target_PARP1_model.pkl").open("rb") as handle:
        parp1_model = pickle.load(handle)
    with (args.model_dir / "target_BRD4_model.pkl").open("rb") as handle:
        brd4_model = pickle.load(handle)

    records: list[dict[str, object]] = []
    top_frames: list[pd.DataFrame] = []
    for method, relative in METHODS:
        method_root = args.experiment_root / relative
        method_top: list[pd.DataFrame] = []
        for seed in SEEDS:
            evaluation = method_root / f"formal_10240_seed{seed}" / "anytime" / "budget_10240"
            molecule_path = evaluation / "standardized_molecules.csv"
            quality_path = evaluation / "quality_constrained" / "quality_annotated_molecules.csv"
            if not molecule_path.exists() or not quality_path.exists():
                continue
            frame = pd.read_csv(molecule_path, encoding="utf-8-sig")
            quality = pd.read_csv(quality_path, encoding="utf-8-sig")
            if len(frame) != len(quality):
                raise ValueError(f"{method} seed {seed}: quality row-count mismatch")
            if not frame["smiles"].astype(str).equals(quality["smiles"].astype(str)):
                raise ValueError(f"{method} seed {seed}: quality row-order mismatch")

            smiles = frame["smiles"].astype(str).tolist()
            parp1, brd4 = predict_pair(smiles, parp1_model, brd4_model)
            quality_pass = parse_bool(quality["quality_pass"]).to_numpy(bool)
            record: dict[str, object] = {"method": method, "seed": seed}
            record.update(summarize_subset("raw", parp1, brd4))
            record.update(summarize_subset("qc", parp1[quality_pass], brd4[quality_pass]))
            records.append(record)

            candidate = frame[["smiles", "qed", "structural_alert"]].copy()
            candidate["method"] = method
            candidate["seed"] = seed
            candidate["original_parp1"] = frame["egfr"].to_numpy(float)
            candidate["original_brd4"] = frame["vegfr2"].to_numpy(float)
            candidate["candidate_parp1"] = parp1
            candidate["candidate_brd4"] = brd4
            candidate["candidate_min"] = np.minimum(parp1, brd4)
            candidate["quality_pass"] = quality_pass
            method_top.append(candidate.nlargest(args.top_per_method, "candidate_min"))
            print(
                f"{method} seed {seed}: dual@6.5={record['raw_dual_at_6_5']:.4%}, "
                f"QC={record['qc_dual_at_6_5']:.4%}",
                flush=True,
            )

        if method_top:
            combined = pd.concat(method_top, ignore_index=True)
            combined = combined.sort_values("candidate_min", ascending=False)
            combined = combined.drop_duplicates("smiles", keep="first")
            top_frames.append(combined.head(args.top_per_method))

    per_seed = pd.DataFrame(records)
    if per_seed.empty:
        raise RuntimeError("No completed formal runs with quality annotations found")

    metric_columns = [column for column in per_seed.columns if column not in {"method", "seed"}]
    aggregates: list[dict[str, object]] = []
    pooled: list[dict[str, object]] = []
    for method, _ in METHODS:
        subset = per_seed[per_seed["method"] == method]
        if subset.empty:
            continue
        for metric in metric_columns:
            values = subset[metric].astype(float)
            aggregates.append(
                {
                    "method": method,
                    "metric": metric,
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)) if len(values) > 1 else math.nan,
                    "n_seeds": int(len(values)),
                }
            )
        row: dict[str, object] = {"method": method, "n_seeds": int(len(subset))}
        for prefix in ("raw", "qc"):
            denominator = int(subset[f"{prefix}_n"].sum())
            row[f"{prefix}_n"] = denominator
            for threshold in THRESHOLDS:
                label = str(threshold).replace(".", "_")
                numerator = int(subset[f"{prefix}_dual_at_{label}_count"].sum())
                row[f"{prefix}_dual_at_{label}_count"] = numerator
                row[f"{prefix}_dual_at_{label}"] = numerator / denominator if denominator else 0.0
        pooled.append(row)

    per_seed.to_csv(args.output / "per_seed_rescored_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(aggregates).to_csv(
        args.output / "aggregate_rescored_metrics.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(pooled).to_csv(
        args.output / "pooled_rescored_metrics.csv", index=False, encoding="utf-8-sig"
    )
    if top_frames:
        pd.concat(top_frames, ignore_index=True).to_csv(
            args.output / "top_balanced_candidates.csv", index=False, encoding="utf-8-sig"
        )
    (args.output / "rescore_metadata.json").write_text(
        json.dumps(
            {
                "experiment_root": str(args.experiment_root),
                "model_dir": str(args.model_dir),
                "thresholds": list(THRESHOLDS),
                "fingerprint": {
                    "type": "Morgan bit vector (ECFP4)",
                    "radius": 2,
                    "n_bits": 2048,
                    "include_chirality": True,
                },
                "quality_denominator": "quality_pass unique valid molecules",
                "formal_outputs_modified": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote rescore audit to {args.output}", flush=True)


if __name__ == "__main__":
    main()
