#!/usr/bin/env python
"""Evaluate a standardized molecule CSV with Pareto and chemistry metrics."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, FilterCatalog, QED
from rdkit.Chem.Scaffolds import MurckoScaffold

from multiobjective_metrics import hypervolume_2d, igd_plus, pareto_front, spacing, spread


# Invalid generator outputs are counted through ``valid_rows``.  Suppress the
# corresponding per-row parser diagnostics so large formal evaluations remain
# readable without changing any metric.
RDLogger.DisableLog("rdApp.error")
RDLogger.DisableLog("rdApp.warning")


def canonicalize(smiles: str) -> tuple[str | None, Chem.Mol | None]:
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None, None
        fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
        if not fragments:
            return None, None
        mol = max(fragments, key=lambda item: item.GetNumHeavyAtoms())
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True), mol
    except (ValueError, RuntimeError, Chem.rdchem.KekulizeException):
        return None, None


def scaffold(mol: Chem.Mol) -> str:
    core = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(core, canonical=True) if core.GetNumAtoms() else ""


def make_alert_catalog() -> FilterCatalog.FilterCatalog:
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
    return FilterCatalog.FilterCatalog(params)


def load_reference(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    frame = pd.read_csv(path)
    return frame[["egfr", "vegfr2"]].dropna().to_numpy(float)


def evaluate(args: argparse.Namespace) -> None:
    frame = pd.read_csv(args.input)
    required = {args.smiles_column, args.egfr_column, args.vegfr2_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    generated = len(frame)
    records = []
    alerts = make_alert_catalog()
    selected = frame[[args.smiles_column, args.egfr_column, args.vegfr2_column]]
    for raw, egfr_raw, vegfr2_raw in selected.itertuples(index=False, name=None):
        can, mol = canonicalize(raw)
        if mol is None:
            continue
        egfr = float(egfr_raw)
        vegfr2 = float(vegfr2_raw)
        if not (math.isfinite(egfr) and math.isfinite(vegfr2)):
            continue
        records.append(
            {
                "smiles": can,
                "egfr": egfr,
                "vegfr2": vegfr2,
                "qed": float(QED.qed(mol)),
                "mol_wt": float(Descriptors.MolWt(mol)),
                "logp": float(Descriptors.MolLogP(mol)),
                "scaffold": scaffold(mol),
                "structural_alert": bool(alerts.HasMatch(mol)),
            }
        )
    valid_rows = len(records)
    valid = pd.DataFrame(records)
    unique = valid.sort_values(["egfr", "vegfr2"], ascending=False).drop_duplicates("smiles")
    points = unique[["egfr", "vegfr2"]].to_numpy(float) if len(unique) else np.empty((0, 2))
    mask = np.zeros(len(unique), dtype=bool)
    if len(unique):
        front_points = pareto_front(points)
        front_keys = {tuple(x) for x in front_points.tolist()}
        mask = np.array([tuple(x) in front_keys for x in points.tolist()])
    front = unique.loc[mask].copy()
    front_points = front[["egfr", "vegfr2"]].to_numpy(float) if len(front) else np.empty((0, 2))

    scaffold_counts = Counter(x for x in unique.get("scaffold", []) if x)
    total_scaffolds = sum(scaffold_counts.values())
    scaffold_entropy = 0.0
    if total_scaffolds:
        probabilities = np.asarray(list(scaffold_counts.values()), dtype=float) / total_scaffolds
        scaffold_entropy = float(-(probabilities * np.log(probabilities)).sum())

    reference = load_reference(args.reference_front)
    metrics = {
        "generated_rows": generated,
        "valid_rows": valid_rows,
        "validity": valid_rows / generated if generated else 0.0,
        "unique_valid": len(unique),
        "uniqueness_valid": len(unique) / valid_rows if valid_rows else 0.0,
        "pareto_size": len(front),
        "hypervolume": hypervolume_2d(front_points, args.hv_reference),
        "igd_plus": igd_plus(front_points, reference) if reference is not None else None,
        "spacing": spacing(front_points),
        "spread": spread(front_points),
        "egfr_max": float(unique["egfr"].max()) if len(unique) else None,
        "vegfr2_max": float(unique["vegfr2"].max()) if len(unique) else None,
        "best_min_activity": float(np.minimum(unique["egfr"], unique["vegfr2"]).max()) if len(unique) else None,
        "qed_mean": float(unique["qed"].mean()) if len(unique) else None,
        "structural_alert_rate": float(unique["structural_alert"].mean()) if len(unique) else None,
        "scaffold_count": len(scaffold_counts),
        "scaffold_diversity": len(scaffold_counts) / len(unique) if len(unique) else 0.0,
        "scaffold_entropy": scaffold_entropy,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    unique.to_csv(args.output / "standardized_molecules.csv", index=False, encoding="utf-8-sig")
    front.to_csv(args.output / "pareto_front.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([metrics]).to_csv(args.output / "evaluation_summary.csv", index=False, encoding="utf-8-sig")
    json_metrics = {
        key: (None if isinstance(value, float) and not math.isfinite(value) else value)
        for key, value in metrics.items()
    }
    (args.output / "evaluation_summary.json").write_text(
        json.dumps(json_metrics, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--egfr-column", default="egfr")
    parser.add_argument("--vegfr2-column", default="vegfr2")
    parser.add_argument("--reference-front", type=Path)
    parser.add_argument("--hv-reference", nargs=2, type=float, default=(0.0, 0.0))
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
