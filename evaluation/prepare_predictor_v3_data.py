#!/usr/bin/env python
"""Build frozen assay-format and EGFR-variant-aware datasets for predictor V3."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from audit_predictor_v3_data import exact_base_filter, variant_class
from validate_target_predictors import canonicalize, safe_scaffold


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def measurement_frame(rows: list[dict]) -> pd.DataFrame:
    records = []
    for row in rows:
        if not exact_base_filter(row):
            continue
        canonical, mol = canonicalize(row.get("canonical_smiles", ""))
        if canonical is None or mol is None:
            continue
        try:
            pactivity = float(row["pchembl_value"])
            year = int(row["document_year"])
        except (TypeError, ValueError, KeyError):
            continue
        if not 2.0 <= pactivity <= 12.0:
            continue
        records.append(
            {
                "smiles": canonical,
                "pactivity": pactivity,
                "document_year": year,
                "activity_id": row.get("activity_id"),
                "molecule_chembl_id": row.get("molecule_chembl_id"),
                "document_chembl_id": row.get("document_chembl_id"),
                "assay_chembl_id": str(row.get("assay_chembl_id")),
                "bao_format": row.get("bao_format"),
                "bao_label": row.get("bao_label"),
                "variant_class": variant_class(row),
                "assay_description": row.get("assay_description"),
                "scaffold": safe_scaffold(canonical, mol),
            }
        )
    return pd.DataFrame(records)


def aggregate(frame: pd.DataFrame, max_span: float) -> tuple[pd.DataFrame, int]:
    grouped = frame.groupby("smiles", sort=True, as_index=False).agg(
        pactivity=("pactivity", "median"),
        n_measurements=("pactivity", "size"),
        pactivity_sd=("pactivity", "std"),
        pactivity_min=("pactivity", "min"),
        pactivity_max=("pactivity", "max"),
        first_document_year=("document_year", "min"),
        last_document_year=("document_year", "max"),
        n_documents=("document_chembl_id", "nunique"),
        n_assays=("assay_chembl_id", "nunique"),
        scaffold=("scaffold", "first"),
    )
    grouped["pactivity_sd"] = grouped["pactivity_sd"].fillna(0.0)
    grouped["pactivity_span"] = grouped["pactivity_max"] - grouped["pactivity_min"]
    excluded = int((grouped["pactivity_span"] > max_span).sum())
    return grouped[grouped["pactivity_span"] <= max_span].reset_index(drop=True), excluded


def describe(frame: pd.DataFrame, threshold: float) -> dict:
    if frame.empty:
        return {"n": 0}
    return {
        "n": int(len(frame)),
        "n_scaffolds": int(frame["scaffold"].nunique()),
        "pactivity_median": float(frame["pactivity"].median()),
        "pactivity_iqr": float(frame["pactivity"].quantile(0.75) - frame["pactivity"].quantile(0.25)),
        "active_rate": float((frame["pactivity"] >= threshold).mean()),
        "year_min": int(frame["first_document_year"].min()),
        "year_max": int(frame["first_document_year"].max()),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=root / "config" / "predictor_retraining_v3.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output = root / config["output_dir"]
    data_root = output / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": config["run_id"],
        "config": str(args.config.resolve()),
        "config_sha256": sha256(args.config),
        "selection_policy": config["selection_policy"],
        "targets": {},
    }
    for target, relative in config["sources"].items():
        source = root / relative
        base = measurement_frame(load_jsonl(source))
        assay_sizes = base.groupby("assay_chembl_id").size()
        target_manifest = {
            "source": str(source.resolve()),
            "source_sha256": sha256(source),
            "base_measurements": int(len(base)),
            "base_unique_molecules": int(base["smiles"].nunique()),
            "variant_counts": {str(k): int(v) for k, v in Counter(base["variant_class"]).most_common()},
            "profiles": {},
        }
        for profile_name, spec in config["profiles"].items():
            selected = base[
                base["bao_format"].isin(spec["bao_formats"])
                & base["variant_class"].isin(spec["variant_classes"])
                & base["assay_chembl_id"].map(assay_sizes).ge(spec["minimum_assay_size"])
            ].copy()
            aggregated, excluded = aggregate(
                selected, float(config["max_within_compound_pactivity_span"])
            )
            profile_dir = data_root / target.lower() / profile_name
            profile_dir.mkdir(parents=True, exist_ok=True)
            selected.to_csv(profile_dir / "eligible_measurements.csv", index=False, encoding="utf-8-sig")
            aggregated.to_csv(profile_dir / "all_compounds.csv", index=False, encoding="utf-8-sig")
            profile_manifest = {
                "specification": spec,
                "measurements": int(len(selected)),
                "assays": int(selected["assay_chembl_id"].nunique()),
                "excluded_inconsistent_compounds": excluded,
                "all_compounds": describe(aggregated, float(config["activity_threshold"])),
                "folds": {},
            }
            for fold in config["rolling_folds"]:
                train = aggregated[
                    aggregated["first_document_year"] <= fold["train_end_year"]
                ].reset_index(drop=True)
                validation = aggregated[
                    aggregated["first_document_year"].between(
                        fold["validation_start_year"], fold["validation_end_year"]
                    )
                ].reset_index(drop=True)
                validation["scaffold_seen_in_train"] = validation["scaffold"].isin(set(train["scaffold"]))
                fold_dir = profile_dir / fold["name"]
                fold_dir.mkdir(exist_ok=True)
                train.to_csv(fold_dir / "train.csv", index=False, encoding="utf-8-sig")
                validation.to_csv(fold_dir / "validation.csv", index=False, encoding="utf-8-sig")
                profile_manifest["folds"][fold["name"]] = {
                    "train": describe(train, float(config["activity_threshold"])),
                    "validation": describe(validation, float(config["activity_threshold"])),
                    "validation_unseen_scaffold_rate": float((~validation["scaffold_seen_in_train"]).mean()) if len(validation) else None,
                }
            development = aggregated[
                aggregated["first_document_year"] <= config["development_end_year"]
            ].reset_index(drop=True)
            exploratory = aggregated[
                aggregated["first_document_year"] >= config["exploratory_start_year"]
            ].reset_index(drop=True)
            development.to_csv(profile_dir / "development_through_2023.csv", index=False, encoding="utf-8-sig")
            exploratory.to_csv(profile_dir / "exploratory_2024plus.csv", index=False, encoding="utf-8-sig")
            profile_manifest["development"] = describe(development, float(config["activity_threshold"]))
            profile_manifest["exploratory"] = describe(exploratory, float(config["activity_threshold"]))
            target_manifest["profiles"][profile_name] = profile_manifest
        manifest["targets"][target] = target_manifest
    (output / "data_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest["targets"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
