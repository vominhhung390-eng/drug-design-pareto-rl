#!/usr/bin/env python
"""Build assay-homogeneous IC50 datasets and frozen rolling-time folds."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "smiles",
    "pactivity",
    "document_year",
    "standard_type",
    "activity_id",
    "document_chembl_id",
    "assay_chembl_id",
    "scaffold",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def aggregate_ic50(frame: pd.DataFrame) -> pd.DataFrame:
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
    grouped["first_document_year"] = grouped["first_document_year"].astype(int)
    grouped["last_document_year"] = grouped["last_document_year"].astype(int)
    return grouped


def summary(frame: pd.DataFrame, threshold: float) -> dict[str, float | int]:
    return {
        "n": int(len(frame)),
        "n_scaffolds": int(frame["scaffold"].fillna("").nunique()),
        "pactivity_mean": float(frame["pactivity"].mean()),
        "pactivity_sd": float(frame["pactivity"].std()),
        "active_rate": float((frame["pactivity"] >= threshold).mean()),
        "first_year_min": int(frame["first_document_year"].min()),
        "first_year_max": int(frame["first_document_year"].max()),
    }


def add_fold_annotations(
    train: pd.DataFrame, validation: pd.DataFrame
) -> pd.DataFrame:
    historical_scaffolds = set(train["scaffold"].fillna(""))
    output = validation.copy()
    output["scaffold_seen_in_train"] = output["scaffold"].fillna("").isin(
        historical_scaffolds
    )
    return output


def write_frame(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "config" / "predictor_retraining_round2.json",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    source_dir = root / config["source_dir"]
    output_dir = root / config["output_dir"]
    data_root = output_dir / "data"
    data_root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "run_id": config["run_id"],
        "config_sha256": sha256(args.config),
        "endpoint": config["endpoint"],
        "max_within_compound_pactivity_span": config[
            "max_within_compound_pactivity_span"
        ],
        "selection_rule": config["selection_rule"],
        "targets": {},
    }
    threshold = float(config["activity_threshold"])

    for target, filename in config["targets"].items():
        source = source_dir / filename
        raw = pd.read_csv(source)
        missing = REQUIRED_COLUMNS - set(raw.columns)
        if missing:
            raise ValueError(f"{source} is missing columns: {sorted(missing)}")
        raw["document_year"] = pd.to_numeric(raw["document_year"], errors="coerce")
        eligible = raw.loc[
            raw["standard_type"].eq(config["endpoint"])
            & raw["document_year"].notna()
        ].copy()
        aggregated_before_consistency = aggregate_ic50(eligible)
        aggregated = aggregated_before_consistency.loc[
            aggregated_before_consistency["pactivity_span"]
            <= config["max_within_compound_pactivity_span"]
        ].reset_index(drop=True)
        if aggregated["smiles"].duplicated().any():
            raise RuntimeError(f"Duplicate SMILES remain for {target}")

        target_dir = data_root / target.lower()
        target_dir.mkdir(parents=True, exist_ok=True)
        all_path = target_dir / "all_ic50_consistent.csv"
        write_frame(aggregated, all_path)

        target_manifest: dict[str, object] = {
            "source": str(source.resolve()),
            "source_sha256": sha256(source),
            "input_measurements": int(len(raw)),
            "ic50_measurements_with_year": int(len(eligible)),
            "ic50_unique_compounds_before_consistency_filter": int(
                len(aggregated_before_consistency)
            ),
            "excluded_inconsistent_compounds": int(
                len(aggregated_before_consistency) - len(aggregated)
            ),
            "all_consistent_compounds": summary(aggregated, threshold),
            "folds": {},
        }

        for fold in config["rolling_folds"]:
            train = aggregated.loc[
                aggregated["first_document_year"] <= fold["train_end_year"]
            ].reset_index(drop=True)
            validation = aggregated.loc[
                aggregated["first_document_year"].between(
                    fold["validation_start_year"], fold["validation_end_year"]
                )
            ].reset_index(drop=True)
            validation = add_fold_annotations(train, validation)
            fold_dir = target_dir / fold["name"]
            fold_dir.mkdir(parents=True, exist_ok=True)
            train_path = fold_dir / "train.csv"
            validation_path = fold_dir / "validation.csv"
            write_frame(train, train_path)
            write_frame(validation, validation_path)
            target_manifest["folds"][fold["name"]] = {
                "train": summary(train, threshold),
                "validation": summary(validation, threshold),
                "validation_unseen_scaffold_n": int(
                    (~validation["scaffold_seen_in_train"]).sum()
                ),
                "validation_unseen_scaffold_rate": float(
                    (~validation["scaffold_seen_in_train"]).mean()
                ),
                "train_sha256": sha256(train_path),
                "validation_sha256": sha256(validation_path),
            }

        development = aggregated.loc[
            aggregated["first_document_year"] <= config["development_end_year"]
        ].reset_index(drop=True)
        exploratory = aggregated.loc[
            aggregated["first_document_year"] >= config["exploratory_start_year"]
        ].reset_index(drop=True)
        exploratory = add_fold_annotations(development, exploratory)
        development_path = target_dir / "development_through_2023.csv"
        exploratory_path = target_dir / "exploratory_2024plus.csv"
        write_frame(development, development_path)
        write_frame(exploratory, exploratory_path)
        target_manifest["development"] = summary(development, threshold)
        target_manifest["exploratory"] = {
            **summary(exploratory, threshold),
            "unseen_scaffold_n": int(
                (~exploratory["scaffold_seen_in_train"]).sum()
            ),
            "unseen_scaffold_rate": float(
                (~exploratory["scaffold_seen_in_train"]).mean()
            ),
            "policy": "exploratory only; already viewed in round 1",
        }
        target_manifest["all_data_sha256"] = sha256(all_path)
        target_manifest["development_sha256"] = sha256(development_path)
        target_manifest["exploratory_sha256"] = sha256(exploratory_path)
        manifest["targets"][target] = target_manifest

    manifest_path = output_dir / "data_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest["targets"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
