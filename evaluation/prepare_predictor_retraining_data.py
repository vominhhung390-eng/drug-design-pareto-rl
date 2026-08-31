#!/usr/bin/env python
"""Freeze leak-controlled Chemprop splits for EGFR and VEGFR2 retraining."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


REQUIRED_COLUMNS = {
    "smiles",
    "pactivity",
    "n_measurements",
    "pactivity_sd",
    "pactivity_span",
    "first_document_year",
    "last_document_year",
    "scaffold",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scaffold_split(
    historical: pd.DataFrame, validation_fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=validation_fraction, random_state=seed
    )
    train_idx, validation_idx = next(
        splitter.split(historical, groups=historical["scaffold"].fillna(""))
    )
    train = historical.iloc[train_idx].copy()
    validation = historical.iloc[validation_idx].copy()
    overlap = set(train["scaffold"].fillna("")) & set(
        validation["scaffold"].fillna("")
    )
    if overlap:
        raise RuntimeError(f"Scaffold leakage detected: {len(overlap)} shared groups")
    return train, validation


def write_chemprop_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.loc[:, ["smiles", "pactivity"]].to_csv(
        path, index=False, encoding="utf-8"
    )


def cohort_summary(frame: pd.DataFrame) -> dict[str, float | int]:
    return {
        "n": int(len(frame)),
        "n_scaffolds": int(frame["scaffold"].fillna("").nunique()),
        "pactivity_mean": float(frame["pactivity"].mean()),
        "pactivity_sd": float(frame["pactivity"].std()),
        "active_rate_at_6_5": float((frame["pactivity"] >= 6.5).mean()),
        "high_measurement_disagreement_rate": float(
            (frame["pactivity_span"] > 1.0).mean()
        ),
        "first_year_min": int(frame["first_document_year"].min()),
        "first_year_max": int(frame["first_document_year"].max()),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=root / "config" / "predictor_retraining.json"
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    source_dir = root / config["source_dir"]
    output_dir = root / config["output_dir"]
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "run_id": config["run_id"],
        "config_sha256": sha256(args.config),
        "split_policy": {
            "train_and_validation": (
                f"first_document_year <= {config['historical_end_year']}"
            ),
            "temporal_test": (
                f"first_document_year >= {config['temporal_start_year']}"
            ),
            "validation": "GroupShuffleSplit grouped by Bemis-Murcko scaffold",
            "validation_fraction": config["validation_fraction"],
            "random_seed": config["random_seed"],
        },
        "targets": {},
    }

    for target, filename in config["targets"].items():
        source = source_dir / filename
        frame = pd.read_csv(source)
        missing = REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"{source} is missing columns: {sorted(missing)}")
        if frame["smiles"].duplicated().any():
            raise ValueError(f"{source} contains duplicate canonical SMILES")

        historical = frame.loc[
            frame["first_document_year"] <= config["historical_end_year"]
        ].reset_index(drop=True)
        temporal = frame.loc[
            frame["first_document_year"] >= config["temporal_start_year"]
        ].reset_index(drop=True)
        if len(historical) + len(temporal) != len(frame):
            raise ValueError(f"Year cohorts do not partition all rows for {target}")

        train, validation = scaffold_split(
            historical, config["validation_fraction"], config["random_seed"]
        )
        train = train.sort_values("smiles").reset_index(drop=True)
        validation = validation.sort_values("smiles").reset_index(drop=True)
        temporal = temporal.sort_values("smiles").reset_index(drop=True)

        target_dir = data_dir / target.lower()
        target_dir.mkdir(parents=True, exist_ok=True)
        write_chemprop_csv(train, target_dir / "train.csv")
        write_chemprop_csv(validation, target_dir / "validation.csv")
        write_chemprop_csv(temporal, target_dir / "temporal_test.csv")

        annotated = pd.concat(
            [
                train.assign(cohort="train"),
                validation.assign(cohort="scaffold_validation"),
                temporal.assign(cohort="temporal_test"),
            ],
            ignore_index=True,
        )
        annotated.to_csv(
            target_dir / "all_compounds_with_cohort.csv",
            index=False,
            encoding="utf-8-sig",
        )

        temporal_scaffold_overlap = set(temporal["scaffold"].fillna("")) & set(
            historical["scaffold"].fillna("")
        )
        manifest["targets"][target] = {
            "source": str(source.resolve()),
            "source_sha256": sha256(source),
            "cohorts": {
                "train": cohort_summary(train),
                "scaffold_validation": cohort_summary(validation),
                "temporal_test": cohort_summary(temporal),
            },
            "train_validation_scaffold_overlap": 0,
            "temporal_scaffolds_seen_historically": len(temporal_scaffold_overlap),
            "temporal_scaffold_overlap_rate": (
                len(temporal_scaffold_overlap)
                / max(1, temporal["scaffold"].fillna("").nunique())
            ),
            "files": {
                name: {
                    "path": str((target_dir / name).resolve()),
                    "sha256": sha256(target_dir / name),
                }
                for name in (
                    "train.csv",
                    "validation.csv",
                    "temporal_test.csv",
                    "all_compounds_with_cohort.csv",
                )
            },
        }

    (output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest["targets"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
