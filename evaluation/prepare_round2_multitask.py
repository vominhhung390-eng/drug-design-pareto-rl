#!/usr/bin/env python
"""Prepare molecule-disjoint EGFR/VEGFR2 rolling-time multitask folds."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import GroupShuffleSplit


def target_frame(path: Path, target: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return frame.rename(
        columns={
            "pactivity": f"{target.lower()}_pactivity",
            "first_document_year": f"{target.lower()}_first_year",
        }
    )[["smiles", f"{target.lower()}_pactivity", f"{target.lower()}_first_year"]]


def scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (root / "config" / "predictor_retraining_round2.json").read_text(
            encoding="utf-8"
        )
    )
    run_dir = root / config["output_dir"]
    egfr = target_frame(
        run_dir / "data" / "egfr" / "all_ic50_consistent.csv", "EGFR"
    )
    vegfr2 = target_frame(
        run_dir / "data" / "vegfr2" / "all_ic50_consistent.csv", "VEGFR2"
    )
    output_root = run_dir / "multitask_data"
    output_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {}

    for fold in config["rolling_folds"]:
        egfr_validation = egfr.loc[
            egfr["egfr_first_year"].between(
                fold["validation_start_year"], fold["validation_end_year"]
            )
        ].copy()
        vegfr2_validation = vegfr2.loc[
            vegfr2["vegfr2_first_year"].between(
                fold["validation_start_year"], fold["validation_end_year"]
            )
        ].copy()
        validation_smiles = set(egfr_validation["smiles"]) | set(
            vegfr2_validation["smiles"]
        )

        egfr_train = egfr.loc[
            (egfr["egfr_first_year"] <= fold["train_end_year"])
            & ~egfr["smiles"].isin(validation_smiles)
        ].copy()
        vegfr2_train = vegfr2.loc[
            (vegfr2["vegfr2_first_year"] <= fold["train_end_year"])
            & ~vegfr2["smiles"].isin(validation_smiles)
        ].copy()
        train = egfr_train.merge(vegfr2_train, on="smiles", how="outer")
        validation = egfr_validation.merge(
            vegfr2_validation, on="smiles", how="outer"
        )
        if set(train["smiles"]) & set(validation["smiles"]):
            raise RuntimeError(f"Molecule leakage in {fold['name']}")

        columns = ["smiles", "egfr_pactivity", "vegfr2_pactivity"]
        fold_dir = output_root / fold["name"]
        fold_dir.mkdir(parents=True, exist_ok=True)
        train[columns].to_csv(
            fold_dir / "train.csv", index=False, encoding="utf-8"
        )
        validation[columns].to_csv(
            fold_dir / "validation.csv", index=False, encoding="utf-8"
        )
        manifest[fold["name"]] = {
            "train_molecules": int(len(train)),
            "validation_molecules": int(len(validation)),
            "train_egfr_labels": int(train["egfr_pactivity"].notna().sum()),
            "train_vegfr2_labels": int(train["vegfr2_pactivity"].notna().sum()),
            "validation_egfr_labels": int(
                validation["egfr_pactivity"].notna().sum()
            ),
            "validation_vegfr2_labels": int(
                validation["vegfr2_pactivity"].notna().sum()
            ),
            "train_validation_smiles_overlap": 0,
        }

    egfr_exploratory = egfr.loc[
        egfr["egfr_first_year"] >= config["exploratory_start_year"]
    ].copy()
    vegfr2_exploratory = vegfr2.loc[
        vegfr2["vegfr2_first_year"] >= config["exploratory_start_year"]
    ].copy()
    exploratory_smiles = set(egfr_exploratory["smiles"]) | set(
        vegfr2_exploratory["smiles"]
    )
    egfr_development = egfr.loc[
        (egfr["egfr_first_year"] <= config["development_end_year"])
        & ~egfr["smiles"].isin(exploratory_smiles)
    ].copy()
    vegfr2_development = vegfr2.loc[
        (vegfr2["vegfr2_first_year"] <= config["development_end_year"])
        & ~vegfr2["smiles"].isin(exploratory_smiles)
    ].copy()
    final_train = egfr_development.merge(
        vegfr2_development, on="smiles", how="outer"
    )
    final_exploratory = egfr_exploratory.merge(
        vegfr2_exploratory, on="smiles", how="outer"
    )
    deployment = egfr.merge(vegfr2, on="smiles", how="outer")
    columns = ["smiles", "egfr_pactivity", "vegfr2_pactivity"]
    final_train[columns].to_csv(
        output_root / "final_research_train_through_2023.csv",
        index=False,
        encoding="utf-8",
    )
    groups = final_train["smiles"].map(scaffold)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=42)
    fit_idx, calibration_idx = next(splitter.split(final_train, groups=groups))
    final_fit = final_train.iloc[fit_idx].reset_index(drop=True)
    final_calibration = final_train.iloc[calibration_idx].reset_index(drop=True)
    final_fit[columns].to_csv(
        output_root / "final_research_fit.csv", index=False, encoding="utf-8"
    )
    final_calibration[columns].to_csv(
        output_root / "final_research_calibration.csv",
        index=False,
        encoding="utf-8",
    )
    final_exploratory[columns].to_csv(
        output_root / "final_exploratory_2024plus.csv",
        index=False,
        encoding="utf-8",
    )
    deployment[columns].to_csv(
        output_root / "deployment_all_available.csv",
        index=False,
        encoding="utf-8",
    )
    manifest["final"] = {
        "research_train_molecules": int(len(final_train)),
        "research_fit_molecules": int(len(final_fit)),
        "research_calibration_molecules": int(len(final_calibration)),
        "research_fit_calibration_scaffold_overlap": 0,
        "exploratory_molecules": int(len(final_exploratory)),
        "research_exploratory_smiles_overlap": 0,
        "deployment_molecules": int(len(deployment)),
    }

    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
