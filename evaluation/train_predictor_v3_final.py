#!/usr/bin/env python
"""Fit frozen V3 candidate predictors after model selection on historical folds."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

from benchmark_predictor_round2 import featurize, recency_weights


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results" / "predictor_retraining_v3_20260731"
OUT = RUN / "final_candidate"
SEED = 42
THRESHOLD = 6.5


def morgan_fps(smiles: pd.Series):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return [generator.GetFingerprint(Chem.MolFromSmiles(smi)) for smi in smiles]


def fit_regressor(data: pd.DataFrame, weighted: bool, end_year: int = 2023) -> ExtraTreesRegressor:
    model = ExtraTreesRegressor(
        n_estimators=1000,
        max_features="sqrt",
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=SEED,
    )
    kwargs = {}
    if weighted:
        kwargs["sample_weight"] = recency_weights(data["first_document_year"], end_year)
    model.fit(featurize(data["smiles"]), data["pactivity"].to_numpy(float), **kwargs)
    return model


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    egfr_dir = RUN / "data" / "egfr" / "single_protein_assay_ge10"
    egfr = pd.read_csv(egfr_dir / "development_through_2023.csv")
    egfr_model = fit_regressor(egfr, weighted=True)
    egfr_bundle = {
        "schema_version": "predictor-v3.0",
        "target": "EGFR",
        "profile": "human binding IC50; single-protein; WT or unspecified; assays with >=10 eligible measurements",
        "trained_through_year": 2023,
        "regressor": egfr_model,
        "knn_fingerprints": morgan_fps(egfr["smiles"]),
        "knn_targets": egfr["pactivity"].to_numpy(np.float32),
        "knn_k": 20,
        "knn_similarity_power": 3,
        "ensemble_weights": {"extratrees": 0.5, "similarity_knn": 0.5},
        "activity_threshold_pIC50": THRESHOLD,
        "validated_use": "candidate_ranking_only",
        "absolute_activity_use_allowed": False,
    }
    joblib.dump(egfr_bundle, OUT / "egfr_v3_candidate.joblib", compress=3)

    veg_reg_dir = RUN / "data" / "vegfr2" / "single_protein_assay_ge10"
    veg_cls_dir = RUN / "data" / "vegfr2" / "single_protein_assay_ge5"
    veg_reg = pd.read_csv(veg_reg_dir / "development_through_2023.csv")
    veg_cls = pd.read_csv(veg_cls_dir / "development_through_2023.csv")
    veg_regressor = fit_regressor(veg_reg, weighted=False)
    veg_classifier = ExtraTreesClassifier(
        n_estimators=1000,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=SEED,
    )
    veg_classifier.fit(featurize(veg_cls["smiles"]), (veg_cls["pactivity"] >= THRESHOLD).astype(int))
    veg_bundle = {
        "schema_version": "predictor-v3.0",
        "target": "VEGFR2",
        "profile": "human binding IC50; single-protein; WT or unspecified; assay-size filters selected on frozen folds",
        "trained_through_year": 2023,
        "regressor": veg_regressor,
        "classifier": veg_classifier,
        "ad_fingerprints": morgan_fps(veg_reg["smiles"]),
        "activity_threshold_pIC50": THRESHOLD,
        "validated_use": "candidate_ranking_and_approximate_activity",
        "absolute_activity_use_allowed": True,
    }
    joblib.dump(veg_bundle, OUT / "vegfr2_v3_candidate.joblib", compress=3)

    metadata = {
        "run_id": "predictor_retraining_v3_20260731",
        "frozen_at": "2026-07-31",
        "data_cutoff": 2023,
        "egfr_training_compounds": int(len(egfr)),
        "vegfr2_regression_training_compounds": int(len(veg_reg)),
        "vegfr2_classification_training_compounds": int(len(veg_cls)),
        "selection_was_based_only_on": ["fold_a: train<=2019, validate=2020-2021", "fold_b: train<=2021, validate=2022-2023"],
        "2024plus_data_used_for_selection": False,
        "model_files": ["egfr_v3_candidate.joblib", "vegfr2_v3_candidate.joblib"],
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
