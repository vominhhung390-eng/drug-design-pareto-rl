#!/usr/bin/env python
"""Train all-available V3 deployment bundles after frozen historical evaluation."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import ExtraTreesClassifier

from benchmark_predictor_v3_bindingdb_augmented import aggregate_training, clean_bindingdb
from train_predictor_v3_final import RUN, fit_regressor, morgan_fps
from benchmark_predictor_round2 import featurize


ROOT = Path(__file__).resolve().parents[1]
OUT = RUN / "final_candidate" / "deployment_all_available"


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    egfr_ch = pd.read_csv(RUN / "data" / "egfr" / "single_protein_assay_ge10" / "all_compounds.csv")
    bindingdb = clean_bindingdb()
    egfr_aug = aggregate_training(egfr_ch, bindingdb, 2025, set())
    ch_model = fit_regressor(egfr_ch, weighted=True, end_year=2025)
    aug_model = fit_regressor(egfr_aug, weighted=True, end_year=2025)
    egfr_bundle = {
        "schema_version": "predictor-v3.1-consensus",
        "target": "EGFR",
        "trained_through_year": 2025,
        "chembl_regressor": ch_model,
        "chembl_knn_fingerprints": morgan_fps(egfr_ch.smiles),
        "chembl_knn_targets": egfr_ch.pactivity.to_numpy(np.float32),
        "cross_source_regressor": aug_model,
        "cross_source_knn_fingerprints": morgan_fps(egfr_aug.smiles),
        "cross_source_knn_targets": egfr_aug.pactivity.to_numpy(np.float32),
        "knn_k": 20,
        "knn_similarity_power": 3,
        "within_model_weights": {"extratrees": 0.5, "similarity_knn": 0.5},
        "consensus_weights": {"chembl": 0.5, "cross_source": 0.5},
        "consensus_disagreement_limit_pIC50": 0.25,
        "validated_use": "ranking only when consensus_supported; otherwise diagnostic",
        "absolute_activity_use_allowed": False,
    }
    joblib.dump(egfr_bundle, OUT / "egfr_v3_consensus.joblib", compress=3)

    veg_reg = pd.read_csv(RUN / "data" / "vegfr2" / "single_protein_assay_ge10" / "all_compounds.csv")
    veg_cls = pd.read_csv(RUN / "data" / "vegfr2" / "single_protein_assay_ge5" / "all_compounds.csv")
    veg_regressor = fit_regressor(veg_reg, weighted=False)
    veg_classifier = ExtraTreesClassifier(n_estimators=1000, max_features="sqrt", min_samples_leaf=2,
                                          class_weight="balanced", n_jobs=-1, random_state=42)
    veg_classifier.fit(featurize(veg_cls.smiles), (veg_cls.pactivity >= 6.5).astype(int))
    veg_bundle = {
        "schema_version": "predictor-v3.1",
        "target": "VEGFR2", "trained_through_year": 2025,
        "regressor": veg_regressor, "classifier": veg_classifier,
        "ad_fingerprints": morgan_fps(veg_reg.smiles),
        "validated_use": "candidate ranking and approximate activity",
        "absolute_activity_use_allowed": True,
    }
    joblib.dump(veg_bundle, OUT / "vegfr2_v3_candidate.joblib", compress=3)
    metadata = {
        "status": "deployment candidate; not a replacement for wet-lab validation",
        "training_cutoff": 2025,
        "egfr_chembl_compounds": int(len(egfr_ch)),
        "egfr_cross_source_compounds": int(len(egfr_aug)),
        "vegfr2_regression_compounds": int(len(veg_reg)),
        "vegfr2_classification_compounds": int(len(veg_cls)),
        "egfr_gate": "abs(chembl_score - cross_source_score) <= 0.25 and both models within structural AD",
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
