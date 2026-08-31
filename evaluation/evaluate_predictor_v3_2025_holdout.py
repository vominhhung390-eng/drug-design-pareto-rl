#!/usr/bin/env python
"""Locked-architecture evaluation: refit through 2024 and test on 2025 records."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import average_precision_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score

from benchmark_predictor_round2 import featurize, recency_weights


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results" / "predictor_retraining_v3_20260731"
OUT = RUN / "holdout_2025"


def reg_metrics(y, pred):
    return {"n": int(len(y)), "spearman": float(spearmanr(y, pred).statistic),
            "rmse": float(mean_squared_error(y, pred) ** 0.5), "mae": float(mean_absolute_error(y, pred)),
            "r2": float(r2_score(y, pred)), "bias": float(np.mean(pred - y))}


def forest(weighted, data):
    model = ExtraTreesRegressor(n_estimators=1000, max_features="sqrt", min_samples_leaf=2,
                                n_jobs=-1, random_state=42)
    kwargs = {"sample_weight": recency_weights(data.first_document_year, 2024)} if weighted else {}
    model.fit(featurize(data.smiles), data.pactivity, **kwargs)
    return model


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    egfr_dir = RUN / "data" / "egfr" / "single_protein_assay_ge10"
    egfr_old = pd.read_csv(egfr_dir / "development_through_2023.csv")
    egfr_new = pd.read_csv(egfr_dir / "exploratory_2024plus.csv")
    egfr_train = pd.concat([egfr_old, egfr_new[egfr_new.first_document_year.eq(2024)]], ignore_index=True)
    egfr_test = egfr_new[egfr_new.first_document_year.eq(2025)].reset_index(drop=True)
    et = forest(True, egfr_train)
    et_pred = et.predict(featurize(egfr_test.smiles))
    train_fp = [fpgen.GetFingerprint(Chem.MolFromSmiles(s)) for s in egfr_train.smiles]
    test_fp = [fpgen.GetFingerprint(Chem.MolFromSmiles(s)) for s in egfr_test.smiles]
    y_train = egfr_train.pactivity.to_numpy(float)
    knn = []
    similarity = []
    for fp in test_fp:
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, train_fp))
        idx = np.argsort(sims)[::-1][:20]
        knn.append(np.average(y_train[idx], weights=np.maximum(sims[idx], 1e-6) ** 3))
        similarity.append(sims[idx[0]])
    egfr_pred = 0.5 * et_pred + 0.5 * np.asarray(knn)
    results["EGFR"] = reg_metrics(egfr_test.pactivity, egfr_pred)
    egfr_test.assign(prediction=egfr_pred, max_train_similarity=similarity).to_csv(OUT / "egfr_2025_predictions.csv", index=False)

    veg_dir = RUN / "data" / "vegfr2" / "single_protein_assay_ge10"
    veg_old = pd.read_csv(veg_dir / "development_through_2023.csv")
    veg_new = pd.read_csv(veg_dir / "exploratory_2024plus.csv")
    veg_train = pd.concat([veg_old, veg_new[veg_new.first_document_year.eq(2024)]], ignore_index=True)
    veg_test = veg_new[veg_new.first_document_year.eq(2025)].reset_index(drop=True)
    veg_model = forest(False, veg_train)
    veg_pred = veg_model.predict(featurize(veg_test.smiles))
    results["VEGFR2"] = reg_metrics(veg_test.pactivity, veg_pred)

    cls_dir = RUN / "data" / "vegfr2" / "single_protein_assay_ge5"
    cls_old = pd.read_csv(cls_dir / "development_through_2023.csv")
    cls_new = pd.read_csv(cls_dir / "exploratory_2024plus.csv")
    cls_train = pd.concat([cls_old, cls_new[cls_new.first_document_year.eq(2024)]], ignore_index=True)
    classifier = ExtraTreesClassifier(n_estimators=1000, max_features="sqrt", min_samples_leaf=2,
                                      class_weight="balanced", n_jobs=-1, random_state=42)
    classifier.fit(featurize(cls_train.smiles), (cls_train.pactivity >= 6.5).astype(int))
    probability = classifier.predict_proba(featurize(veg_test.smiles))[:, 1]
    label = (veg_test.pactivity >= 6.5).astype(int)
    results["VEGFR2"].update({"auroc": float(roc_auc_score(label, probability)),
                               "auprc": float(average_precision_score(label, probability))})
    veg_test.assign(prediction=veg_pred, active_probability=probability).to_csv(OUT / "vegfr2_2025_predictions.csv", index=False)
    (OUT / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
