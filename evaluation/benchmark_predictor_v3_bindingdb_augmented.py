#!/usr/bin/env python
"""Evaluate locked EGFR hybrid model after leakage-safe BindingDB augmentation."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from benchmark_predictor_round2 import featurize, recency_weights


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results" / "predictor_retraining_v3_20260731"
OUT = RUN / "bindingdb_augmented"
RDLogger.DisableLog("rdApp.error")


def clean_bindingdb() -> pd.DataFrame:
    cache = OUT / "bindingdb_egfr_exact_ic50.csv"
    if cache.exists():
        return pd.read_csv(cache)
    source = ROOT / "data" / "external" / "bindingdb" / "P00533_EGFR_ligands_20260729.json"
    years_path = ROOT / "data" / "external" / "bindingdb" / "pubmed_publication_years.json"
    rows = json.loads(source.read_text(encoding="utf-8"))["getLindsByUniprotsResponse"]["affinities"]
    years = {str(k): int(v) for k, v in json.loads(years_path.read_text(encoding="utf-8")).items()}
    cleaned = []
    for row in rows:
        if str(row.get("affinity_type")) != "IC50":
            continue
        text = str(row.get("affinity", "")).strip()
        try:
            value = float(text)
        except ValueError:
            continue
        if not np.isfinite(value) or value <= 0:
            continue
        pactivity = 9.0 - math.log10(value)
        if not 2.0 <= pactivity <= 12.0:
            continue
        year = years.get(str(row.get("pmid") or ""))
        if year is None:
            continue
        mol = Chem.MolFromSmiles(str(row.get("smile") or ""))
        if mol is None:
            continue
        fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
        if fragments:
            mol = max(fragments, key=lambda m: m.GetNumHeavyAtoms())
        cleaned.append({"smiles": Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True),
                        "pactivity": pactivity, "document_year": year, "pmid": row.get("pmid")})
    frame = pd.DataFrame(cleaned).drop_duplicates(["smiles", "pactivity", "document_year", "pmid"])
    frame.to_csv(cache, index=False)
    return frame


def aggregate_training(chembl: pd.DataFrame, bindingdb: pd.DataFrame, cutoff: int, excluded_smiles: set[str]) -> pd.DataFrame:
    bd = bindingdb[(bindingdb.document_year <= cutoff) & ~bindingdb.smiles.isin(excluded_smiles)].copy()
    bd = bd.groupby("smiles", as_index=False).agg(
        pactivity=("pactivity", "median"), pactivity_min=("pactivity", "min"),
        pactivity_max=("pactivity", "max"), first_document_year=("document_year", "min")
    )
    bd = bd[(bd.pactivity_max - bd.pactivity_min) <= 1.5][["smiles", "pactivity", "first_document_year"]]
    ch = chembl[["smiles", "pactivity", "first_document_year"]].copy()
    union = pd.concat([ch.assign(source_weight=2.0), bd.assign(source_weight=1.0)], ignore_index=True)
    # Preserve ChEMBL priority while allowing non-overlapping BindingDB chemistry to expand coverage.
    combined = []
    for smiles, group in union.groupby("smiles", sort=False):
        combined.append({"smiles": smiles,
                         "pactivity": float(np.average(group.pactivity, weights=group.source_weight)),
                         "first_document_year": int(group.first_document_year.min()),
                         "n_sources": int(len(group))})
    return pd.DataFrame(combined)


def predict_hybrid(train: pd.DataFrame, test: pd.DataFrame, cutoff: int):
    model = ExtraTreesRegressor(n_estimators=1000, max_features="sqrt", min_samples_leaf=2,
                                n_jobs=-1, random_state=42)
    model.fit(featurize(train.smiles), train.pactivity,
              sample_weight=recency_weights(train.first_document_year, cutoff))
    et = model.predict(featurize(test.smiles))
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    train_fp = [fpgen.GetFingerprint(Chem.MolFromSmiles(s)) for s in train.smiles]
    y = train.pactivity.to_numpy(float)
    knn, max_sim = [], []
    for smiles in test.smiles:
        fp = fpgen.GetFingerprint(Chem.MolFromSmiles(smiles))
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, train_fp))
        idx = np.argsort(sims)[::-1][:20]
        knn.append(float(np.average(y[idx], weights=np.maximum(sims[idx], 1e-6) ** 3)))
        max_sim.append(float(sims[idx[0]]))
    return 0.5 * et + 0.5 * np.asarray(knn), np.asarray(max_sim)


def metrics(y, prediction):
    return {"n": int(len(y)), "spearman": float(spearmanr(y, prediction).statistic),
            "rmse": float(mean_squared_error(y, prediction) ** 0.5),
            "mae": float(mean_absolute_error(y, prediction)), "r2": float(r2_score(y, prediction)),
            "bias": float(np.mean(prediction - y))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bindingdb = clean_bindingdb()
    data_dir = RUN / "data" / "egfr" / "single_protein_assay_ge10"
    rows = []
    for fold, cutoff in [("fold_a", 2019), ("fold_b", 2021)]:
        train = pd.read_csv(data_dir / fold / "train.csv")
        test = pd.read_csv(data_dir / fold / "validation.csv")
        augmented = aggregate_training(train, bindingdb, cutoff, set(test.smiles))
        pred, similarity = predict_hybrid(augmented, test, cutoff)
        result = {"cohort": fold, "train_cutoff": cutoff, "training_compounds": len(augmented), **metrics(test.pactivity, pred)}
        rows.append(result)
        test.assign(prediction=pred, max_train_similarity=similarity).to_csv(OUT / f"{fold}_predictions.csv", index=False)
        print(result, flush=True)

    # Architecture is unchanged; refit through 2024 and retain 2025 as the latest small holdout.
    old = pd.read_csv(data_dir / "development_through_2023.csv")
    recent = pd.read_csv(data_dir / "exploratory_2024plus.csv")
    train_2024 = pd.concat([old, recent[recent.first_document_year.eq(2024)]], ignore_index=True)
    test_2025 = recent[recent.first_document_year.eq(2025)].reset_index(drop=True)
    augmented = aggregate_training(train_2024, bindingdb, 2024, set(test_2025.smiles))
    pred, similarity = predict_hybrid(augmented, test_2025, 2024)
    result = {"cohort": "holdout_2025", "train_cutoff": 2024, "training_compounds": len(augmented), **metrics(test_2025.pactivity, pred)}
    rows.append(result)
    test_2025.assign(prediction=pred, max_train_similarity=similarity).to_csv(OUT / "holdout_2025_predictions.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / "metrics.csv", index=False)
    audit = {"bindingdb_exact_ic50_known_year_measurements": int(len(bindingdb)),
             "bindingdb_unique_compounds": int(bindingdb.smiles.nunique()), "results": rows}
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(result, flush=True)


if __name__ == "__main__":
    main()
