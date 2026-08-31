#!/usr/bin/env python
"""Focused advanced benchmarks for the best V3 assay profile."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRanker, XGBRegressor

from benchmark_predictor_round2 import recency_weights


def multiradius_features(smiles: pd.Series) -> np.ndarray:
    generators = [
        rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=2048, includeChirality=True)
        for radius in (2, 3, 4, 5)
    ]
    output = np.empty((len(smiles), 8198), dtype=np.float32)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(smi)
        for j, generator in enumerate(generators):
            fp = generator.GetFingerprint(mol)
            DataStructs.ConvertToNumpyArray(fp, output[i, j * 2048 : (j + 1) * 2048])
        output[i, 8192:] = (
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumRotatableBonds(mol),
        )
    return output


def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "n": len(y),
        "r2": float(r2_score(y, pred)),
        "rmse": float(math.sqrt(mean_squared_error(y, pred))),
        "mae": float(mean_absolute_error(y, pred)),
        "spearman": float(spearmanr(y, pred).statistic),
        "bias": float(np.mean(pred - y)),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=root / "config" / "predictor_retraining_v3.json")
    parser.add_argument("--target", default="EGFR")
    parser.add_argument("--profile", default="single_protein_assay_ge10")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    run_dir = root / config["output_dir"]
    output = run_dir / "advanced_benchmark"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for fold in config["rolling_folds"]:
        profile_dir = run_dir / "data" / args.target.lower() / args.profile
        train = pd.read_csv(profile_dir / fold["name"] / "train.csv")
        validation = pd.read_csv(profile_dir / fold["name"] / "validation.csv")
        x_train = multiradius_features(train["smiles"])
        x_val = multiradius_features(validation["smiles"])
        y_train = train["pactivity"].to_numpy(float)
        y_val = validation["pactivity"].to_numpy(float)
        weights = recency_weights(train["first_document_year"], fold["train_end_year"])
        models = {
            "multiradius_rf": (
                RandomForestRegressor(n_estimators=800, max_features="sqrt", min_samples_leaf=2, n_jobs=-1, random_state=42),
                None,
            ),
            "multiradius_et_recent": (
                ExtraTreesRegressor(n_estimators=900, max_features="sqrt", min_samples_leaf=2, n_jobs=-1, random_state=42),
                weights,
            ),
            "multiradius_xgb_recent": (
                XGBRegressor(
                    n_estimators=2200, learning_rate=0.02, max_depth=7, min_child_weight=6,
                    subsample=0.85, colsample_bytree=0.45, reg_lambda=10.0, reg_alpha=0.1,
                    objective="reg:squarederror", eval_metric="rmse", tree_method="hist", device="cuda",
                    early_stopping_rounds=120, random_state=42, n_jobs=-1,
                ),
                weights,
            ),
        }
        prediction_columns = {}
        for name, (model, sample_weight) in models.items():
            kwargs = {}
            if sample_weight is not None:
                kwargs["sample_weight"] = sample_weight
            if name.startswith("multiradius_xgb"):
                kwargs.update(eval_set=[(x_val, y_val)], verbose=False)
            started = time.perf_counter()
            model.fit(x_train, y_train, **kwargs)
            pred = model.predict(x_val)
            prediction_columns[name] = pred
            result = metrics(y_val, pred)
            rows.append({"target": args.target, "profile": args.profile, "fold": fold["name"], "model": name, "fit_seconds": time.perf_counter() - started, **result})
            print(fold["name"], name, f"rho={result['spearman']:.3f}", f"rmse={result['rmse']:.3f}", flush=True)

        # Assay-relative pairwise ranking. Fit only on measurements available by the fold cutoff.
        measurements = pd.read_csv(profile_dir / "eligible_measurements.csv")
        allowed_smiles = set(train["smiles"])
        rank_train = measurements[
            measurements["smiles"].isin(allowed_smiles)
            & (measurements["document_year"] <= fold["train_end_year"])
        ].copy()
        rank_train = rank_train.sort_values("assay_chembl_id").reset_index(drop=True)
        group_sizes = rank_train.groupby("assay_chembl_id", sort=True).size().to_numpy()
        x_rank = multiradius_features(rank_train["smiles"])
        ranker = XGBRanker(
            n_estimators=1200, learning_rate=0.025, max_depth=6, min_child_weight=5,
            subsample=0.85, colsample_bytree=0.45, reg_lambda=10.0, reg_alpha=0.1,
            objective="rank:pairwise", eval_metric="ndcg", tree_method="hist", device="cuda",
            random_state=42, n_jobs=-1,
        )
        started = time.perf_counter()
        ranker.fit(x_rank, rank_train["pactivity"].to_numpy(float), group=group_sizes, verbose=False)
        train_score = ranker.predict(x_train)
        val_score = ranker.predict(x_val)
        calibrator = HuberRegressor(epsilon=1.5, alpha=1.0).fit(train_score.reshape(-1, 1), y_train)
        rank_pred = calibrator.predict(val_score.reshape(-1, 1))
        prediction_columns["assay_pairwise_ranker"] = rank_pred
        result = metrics(y_val, rank_pred)
        rows.append({"target": args.target, "profile": args.profile, "fold": fold["name"], "model": "assay_pairwise_ranker", "fit_seconds": time.perf_counter() - started, **result})
        print(fold["name"], "assay_pairwise_ranker", f"rho={result['spearman']:.3f}", f"rmse={result['rmse']:.3f}", flush=True)

        output_frame = validation.copy()
        for name, pred in prediction_columns.items():
            output_frame[name] = pred
        output_frame.to_csv(output / f"{args.target.lower()}_{args.profile}_{fold['name']}_predictions.csv", index=False, encoding="utf-8-sig")

    result_frame = pd.DataFrame(rows)
    result_frame.to_csv(output / f"{args.target.lower()}_{args.profile}_metrics.csv", index=False, encoding="utf-8-sig")
    ranking = result_frame.groupby("model", as_index=False).agg(
        mean_spearman=("spearman", "mean"), worst_spearman=("spearman", "min"),
        mean_rmse=("rmse", "mean"), worst_rmse=("rmse", "max"), total_fit_seconds=("fit_seconds", "sum")
    ).sort_values(["mean_spearman", "mean_rmse"], ascending=[False, True])
    ranking.to_csv(output / f"{args.target.lower()}_{args.profile}_ranking.csv", index=False, encoding="utf-8-sig")
    print(ranking.to_string(index=False))


if __name__ == "__main__":
    main()
