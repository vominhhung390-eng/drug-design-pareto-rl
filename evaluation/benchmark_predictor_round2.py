#!/usr/bin/env python
"""Screen strong fingerprint regressors using only frozen rolling-time folds."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdFingerprintGenerator
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from xgboost import XGBRegressor


def featurize(smiles: pd.Series) -> np.ndarray:
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048, includeChirality=True
    )
    output = np.empty((len(smiles), 2058), dtype=np.float32)
    for index, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smi}")
        fp = generator.GetFingerprint(mol)
        DataStructs.ConvertToNumpyArray(fp, output[index, :2048])
        output[index, 2048:] = (
            Descriptors.MolWt(mol),
            Crippen.MolLogP(mol),
            Descriptors.TPSA(mol),
            Lipinski.NumHDonors(mol),
            Lipinski.NumHAcceptors(mol),
            Lipinski.NumRotatableBonds(mol),
            Lipinski.RingCount(mol),
            Lipinski.FractionCSP3(mol),
            Lipinski.HeavyAtomCount(mol),
            Chem.GetFormalCharge(mol),
        )
    return output


def recency_weights(years: pd.Series, end_year: int, half_life: float = 6.0) -> np.ndarray:
    age = np.maximum(0.0, end_year - years.to_numpy(float))
    weights = np.maximum(0.10, np.power(0.5, age / half_life))
    return weights / weights.mean()


def model_specs(seed: int) -> dict[str, tuple[object, bool]]:
    common_forest = dict(
        n_estimators=200,
        max_features="sqrt",
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=seed,
    )
    common_xgb = dict(
        n_estimators=2000,
        learning_rate=0.025,
        max_depth=8,
        min_child_weight=5,
        subsample=0.85,
        colsample_bytree=0.60,
        reg_lambda=5.0,
        reg_alpha=0.05,
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        device="cuda",
        early_stopping_rounds=100,
        random_state=seed,
        n_jobs=-1,
    )
    return {
        "rf_uniform": (RandomForestRegressor(**common_forest), False),
        "extratrees_uniform": (ExtraTreesRegressor(**common_forest), False),
        "extratrees_recent_hl6": (ExtraTreesRegressor(**common_forest), True),
        "xgb_uniform": (XGBRegressor(**common_xgb), False),
        "xgb_recent_hl6": (XGBRegressor(**common_xgb), True),
    }


def metrics(frame: pd.DataFrame, prediction: np.ndarray, threshold: float) -> dict[str, float | int]:
    observed = frame["pactivity"].to_numpy(float)
    active = observed >= threshold
    result: dict[str, float | int] = {
        "n": int(len(frame)),
        "r2": float(r2_score(observed, prediction)),
        "rmse": float(math.sqrt(mean_squared_error(observed, prediction))),
        "mae": float(mean_absolute_error(observed, prediction)),
        "pearson": float(pearsonr(observed, prediction).statistic),
        "spearman": float(spearmanr(observed, prediction).statistic),
        "bias": float(np.mean(prediction - observed)),
    }
    try:
        result["auroc_at_6_5"] = float(roc_auc_score(active, prediction))
    except ValueError:
        result["auroc_at_6_5"] = math.nan
    return result


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
    run_dir = root / config["output_dir"]
    output_dir = run_dir / "classical_benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)
    threshold = float(config["activity_threshold"])
    rows: list[dict[str, object]] = []

    for target in config["targets"]:
        target_key = target.lower()
        for fold in config["rolling_folds"]:
            fold_dir = run_dir / "data" / target_key / fold["name"]
            train = pd.read_csv(fold_dir / "train.csv")
            validation = pd.read_csv(fold_dir / "validation.csv")
            x_train = featurize(train["smiles"])
            x_validation = featurize(validation["smiles"])
            y_train = train["pactivity"].to_numpy(float)
            weights = recency_weights(
                train["first_document_year"], fold["train_end_year"]
            )

            for model_name, (model, use_weights) in model_specs(
                config["random_seed"]
            ).items():
                started = time.perf_counter()
                fit_kwargs: dict[str, object] = {}
                if use_weights:
                    fit_kwargs["sample_weight"] = weights
                if model_name.startswith("xgb_"):
                    fit_kwargs["eval_set"] = [(x_validation, validation["pactivity"])]
                    fit_kwargs["verbose"] = False
                model.fit(x_train, y_train, **fit_kwargs)
                prediction = model.predict(x_validation)
                elapsed = time.perf_counter() - started

                for cohort, mask in {
                    "all_temporal_validation": np.ones(len(validation), dtype=bool),
                    "unseen_scaffold_only": ~validation[
                        "scaffold_seen_in_train"
                    ].to_numpy(bool),
                }.items():
                    selected = validation.loc[mask].reset_index(drop=True)
                    result = metrics(selected, prediction[mask], threshold)
                    rows.append(
                        {
                            "target": target,
                            "fold": fold["name"],
                            "model": model_name,
                            "cohort": cohort,
                            "fit_seconds": elapsed,
                            **result,
                        }
                    )

                prediction_frame = validation.copy()
                prediction_frame["prediction"] = prediction
                prediction_frame.to_csv(
                    output_dir
                    / f"{target_key}_{fold['name']}_{model_name}_predictions.csv",
                    index=False,
                    encoding="utf-8-sig",
                )
                print(
                    target,
                    fold["name"],
                    model_name,
                    f"RMSE={rows[-2]['rmse']:.3f}",
                    f"rho={rows[-2]['spearman']:.3f}",
                    f"seconds={elapsed:.1f}",
                    flush=True,
                )

    metrics_frame = pd.DataFrame(rows)
    metrics_frame.to_csv(
        output_dir / "fold_metrics.csv", index=False, encoding="utf-8-sig"
    )
    primary = metrics_frame.loc[
        metrics_frame["cohort"].eq("all_temporal_validation")
    ]
    ranking = (
        primary.groupby(["target", "model"], as_index=False)
        .agg(
            mean_rmse=("rmse", "mean"),
            worst_rmse=("rmse", "max"),
            mean_mae=("mae", "mean"),
            mean_spearman=("spearman", "mean"),
            mean_auroc_at_6_5=("auroc_at_6_5", "mean"),
            total_fit_seconds=("fit_seconds", "sum"),
        )
        .sort_values(["target", "mean_rmse", "mean_spearman"], ascending=[True, True, False])
    )
    ranking["rank"] = ranking.groupby("target")["mean_rmse"].rank(
        method="first"
    ).astype(int)
    ranking.to_csv(output_dir / "model_ranking.csv", index=False, encoding="utf-8-sig")
    winners = {
        target: ranking.loc[
            ranking["target"].eq(target) & ranking["rank"].eq(1), "model"
        ].iloc[0]
        for target in config["targets"]
    }
    (output_dir / "selected_classical_models.json").write_text(
        json.dumps(
            {
                "selection_basis": config["selection_rule"],
                "winners": winners,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(ranking.to_string(index=False))


if __name__ == "__main__":
    main()
