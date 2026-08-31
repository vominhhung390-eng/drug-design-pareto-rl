#!/usr/bin/env python
"""Benchmark V3 regression and active-probability models on frozen time folds."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from xgboost import XGBClassifier, XGBRegressor

from benchmark_predictor_round2 import featurize, recency_weights


def regression_models(seed: int) -> dict[str, tuple[object, bool]]:
    forest = dict(
        n_estimators=500,
        max_features="sqrt",
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=seed,
    )
    xgb = dict(
        n_estimators=1800,
        learning_rate=0.025,
        max_depth=7,
        min_child_weight=6,
        subsample=0.85,
        colsample_bytree=0.65,
        reg_lambda=8.0,
        reg_alpha=0.10,
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        device="cuda",
        early_stopping_rounds=100,
        random_state=seed,
        n_jobs=-1,
    )
    return {
        "rf_uniform": (RandomForestRegressor(**forest), False),
        "extratrees_uniform": (ExtraTreesRegressor(**forest), False),
        "extratrees_recent_hl6": (ExtraTreesRegressor(**forest), True),
        "xgb_uniform": (XGBRegressor(**xgb), False),
        "xgb_recent_hl6": (XGBRegressor(**xgb), True),
    }


def classifier_models(seed: int, scale_pos_weight: float) -> dict[str, object]:
    return {
        "extratrees_balanced": ExtraTreesClassifier(
            n_estimators=700,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        ),
        "xgb_balanced": XGBClassifier(
            n_estimators=1600,
            learning_rate=0.025,
            max_depth=7,
            min_child_weight=6,
            subsample=0.85,
            colsample_bytree=0.65,
            reg_lambda=8.0,
            reg_alpha=0.10,
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            device="cuda",
            early_stopping_rounds=100,
            scale_pos_weight=scale_pos_weight,
            random_state=seed,
            n_jobs=-1,
        ),
    }


def regression_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "n": len(y),
        "r2": float(r2_score(y, pred)),
        "rmse": float(math.sqrt(mean_squared_error(y, pred))),
        "mae": float(mean_absolute_error(y, pred)),
        "spearman": float(spearmanr(y, pred).statistic),
        "bias": float(np.mean(pred - y)),
    }


def classification_metrics(y: np.ndarray, prob: np.ndarray) -> dict:
    pred = prob >= 0.5
    if len(np.unique(y)) < 2:
        auroc = auprc = math.nan
    else:
        auroc = float(roc_auc_score(y, prob))
        auprc = float(average_precision_score(y, prob))
    return {
        "n": len(y),
        "positive_rate": float(y.mean()),
        "auroc": auroc,
        "auprc": auprc,
        "balanced_accuracy_at_0_5": float(balanced_accuracy_score(y, pred)),
        "brier": float(brier_score_loss(y, prob)),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=root / "config" / "predictor_retraining_v3.json")
    parser.add_argument("--targets", nargs="*", default=["EGFR", "VEGFR2"])
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    run_dir = root / config["output_dir"]
    output = run_dir / "classical_benchmark"
    output.mkdir(parents=True, exist_ok=True)
    threshold = float(config["activity_threshold"])
    regression_rows = []
    classification_rows = []

    for target in args.targets:
        for profile in config["profiles"]:
            profile_dir = run_dir / "data" / target.lower() / profile
            if not (profile_dir / "all_compounds.csv").exists():
                continue
            for fold in config["rolling_folds"]:
                fold_dir = profile_dir / fold["name"]
                train = pd.read_csv(fold_dir / "train.csv")
                validation = pd.read_csv(fold_dir / "validation.csv")
                if len(train) < 100 or len(validation) < 30:
                    continue
                x_train = featurize(train["smiles"])
                x_val = featurize(validation["smiles"])
                y_train = train["pactivity"].to_numpy(float)
                y_val = validation["pactivity"].to_numpy(float)
                weights = recency_weights(train["first_document_year"], fold["train_end_year"])

                for name, (model, weighted) in regression_models(config["random_seed"]).items():
                    kwargs = {}
                    if weighted:
                        kwargs["sample_weight"] = weights
                    if name.startswith("xgb"):
                        kwargs.update(eval_set=[(x_val, y_val)], verbose=False)
                    started = time.perf_counter()
                    model.fit(x_train, y_train, **kwargs)
                    pred = model.predict(x_val)
                    elapsed = time.perf_counter() - started
                    for cohort, mask in {
                        "all": np.ones(len(validation), dtype=bool),
                        "unseen_scaffold": ~validation["scaffold_seen_in_train"].to_numpy(bool),
                    }.items():
                        regression_rows.append(
                            {
                                "target": target,
                                "profile": profile,
                                "fold": fold["name"],
                                "model": name,
                                "cohort": cohort,
                                "fit_seconds": elapsed,
                                **regression_metrics(y_val[mask], pred[mask]),
                            }
                        )
                    pd.DataFrame(
                        {**validation.to_dict(orient="list"), "prediction": pred}
                    ).to_csv(
                        output / f"{target.lower()}_{profile}_{fold['name']}_{name}_regression.csv",
                        index=False,
                        encoding="utf-8-sig",
                    )
                    print(target, profile, fold["name"], name, f"rho={regression_rows[-2]['spearman']:.3f}", f"rmse={regression_rows[-2]['rmse']:.3f}", flush=True)

                active_train = (y_train >= threshold).astype(int)
                active_val = (y_val >= threshold).astype(int)
                n_pos = max(1, int(active_train.sum()))
                n_neg = max(1, int(len(active_train) - n_pos))
                for name, model in classifier_models(config["random_seed"], n_neg / n_pos).items():
                    kwargs = {}
                    if name.startswith("xgb"):
                        kwargs.update(eval_set=[(x_val, active_val)], verbose=False)
                    started = time.perf_counter()
                    model.fit(x_train, active_train, **kwargs)
                    prob = model.predict_proba(x_val)[:, 1]
                    elapsed = time.perf_counter() - started
                    for cohort, mask in {
                        "all": np.ones(len(validation), dtype=bool),
                        "unseen_scaffold": ~validation["scaffold_seen_in_train"].to_numpy(bool),
                    }.items():
                        classification_rows.append(
                            {
                                "target": target,
                                "profile": profile,
                                "fold": fold["name"],
                                "model": name,
                                "cohort": cohort,
                                "fit_seconds": elapsed,
                                **classification_metrics(active_val[mask], prob[mask]),
                            }
                        )
                    validation.assign(active_label=active_val, active_probability=prob).to_csv(
                        output / f"{target.lower()}_{profile}_{fold['name']}_{name}_classification.csv",
                        index=False,
                        encoding="utf-8-sig",
                    )
                    print(target, profile, fold["name"], name, f"auroc={classification_rows[-2]['auroc']:.3f}", flush=True)

    reg = pd.DataFrame(regression_rows)
    cls = pd.DataFrame(classification_rows)
    reg.to_csv(output / "regression_fold_metrics.csv", index=False, encoding="utf-8-sig")
    cls.to_csv(output / "classification_fold_metrics.csv", index=False, encoding="utf-8-sig")
    reg_rank = (
        reg[reg["cohort"].eq("all")]
        .groupby(["target", "profile", "model"], as_index=False)
        .agg(mean_spearman=("spearman", "mean"), worst_spearman=("spearman", "min"), mean_rmse=("rmse", "mean"), worst_rmse=("rmse", "max"), mean_mae=("mae", "mean"), total_fit_seconds=("fit_seconds", "sum"), min_fold_n=("n", "min"))
        .sort_values(["target", "mean_spearman", "mean_rmse"], ascending=[True, False, True])
    )
    cls_rank = (
        cls[cls["cohort"].eq("all")]
        .groupby(["target", "profile", "model"], as_index=False)
        .agg(mean_auroc=("auroc", "mean"), worst_auroc=("auroc", "min"), mean_auprc=("auprc", "mean"), mean_balanced_accuracy=("balanced_accuracy_at_0_5", "mean"), mean_brier=("brier", "mean"), total_fit_seconds=("fit_seconds", "sum"), min_fold_n=("n", "min"))
        .sort_values(["target", "mean_auroc", "mean_brier"], ascending=[True, False, True])
    )
    reg_rank.to_csv(output / "regression_ranking.csv", index=False, encoding="utf-8-sig")
    cls_rank.to_csv(output / "classification_ranking.csv", index=False, encoding="utf-8-sig")
    print("\nREGRESSION RANKING\n", reg_rank.groupby("target").head(8).to_string(index=False))
    print("\nCLASSIFICATION RANKING\n", cls_rank.groupby("target").head(8).to_string(index=False))


if __name__ == "__main__":
    main()
