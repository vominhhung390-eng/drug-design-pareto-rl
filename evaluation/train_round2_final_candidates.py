#!/usr/bin/env python
"""Train versioned round-2 classical ensembles and audit their applicability domain."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdFingerprintGenerator
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from xgboost import XGBRegressor


SEEDS = (42, 43, 44, 45, 46)


def featurize(smiles: pd.Series) -> tuple[np.ndarray, list[DataStructs.ExplicitBitVect]]:
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048, includeChirality=True
    )
    output = np.empty((len(smiles), 2058), dtype=np.float32)
    bitvectors: list[DataStructs.ExplicitBitVect] = []
    for index, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smi}")
        fp = generator.GetFingerprint(mol)
        bitvectors.append(fp)
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
    return output, bitvectors


def max_similarity(
    queries: list[DataStructs.ExplicitBitVect],
    references: list[DataStructs.ExplicitBitVect],
) -> np.ndarray:
    return np.asarray(
        [max(DataStructs.BulkTanimotoSimilarity(query, references)) for query in queries],
        dtype=float,
    )


def recency_weights(years: pd.Series, end_year: int, half_life: float = 6.0) -> np.ndarray:
    age = np.maximum(0.0, end_year - years.to_numpy(float))
    weights = np.maximum(0.10, np.power(0.5, age / half_life))
    return weights / weights.mean()


def make_model(name: str, seed: int):
    if name == "xgb_recent_hl6":
        return XGBRegressor(
            n_estimators=300,
            learning_rate=0.025,
            max_depth=8,
            min_child_weight=5,
            subsample=0.85,
            colsample_bytree=0.60,
            reg_lambda=5.0,
            reg_alpha=0.05,
            objective="reg:squarederror",
            tree_method="hist",
            device="cuda",
            random_state=seed,
            n_jobs=-1,
        )
    if name == "extratrees_recent_hl6":
        return ExtraTreesRegressor(
            n_estimators=600,
            max_features="sqrt",
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=seed,
        )
    raise ValueError(f"Unsupported selected model: {name}")


def regression_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    active = observed >= 6.5
    try:
        auroc = float(roc_auc_score(active, predicted))
    except ValueError:
        auroc = math.nan
    return {
        "n": int(len(observed)),
        "r2": float(r2_score(observed, predicted)),
        "rmse": float(math.sqrt(mean_squared_error(observed, predicted))),
        "mae": float(mean_absolute_error(observed, predicted)),
        "spearman": float(spearmanr(observed, predicted).statistic),
        "bias": float(np.mean(predicted - observed)),
        "auroc_at_6_5": auroc,
    }


def conformal_radius(errors: np.ndarray, coverage: float = 0.90) -> float:
    level = min(1.0, math.ceil((len(errors) + 1) * coverage) / len(errors))
    return float(np.quantile(errors, level, method="higher"))


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
    benchmark_dir = run_dir / "classical_benchmark"
    selected = json.loads(
        (benchmark_dir / "selected_classical_models.json").read_text(encoding="utf-8")
    )["winners"]
    model_root = run_dir / "final_candidates"
    model_root.mkdir(parents=True, exist_ok=True)
    ranking = pd.read_csv(benchmark_dir / "model_ranking.csv")
    all_metrics: list[dict[str, object]] = []
    qualification: dict[str, object] = {}

    for target, model_name in selected.items():
        key = target.lower()
        data_dir = run_dir / "data" / key
        development = pd.read_csv(data_dir / "development_through_2023.csv")
        exploratory = pd.read_csv(data_dir / "exploratory_2024plus.csv")
        x_development, fp_development = featurize(development["smiles"])
        x_exploratory, fp_exploratory = featurize(exploratory["smiles"])
        sample_weights = recency_weights(
            development["first_document_year"], config["development_end_year"]
        )
        target_model_dir = model_root / key / "research_through_2023"
        target_model_dir.mkdir(parents=True, exist_ok=True)
        predictions: list[np.ndarray] = []
        for seed in SEEDS:
            model = make_model(model_name, seed)
            model.fit(
                x_development,
                development["pactivity"].to_numpy(float),
                sample_weight=sample_weights,
            )
            joblib.dump(model, target_model_dir / f"model_seed_{seed}.joblib", compress=3)
            predictions.append(model.predict(x_exploratory))
        prediction_matrix = np.vstack(predictions)
        exploratory["prediction"] = prediction_matrix.mean(axis=0)
        exploratory["ensemble_std"] = prediction_matrix.std(axis=0, ddof=1)
        exploratory["max_tanimoto_to_development"] = max_similarity(
            fp_exploratory, fp_development
        )

        rolling_prediction_frames = []
        for fold in config["rolling_folds"]:
            path = benchmark_dir / f"{key}_{fold['name']}_{model_name}_predictions.csv"
            rolling_prediction_frames.append(pd.read_csv(path))
        rolling_predictions = pd.concat(rolling_prediction_frames, ignore_index=True)
        radius = conformal_radius(
            np.abs(
                rolling_predictions["pactivity"].to_numpy(float)
                - rolling_predictions["prediction"].to_numpy(float)
            )
        )
        exploratory["conformal_lower_90"] = exploratory["prediction"] - radius
        exploratory["conformal_upper_90"] = exploratory["prediction"] + radius
        exploratory["within_applicability_domain"] = (
            exploratory["max_tanimoto_to_development"] >= 0.60
        )
        exploratory.to_csv(
            target_model_dir / "exploratory_2024plus_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )

        observed = exploratory["pactivity"].to_numpy(float)
        predicted = exploratory["prediction"].to_numpy(float)
        for cohort, mask in {
            "exploratory_all": np.ones(len(exploratory), dtype=bool),
            "exploratory_similarity_ge_0_60": exploratory[
                "within_applicability_domain"
            ].to_numpy(bool),
            "exploratory_similarity_lt_0_60": ~exploratory[
                "within_applicability_domain"
            ].to_numpy(bool),
        }.items():
            if mask.sum() >= 10:
                all_metrics.append(
                    {
                        "target": target,
                        "model": model_name,
                        "cohort": cohort,
                        **regression_metrics(observed[mask], predicted[mask]),
                    }
                )

        temporal_coverage = float(
            np.mean(
                (observed >= exploratory["conformal_lower_90"])
                & (observed <= exploratory["conformal_upper_90"])
            )
        )
        rank_row = ranking.loc[
            ranking["target"].eq(target) & ranking["model"].eq(model_name)
        ].iloc[0]
        passes_historical_time_validation = bool(
            rank_row["mean_rmse"] <= 1.0 and rank_row["mean_spearman"] >= 0.50
        )
        qualification[target] = {
            "selected_model": model_name,
            "rolling_mean_rmse": float(rank_row["mean_rmse"]),
            "rolling_mean_spearman": float(rank_row["mean_spearman"]),
            "rolling_worst_rmse": float(rank_row["worst_rmse"]),
            "passes_historical_time_validation": passes_historical_time_validation,
            "conformal_radius_pactivity": radius,
            "exploratory_coverage_90": temporal_coverage,
            "applicability_domain": "max ECFP4 Tanimoto to development >= 0.60",
            "eligible_for_formal_oracle_replacement": False,
            "replacement_reason": (
                "A new untouched external test set is still required; the 2024+ set was viewed in round 1."
            ),
        }
        (target_model_dir / "model_card.json").write_text(
            json.dumps(qualification[target], indent=2), encoding="utf-8"
        )

        deployment = pd.read_csv(data_dir / "all_ic50_consistent.csv")
        x_deployment, _ = featurize(deployment["smiles"])
        deployment_weights = recency_weights(
            deployment["first_document_year"],
            int(deployment["first_document_year"].max()),
        )
        deployment_dir = model_root / key / "deployment_all_available"
        deployment_dir.mkdir(parents=True, exist_ok=True)
        for seed in SEEDS:
            model = make_model(model_name, seed)
            model.fit(
                x_deployment,
                deployment["pactivity"].to_numpy(float),
                sample_weight=deployment_weights,
            )
            joblib.dump(model, deployment_dir / f"model_seed_{seed}.joblib", compress=3)
        (deployment_dir / "NOT_FOR_PRIMARY_PAPER_COMPARISONS.txt").write_text(
            "This deployment candidate uses all available IC50 data through 2025 and has no untouched future test set.\n",
            encoding="utf-8",
        )

    metrics_frame = pd.DataFrame(all_metrics)
    metrics_frame.to_csv(
        model_root / "exploratory_metrics.csv", index=False, encoding="utf-8-sig"
    )
    (model_root / "qualification.json").write_text(
        json.dumps(qualification, indent=2), encoding="utf-8"
    )
    print(metrics_frame.to_string(index=False))
    print(json.dumps(qualification, indent=2))


if __name__ == "__main__":
    main()
