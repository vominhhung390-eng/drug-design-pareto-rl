#!/usr/bin/env python
"""Evaluate D-MPNN ensembles on frozen scaffold and temporal cohorts."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


def safe_metric(function, *args) -> float:
    try:
        return float(function(*args))
    except (ValueError, TypeError):
        return math.nan


def prediction_column(frame: pd.DataFrame) -> str:
    preferred = ["pactivity", "pactivity_pred", "prediction", "pred"]
    for column in preferred:
        if column in frame.columns and pd.api.types.is_numeric_dtype(frame[column]):
            return column
    candidates = [
        column
        for column in frame.columns
        if pd.api.types.is_numeric_dtype(frame[column])
        and "unc" not in column.lower()
        and "variance" not in column.lower()
        and "std" not in column.lower()
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"Could not identify a unique prediction column. Columns: {list(frame.columns)}"
        )
    return candidates[0]


def load_predictions(truth_path: Path, prediction_path: Path) -> pd.DataFrame:
    truth = pd.read_csv(truth_path)
    predictions = pd.read_csv(prediction_path)
    if len(truth) != len(predictions):
        raise ValueError(
            f"Row mismatch: {truth_path} has {len(truth)}, "
            f"{prediction_path} has {len(predictions)}"
        )
    column = prediction_column(predictions)
    output = truth.copy()
    output["prediction"] = pd.to_numeric(predictions[column], errors="raise")
    for candidate in predictions.columns:
        name = candidate.lower()
        if candidate != column and any(
            token in name for token in ("unc", "variance", "std")
        ):
            output[f"chemprop_{candidate}"] = predictions[candidate].to_numpy()
    return output


def metrics(frame: pd.DataFrame, threshold: float) -> dict[str, float | int]:
    observed = frame["pactivity"].to_numpy(float)
    predicted = frame["prediction"].to_numpy(float)
    active = observed >= threshold
    return {
        "n": int(len(frame)),
        "r2": safe_metric(r2_score, observed, predicted),
        "rmse": math.sqrt(safe_metric(mean_squared_error, observed, predicted)),
        "mae": safe_metric(mean_absolute_error, observed, predicted),
        "pearson": safe_metric(lambda x, y: pearsonr(x, y).statistic, observed, predicted),
        "spearman": safe_metric(
            lambda x, y: spearmanr(x, y).statistic, observed, predicted
        ),
        "bias": float(np.mean(predicted - observed)),
        "auroc_at_6_5": safe_metric(roc_auc_score, active, predicted),
        "auprc_at_6_5": safe_metric(average_precision_score, active, predicted),
    }


def conformal_radius(errors: np.ndarray, coverage: float = 0.90) -> float:
    n = len(errors)
    level = min(1.0, math.ceil((n + 1) * coverage) / n)
    return float(np.quantile(errors, level, method="higher"))


def passes(metrics_row: dict[str, float | int], gates: dict[str, float]) -> bool:
    return bool(
        metrics_row["r2"] >= gates["r2_min"]
        and metrics_row["rmse"] <= gates["rmse_max"]
        and metrics_row["spearman"] >= gates["spearman_min"]
        and metrics_row["auroc_at_6_5"] >= gates["auroc_min"]
    )


def plot_target(target: str, frames: dict[str, pd.DataFrame], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0), constrained_layout=True)
    for axis, (cohort, frame) in zip(axes, frames.items()):
        axis.scatter(
            frame["pactivity"], frame["prediction"], s=18, alpha=0.62, edgecolors="none"
        )
        low = min(frame["pactivity"].min(), frame["prediction"].min())
        high = max(frame["pactivity"].max(), frame["prediction"].max())
        axis.plot([low, high], [low, high], "--", color="0.25", linewidth=1)
        axis.set(
            title=cohort.replace("_", " "),
            xlabel="Observed pActivity",
            ylabel="D-MPNN ensemble prediction",
        )
    fig.suptitle(target)
    fig.savefig(output, dpi=300)
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=root / "config" / "predictor_retraining.json"
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    run_dir = root / config["output_dir"]
    prediction_dir = run_dir / "predictions"
    evaluation_dir = run_dir / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    threshold = float(config["activity_threshold"])

    rows: list[dict[str, object]] = []
    calibration: dict[str, object] = {}
    qualification: dict[str, object] = {}
    for target in config["targets"]:
        target_key = target.lower()
        data_dir = run_dir / "data" / target_key
        frames = {
            "scaffold_validation": load_predictions(
                data_dir / "validation.csv",
                prediction_dir / target_key / "validation_predictions.csv",
            ),
            "temporal_test": load_predictions(
                data_dir / "temporal_test.csv",
                prediction_dir / target_key / "temporal_predictions.csv",
            ),
        }
        target_metrics: dict[str, dict[str, float | int]] = {}
        for cohort, frame in frames.items():
            result = metrics(frame, threshold)
            target_metrics[cohort] = result
            rows.append({"target": target, "cohort": cohort, **result})
            frame.to_csv(
                evaluation_dir / f"{target_key}_{cohort}_predictions.csv",
                index=False,
                encoding="utf-8-sig",
            )

        validation_errors = np.abs(
            frames["scaffold_validation"]["pactivity"]
            - frames["scaffold_validation"]["prediction"]
        ).to_numpy(float)
        radius = conformal_radius(validation_errors, 0.90)
        temporal_errors = np.abs(
            frames["temporal_test"]["pactivity"]
            - frames["temporal_test"]["prediction"]
        ).to_numpy(float)
        calibration[target] = {
            "method": "split conformal absolute residual",
            "calibration_cohort": "scaffold_validation",
            "nominal_coverage": 0.90,
            "radius_pactivity": radius,
            "temporal_empirical_coverage": float(np.mean(temporal_errors <= radius)),
            "temporal_mean_interval_width": 2.0 * radius,
        }

        scaffold_pass = passes(
            target_metrics["scaffold_validation"],
            config["acceptance_gates"]["scaffold_validation"],
        )
        temporal_pass = passes(
            target_metrics["temporal_test"],
            config["acceptance_gates"]["temporal_test"],
        )
        qualification[target] = {
            "scaffold_validation_pass": scaffold_pass,
            "temporal_test_pass": temporal_pass,
            "qualified_for_oracle_replacement": scaffold_pass and temporal_pass,
            "policy": "Both pre-registered cohorts must pass; no test-set tuning.",
        }
        plot_target(
            target,
            frames,
            evaluation_dir / f"{target_key}_observed_vs_predicted.png",
        )

    metrics_frame = pd.DataFrame(rows)
    metrics_frame.to_csv(
        evaluation_dir / "metrics.csv", index=False, encoding="utf-8-sig"
    )
    (evaluation_dir / "calibration.json").write_text(
        json.dumps(calibration, indent=2), encoding="utf-8"
    )
    (evaluation_dir / "qualification.json").write_text(
        json.dumps(qualification, indent=2), encoding="utf-8"
    )

    lines = [
        "# Retrained target predictor evaluation",
        "",
        "The 2024+ temporal cohort was not used for fitting, early stopping, calibration, or hyperparameter selection.",
        "",
        "| Target | Cohort | n | R2 | RMSE | MAE | Pearson | Spearman | AUROC@6.5 | AUPRC@6.5 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['target']} | {row['cohort']} | {row['n']} | "
            f"{row['r2']:.3f} | {row['rmse']:.3f} | {row['mae']:.3f} | "
            f"{row['pearson']:.3f} | {row['spearman']:.3f} | "
            f"{row['auroc_at_6_5']:.3f} | {row['auprc_at_6_5']:.3f} |"
        )
    lines.extend(["", "## Qualification", ""])
    for target, result in qualification.items():
        lines.append(
            f"- {target}: scaffold={'PASS' if result['scaffold_validation_pass'] else 'FAIL'}, "
            f"temporal={'PASS' if result['temporal_test_pass'] else 'FAIL'}, "
            f"replacement={'YES' if result['qualified_for_oracle_replacement'] else 'NO'}"
        )
    (evaluation_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(metrics_frame.to_string(index=False))
    print(json.dumps(qualification, indent=2))


if __name__ == "__main__":
    main()
