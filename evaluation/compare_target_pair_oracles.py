#!/usr/bin/env python
"""Compare current and legacy-aligned PARP1/BRD4 RF oracles."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


THRESHOLD = 6.5
TARGETS = ("parp1", "brd4")
COHORTS = ("fold_a", "fold_b", "locked_2024plus")


def metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    observed = frame["pactivity"].to_numpy(float)
    predicted = frame["prediction"].to_numpy(float)
    active = observed >= THRESHOLD
    predicted_active = predicted >= THRESHOLD
    result: dict[str, float | int] = {
        "n": int(len(frame)),
        "observed_positive_rate": float(active.mean()),
        "predicted_positive_rate": float(predicted_active.mean()),
        "r2": float(r2_score(observed, predicted)),
        "rmse": float(math.sqrt(mean_squared_error(observed, predicted))),
        "mae": float(mean_absolute_error(observed, predicted)),
        "spearman": float(spearmanr(observed, predicted).statistic),
        "bias": float(np.mean(predicted - observed)),
    }
    if len(np.unique(active)) == 2:
        result.update(
            {
                "auroc_at_6_5": float(roc_auc_score(active, predicted)),
                "auprc_at_6_5": float(average_precision_score(active, predicted)),
                "balanced_accuracy_at_6_5": float(
                    balanced_accuracy_score(active, predicted_active)
                ),
                "sensitivity_at_6_5": float(predicted_active[active].mean()),
                "specificity_at_6_5": float((~predicted_active[~active]).mean()),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("current_validation", type=Path)
    parser.add_argument("candidate_validation", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    sources = (
        ("current_sqrt_leaf2", args.current_validation),
        ("legacy_aligned", args.candidate_validation),
    )
    for model, source in sources:
        for target in TARGETS:
            for cohort in COHORTS:
                path = source / f"{target}_{cohort}_predictions.csv"
                frame = pd.read_csv(path, encoding="utf-8-sig")
                rows.append(
                    {
                        "model": model,
                        "target": target.upper(),
                        "cohort": cohort,
                        **metrics(frame),
                    }
                )
                if cohort == "locked_2024plus" and "first_document_year" in frame.columns:
                    latest = frame[frame["first_document_year"] >= 2025].copy()
                    if len(latest):
                        rows.append(
                            {
                                "model": model,
                                "target": target.upper(),
                                "cohort": "2025plus_sensitivity",
                                **metrics(latest),
                            }
                        )

    comparison = pd.DataFrame(rows)
    comparison.to_csv(args.output / "oracle_comparison.csv", index=False, encoding="utf-8-sig")

    rolling = comparison[comparison["cohort"].isin(["fold_a", "fold_b"])]
    summary = (
        rolling.groupby(["model", "target"], as_index=False)
        .agg(
            mean_rolling_spearman=("spearman", "mean"),
            worst_rolling_spearman=("spearman", "min"),
            mean_rolling_rmse=("rmse", "mean"),
            mean_rolling_auroc=("auroc_at_6_5", "mean"),
            mean_rolling_balanced_accuracy=("balanced_accuracy_at_6_5", "mean"),
        )
    )
    summary.to_csv(args.output / "rolling_comparison.csv", index=False, encoding="utf-8-sig")
    print(comparison.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
