#!/usr/bin/env python
"""Summarize V4-B on the common baseline seeds 42-51.

This is a post-selection descriptive comparison.  The independent prospective
confirmation on seeds 82-91 remains the primary anti-selection-bias evidence.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scipy import stats

from multiobjective_metrics import hypervolume_2d


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "own_method_v4" / "common_seeds_42_51_10240"
SEEDS = list(range(42, 52))
BASELINES = {
    "polygon_original": ROOT / "results" / "baselines" / "polygon_original",
    "drugex_v2": ROOT / "results" / "baselines" / "drugex_v2",
    "reinvent4": ROOT / "results" / "baselines" / "reinvent4",
    "graphpareto_nsga2": ROOT / "results" / "baselines" / "graphpareto_nsga2",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def hv_from_standardized(path: Path) -> float:
    frame = pd.read_csv(path)
    points = frame[["egfr", "vegfr2"]].dropna().to_numpy(float)
    return float(hypervolume_2d(points))


def own_row(seed: int) -> dict:
    base = OUT / f"v4_b_raw_mean_seed{seed}" / "evaluation"
    evaluation = read_json(base / "evaluation_summary.json")
    quality = read_json(
        base / "quality_constrained" / "quality_constrained_summary.json"
    )
    return {
        "method": "own_v4_b_raw_mean",
        "seed": seed,
        "hv": float(evaluation["hypervolume"]),
        "validity": float(evaluation["validity"]),
        "unique_valid": int(evaluation["unique_valid"]),
        "best_min_activity": float(evaluation["best_min_activity"]),
        "qed_mean": float(evaluation["qed_mean"]),
        "structural_alert_rate": float(evaluation["structural_alert_rate"]),
        "quality_hv": float(quality["quality_constrained_hypervolume"]),
        "quality_pass_rate": float(quality["quality_pass_rate"]),
    }


def baseline_row(method: str, seed: int) -> dict:
    base = BASELINES[method] / f"formal_10240_seed{seed}" / "anytime" / "budget_10240"
    evaluation = read_json(base / "evaluation_summary.json")
    return {
        "method": method,
        "seed": seed,
        "hv": hv_from_standardized(base / "standardized_molecules.csv"),
        "validity": float(evaluation["validity"]),
        "unique_valid": int(evaluation["unique_valid"]),
        "best_min_activity": float(evaluation["best_min_activity"]),
        "qed_mean": float(evaluation["qed_mean"]),
        "structural_alert_rate": float(evaluation["structural_alert_rate"]),
    }


def v1_row(seed: int) -> dict:
    base = ROOT / "results" / "own_method" / f"formal_10240_seed{seed}" / "anytime" / "budget_10240"
    evaluation = read_json(base / "evaluation_summary.json")
    return {
        "method": "own_v1",
        "seed": seed,
        "hv": hv_from_standardized(base / "standardized_molecules.csv"),
        "validity": float(evaluation["validity"]),
        "unique_valid": int(evaluation["unique_valid"]),
        "best_min_activity": float(evaluation["best_min_activity"]),
        "qed_mean": float(evaluation["qed_mean"]),
        "structural_alert_rate": float(evaluation["structural_alert_rate"]),
    }


def main() -> None:
    rows = [own_row(seed) for seed in SEEDS]
    rows += [v1_row(seed) for seed in SEEDS]
    rows += [
        baseline_row(method, seed)
        for method in BASELINES
        for seed in SEEDS
    ]
    detail = pd.DataFrame(rows).sort_values(["method", "seed"])
    own = detail[detail.method == "own_v4_b_raw_mean"].set_index("seed")
    aggregates = []
    paired = []
    for method, group in detail.groupby("method"):
        numeric = group.select_dtypes("number")
        record = {"method": method, "n_seeds": len(group)}
        for column in numeric.columns:
            if column == "seed":
                continue
            record[f"{column}_mean"] = numeric[column].mean()
            record[f"{column}_sd"] = numeric[column].std(ddof=1)
        aggregates.append(record)
        if method == "own_v4_b_raw_mean":
            continue
        baseline = group.set_index("seed")
        difference = own.hv - baseline.hv
        ci = stats.t.interval(
            0.95,
            len(difference) - 1,
            loc=difference.mean(),
            scale=stats.sem(difference),
        )
        paired.append(
            {
                "comparison": f"own_v4_b_raw_mean_minus_{method}",
                "hv_mean_difference": difference.mean(),
                "hv_ci95_low": ci[0],
                "hv_ci95_high": ci[1],
                "wins": int((difference > 0).sum()),
                "losses": int((difference < 0).sum()),
                "paired_t_p": stats.ttest_rel(own.hv, baseline.hv).pvalue,
                "wilcoxon_p": stats.wilcoxon(difference).pvalue,
            }
        )

    polygon_quality = []
    for seed in SEEDS:
        path = (
            BASELINES["polygon_original"]
            / f"formal_10240_seed{seed}"
            / "anytime"
            / "budget_10240"
            / "quality_constrained"
            / "quality_constrained_summary.json"
        )
        polygon_quality.append(read_json(path))
    polygon_qhv = pd.Series(
        [row["quality_constrained_hypervolume"] for row in polygon_quality],
        index=SEEDS,
    )
    polygon_qpass = pd.Series(
        [row["quality_pass_rate"] for row in polygon_quality], index=SEEDS
    )
    q_difference = own.quality_hv - polygon_qhv
    q_ci = stats.t.interval(
        0.95,
        len(q_difference) - 1,
        loc=q_difference.mean(),
        scale=stats.sem(q_difference),
    )

    aggregate = pd.DataFrame(aggregates).sort_values("hv_mean", ascending=False)
    paired_frame = pd.DataFrame(paired).sort_values("hv_mean_difference")
    detail.to_csv(OUT / "common_seed_detail.csv", index=False, encoding="utf-8-sig")
    aggregate.to_csv(
        OUT / "common_seed_aggregate.csv", index=False, encoding="utf-8-sig"
    )
    paired_frame.to_csv(
        OUT / "common_seed_paired_tests.csv", index=False, encoding="utf-8-sig"
    )

    polygon = detail[detail.method == "polygon_original"].set_index("seed")
    hv_difference = own.hv - polygon.hv
    hv_test = stats.ttest_rel(own.hv, polygon.hv)
    lines = [
        "# V4-B common-seed comparison (42-51)",
        "",
        "Post-selection descriptive comparison at 10,240 oracle calls per seed. Parameters were frozen from the prospective V4 confirmation.",
        "",
        "| method | HV mean +/- SD | validity | best minimum activity |",
        "|---|---:|---:|---:|",
    ]
    for _, row in aggregate.iterrows():
        lines.append(
            f"| {row.method} | {row.hv_mean:.4f} +/- {row.hv_sd:.4f} | "
            f"{row.validity_mean:.4f} | {row.best_min_activity_mean:.4f} |"
        )
    lines += [
        "",
        "## Paired V4-B versus POLYGON",
        "",
        f"- HV difference: {hv_difference.mean():+.4f}; wins {(hv_difference > 0).sum()}/10; paired p={hv_test.pvalue:.6f}.",
        f"- Quality-HV difference: {q_difference.mean():+.4f}, 95% CI [{q_ci[0]:+.4f}, {q_ci[1]:+.4f}], paired p={stats.ttest_rel(own.quality_hv, polygon_qhv).pvalue:.6f}.",
        f"- Quality pass rate: own {own.quality_pass_rate.mean():.4f}, POLYGON {polygon_qpass.mean():.4f}.",
        "- Interpretation: use this table for same-seed comparison to all completed baselines; retain seeds 82-91 as the primary prospective confirmation.",
    ]
    (OUT / "common_seed_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
