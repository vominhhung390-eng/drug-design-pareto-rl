#!/usr/bin/env python
"""Summarize the pre-registered V3-B versus official POLYGON confirmation."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "results" / "own_method_v3" / "confirmation_10240"
CONFIG = json.loads(
    (ROOT / "config" / "v3_min_elite_confirmation_10240.json").read_text(
        encoding="utf-8"
    )
)
SEEDS = CONFIG["seeds"]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_own(seed: int) -> dict:
    run = RUN_ROOT / f"v3_b_min_elite_seed{seed}" / "evaluation"
    evaluation = read_json(run / "evaluation_summary.json")
    quality = read_json(run / "quality_constrained" / "quality_constrained_summary.json")
    return {
        "method": "own_v3_b",
        "seed": seed,
        "hv": float(evaluation["hypervolume"]),
        "validity": float(evaluation["validity"]),
        "quality_hv": float(quality["quality_constrained_hypervolume"]),
        "quality_pass_rate": float(quality["quality_pass_rate"]),
        "best_min_activity": float(evaluation["best_min_activity"]),
    }


def load_polygon(seed: int) -> dict:
    run = (
        ROOT
        / "results"
        / "baselines"
        / "polygon_original"
        / f"formal_10240_seed{seed}"
        / "anytime"
        / "budget_10240"
    )
    evaluation = read_json(run / "evaluation_summary.json")
    quality = read_json(run / "quality_constrained" / "quality_constrained_summary.json")
    return {
        "method": "polygon_original",
        "seed": seed,
        "hv": float(evaluation["hypervolume"]),
        "validity": float(evaluation["validity"]),
        "quality_hv": float(quality["quality_constrained_hypervolume"]),
        "quality_pass_rate": float(quality["quality_pass_rate"]),
        "best_min_activity": float(evaluation["best_min_activity"]),
    }


def mean_sd(values: pd.Series) -> str:
    return f"{values.mean():.4f} +/- {values.std(ddof=1):.4f}"


def main() -> None:
    rows = [load(seed) for load in (load_own, load_polygon) for seed in SEEDS]
    detail = pd.DataFrame(rows).sort_values(["method", "seed"])
    own = detail[detail.method == "own_v3_b"].set_index("seed")
    polygon = detail[detail.method == "polygon_original"].set_index("seed")
    hv_diff = own.hv - polygon.hv
    quality_diff = own.quality_hv - polygon.quality_hv
    ci = stats.t.interval(
        0.95, len(hv_diff) - 1, loc=hv_diff.mean(), scale=stats.sem(hv_diff)
    )
    ttest = stats.ttest_rel(own.hv, polygon.hv)
    wilcoxon = stats.wilcoxon(hv_diff)
    gate = CONFIG["acceptance_gate"]
    gates = {
        "mean_hv_gain": hv_diff.mean()
        >= float(gate["paired_mean_hv_gain_vs_polygon"]),
        "wins": int((hv_diff > 0).sum())
        >= int(gate["wins_vs_polygon_required"]),
        "validity": own.validity.mean() >= float(gate["validity_minimum"]),
        "quality": quality_diff.mean() >= 0.0,
        "paired_test": ttest.pvalue < float(gate["paired_test_alpha"])
        and hv_diff.mean() > 0,
    }
    aggregate = pd.DataFrame(
        [
            {
                "method": method,
                "hv_mean": group.hv.mean(),
                "hv_sd": group.hv.std(ddof=1),
                "validity_mean": group.validity.mean(),
                "quality_hv_mean": group.quality_hv.mean(),
                "quality_hv_sd": group.quality_hv.std(ddof=1),
                "quality_pass_rate_mean": group.quality_pass_rate.mean(),
                "best_min_activity_mean": group.best_min_activity.mean(),
            }
            for method, group in (("own_v3_b", own), ("polygon_original", polygon))
        ]
    )
    paired = pd.DataFrame(
        [
            {
                "comparison": "own_v3_b_minus_polygon",
                "hv_mean_difference": hv_diff.mean(),
                "hv_ci95_low": ci[0],
                "hv_ci95_high": ci[1],
                "wins": int((hv_diff > 0).sum()),
                "losses": int((hv_diff < 0).sum()),
                "paired_t_p": ttest.pvalue,
                "wilcoxon_p": wilcoxon.pvalue,
                "quality_hv_mean_difference": quality_diff.mean(),
                **{f"gate_{name}": value for name, value in gates.items()},
                "all_gates": all(gates.values()),
            }
        ]
    )
    detail.to_csv(RUN_ROOT / "confirmation_detail.csv", index=False, encoding="utf-8-sig")
    aggregate.to_csv(
        RUN_ROOT / "confirmation_aggregate.csv", index=False, encoding="utf-8-sig"
    )
    paired.to_csv(
        RUN_ROOT / "confirmation_paired_test.csv", index=False, encoding="utf-8-sig"
    )
    lines = [
        "# V3-B prospective paired confirmation (10,240 oracle calls)",
        "",
        "Seeds 72-81 and all acceptance gates were recorded before these runs. No tuning used these results.",
        "",
        "| method | HV mean +/- SD | validity | quality HV mean +/- SD | quality pass rate |",
        "|---|---:|---:|---:|---:|",
        f"| Own V3-B | {mean_sd(own.hv)} | {own.validity.mean():.4f} | {mean_sd(own.quality_hv)} | {own.quality_pass_rate.mean():.4f} |",
        f"| Official POLYGON | {mean_sd(polygon.hv)} | {polygon.validity.mean():.4f} | {mean_sd(polygon.quality_hv)} | {polygon.quality_pass_rate.mean():.4f} |",
        "",
        "## Paired result",
        "",
        f"- HV difference own minus POLYGON: {hv_diff.mean():+.4f}, 95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}].",
        f"- Wins: {(hv_diff > 0).sum()}/10; paired t-test p={ttest.pvalue:.4f}; Wilcoxon p={wilcoxon.pvalue:.4f}.",
        f"- Quality-HV difference: {quality_diff.mean():+.4f}.",
        f"- Acceptance gates: {'PASS' if all(gates.values()) else 'FAIL'}.",
        "- Decision: V3-B is rejected as the formal method because the prospective gates did not pass.",
    ]
    (RUN_ROOT / "confirmation_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
