#!/usr/bin/env python
"""Summarize prospective paired V2.1-C versus V1 confirmation."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "results" / "own_method_v21" / "prospective_confirmation_10240"
CONFIG = json.loads(
    (ROOT / "config" / "v21_prospective_confirmation_10240.json").read_text(encoding="utf-8")
)
SEEDS = CONFIG["seeds"]


def load(variant: str, seed: int) -> dict:
    run = RUN_ROOT / f"{variant}_seed{seed}"
    evaluation = json.loads(
        (run / "evaluation" / "evaluation_summary.json").read_text(encoding="utf-8")
    )
    quality = json.loads(
        (run / "evaluation" / "quality_constrained" / "quality_constrained_summary.json")
        .read_text(encoding="utf-8")
    )
    metrics = pd.read_csv(run / "metrics.csv")
    triggered = metrics["archive_stagnation_triggered"].astype(str).str.lower().eq("true")
    return {
        "method": variant,
        "seed": seed,
        "hv": float(evaluation["hypervolume"]),
        "validity": float(evaluation["validity"]),
        "quality_hv": float(quality["quality_constrained_hypervolume"]),
        "quality_pass_rate": float(quality["quality_pass_rate"]),
        "best_min_activity": float(evaluation["best_min_activity"]),
        "triggered_epochs": int(triggered.sum()),
    }


def mean_sd(values: pd.Series) -> str:
    return f"{values.mean():.4f} +/- {values.std(ddof=1):.4f}"


def main() -> None:
    v2_name = "v21_c_stagnation_balanced25"
    rows = [load(method, seed) for method in ("v1_control", v2_name) for seed in SEEDS]
    detail = pd.DataFrame(rows).sort_values(["method", "seed"])
    v1 = detail[detail.method == "v1_control"].set_index("seed")
    v2 = detail[detail.method == v2_name].set_index("seed")
    hv_diff = v2.hv - v1.hv
    quality_diff = v2.quality_hv - v1.quality_hv
    ci = stats.t.interval(0.95, len(hv_diff) - 1, loc=hv_diff.mean(), scale=stats.sem(hv_diff))
    ttest = stats.ttest_rel(v2.hv, v1.hv)
    wilcoxon = stats.wilcoxon(v2.hv, v1.hv)
    gate = CONFIG["acceptance_gate"]
    gates = {
        "mean_hv_gain": hv_diff.mean() >= float(gate["paired_mean_hv_gain_vs_v1"]),
        "wins": int((hv_diff > 0).sum()) >= int(gate["wins_vs_v1_required"]),
        "validity": v2.validity.mean() >= float(gate["validity_minimum"]),
        "quality": quality_diff.mean() >= 0.0,
        "paired_test": ttest.pvalue < float(gate["paired_test_alpha"]) and hv_diff.mean() > 0,
    }
    aggregate = pd.DataFrame([
        {
            "method": method,
            "hv_mean": group.hv.mean(),
            "hv_sd": group.hv.std(ddof=1),
            "validity_mean": group.validity.mean(),
            "quality_hv_mean": group.quality_hv.mean(),
            "quality_hv_sd": group.quality_hv.std(ddof=1),
            "quality_pass_rate_mean": group.quality_pass_rate.mean(),
            "triggered_epochs_mean": group.triggered_epochs.mean(),
        }
        for method, group in (("v1_control", v1), (v2_name, v2))
    ])
    paired = pd.DataFrame([{
        "comparison": "v21_c_minus_v1",
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
    }])
    detail.to_csv(RUN_ROOT / "confirmation_detail.csv", index=False, encoding="utf-8-sig")
    aggregate.to_csv(RUN_ROOT / "confirmation_aggregate.csv", index=False, encoding="utf-8-sig")
    paired.to_csv(RUN_ROOT / "confirmation_paired_test.csv", index=False, encoding="utf-8-sig")
    lines = [
        "# V2.1-C prospective paired confirmation (10,240 oracle calls)",
        "",
        "Seeds 62-71 and the acceptance gate were recorded before these runs. No parameter tuning used these results.",
        "",
        "| method | HV mean +/- SD | validity | quality HV mean +/- SD | quality pass rate |",
        "|---|---:|---:|---:|---:|",
        f"| V1 paired control | {mean_sd(v1.hv)} | {v1.validity.mean():.4f} | {mean_sd(v1.quality_hv)} | {v1.quality_pass_rate.mean():.4f} |",
        f"| V2.1-C | {mean_sd(v2.hv)} | {v2.validity.mean():.4f} | {mean_sd(v2.quality_hv)} | {v2.quality_pass_rate.mean():.4f} |",
        "",
        "## Paired result",
        "",
        f"- HV difference V2.1-C minus V1: {hv_diff.mean():+.4f}, 95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}].",
        f"- Wins: {(hv_diff > 0).sum()}/10; paired t-test p={ttest.pvalue:.4f}; Wilcoxon p={wilcoxon.pvalue:.4f}.",
        f"- Quality-HV difference: {quality_diff.mean():+.4f}.",
        f"- Mean stagnation-triggered epochs: {v2.triggered_epochs.mean():.1f}/160.",
        f"- Acceptance gates: {'PASS' if all(gates.values()) else 'FAIL'}.",
        "- Decision: keep V1 as the formal method unless all prospective gates pass.",
    ]
    (RUN_ROOT / "confirmation_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
