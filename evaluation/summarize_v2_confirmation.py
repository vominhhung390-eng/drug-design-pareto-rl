#!/usr/bin/env python
"""Summarize paired V2 archive and V1 held-out confirmation runs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
V2_ROOT = ROOT / "results" / "own_method_v2" / "confirmation_10240"
V1_ROOT = ROOT / "results" / "own_method_v2" / "v1_extended_control_10240"
SEEDS = range(52, 62)


def load(method: str, seed: int) -> dict:
    if method == "v2_archive":
        run = V2_ROOT / f"v2_archive_seed{seed}"
    else:
        run = V1_ROOT / f"v1_control_seed{seed}"
    evaluation = json.loads(
        (run / "evaluation" / "evaluation_summary.json").read_text(encoding="utf-8")
    )
    quality = json.loads(
        (run / "evaluation" / "quality_constrained" / "quality_constrained_summary.json")
        .read_text(encoding="utf-8")
    )
    return {
        "method": method,
        "seed": seed,
        "hv": float(evaluation["hypervolume"]),
        "validity": float(evaluation["validity"]),
        "unique_valid": int(evaluation["unique_valid"]),
        "quality_hv": float(quality["quality_constrained_hypervolume"]),
        "quality_pass_rate": float(quality["quality_pass_rate"]),
        "best_min_activity": float(evaluation["best_min_activity"]),
    }


def fmt(series: pd.Series) -> str:
    return f"{series.mean():.4f} +/- {series.std(ddof=1):.4f}"


def main() -> None:
    detail = pd.DataFrame(
        [load(method, seed) for method in ("v1", "v2_archive") for seed in SEEDS]
    )
    v1 = detail[detail.method == "v1"].set_index("seed")
    v2 = detail[detail.method == "v2_archive"].set_index("seed")
    hv_diff = v2.hv - v1.hv
    qhv_diff = v2.quality_hv - v1.quality_hv
    hv_ci = stats.t.interval(
        0.95, len(hv_diff) - 1, loc=hv_diff.mean(), scale=stats.sem(hv_diff)
    )
    ttest = stats.ttest_rel(v2.hv, v1.hv)
    wilcoxon = stats.wilcoxon(v2.hv, v1.hv)

    aggregate = pd.DataFrame([
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
        for method, group in (("v1", v1), ("v2_archive", v2))
    ])
    paired = pd.DataFrame([{
        "comparison": "v2_archive_minus_v1",
        "hv_mean_difference": hv_diff.mean(),
        "hv_ci95_low": hv_ci[0],
        "hv_ci95_high": hv_ci[1],
        "wins": int((hv_diff > 0).sum()),
        "ties": int((hv_diff == 0).sum()),
        "losses": int((hv_diff < 0).sum()),
        "paired_t_p": ttest.pvalue,
        "wilcoxon_p": wilcoxon.pvalue,
        "quality_hv_mean_difference": qhv_diff.mean(),
    }])

    detail.to_csv(V2_ROOT / "confirmation_detail.csv", index=False, encoding="utf-8-sig")
    aggregate.to_csv(V2_ROOT / "confirmation_aggregate.csv", index=False, encoding="utf-8-sig")
    paired.to_csv(V2_ROOT / "confirmation_paired_test.csv", index=False, encoding="utf-8-sig")
    lines = [
        "# V2 archive held-out confirmation (10,240 oracle calls)",
        "",
        "Seeds 52-61 were not used in the 42-44 development screen. V1 was rerun on the same seeds and budget.",
        "",
        "| method | HV mean +/- SD | validity | quality HV mean +/- SD | quality pass rate |",
        "|---|---:|---:|---:|---:|",
        f"| V1 paired control | {fmt(v1.hv)} | {v1.validity.mean():.4f} | {fmt(v1.quality_hv)} | {v1.quality_pass_rate.mean():.4f} |",
        f"| V2 archive | {fmt(v2.hv)} | {v2.validity.mean():.4f} | {fmt(v2.quality_hv)} | {v2.quality_pass_rate.mean():.4f} |",
        "",
        "## Paired result",
        "",
        f"- HV difference V2-V1: {hv_diff.mean():+.4f}, 95% CI [{hv_ci[0]:+.4f}, {hv_ci[1]:+.4f}].",
        f"- Paired wins: {(hv_diff > 0).sum()}/10; paired t-test p={ttest.pvalue:.4f}; Wilcoxon p={wilcoxon.pvalue:.4f}.",
        f"- Quality-HV difference: {qhv_diff.mean():+.4f}.",
        "- Decision: archive-only does not replace V1; the held-out paired result shows no reliable gain.",
    ]
    (V2_ROOT / "confirmation_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
