#!/usr/bin/env python
"""Aggregate the prospectively registered V4-B formal ablation."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


REFERENCE = "a0_full_v4b"
SINGLE_FACTOR = [
    "a1_batch256",
    "a2_no_online_vae",
    "a3_legacy_elite",
    "a4_frozen_actor",
    "a5_fixed_weight",
    "a6_single_critic",
    "a7_nonadaptive_channels",
]
METRICS = {
    "hv_final": True,
    "quality_hv": True,
    "validity": True,
    "best_min_activity": True,
    "dual_active_7_rate": True,
    "scaffold_diversity": True,
    "qed_mean": True,
    "runtime_sec": False,
}


def holm(pvalues: list[float]) -> list[float]:
    values = np.asarray(pvalues, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def parse_run_name(name: str) -> tuple[str, int]:
    variant, seed = name.rsplit("_seed", 1)
    return variant, int(seed)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def t_ci(values: np.ndarray) -> tuple[float, float]:
    if len(values) < 2:
        return math.nan, math.nan
    sem = stats.sem(values)
    return tuple(map(float, stats.t.interval(0.95, len(values) - 1, loc=np.mean(values), scale=sem)))


def paired_test(reference: np.ndarray, ablation: np.ndarray, higher: bool) -> dict:
    delta = reference - ablation if higher else ablation - reference
    delta = np.asarray(delta, dtype=float)
    low, high = t_ci(delta)
    try:
        t_p = float(stats.ttest_1samp(delta, 0.0).pvalue)
    except ValueError:
        t_p = math.nan
    try:
        w_p = float(stats.wilcoxon(delta).pvalue)
    except ValueError:
        w_p = 1.0
    sd = float(np.std(delta, ddof=1)) if len(delta) > 1 else math.nan
    return {
        "n": len(delta),
        "reference_mean": float(np.mean(reference)),
        "ablation_mean": float(np.mean(ablation)),
        "reference_better_delta": float(np.mean(delta)),
        "ci95_low": low,
        "ci95_high": high,
        "paired_t_p": t_p,
        "wilcoxon_p": w_p,
        "effect_dz": float(np.mean(delta) / sd) if sd > 0 else math.nan,
        "reference_wins": int(np.sum(delta > 0)),
    }


def tost(values: np.ndarray, margin: float) -> dict:
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1))
    se = sd / np.sqrt(len(values))
    if se == 0:
        pvalue = 0.0 if abs(mean) < margin else 1.0
    else:
        lower_t = (mean + margin) / se
        upper_t = (mean - margin) / se
        pvalue = max(float(stats.t.sf(lower_t, len(values) - 1)), float(stats.t.cdf(upper_t, len(values) - 1)))
    return {"margin": margin, "mean_delta": mean, "tost_p": pvalue, "equivalent": pvalue < 0.05}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    config = read_json(args.config.resolve())
    labels = {name: item["label"] for name, item in config["variants"].items()}
    rows = []
    for run in sorted(root.glob("*_seed*")):
        if not run.is_dir() or not (run / "summary.csv").exists():
            continue
        variant, seed = parse_run_name(run.name)
        base = pd.read_csv(run / "summary.csv").iloc[0].to_dict()
        evaluation = read_json(run / "evaluation" / "evaluation_summary.json")
        quality = read_json(run / "evaluation" / "quality_constrained" / "quality_constrained_summary.json")
        row = {
            "variant": variant,
            "label": labels.get(variant, variant),
            "seed": seed,
            "hv_final": float(base["hv_final"]),
            "quality_hv": float(quality["quality_constrained_hypervolume"]),
            "validity": float(evaluation["validity"]),
            "best_min_activity": float(evaluation["best_min_activity"]),
            "dual_active_7_rate": float(quality["dual_active_7_rate"]),
            "scaffold_diversity": float(evaluation["scaffold_diversity"]),
            "qed_mean": float(evaluation["qed_mean"]),
            "runtime_sec": float(base["runtime_sec"]),
        }
        for budget in (1024, 2048, 5120, 10240):
            anytime = read_json(run / "anytime" / f"budget_{budget}" / "evaluation_summary.json")
            row[f"hv_{budget}"] = float(anytime["hypervolume"])
        rows.append(row)
    data = pd.DataFrame(rows)
    expected = len(config["seeds"]) * len(config["variants"])
    if len(data) != expected:
        raise RuntimeError(f"Expected {expected} evaluated runs, found {len(data)}")
    data.to_csv(root / "ablation_all_runs.csv", index=False, encoding="utf-8-sig")

    aggregates = []
    for variant, group in data.groupby("variant", sort=False):
        record = {"variant": variant, "label": labels.get(variant, variant), "n": len(group)}
        for metric in METRICS:
            record[f"{metric}_mean"] = float(group[metric].mean())
            record[f"{metric}_sd"] = float(group[metric].std(ddof=1))
        for budget in (1024, 2048, 5120, 10240):
            record[f"hv_{budget}_mean"] = float(group[f"hv_{budget}"].mean())
            record[f"hv_{budget}_sd"] = float(group[f"hv_{budget}"].std(ddof=1))
        aggregates.append(record)
    aggregate = pd.DataFrame(aggregates)
    aggregate.to_csv(root / "ablation_aggregate.csv", index=False, encoding="utf-8-sig")

    reference = data[data.variant == REFERENCE].set_index("seed")
    comparisons = []
    for variant in config["variants"]:
        if variant == REFERENCE:
            continue
        candidate = data[data.variant == variant].set_index("seed").loc[reference.index]
        for metric, higher in METRICS.items():
            result = paired_test(reference[metric].to_numpy(), candidate[metric].to_numpy(), higher)
            result.update({"reference": REFERENCE, "ablation": variant, "metric": metric})
            comparisons.append(result)
    comparisons = pd.DataFrame(comparisons)
    comparisons["holm_p"] = np.nan
    for metric in METRICS:
        mask = (comparisons.metric == metric) & comparisons.ablation.isin(SINGLE_FACTOR)
        comparisons.loc[mask, "holm_p"] = holm(comparisons.loc[mask, "paired_t_p"].tolist())
    comparisons.to_csv(root / "ablation_paired_tests.csv", index=False, encoding="utf-8-sig")

    batch_delta = reference["hv_final"].to_numpy() - data[data.variant == "a1_batch256"].set_index("seed").loc[reference.index, "hv_final"].to_numpy()
    equivalence = tost(batch_delta, float(config["equivalence_margin_hv"]))
    (root / "batch256_equivalence.json").write_text(json.dumps(equivalence, indent=2), encoding="utf-8")

    factorial_rows = []
    for metric, higher in {"hv_final": True, "quality_hv": True}.items():
        pivot = data.pivot(index="seed", columns="variant", values=metric).loc[reference.index]
        interaction = (pivot[REFERENCE] - pivot["a5_fixed_weight"]) - (pivot["a6_single_critic"] - pivot["a8_single_fixed"])
        if not higher:
            interaction = -interaction
        low, high = t_ci(interaction.to_numpy())
        factorial_rows.append({
            "interaction": "critic_x_weight",
            "metric": metric,
            "difference_in_differences": float(interaction.mean()),
            "ci95_low": low,
            "ci95_high": high,
            "paired_t_p": float(stats.ttest_1samp(interaction, 0.0).pvalue),
        })
    pd.DataFrame(factorial_rows).to_csv(root / "ablation_factorial_interactions.csv", index=False, encoding="utf-8-sig")

    order = list(config["variants"])
    table = aggregate.set_index("variant").loc[order]
    lines = [
        "# V4-B formal ablation (seeds 92-101)", "",
        f"- Completed runs: {len(data)}/{expected}",
        "- Primary endpoint: normalized activity hypervolume at 10,240 oracle calls.",
        "- Positive paired delta means Full V4-B is better.", "",
        "| Variant | HV | Quality-HV | Validity | Best min activity | Runtime (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant, row in table.iterrows():
        lines.append(
            f"| {labels[variant]} | {row.hv_final_mean:.4f} +/- {row.hv_final_sd:.4f} | "
            f"{row.quality_hv_mean:.4f} +/- {row.quality_hv_sd:.4f} | "
            f"{row.validity_mean:.4f} | {row.best_min_activity_mean:.4f} | {row.runtime_sec_mean:.1f} |"
        )
    lines += ["", "## Paired primary comparisons", "", "| Ablation | Delta HV | 95% CI | Wins | Raw p | Holm p |", "|---|---:|---:|---:|---:|---:|"]
    primary = comparisons[comparisons.metric == "hv_final"].set_index("ablation")
    for variant in order[1:]:
        row = primary.loc[variant]
        lines.append(
            f"| {labels[variant]} | {row.reference_better_delta:+.4f} | [{row.ci95_low:+.4f}, {row.ci95_high:+.4f}] | "
            f"{int(row.reference_wins)}/10 | {row.paired_t_p:.4g} | {row.holm_p:.4g} |"
        )
    lines += ["", "## Batch-size equivalence", "", f"- TOST margin: +/-{equivalence['margin']:.3f} HV", f"- Mean Full-minus-batch256 delta: {equivalence['mean_delta']:+.4f}", f"- TOST p: {equivalence['tost_p']:.4g}; equivalent: {equivalence['equivalent']}"]
    (root / "ablation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
