#!/usr/bin/env python
"""Paired multi-seed statistical report for formal experiments."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


DEFAULT_METRICS = [
    "hv_final", "pareto_size", "egfr_max", "vegfr2_max",
    "best_balanced_score", "valid_rate", "novelty", "qed_mean", "sa_mean",
]


def parse_name(path: Path):
    name = path.parent.name
    variant, seed = name.rsplit("_seed", 1)
    return variant, int(seed)


def load(roots):
    rows = []
    for root in roots:
        for path in root.glob("*_seed*/summary.csv"):
            variant, seed = parse_name(path)
            row = pd.read_csv(path).iloc[0].to_dict()
            row.update({"variant": variant, "seed": seed, "root": str(root)})
            rows.append(row)
    if not rows:
        raise FileNotFoundError("No summary.csv files were found")
    return pd.DataFrame(rows).drop_duplicates(["variant", "seed"], keep="last")


def bootstrap_ci(values, rng, iterations=10000):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return np.nan, np.nan
    samples = rng.choice(values, size=(iterations, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(samples, [0.025, 0.975]))


def holm_adjust(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues)
    adjusted = np.empty_like(pvalues)
    running = 0.0
    count = len(pvalues)
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * pvalues[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--reference", default="ours_full")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    runs = load(args.roots)
    runs.to_csv(args.output / "all_formal_runs.csv", index=False, encoding="utf-8-sig")
    metrics = [metric for metric in DEFAULT_METRICS if metric in runs.columns]
    reference = runs[runs.variant == args.reference].set_index("seed")
    if reference.empty:
        raise ValueError(f"Reference variant {args.reference!r} was not found")

    rng = np.random.default_rng(args.seed)
    rows = []
    variants = sorted(set(runs.variant) - {args.reference})
    for variant in variants:
        candidate = runs[runs.variant == variant].set_index("seed")
        seeds = reference.index.intersection(candidate.index)
        for metric in metrics:
            paired = pd.DataFrame({
                "reference": pd.to_numeric(reference.loc[seeds, metric], errors="coerce"),
                "competitor": pd.to_numeric(candidate.loc[seeds, metric], errors="coerce"),
            }).dropna()
            if paired.empty:
                continue
            ref = paired["reference"].to_numpy()
            other = paired["competitor"].to_numpy()
            # Positive delta always means the reference is better.
            delta = other - ref if metric == "sa_mean" else ref - other
            low, high = bootstrap_ci(delta, rng, args.bootstrap)
            try:
                pvalue = float(wilcoxon(delta, alternative="two-sided").pvalue)
            except ValueError:
                pvalue = 1.0
            sd = float(np.std(delta, ddof=1)) if len(delta) > 1 else np.nan
            rows.append(
                {
                    "reference": args.reference,
                    "competitor": variant,
                    "metric": metric,
                    "paired_seeds": len(paired),
                    "reference_mean": float(np.mean(ref)),
                    "competitor_mean": float(np.mean(other)),
                    "reference_better_delta_mean": float(np.mean(delta)),
                    "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high,
                    "wilcoxon_p": pvalue,
                    "paired_effect_size_dz": float(np.mean(delta) / sd) if sd and np.isfinite(sd) else np.nan,
                    "reference_wins": int(np.sum(delta > 0)),
                }
            )
    report = pd.DataFrame(rows)
    report["holm_p"] = np.nan
    for metric, indexes in report.groupby("metric").groups.items():
        valid = report.loc[indexes, "wilcoxon_p"].dropna()
        report.loc[valid.index, "holm_p"] = holm_adjust(valid)
    report.to_csv(args.output / "paired_statistical_tests.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
