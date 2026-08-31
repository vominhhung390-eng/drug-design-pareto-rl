#!/usr/bin/env python
"""Compare V5 BalanceSync with V4-B and POLYGON on two target pairs."""
from __future__ import annotations

import argparse
import gzip
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import RDLogger

from build_target_pair_current_metrics import (
    hypervolume_2d,
    igd_plus,
    internal_diversity,
    normalize,
    pareto_front_2d,
    parse_bool,
)


SEEDS = tuple(range(42, 52))
METRICS = (
    "validity",
    "uniqueness",
    "novelty",
    "diversity",
    "hypervolume",
    "igd_plus",
    "pareto_size",
    "target_1_mean",
    "target_2_mean",
    "best_min",
    "dual_at_6",
    "dual_at_6_5",
    "dual_at_7",
    "quality_pass",
    "alert_free",
    "scaffold_diversity",
    "qc_hypervolume",
    "qc_dual_at_6",
    "qc_dual_at_6_5",
    "qc_dual_at_7",
    "qc_best_min",
)
LOWER_IS_BETTER = {"igd_plus"}


def load_training_cache(path: Path) -> set[str]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return {line.rstrip("\n") for line in handle if line.strip()}


def run_specs(project: Path, config: dict, pair_name: str) -> list[dict[str, object]]:
    pair = config["target_pairs"][pair_name]
    v4b_root = project / pair["v4_b_root"]
    polygon_root = project / pair["polygon_root"]
    new_method_root = project / pair["new_method_root"]
    if pair_name == "egfr_vegfr2":
        v4b_run = lambda seed: v4b_root / f"v4_b_raw_mean_seed{seed}"
        v4b_quality = lambda run: run / "evaluation" / "quality_constrained"
    else:
        v4b_run = lambda seed: v4b_root / f"formal_10240_seed{seed}"
        v4b_quality = lambda run: run / "anytime" / "budget_10240" / "quality_constrained"
    return [
        {
            "method": "V4-B",
            "run": v4b_run,
            "evaluation": lambda run: run / "anytime" / "budget_10240",
            "quality": v4b_quality,
        },
        {
            "method": "V5 BalanceSync",
            "run": lambda seed: new_method_root / f"formal_10240_seed{seed}",
            "evaluation": lambda run: run / "evaluation",
            "quality": lambda run: run / "evaluation" / "quality_constrained",
        },
        {
            "method": "POLYGON",
            "run": lambda seed: polygon_root / f"formal_10240_seed{seed}",
            "evaluation": lambda run: run / "anytime" / "budget_10240",
            "quality": lambda run: run / "anytime" / "budget_10240" / "quality_constrained",
        },
    ]


def exact_sign_flip_pvalue(differences: np.ndarray) -> float:
    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences)]
    if not len(differences):
        return math.nan
    observed = abs(float(differences.mean()))
    extreme = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
        statistic = abs(float(np.mean(differences * np.asarray(signs))))
        extreme += statistic >= observed - 1e-15
        total += 1
    return extreme / total


def mean_ci95(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return math.nan, math.nan
    # t(0.975, 9)=2.262; the experiment has ten paired seeds.
    critical = 2.262 if len(values) == 10 else 1.96
    half_width = critical * float(values.std(ddof=1)) / math.sqrt(len(values))
    mean = float(values.mean())
    return mean - half_width, mean + half_width


def collect_pair(
    project: Path,
    config: dict,
    pair_name: str,
    training: set[str],
) -> tuple[pd.DataFrame, np.ndarray]:
    records: list[dict[str, object]] = []
    fronts: dict[tuple[str, int], np.ndarray] = {}
    for spec in run_specs(project, config, pair_name):
        method = str(spec["method"])
        for seed in SEEDS:
            run = spec["run"](seed)
            evaluation = spec["evaluation"](run)
            quality_dir = spec["quality"](run)
            molecule_path = evaluation / "standardized_molecules.csv"
            summary_path = evaluation / "evaluation_summary.json"
            quality_path = quality_dir / "quality_annotated_molecules.csv"
            for required in (molecule_path, summary_path, quality_path):
                if not required.exists():
                    raise FileNotFoundError(required)

            frame = pd.read_csv(molecule_path, encoding="utf-8-sig")
            quality = pd.read_csv(quality_path, encoding="utf-8-sig")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if len(frame) != int(summary["unique_valid"]):
                raise ValueError(f"{pair_name} {method} seed {seed}: row mismatch")
            if len(frame) != len(quality):
                raise ValueError(f"{pair_name} {method} seed {seed}: quality mismatch")
            if set(frame["smiles"].astype(str)) != set(quality["smiles"].astype(str)):
                raise ValueError(f"{pair_name} {method} seed {seed}: molecule mismatch")

            points = frame[["egfr", "vegfr2"]].to_numpy(float)
            normalized = normalize(points)
            front = np.unique(pareto_front_2d(normalized), axis=0)
            fronts[(method, seed)] = front
            smiles = frame["smiles"].astype(str).tolist()
            diversity, diversity_n = internal_diversity(smiles, method, seed)

            quality = quality.set_index("smiles").loc[frame["smiles"].astype(str)].reset_index()
            quality_pass = parse_bool(quality["quality_pass"])
            alert_free = ~parse_bool(quality["structural_alert"])
            qc = quality.loc[quality_pass]
            qc_points = qc[["egfr", "vegfr2"]].to_numpy(float) if len(qc) else np.empty((0, 2))
            nonempty_scaffolds = frame["scaffold"].fillna("").astype(str).str.len() > 0
            scaffold_count = frame.loc[nonempty_scaffolds, "scaffold"].nunique()
            min_scores = np.minimum(frame["egfr"], frame["vegfr2"])

            records.append(
                {
                    "target_pair": pair_name,
                    "method": method,
                    "seed": seed,
                    "generated_rows": int(summary["generated_rows"]),
                    "valid_rows": int(summary["valid_rows"]),
                    "unique_valid": int(summary["unique_valid"]),
                    "diversity_sample_n": diversity_n,
                    "validity": float(summary["validity"]),
                    "uniqueness": float(summary["uniqueness_valid"]),
                    "novelty": sum(value not in training for value in smiles) / len(smiles) if smiles else 0.0,
                    "diversity": diversity,
                    "hypervolume": hypervolume_2d(normalized),
                    "igd_plus": math.nan,
                    "pareto_size": len(front),
                    "target_1_mean": float(frame["egfr"].mean()),
                    "target_2_mean": float(frame["vegfr2"].mean()),
                    "best_min": float(min_scores.max()) if len(frame) else math.nan,
                    "dual_at_6": float((min_scores >= 6.0).mean()) if len(frame) else 0.0,
                    "dual_at_6_5": float((min_scores >= 6.5).mean()) if len(frame) else 0.0,
                    "dual_at_7": float((min_scores >= 7.0).mean()) if len(frame) else 0.0,
                    "quality_pass": float(quality_pass.mean()) if len(quality) else 0.0,
                    "alert_free": float(alert_free.mean()) if len(quality) else 0.0,
                    "scaffold_diversity": float(scaffold_count / len(frame)) if len(frame) else 0.0,
                    "qc_hypervolume": hypervolume_2d(normalize(qc_points)) if len(qc) else 0.0,
                    "qc_dual_at_6": float((np.minimum(qc["egfr"], qc["vegfr2"]) >= 6.0).mean()) if len(qc) else 0.0,
                    "qc_dual_at_6_5": float((np.minimum(qc["egfr"], qc["vegfr2"]) >= 6.5).mean()) if len(qc) else 0.0,
                    "qc_dual_at_7": float((np.minimum(qc["egfr"], qc["vegfr2"]) >= 7.0).mean()) if len(qc) else 0.0,
                    "qc_best_min": float(np.minimum(qc["egfr"], qc["vegfr2"]).max()) if len(qc) else math.nan,
                }
            )
            print(f"{pair_name} {method} seed {seed} loaded", flush=True)

    pooled = np.vstack(list(fronts.values()))
    reference = np.unique(pareto_front_2d(np.unique(pooled, axis=0)), axis=0)
    reference = reference[np.argsort(reference[:, 0])]
    for record in records:
        key = (str(record["method"]), int(record["seed"]))
        record["igd_plus"] = igd_plus(fronts[key], reference)
    return pd.DataFrame(records), reference


def aggregate(per_seed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (pair, method), subset in per_seed.groupby(["target_pair", "method"], sort=False):
        for metric in METRICS:
            values = subset[metric].astype(float)
            rows.append(
                {
                    "target_pair": pair,
                    "method": method,
                    "metric": metric,
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)),
                    "n": int(values.notna().sum()),
                }
            )
    return pd.DataFrame(rows)


def paired_comparisons(per_seed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pair, pair_frame in per_seed.groupby("target_pair", sort=False):
        for reference_method in ("V4-B", "POLYGON"):
            left = pair_frame[pair_frame["method"] == reference_method].set_index("seed")
            right = pair_frame[pair_frame["method"] == "V5 BalanceSync"].set_index("seed")
            common = left.index.intersection(right.index)
            for metric in METRICS:
                baseline = left.loc[common, metric].to_numpy(float)
                candidate = right.loc[common, metric].to_numpy(float)
                delta = candidate - baseline
                oriented = -delta if metric in LOWER_IS_BETTER else delta
                ci_low, ci_high = mean_ci95(delta)
                baseline_mean = float(np.mean(baseline))
                rows.append(
                    {
                        "target_pair": pair,
                        "comparison": f"V5 vs {reference_method}",
                        "metric": metric,
                        "reference_mean": baseline_mean,
                        "v5_mean": float(np.mean(candidate)),
                        "delta_v5_minus_reference": float(np.mean(delta)),
                        "delta_ci95_low": ci_low,
                        "delta_ci95_high": ci_high,
                        "oriented_improvement": float(np.mean(oriented)),
                        "relative_change": float(np.mean(delta) / abs(baseline_mean)) if baseline_mean else math.nan,
                        "v5_seed_wins": int(np.sum(oriented > 0)),
                        "ties": int(np.sum(np.isclose(oriented, 0.0))),
                        "n": len(common),
                        "exact_sign_flip_p": exact_sign_flip_pvalue(delta),
                    }
                )
    return pd.DataFrame(rows)


def write_markdown(aggregate_frame: pd.DataFrame, paired: pd.DataFrame, output: Path) -> None:
    labels = {
        "egfr_vegfr2": "EGFR-VEGFR2",
        "parp1_brd4": "PARP1-BRD4",
    }
    key_metrics = [
        "validity", "hypervolume", "igd_plus", "dual_at_6", "dual_at_6_5",
        "quality_pass", "qc_dual_at_6_5", "scaffold_diversity",
    ]
    lines = [
        "# V5 BalanceSync two-target-pair diagnostic comparison",
        "",
        "V5 changes only generator elite ranking. V4-B already standardized each objective's Advantage before scalarization.",
        "Seeds 42-51 are a paired retrospective diagnostic set, not a new independent confirmation set.",
    ]
    for pair in labels:
        lines.extend(["", f"## {labels[pair]}", "", "| metric | V4-B | V5 | POLYGON |", "|---|---:|---:|---:|"])
        for metric in key_metrics:
            values = aggregate_frame[
                (aggregate_frame["target_pair"] == pair)
                & (aggregate_frame["metric"] == metric)
            ].set_index("method")
            formatted = []
            for method in ("V4-B", "V5 BalanceSync", "POLYGON"):
                mean = float(values.loc[method, "mean"])
                sd = float(values.loc[method, "sd"])
                if "dual" in metric or metric in {"validity", "quality_pass", "scaffold_diversity"}:
                    formatted.append(f"{100 * mean:.2f}+/-{100 * sd:.2f}%")
                else:
                    formatted.append(f"{mean:.4f}+/-{sd:.4f}")
            lines.append(f"| {metric} | {formatted[0]} | {formatted[1]} | {formatted[2]} |")

        subset = paired[
            (paired["target_pair"] == pair)
            & (paired["comparison"] == "V5 vs V4-B")
            & (paired["metric"].isin(key_metrics))
        ]
        lines.extend(["", "Paired V5 minus V4-B:", "", "| metric | delta | seed wins | exact p |", "|---|---:|---:|---:|"])
        for row in subset.itertuples(index=False):
            delta = row.delta_v5_minus_reference
            if "dual" in row.metric or row.metric in {"validity", "quality_pass", "scaffold_diversity"}:
                delta_text = f"{100 * delta:+.2f} pp"
            else:
                delta_text = f"{delta:+.4f}"
            lines.append(f"| {row.metric} | {delta_text} | {row.v5_seed_wins}/{row.n} | {row.exact_sign_flip_p:.4f} |")
    (output / "comparison_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/v5_balancesync_two_pair_20260806.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/own_method_v5_balancesync_20260806/analysis"),
    )
    args = parser.parse_args()
    project = args.project.resolve()
    config_path = args.config if args.config.is_absolute() else project / args.config
    output = args.output if args.output.is_absolute() else project / args.output
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    training_cache = project / "outputs/019f5762-58d7-7670-9168-54fe5fbeb2b3/training_smiles_canonical.txt.gz"
    training = load_training_cache(training_cache)
    RDLogger.DisableLog("rdApp.error")
    RDLogger.DisableLog("rdApp.warning")

    frames = []
    for pair_name in config["target_pairs"]:
        frame, reference = collect_pair(project, config, pair_name, training)
        frames.append(frame)
        pd.DataFrame(reference, columns=["target_1_normalized", "target_2_normalized"]).to_csv(
            output / f"{pair_name}_igd_reference_front.csv", index=False, encoding="utf-8-sig"
        )
    per_seed = pd.concat(frames, ignore_index=True)
    aggregate_frame = aggregate(per_seed)
    paired = paired_comparisons(per_seed)
    per_seed.to_csv(output / "per_seed_metrics.csv", index=False, encoding="utf-8-sig")
    aggregate_frame.to_csv(output / "aggregate_metrics.csv", index=False, encoding="utf-8-sig")
    paired.to_csv(output / "paired_comparisons.csv", index=False, encoding="utf-8-sig")
    write_markdown(aggregate_frame, paired, output)
    (output / "analysis_metadata.json").write_text(
        json.dumps(
            {
                "config": str(config_path),
                "training_cache": str(training_cache),
                "seeds": list(SEEDS),
                "methods": ["V4-B", "V5 BalanceSync", "POLYGON"],
                "igd_reference": "joint nondominated front across these three methods and ten paired seeds within each target pair",
                "statistical_test": "exact two-sided paired sign-flip test over seed-level differences",
                "interpretation": "retrospective paired diagnostic; independent confirmation required before replacing V4-B",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Wrote V5 comparison to {output}", flush=True)


if __name__ == "__main__":
    main()
