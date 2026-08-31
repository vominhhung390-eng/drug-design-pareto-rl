#!/usr/bin/env python
"""Unified two-target-pair comparison of V4-B, V5 and all five baselines."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import RDLogger

from build_target_pair_current_metrics import (
    SEEDS,
    hypervolume_2d,
    igd_plus,
    internal_diversity,
    load_training_cache,
    normalize,
    pareto_front_2d,
    parse_bool,
)


METHODS = (
    "V4-B",
    "V5 BalanceSync",
    "POLYGON",
    "REINVENT4",
    "DrugEx v2",
    "MO-LSO",
    "GraphPareto-NSGA-II",
)

METRICS = (
    "validity",
    "uniqueness",
    "novelty",
    "diversity",
    "hypervolume",
    "igd_plus",
    "pareto_size",
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


def pair_specs(project: Path, pair: str) -> list[dict[str, object]]:
    if pair == "egfr_vegfr2":
        v4_root = project / "results/own_method_v4/common_seeds_42_51_10240"
        v5_root = project / "results/own_method_v5_balancesync_20260806/egfr_vegfr2"
        baseline_root = project / "results/baselines"
        v4_run = lambda seed: v4_root / f"v4_b_raw_mean_seed{seed}"
        v4_quality = lambda run: run / "evaluation/quality_constrained"
    elif pair == "parp1_brd4":
        experiment = project / "results/target_pairs/parp1_brd4_20260804"
        v4_root = experiment / "own_method"
        v5_root = project / "results/own_method_v5_balancesync_20260806/parp1_brd4"
        baseline_root = experiment / "baselines"
        v4_run = lambda seed: v4_root / f"formal_10240_seed{seed}"
        v4_quality = lambda run: run / "anytime/budget_10240/quality_constrained"
    else:
        raise ValueError(pair)

    specs: list[dict[str, object]] = [
        {
            "method": "V4-B",
            "run": v4_run,
            "evaluation": lambda run: run / "anytime/budget_10240",
            "quality": v4_quality,
        },
        {
            "method": "V5 BalanceSync",
            "run": lambda seed: v5_root / f"formal_10240_seed{seed}",
            "evaluation": lambda run: run / "evaluation",
            "quality": lambda run: run / "evaluation/quality_constrained",
        },
    ]
    for method, folder in (
        ("POLYGON", "polygon_original"),
        ("REINVENT4", "reinvent4"),
        ("DrugEx v2", "drugex_v2"),
        ("MO-LSO", "mo_lso"),
        ("GraphPareto-NSGA-II", "graphpareto_nsga2"),
    ):
        root = baseline_root / folder
        specs.append({
            "method": method,
            "run": lambda seed, root=root: root / f"formal_10240_seed{seed}",
            "evaluation": lambda run: run / "anytime/budget_10240",
            "quality": lambda run: run / "anytime/budget_10240/quality_constrained",
        })
    return specs


def collect_pair(
    project: Path,
    pair: str,
    training: set[str],
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    records: list[dict[str, object]] = []
    fronts: dict[tuple[str, int], np.ndarray] = {}
    completion: list[dict[str, object]] = []
    for spec in pair_specs(project, pair):
        method = str(spec["method"])
        completed = 0
        for seed in SEEDS:
            run = spec["run"](seed)
            evaluation = spec["evaluation"](run)
            quality_dir = spec["quality"](run)
            molecule_path = evaluation / "standardized_molecules.csv"
            summary_path = evaluation / "evaluation_summary.json"
            quality_path = quality_dir / "quality_annotated_molecules.csv"
            if not all(path.exists() for path in (molecule_path, summary_path, quality_path)):
                continue

            frame = pd.read_csv(molecule_path, encoding="utf-8-sig")
            quality = pd.read_csv(quality_path, encoding="utf-8-sig")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if len(frame) != int(summary["unique_valid"]):
                raise ValueError(f"{pair} {method} seed {seed}: standardized row mismatch")
            if len(quality) != len(frame):
                raise ValueError(f"{pair} {method} seed {seed}: quality row mismatch")
            if set(frame["smiles"].astype(str)) != set(quality["smiles"].astype(str)):
                raise ValueError(f"{pair} {method} seed {seed}: quality identity mismatch")

            completed += 1
            smiles = frame["smiles"].astype(str).tolist()
            points = frame[["egfr", "vegfr2"]].to_numpy(float)
            normalized = normalize(points)
            front = np.unique(pareto_front_2d(normalized), axis=0)
            fronts[(method, seed)] = front
            diversity, sample_n = internal_diversity(smiles, f"{pair}:{method}", seed)

            quality = quality.set_index("smiles").loc[frame["smiles"].astype(str)].reset_index()
            quality_pass = parse_bool(quality["quality_pass"])
            alert_free = ~parse_bool(quality["structural_alert"])
            qc = quality.loc[quality_pass]
            qc_points = qc[["egfr", "vegfr2"]].to_numpy(float) if len(qc) else np.empty((0, 2))
            min_score = np.minimum(frame["egfr"], frame["vegfr2"])
            qc_min = np.minimum(qc["egfr"], qc["vegfr2"]) if len(qc) else np.asarray([])
            nonempty_scaffolds = frame["scaffold"].fillna("").astype(str).str.len() > 0
            scaffold_count = frame.loc[nonempty_scaffolds, "scaffold"].nunique()

            records.append({
                "target_pair": pair,
                "method": method,
                "seed": seed,
                "generated_rows": int(summary["generated_rows"]),
                "valid_rows": int(summary["valid_rows"]),
                "unique_valid": int(summary["unique_valid"]),
                "diversity_sample_n": sample_n,
                "validity": float(summary["validity"]),
                "uniqueness": float(summary["uniqueness_valid"]),
                "novelty": sum(value not in training for value in smiles) / len(smiles) if smiles else 0.0,
                "diversity": diversity,
                "hypervolume": hypervolume_2d(normalized),
                "igd_plus": math.nan,
                "pareto_size": int(len(front)),
                "dual_at_6": float((min_score >= 6.0).mean()) if len(frame) else 0.0,
                "dual_at_6_5": float((min_score >= 6.5).mean()) if len(frame) else 0.0,
                "dual_at_7": float((min_score >= 7.0).mean()) if len(frame) else 0.0,
                "quality_pass": float(quality_pass.mean()) if len(quality) else 0.0,
                "alert_free": float(alert_free.mean()) if len(quality) else 0.0,
                "scaffold_diversity": float(scaffold_count / len(frame)) if len(frame) else 0.0,
                "qc_hypervolume": hypervolume_2d(normalize(qc_points)) if len(qc) else 0.0,
                "qc_dual_at_6": float((qc_min >= 6.0).mean()) if len(qc) else 0.0,
                "qc_dual_at_6_5": float((qc_min >= 6.5).mean()) if len(qc) else 0.0,
                "qc_dual_at_7": float((qc_min >= 7.0).mean()) if len(qc) else 0.0,
                "qc_best_min": float(qc_min.max()) if len(qc) else math.nan,
            })
            print(f"{pair} {method} seed {seed} loaded", flush=True)
        completion.append({
            "target_pair": pair,
            "method": method,
            "completed_seeds": completed,
            "planned_seeds": len(SEEDS),
        })

    if not records:
        raise RuntimeError(f"No completed evaluations for {pair}")
    pooled = np.vstack(list(fronts.values()))
    reference = np.unique(pareto_front_2d(np.unique(pooled, axis=0)), axis=0)
    reference = reference[np.argsort(reference[:, 0])]
    for record in records:
        key = (str(record["method"]), int(record["seed"]))
        record["igd_plus"] = igd_plus(fronts[key], reference)
    return pd.DataFrame(records), reference, pd.DataFrame(completion)


def aggregate(per_seed: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for pair in ("egfr_vegfr2", "parp1_brd4"):
        for method in METHODS:
            subset = per_seed[(per_seed["target_pair"] == pair) & (per_seed["method"] == method)]
            if subset.empty:
                continue
            for metric in METRICS:
                values = subset[metric].astype(float)
                rows.append({
                    "target_pair": pair,
                    "method": method,
                    "metric": metric,
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)) if len(values) > 1 else math.nan,
                    "n": int(values.notna().sum()),
                })
    return pd.DataFrame(rows)


def fmt(mean: float, sd: float, percent: bool = False, digits: int = 4) -> str:
    if percent:
        return f"{100 * mean:.2f}+/-{100 * sd:.2f}%"
    return f"{mean:.{digits}f}+/-{sd:.{digits}f}"


def write_summary(aggregate_frame: pd.DataFrame, completion: pd.DataFrame, output: Path) -> None:
    lookup = aggregate_frame.set_index(["target_pair", "method", "metric"])
    labels = {"egfr_vegfr2": "EGFR-VEGFR2", "parp1_brd4": "PARP1-BRD4"}
    lines = [
        "# V4-B and V5 versus all baselines on two target pairs",
        "",
        "Values are mean +/- SD across seeds 42-51 at an oracle budget of 10,240 per seed.",
        "IGD+ uses one pooled reference front per target pair containing every completed method and seed.",
        "",
    ]
    for pair, label in labels.items():
        lines.extend([
            f"## {label}: main generation performance",
            "",
            "| method | n | validity | uniqueness | novelty | diversity | HV | IGD+ | Pareto size | Dual@6 | Dual@6.5 | Dual@7 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for method in METHODS:
            key = (pair, method, "validity")
            if key not in lookup.index:
                continue
            n = int(lookup.loc[key, "n"])
            def value(metric: str) -> tuple[float, float]:
                row = lookup.loc[(pair, method, metric)]
                return float(row["mean"]), float(row["sd"])
            lines.append(
                f"| {method} | {n} | {fmt(*value('validity'), percent=True)} | "
                f"{fmt(*value('uniqueness'), percent=True)} | {fmt(*value('novelty'), percent=True)} | "
                f"{fmt(*value('diversity'))} | {fmt(*value('hypervolume'))} | {fmt(*value('igd_plus'))} | "
                f"{fmt(*value('pareto_size'), digits=1)} | {fmt(*value('dual_at_6'), percent=True)} | "
                f"{fmt(*value('dual_at_6_5'), percent=True)} | {fmt(*value('dual_at_7'), percent=True)} |"
            )
        lines.extend([
            "", f"## {label}: quality-constrained performance", "",
            "| method | n | quality pass | alert-free | scaffold diversity | QC-HV | QC-Dual@6 | QC-Dual@6.5 | QC-Dual@7 | QC best-min |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for method in METHODS:
            key = (pair, method, "quality_pass")
            if key not in lookup.index:
                continue
            n = int(lookup.loc[key, "n"])
            def value(metric: str) -> tuple[float, float]:
                row = lookup.loc[(pair, method, metric)]
                return float(row["mean"]), float(row["sd"])
            lines.append(
                f"| {method} | {n} | {fmt(*value('quality_pass'), percent=True)} | "
                f"{fmt(*value('alert_free'), percent=True)} | {fmt(*value('scaffold_diversity'), percent=True)} | "
                f"{fmt(*value('qc_hypervolume'))} | {fmt(*value('qc_dual_at_6'), percent=True)} | "
                f"{fmt(*value('qc_dual_at_6_5'), percent=True)} | {fmt(*value('qc_dual_at_7'), percent=True)} | "
                f"{fmt(*value('qc_best_min'), digits=3)} |"
            )
        lines.append("")

    incomplete = completion[completion["completed_seeds"] < completion["planned_seeds"]]
    if len(incomplete):
        lines.extend(["## Incomplete cells", ""])
        for row in incomplete.itertuples(index=False):
            lines.append(
                f"- {row.target_pair} / {row.method}: {row.completed_seeds}/{row.planned_seeds} seeds."
            )
        lines.append("")
    (output / "all_baselines_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training-cache",
        type=Path,
        default=Path("outputs/019f5762-58d7-7670-9168-54fe5fbeb2b3/training_smiles_canonical.txt.gz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/own_method_v5_balancesync_20260806/analysis/all_baselines_two_pairs"),
    )
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    training_path = args.training_cache if args.training_cache.is_absolute() else project / args.training_cache
    output = args.output if args.output.is_absolute() else project / args.output
    output.mkdir(parents=True, exist_ok=True)
    RDLogger.DisableLog("rdApp.error")
    RDLogger.DisableLog("rdApp.warning")
    training = load_training_cache(training_path)

    pair_frames = []
    completion_frames = []
    for pair in ("egfr_vegfr2", "parp1_brd4"):
        frame, reference, completion = collect_pair(project, pair, training)
        pair_frames.append(frame)
        completion_frames.append(completion)
        pd.DataFrame(reference, columns=["target_1_normalized", "target_2_normalized"]).to_csv(
            output / f"{pair}_igd_reference_front.csv", index=False, encoding="utf-8-sig"
        )
    per_seed = pd.concat(pair_frames, ignore_index=True)
    completion = pd.concat(completion_frames, ignore_index=True)
    aggregate_frame = aggregate(per_seed)
    per_seed.to_csv(output / "per_seed_metrics.csv", index=False, encoding="utf-8-sig")
    aggregate_frame.to_csv(output / "aggregate_metrics.csv", index=False, encoding="utf-8-sig")
    completion.to_csv(output / "completion.csv", index=False, encoding="utf-8-sig")
    write_summary(aggregate_frame, completion, output)
    metadata = {
        "budget_per_seed": 10240,
        "planned_seeds": list(SEEDS),
        "unit_of_replication": "formal seed",
        "methods": list(METHODS),
        "training_cache": str(training_path),
        "training_unique_canonical": len(training),
        "igd_reference_scope": "pooled within each target pair across all completed methods and seeds",
        "quality_definition": "QED>=0.60; SA<=4.0; no PAINS/Brenk alert; Lipinski pass",
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote {len(per_seed)} seed rows to {output}", flush=True)


if __name__ == "__main__":
    main()
