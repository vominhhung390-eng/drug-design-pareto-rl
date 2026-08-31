#!/usr/bin/env python
"""Aggregate the held-constant 5,120-budget V2 screening matrix."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from multiobjective_metrics import hypervolume_2d


ROOT = Path(__file__).resolve().parents[1]
SCREEN_ROOT = ROOT / "results" / "own_method_v2" / "screening_5120"
MATRIX_PATH = ROOT / "config" / "v2_late_pareto_screening.json"
SEEDS = (42, 43, 44)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def current_hv(standardized_csv: Path) -> float:
    frame = pd.read_csv(standardized_csv)
    points = frame[["egfr", "vegfr2"]].dropna().to_numpy(float)
    return float(hypervolume_2d(points))


def reference_row(method: str, seed: int) -> dict:
    if method == "v1":
        base = (
            ROOT / "results" / "own_method" / f"formal_10240_seed{seed}"
            / "anytime" / "budget_5120"
        )
    elif method == "polygon":
        base = (
            ROOT / "results" / "baselines" / "polygon_original"
            / f"formal_10240_seed{seed}" / "anytime" / "budget_5120"
        )
    else:
        raise ValueError(method)
    standardized = base / "standardized_molecules.csv"
    quality = read_json(base / "quality_constrained" / "quality_constrained_summary.json")
    evaluation = read_json(base / "evaluation_summary.json")
    return {
        "method": method,
        "seed": seed,
        "hv": current_hv(standardized),
        "validity": float(evaluation["validity"]),
        "quality_hv": float(quality["quality_constrained_hypervolume"]),
        "quality_pass_rate": float(quality["quality_pass_rate"]),
    }


def v2_row(run_dir: Path) -> dict:
    seed = int(run_dir.name.rsplit("seed", 1)[1])
    variant = run_dir.name.rsplit("_seed", 1)[0]
    evaluation_dir = run_dir / "evaluation"
    evaluation = read_json(evaluation_dir / "evaluation_summary.json")
    quality = read_json(
        evaluation_dir / "quality_constrained" / "quality_constrained_summary.json"
    )
    summary = pd.read_csv(run_dir / "summary.csv").iloc[0]
    return {
        "method": variant,
        "seed": seed,
        "hv": current_hv(evaluation_dir / "standardized_molecules.csv"),
        "validity": float(evaluation["validity"]),
        "unique_valid": int(evaluation["unique_valid"]),
        "quality_hv": float(quality["quality_constrained_hypervolume"]),
        "quality_pass_rate": float(quality["quality_pass_rate"]),
        "best_min_activity": float(evaluation["best_min_activity"]),
        "runtime_sec": float(summary["runtime_sec"]),
    }


def mean_sd(values: pd.Series) -> str:
    return f"{values.mean():.4f} +/- {values.std(ddof=1):.4f}"


def main() -> None:
    matrix = read_json(MATRIX_PATH)
    rows = []
    for run_dir in sorted(SCREEN_ROOT.glob("v2_*_seed*")):
        if run_dir.is_dir() and (run_dir / "summary.csv").exists():
            rows.append(v2_row(run_dir))
    rows.extend(reference_row(method, seed) for method in ("v1", "polygon") for seed in SEEDS)
    detail = pd.DataFrame(rows).sort_values(["method", "seed"]).reset_index(drop=True)

    v1 = detail[detail["method"] == "v1"].set_index("seed")
    polygon = detail[detail["method"] == "polygon"].set_index("seed")
    summaries = []
    for variant in matrix["variants"]:
        group = detail[detail["method"] == variant].set_index("seed")
        polygon_wins = int((group.loc[list(SEEDS), "hv"] > polygon.loc[list(SEEDS), "hv"]).sum())
        v1_wins = int((group.loc[list(SEEDS), "hv"] > v1.loc[list(SEEDS), "hv"]).sum())
        mean_hv = float(group["hv"].mean())
        mean_validity = float(group["validity"].mean())
        mean_quality_hv = float(group["quality_hv"].mean())
        gate_hv = mean_hv >= float(matrix["selection_gate"]["mean_hv_target"])
        gate_validity = mean_validity >= float(matrix["selection_gate"]["validity_minimum"])
        gate_quality = mean_quality_hv >= float(v1["quality_hv"].mean())
        gate_polygon = polygon_wins >= int(matrix["selection_gate"]["same_seed_polygon_wins_required"])
        summaries.append({
            "variant": variant,
            "hv_mean": mean_hv,
            "hv_sd": float(group["hv"].std(ddof=1)),
            "validity_mean": mean_validity,
            "quality_hv_mean": mean_quality_hv,
            "quality_hv_delta_vs_v1": mean_quality_hv - float(v1["quality_hv"].mean()),
            "quality_pass_rate_mean": float(group["quality_pass_rate"].mean()),
            "best_min_activity_mean": float(group["best_min_activity"].mean()),
            "wins_vs_v1": v1_wins,
            "wins_vs_polygon": polygon_wins,
            "gate_hv": gate_hv,
            "gate_validity": gate_validity,
            "gate_quality": gate_quality,
            "gate_polygon": gate_polygon,
            "all_gates": gate_hv and gate_validity and gate_quality and gate_polygon,
        })
    aggregate = pd.DataFrame(summaries).sort_values("hv_mean", ascending=False)
    winner = aggregate.iloc[0]

    detail.to_csv(SCREEN_ROOT / "screening_detail.csv", index=False, encoding="utf-8-sig")
    aggregate.to_csv(SCREEN_ROOT / "screening_aggregate.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# Own-method V2 late-Pareto screening (5,120 oracle calls)",
        "",
        "Post-hoc development screen; seeds 42-44 are shared with the prior formal comparison and are not held-out confirmation seeds.",
        "",
        "## Aggregate results",
        "",
        "| variant | HV mean +/- SD | validity | quality HV | delta quality HV vs V1 | wins vs V1 | wins vs POLYGON | all gates |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in aggregate.iterrows():
        lines.append(
            f"| {row['variant']} | {row['hv_mean']:.4f} +/- {row['hv_sd']:.4f} | "
            f"{row['validity_mean']:.4f} | {row['quality_hv_mean']:.4f} | "
            f"{row['quality_hv_delta_vs_v1']:+.4f} | {int(row['wins_vs_v1'])}/3 | "
            f"{int(row['wins_vs_polygon'])}/3 | {'PASS' if row['all_gates'] else 'FAIL'} |"
        )
    lines.extend([
        "",
        "## Same-seed references",
        "",
        f"- V1 HV: {mean_sd(v1['hv'])}; quality HV: {mean_sd(v1['quality_hv'])}.",
        f"- POLYGON HV: {mean_sd(polygon['hv'])}; quality HV: {mean_sd(polygon['quality_hv'])}.",
        "",
        "## Selection",
        "",
        f"Highest mean HV: **{winner['variant']}** ({winner['hv_mean']:.4f}).",
        "A 10,240-budget confirmation must use new seeds before this can replace V1 in a paper claim.",
    ])
    (SCREEN_ROOT / "screening_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
