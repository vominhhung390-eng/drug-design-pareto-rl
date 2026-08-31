#!/usr/bin/env python
"""Aggregate V3 actor-plus-generator self-training development screen."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCREEN = ROOT / "results" / "own_method_v3" / "generator_screening_5120"
CONFIG = json.loads(
    (ROOT / "config" / "v3_generator_self_training_screening_5120.json")
    .read_text(encoding="utf-8")
)
SEEDS = CONFIG["seeds"]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_v3(run: Path) -> dict:
    method, seed_text = run.name.rsplit("_seed", 1)
    evaluation = read_json(run / "evaluation" / "evaluation_summary.json")
    quality = read_json(run / "evaluation" / "quality_constrained" / "quality_constrained_summary.json")
    summary = pd.read_csv(run / "summary.csv").iloc[0]
    return {
        "method": method,
        "seed": int(seed_text),
        "hv": float(evaluation["hypervolume"]),
        "validity": float(evaluation["validity"]),
        "quality_hv": float(quality["quality_constrained_hypervolume"]),
        "quality_pass_rate": float(quality["quality_pass_rate"]),
        "best_min_activity": float(evaluation["best_min_activity"]),
        "finetune_count": int(summary["generator_finetune_count"]),
    }


def load_polygon(seed: int) -> dict:
    base = (
        ROOT / "results" / "baselines" / "polygon_original"
        / f"formal_10240_seed{seed}" / "anytime" / "budget_5120"
    )
    evaluation = read_json(base / "evaluation_summary.json")
    quality = read_json(base / "quality_constrained" / "quality_constrained_summary.json")
    # Stored POLYGON summaries predate normalized HV; the quality evaluator
    # already recomputed the current normalized raw HV from standardized rows.
    return {
        "method": "polygon",
        "seed": seed,
        "hv": float(quality["raw_hypervolume"]),
        "validity": float(evaluation["validity"]),
        "quality_hv": float(quality["quality_constrained_hypervolume"]),
        "quality_pass_rate": float(quality["quality_pass_rate"]),
        "best_min_activity": float(evaluation["best_min_activity"]),
        "finetune_count": 5,
    }


def main() -> None:
    rows = [load_v3(run) for run in sorted(SCREEN.glob("v3_*_seed*")) if (run / "summary.csv").exists()]
    rows += [load_polygon(seed) for seed in SEEDS]
    detail = pd.DataFrame(rows).sort_values(["method", "seed"])
    polygon = detail[detail.method == "polygon"].set_index("seed")
    gate = CONFIG["selection_gate"]
    aggregates = []
    for variant in CONFIG["variants"]:
        group = detail[detail.method == variant].set_index("seed")
        delta = group.hv - polygon.hv
        quality_delta = group.quality_hv - polygon.quality_hv
        gates = {
            "gate_hv": group.hv.mean() >= float(gate["mean_hv_target"]),
            "gate_wins": int((delta > 0).sum()) >= int(gate["wins_vs_polygon_required"]),
            "gate_validity": group.validity.mean() >= float(gate["validity_minimum"]),
            "gate_quality": quality_delta.mean() >= 0.0,
        }
        aggregates.append({
            "variant": variant,
            "hv_mean": group.hv.mean(),
            "hv_sd": group.hv.std(ddof=1),
            "hv_gain_vs_polygon": delta.mean(),
            "validity_mean": group.validity.mean(),
            "quality_hv_mean": group.quality_hv.mean(),
            "quality_hv_gain_vs_polygon": quality_delta.mean(),
            "quality_pass_rate_mean": group.quality_pass_rate.mean(),
            "best_min_activity_mean": group.best_min_activity.mean(),
            "wins_vs_polygon": int((delta > 0).sum()),
            **gates,
            "all_gates": all(gates.values()),
        })
    aggregate = pd.DataFrame(aggregates).sort_values("hv_mean", ascending=False)
    winner = aggregate.iloc[0]
    detail.to_csv(SCREEN / "screening_detail.csv", index=False, encoding="utf-8-sig")
    aggregate.to_csv(SCREEN / "screening_aggregate.csv", index=False, encoding="utf-8-sig")
    lines = [
        "# V3 actor plus real-SMILES generator self-training screen",
        "",
        "Budget 5,120; development seeds 42-44; five generator fine-tunes at 1,024-call intervals.",
        "",
        "| variant | HV mean +/- SD | gain vs POLYGON | validity | quality HV gain | wins vs POLYGON | all gates |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in aggregate.iterrows():
        lines.append(
            f"| {row.variant} | {row.hv_mean:.4f} +/- {row.hv_sd:.4f} | {row.hv_gain_vs_polygon:+.4f} | "
            f"{row.validity_mean:.4f} | {row.quality_hv_gain_vs_polygon:+.4f} | "
            f"{int(row.wins_vs_polygon)}/3 | {'PASS' if row.all_gates else 'FAIL'} |"
        )
    lines += [
        "",
        "## Decision",
        "",
        f"Development winner: **{winner.variant}**, mean HV {winner.hv_mean:.4f}.",
        "Only a new-seed paired 10,240-call confirmation can support a claim over POLYGON.",
    ]
    (SCREEN / "screening_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
