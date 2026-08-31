#!/usr/bin/env python
"""Aggregate V2.1 elite-archive screening against same-seed references."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from multiobjective_metrics import hypervolume_2d


ROOT = Path(__file__).resolve().parents[1]
SCREEN = ROOT / "results" / "own_method_v21" / "screening_5120"
MATRIX = json.loads(
    (ROOT / "config" / "v21_elite_archive_screening.json").read_text(encoding="utf-8")
)
SEEDS = (42, 43, 44)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def current_hv(path: Path) -> float:
    frame = pd.read_csv(path)
    return float(hypervolume_2d(frame[["egfr", "vegfr2"]].dropna().to_numpy(float)))


def reference(method: str, seed: int) -> dict:
    if method == "v1":
        base = ROOT / "results" / "own_method" / f"formal_10240_seed{seed}" / "anytime" / "budget_5120"
    else:
        base = ROOT / "results" / "baselines" / "polygon_original" / f"formal_10240_seed{seed}" / "anytime" / "budget_5120"
    evaluation = read_json(base / "evaluation_summary.json")
    quality = read_json(base / "quality_constrained" / "quality_constrained_summary.json")
    return {
        "method": method,
        "seed": seed,
        "hv": current_hv(base / "standardized_molecules.csv"),
        "validity": float(evaluation["validity"]),
        "quality_hv": float(quality["quality_constrained_hypervolume"]),
        "quality_pass_rate": float(quality["quality_pass_rate"]),
        "triggered_epochs": 0,
        "archive_rows": 0,
    }


def v21(run: Path) -> dict:
    variant, seed_text = run.name.rsplit("_seed", 1)
    evaluation = read_json(run / "evaluation" / "evaluation_summary.json")
    quality = read_json(run / "evaluation" / "quality_constrained" / "quality_constrained_summary.json")
    metrics = pd.read_csv(run / "metrics.csv")
    generated = pd.read_csv(run / "all_generated_molecules.csv", usecols=["latent_source"])
    triggered = metrics["archive_stagnation_triggered"].astype(str).str.lower().eq("true")
    return {
        "method": variant,
        "seed": int(seed_text),
        "hv": current_hv(run / "evaluation" / "standardized_molecules.csv"),
        "validity": float(evaluation["validity"]),
        "quality_hv": float(quality["quality_constrained_hypervolume"]),
        "quality_pass_rate": float(quality["quality_pass_rate"]),
        "triggered_epochs": int(triggered.sum()),
        "archive_rows": int((generated["latent_source"] == "pareto_archive").sum()),
    }


def main() -> None:
    rows = [v21(run) for run in sorted(SCREEN.glob("v21_*_seed*")) if (run / "summary.csv").exists()]
    rows += [reference(method, seed) for method in ("v1", "polygon") for seed in SEEDS]
    detail = pd.DataFrame(rows).sort_values(["method", "seed"])
    v1 = detail[detail.method == "v1"].set_index("seed")
    polygon = detail[detail.method == "polygon"].set_index("seed")
    gate = MATRIX["selection_gate"]
    summaries = []
    for variant in MATRIX["variants"]:
        group = detail[detail.method == variant].set_index("seed")
        hv_delta = group.hv - v1.hv
        quality_delta = group.quality_hv - v1.quality_hv
        gates = {
            "gate_hv_gain": hv_delta.mean() >= float(gate["paired_mean_hv_gain_vs_v1"]),
            "gate_validity": group.validity.mean() >= float(gate["validity_minimum"]),
            "gate_wins": int((hv_delta > 0).sum()) >= int(gate["wins_vs_v1_required"]),
            "gate_quality": quality_delta.mean() >= 0.0,
        }
        summaries.append({
            "variant": variant,
            "hv_mean": group.hv.mean(),
            "hv_sd": group.hv.std(ddof=1),
            "hv_gain_vs_v1": hv_delta.mean(),
            "validity_mean": group.validity.mean(),
            "quality_hv_mean": group.quality_hv.mean(),
            "quality_hv_gain_vs_v1": quality_delta.mean(),
            "wins_vs_v1": int((hv_delta > 0).sum()),
            "wins_vs_polygon": int((group.hv > polygon.hv).sum()),
            "triggered_epochs_mean": group.triggered_epochs.mean(),
            "archive_rows_mean": group.archive_rows.mean(),
            **gates,
            "all_gates": all(gates.values()),
        })
    aggregate = pd.DataFrame(summaries).sort_values("hv_mean", ascending=False)
    winner = aggregate.iloc[0]
    detail.to_csv(SCREEN / "screening_detail.csv", index=False, encoding="utf-8-sig")
    aggregate.to_csv(SCREEN / "screening_aggregate.csv", index=False, encoding="utf-8-sig")
    lines = [
        "# V2.1 elite-archive screening (5,120 oracle calls)",
        "",
        "Development seeds 42-44; all variants retain the V1 raw vector reward, dual critics, and shared preference.",
        "",
        "| variant | HV mean +/- SD | gain vs V1 | validity | quality HV gain | wins vs V1 | wins vs POLYGON | trigger epochs | all gates |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in aggregate.iterrows():
        lines.append(
            f"| {row.variant} | {row.hv_mean:.4f} +/- {row.hv_sd:.4f} | {row.hv_gain_vs_v1:+.4f} | "
            f"{row.validity_mean:.4f} | {row.quality_hv_gain_vs_v1:+.4f} | {int(row.wins_vs_v1)}/3 | "
            f"{int(row.wins_vs_polygon)}/3 | {row.triggered_epochs_mean:.1f} | {'PASS' if row.all_gates else 'FAIL'} |"
        )
    lines += [
        "",
        "## Decision",
        "",
        f"Selected development winner: **{winner.variant}** with mean HV {winner.hv_mean:.4f}.",
        "A new-seed paired confirmation is required before replacing V1.",
    ]
    (SCREEN / "screening_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
