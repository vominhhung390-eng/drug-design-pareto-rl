#!/usr/bin/env python
"""Summarize the paired actor-update by online-VAE-adaptation 2x2 factorial.

Three cells come from the prospectively registered V4-B ablation and the
joint-off cell comes from the P0 completion run.  The interaction is reported
as exploratory because the missing cell was specified after the original
single-factor results were available.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


SEEDS = tuple(range(92, 102))
CELLS = {
    "y11_full": ("a0_full_v4b", 1, 1),
    "y10_actor_only": ("a2_no_online_vae", 1, 0),
    "y01_vae_only": ("a4_frozen_actor", 0, 1),
    "y00_neither": ("a9_frozen_actor_no_online_vae", 0, 0),
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def t_ci(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return math.nan, math.nan
    sem = stats.sem(values)
    return tuple(
        map(float, stats.t.interval(0.95, len(values) - 1, loc=np.mean(values), scale=sem))
    )


def load_cell(root: Path, variant: str, seed: int) -> dict[str, float]:
    run = root / f"{variant}_seed{seed}"
    summary_path = run / "summary.csv"
    evaluation_path = run / "evaluation" / "evaluation_summary.json"
    quality_path = run / "evaluation" / "quality_constrained" / "quality_constrained_summary.json"
    missing = [str(path) for path in (summary_path, evaluation_path, quality_path) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing factorial outputs: " + "; ".join(missing))
    summary = pd.read_csv(summary_path).iloc[0]
    evaluation = read_json(evaluation_path)
    quality = read_json(quality_path)
    if int(float(summary["oracle_budget"])) != 10240:
        raise ValueError(f"{run}: oracle budget is not 10240")
    return {
        "hv": float(summary["hv_final"]),
        "qc_hv": float(quality["quality_constrained_hypervolume"]),
        "validity": float(evaluation["validity"]),
        "runtime_sec": float(summary["runtime_sec"]),
    }


def effect_record(name: str, metric: str, values: np.ndarray) -> dict[str, float | str | int]:
    values = np.asarray(values, dtype=float)
    low, high = t_ci(values)
    sd = float(values.std(ddof=1))
    return {
        "effect": name,
        "metric": metric,
        "n": len(values),
        "mean": float(values.mean()),
        "sd": sd,
        "ci95_low": low,
        "ci95_high": high,
        "paired_t_p": float(stats.ttest_1samp(values, 0.0).pvalue),
        "effect_dz": float(values.mean() / sd) if sd > 0 else math.nan,
        "positive_seed_count": int(np.sum(values > 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--completion-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    original_root = args.original_root.resolve()
    completion_root = args.completion_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for seed in SEEDS:
        for cell, (variant, actor_update, online_vae) in CELLS.items():
            root = completion_root if cell == "y00_neither" else original_root
            values = load_cell(root, variant, seed)
            rows.append(
                {
                    "seed": seed,
                    "cell": cell,
                    "variant": variant,
                    "actor_update": actor_update,
                    "online_vae": online_vae,
                    **values,
                }
            )
    data = pd.DataFrame(rows)
    data.to_csv(output / "actor_vae_factorial_all_runs.csv", index=False, encoding="utf-8-sig")

    effects: list[dict[str, object]] = []
    for metric in ("hv", "qc_hv", "validity"):
        pivot = data.pivot(index="seed", columns="cell", values=metric).loc[list(SEEDS)]
        interaction = (
            pivot["y11_full"]
            - pivot["y10_actor_only"]
            - pivot["y01_vae_only"]
            + pivot["y00_neither"]
        )
        actor_main = 0.5 * (
            (pivot["y11_full"] - pivot["y01_vae_only"])
            + (pivot["y10_actor_only"] - pivot["y00_neither"])
        )
        vae_main = 0.5 * (
            (pivot["y11_full"] - pivot["y10_actor_only"])
            + (pivot["y01_vae_only"] - pivot["y00_neither"])
        )
        effects.extend(
            [
                effect_record("actor_x_online_vae", metric, interaction.to_numpy()),
                effect_record("actor_update_main", metric, actor_main.to_numpy()),
                effect_record("online_vae_main", metric, vae_main.to_numpy()),
            ]
        )
    effects_frame = pd.DataFrame(effects)
    effects_frame.to_csv(output / "actor_vae_factorial_effects.csv", index=False, encoding="utf-8-sig")

    aggregate = (
        data.groupby(["cell", "variant", "actor_update", "online_vae"], sort=False)
        .agg(
            n=("seed", "count"),
            hv_mean=("hv", "mean"),
            hv_sd=("hv", "std"),
            qc_hv_mean=("qc_hv", "mean"),
            qc_hv_sd=("qc_hv", "std"),
            validity_mean=("validity", "mean"),
            validity_sd=("validity", "std"),
            runtime_sec_mean=("runtime_sec", "mean"),
        )
        .reset_index()
    )
    aggregate.to_csv(output / "actor_vae_factorial_aggregate.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# Exploratory actor-update by online-VAE-adaptation factorial",
        "",
        "The three previously completed cells use the prospectively registered seeds 92--101. "
        "The joint-off cell was specified after the single-factor results were available; "
        "therefore all interaction tests are exploratory.",
        "",
        "| Cell | Actor update | Online VAE | HV | QC-HV | Validity |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate.itertuples(index=False):
        lines.append(
            f"| {row.cell} | {row.actor_update} | {row.online_vae} | "
            f"{row.hv_mean:.4f} +/- {row.hv_sd:.4f} | "
            f"{row.qc_hv_mean:.4f} +/- {row.qc_hv_sd:.4f} | "
            f"{row.validity_mean:.4f} +/- {row.validity_sd:.4f} |"
        )
    lines.extend(
        [
            "",
            "| Effect | Metric | Mean | 95% CI | Paired p | dz | Positive seeds |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in effects_frame.itertuples(index=False):
        lines.append(
            f"| {row.effect} | {row.metric} | {row.mean:+.4f} | "
            f"[{row.ci95_low:+.4f}, {row.ci95_high:+.4f}] | {row.paired_t_p:.4g} | "
            f"{row.effect_dz:+.3f} | {row.positive_seed_count}/{row.n} |"
        )
    (output / "actor_vae_factorial_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8-sig")

    manifest = {
        "status": "complete",
        "analysis_class": "exploratory_post_hoc_factorial_completion",
        "seeds": list(SEEDS),
        "oracle_budget_per_seed": 10240,
        "original_root": str(original_root),
        "completion_root": str(completion_root),
        "cell_mapping": CELLS,
    }
    (output / "actor_vae_factorial_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
