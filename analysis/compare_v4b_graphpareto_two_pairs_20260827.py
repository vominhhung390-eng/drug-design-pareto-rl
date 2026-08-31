#!/usr/bin/env python
"""Build the final two-pair paper tables with GraphPareto--NSGA-II.

The EGFR--VEGFR2 panel uses the locked original formal runs.  The PARP1--BRD4
panel uses the 2026-08-27 predictor-aligned rerun.  All metric definitions and
pooled within-pair IGD+ construction are inherited from the established
two-pair analysis implementation.
"""
from __future__ import annotations

from pathlib import Path

import compare_v5_all_baselines_two_pairs as base


base.METHODS = (
    "V4-B",
    "POLYGON",
    "REINVENT4",
    "DrugEx v2",
    "MO-LSO",
    "GraphPareto--NSGA-II",
)


def pair_specs(project: Path, pair: str) -> list[dict[str, object]]:
    if pair == "egfr_vegfr2":
        own_root = project / "results/own_method_v4/common_seeds_42_51_10240"
        baseline_root = project / "results/baselines"
        own_run = lambda seed: own_root / f"v4_b_raw_mean_seed{seed}"
        own_quality = lambda run: run / "evaluation/quality_constrained"
    elif pair == "parp1_brd4":
        experiment = (
            project
            / "results/target_pairs/parp1_brd4_egfr_vegfr2_aligned_20260827"
        )
        own_root = experiment / "own_method"
        baseline_root = experiment / "baselines"
        own_run = lambda seed: own_root / f"formal_10240_seed{seed}"
        own_quality = lambda run: run / "anytime/budget_10240/quality_constrained"
    else:
        raise ValueError(pair)

    specs: list[dict[str, object]] = [
        {
            "method": "V4-B",
            "run": own_run,
            "evaluation": lambda run: run / "anytime/budget_10240",
            "quality": own_quality,
        }
    ]
    for method, folder in (
        ("POLYGON", "polygon_original"),
        ("REINVENT4", "reinvent4"),
        ("DrugEx v2", "drugex_v2"),
        ("MO-LSO", "mo_lso"),
        ("GraphPareto--NSGA-II", "graphpareto_nsga2"),
    ):
        root = baseline_root / folder
        specs.append(
            {
                "method": method,
                "run": lambda seed, root=root: root / f"formal_10240_seed{seed}",
                "evaluation": lambda run: run / "anytime/budget_10240",
                "quality": lambda run: run
                / "anytime/budget_10240/quality_constrained",
            }
        )
    return specs


base.pair_specs = pair_specs
_write_summary = base.write_summary


def write_summary(aggregate_frame, completion, output):
    _write_summary(aggregate_frame, completion, output)
    path = output / "all_baselines_summary.md"
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "# V4-B and V5 versus all baselines on two target pairs",
        "# V4-B versus five baselines on two target pairs",
        1,
    )
    path.write_text(text, encoding="utf-8")


base.write_summary = write_summary


if __name__ == "__main__":
    base.main()
