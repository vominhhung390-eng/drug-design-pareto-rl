#!/usr/bin/env python
"""Build an empirical reference front and compare completed experiment runs."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from multiobjective_metrics import coverage, hypervolume_2d, igd_plus, pareto_front, spacing, spread


def load_front(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"egfr", "vegfr2"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{path} does not contain {sorted(required)}")
    frame = frame.dropna(subset=["egfr", "vegfr2"]).copy()
    frame["source"] = path.parent.name
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--pattern", default="**/pareto_front.csv")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--hv-reference", nargs=2, type=float, default=(0.0, 0.0))
    args = parser.parse_args()
    output = args.output or args.root / "comparison"
    output.mkdir(parents=True, exist_ok=True)

    paths = [p for p in args.root.glob(args.pattern) if output not in p.parents]
    if not paths:
        raise FileNotFoundError(f"No fronts matching {args.pattern} under {args.root}")
    fronts = {path.parent.name: load_front(path) for path in paths}
    union = pd.concat(fronts.values(), ignore_index=True)
    union = union.drop_duplicates(["egfr", "vegfr2"])
    union_points = union[["egfr", "vegfr2"]].to_numpy(float)
    reference_points = pareto_front(union_points)
    reference_keys = {tuple(point) for point in reference_points.tolist()}
    reference = union[
        [tuple(point) in reference_keys for point in union[["egfr", "vegfr2"]].to_numpy(float).tolist()]
    ].drop_duplicates(["egfr", "vegfr2"])
    reference.to_csv(output / "empirical_reference_front.csv", index=False, encoding="utf-8-sig")

    metric_rows = []
    for name, frame in fronts.items():
        points = pareto_front(frame[["egfr", "vegfr2"]].to_numpy(float))
        metric_rows.append(
            {
                "run": name,
                "pareto_size": len(points),
                "hypervolume": hypervolume_2d(points, args.hv_reference),
                "igd_plus": igd_plus(points, reference_points),
                "spacing": spacing(points),
                "spread": spread(points),
            }
        )
    pd.DataFrame(metric_rows).sort_values("hypervolume", ascending=False).to_csv(
        output / "front_metrics.csv", index=False, encoding="utf-8-sig"
    )

    names = sorted(fronts)
    matrix = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        points_a = pareto_front(fronts[a][["egfr", "vegfr2"]].to_numpy(float))
        for b in names:
            points_b = pareto_front(fronts[b][["egfr", "vegfr2"]].to_numpy(float))
            matrix.loc[a, b] = coverage(points_a, points_b)
    matrix.index.name = "A_dominates_B"
    matrix.to_csv(output / "coverage_matrix.csv", encoding="utf-8-sig")


if __name__ == "__main__":
    main()
