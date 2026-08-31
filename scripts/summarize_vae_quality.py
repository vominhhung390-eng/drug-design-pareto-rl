#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=40)
    args = parser.parse_args()
    rows: list[dict[str, object]] = []
    for path in args.quality_root.glob("*/vae_quality_by_temperature.csv"):
        with path.open(encoding="utf-8", newline="") as handle:
            for raw in csv.DictReader(handle):
                row: dict[str, object] = dict(raw)
                row["model"] = path.parent.name
                for key in ("temperature", "validity", "uniqueness_valid", "unique_ratio_all", "novelty_unique"):
                    row[key] = float(row[key])
                row["balanced_score"] = min(float(row["validity"]), float(row["novelty_unique"]))
                row["sum_score"] = float(row["validity"]) + float(row["novelty_unique"])
                rows.append(row)
    if not rows:
        raise SystemExit(f"No VAE quality tables found below {args.quality_root}")
    rows.sort(
        key=lambda row: (
            float(row["validity"]) >= 0.90,
            float(row["novelty_unique"]) >= 0.90,
            float(row["balanced_score"]),
            float(row["sum_score"]),
            float(row["uniqueness_valid"]),
        ),
        reverse=True,
    )
    fields = [
        "rank", "model", "temperature", "validity", "novelty_unique", "uniqueness_valid",
        "unique_ratio_all", "balanced_score", "sum_score", "samples", "valid", "unique_valid",
        "novel_unique", "memorized_unique",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(rows[: args.top_k], 1):
            writer.writerow({"rank": rank, **{key: row[key] for key in fields if key != "rank"}})
    print(f"selected={rows[0]['model']} temperature={rows[0]['temperature']} output={args.output}")


if __name__ == "__main__":
    main()
