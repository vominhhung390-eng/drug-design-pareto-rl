#!/usr/bin/env python
"""Run the locked final and anytime evaluators for one V4 ablation run."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def call(*arguments: object) -> None:
    subprocess.run([sys.executable, *map(str, arguments)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    root = Path(__file__).resolve().parent
    source = run_dir / "all_generated_molecules.csv"
    if not source.exists():
        raise FileNotFoundError(source)

    final_dir = run_dir / "evaluation"
    call(root / "evaluate_experiment.py", source, final_dir)
    call(
        root / "quality_constrained_metrics.py",
        final_dir / "standardized_molecules.csv",
        final_dir / "quality_constrained",
    )
    call(
        root / "evaluate_anytime.py",
        source,
        run_dir / "anytime",
        "--checkpoints",
        1024,
        2048,
        5120,
        10240,
    )


if __name__ == "__main__":
    main()
