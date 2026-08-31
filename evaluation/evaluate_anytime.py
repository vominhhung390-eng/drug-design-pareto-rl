"""Evaluate preregistered prefixes of an exact-budget baseline result CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from evaluate_experiment import evaluate, parse_args as _unused  # noqa: F401


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--checkpoints", nargs="+", type=int, default=[1024, 2048, 5120, 10240])
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    for checkpoint in args.checkpoints:
        if len(frame) < checkpoint:
            raise RuntimeError(f"Requested checkpoint {checkpoint}, but only {len(frame)} rows exist")
        checkpoint_dir = args.output / f"budget_{checkpoint}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        prefix_file = checkpoint_dir / "generated_prefix.csv"
        frame.iloc[:checkpoint].to_csv(prefix_file, index=False)
        namespace = argparse.Namespace(
            input=prefix_file,
            output=checkpoint_dir,
            smiles_column="smiles",
            egfr_column="egfr",
            vegfr2_column="vegfr2",
            reference_front=None,
            hv_reference=(0.0, 0.0),
        )
        evaluate(namespace)


if __name__ == "__main__":
    main()
