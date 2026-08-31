#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from rdkit import Chem, RDLogger

from polygon.utils.utils import load_model
from polygon.vae.vae_model import VAE


RDLogger.DisableLog("rdApp.*")


def canonical(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    return None if mol is None else Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def load_training_set(data_path: Path, cache_path: Path) -> set[str]:
    if cache_path.exists():
        return {line.strip() for line in cache_path.open(encoding="utf-8") if line.strip()}
    values: set[str] = set()
    with data_path.open(encoding="utf-8") as handle:
        for line in handle:
            value = canonical(line.strip())
            if value is not None:
                values.add(value)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("\n".join(sorted(values)) + "\n", encoding="utf-8")
    return values


def evaluate(model, training: set[str], args, temperature: float) -> dict[str, object]:
    torch.manual_seed(args.seed)
    if str(args.device).startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)
    samples: list[str] = []
    with torch.no_grad():
        while len(samples) < args.samples:
            count = min(args.batch_size, args.samples - len(samples))
            samples.extend(model.sample(count, max_len=args.max_len, temp=temperature, multinomial=True))
    valid = [value for raw in samples if (value := canonical(raw)) is not None]
    unique = set(valid)
    novel = unique - training
    return {
        "temperature": temperature,
        "samples": len(samples),
        "valid": len(valid),
        "invalid": len(samples) - len(valid),
        "validity": len(valid) / len(samples) if samples else 0.0,
        "unique_valid": len(unique),
        "uniqueness_valid": len(unique) / len(valid) if valid else 0.0,
        "unique_ratio_all": len(unique) / len(samples) if samples else 0.0,
        "novel_unique": len(novel),
        "novelty_unique": len(novel) / len(unique) if unique else 0.0,
        "memorized_unique": len(unique & training),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-len", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperatures", default="1.00,1.05,1.10,1.15,1.18,1.20,1.22,1.25,1.30,1.35")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    training = load_training_set(args.train_data, args.train_cache)
    model = load_model(VAE, str(args.model), args.device)
    model.eval()
    rows = [evaluate(model, training, args, float(value)) for value in args.temperatures.split(",")]
    with (args.output / "vae_quality_by_temperature.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "vae_quality_by_temperature.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
