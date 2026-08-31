#!/usr/bin/env python
from __future__ import annotations

import argparse
import multiprocessing as mp
import random
from pathlib import Path

from rdkit import Chem, RDLogger


RDLogger.DisableLog("rdApp.*")


def randomized_smiles(smiles: str, n_random: int, rng: random.Random) -> list[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    values = {Chem.MolToSmiles(mol, canonical=True)}
    for _ in range(n_random):
        values.add(Chem.MolToSmiles(mol, canonical=False, doRandom=True))
    output = list(values)
    rng.shuffle(output)
    return output


def process_chunk(args: tuple[list[str], int, int, int]) -> tuple[int, int, list[str]]:
    chunk, n_random, seed, chunk_id = args
    rng = random.Random(seed + chunk_id)
    lines: list[str] = []
    seen = 0
    for smiles in chunk:
        smiles = smiles.strip()
        if not smiles:
            continue
        seen += 1
        lines.extend(randomized_smiles(smiles, n_random, rng))
    return chunk_id, seen, lines


def chunked(iterable, size: int):
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def main() -> None:
    parser = argparse.ArgumentParser(description="Create canonical plus randomized-SMILES training data.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-random", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--chunk-size", type=int, default=8000)
    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8") as handle:
        jobs = [
            (chunk, args.n_random, args.seed, index)
            for index, chunk in enumerate(chunked(handle, args.chunk_size))
        ]
    written = 0
    seen = 0
    with output_path.open("w", encoding="utf-8") as output:
        with mp.Pool(processes=args.workers) as pool:
            for _, count, lines in pool.imap_unordered(process_chunk, jobs):
                seen += count
                output.writelines(value + "\n" for value in lines)
                written += len(lines)
                print(f"processed={seen} written={written}", flush=True)
    print(f"input_molecules={seen} written_smiles={written} output={output_path}")


if __name__ == "__main__":
    main()
