"""Encode the shared SMILES dataset with DrugEx v2's native tokenizer."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np


TOKEN_PATTERN = re.compile(r"(\[[^\[\]]{1,6}\])")


def tokenize(smiles: str) -> list[str]:
    smiles = smiles.replace("Cl", "L").replace("Br", "R")
    tokens: list[str] = []
    for word in TOKEN_PATTERN.split(smiles):
        if not word:
            continue
        tokens.extend([word] if word.startswith("[") else list(word))
    return tokens


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, default=Path("../../data/train_smiles_only.txt")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../../results/baselines/drugex_v2/data"),
    )
    parser.add_argument("--max-len", type=int, default=100)
    args = parser.parse_args()
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    expected_hash = "4301e7f6118839465012eb93510328681ef4b7b24642e8748c4ad40971f4a304"
    actual_hash = file_sha256(input_path)
    if actual_hash != expected_hash:
        raise RuntimeError(f"Unexpected training dataset hash: {actual_hash}")

    vocabulary: set[str] = set()
    line_count = 0
    max_tokens = 0
    rejected = 0
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            smiles = line.strip()
            if not smiles:
                continue
            tokens = tokenize(smiles)
            line_count += 1
            max_tokens = max(max_tokens, len(tokens))
            if len(tokens) > args.max_len:
                rejected += 1
                continue
            vocabulary.update(tokens)

    words = sorted(vocabulary)
    token_to_index = {token: index + 2 for index, token in enumerate(words)}
    vocab_path = output_dir / "common_voc.txt"
    vocab_path.write_text("\n".join(words) + "\n", encoding="utf-8")

    accepted = line_count - rejected
    token_path = output_dir / "common_tokens.npy"
    array = np.lib.format.open_memmap(
        token_path, mode="w+", dtype=np.uint8, shape=(accepted, args.max_len)
    )
    output_index = 0
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            smiles = line.strip()
            if not smiles:
                continue
            tokens = tokenize(smiles)
            if len(tokens) > args.max_len:
                continue
            array[output_index, : len(tokens)] = [token_to_index[item] for item in tokens]
            output_index += 1
    array.flush()

    metadata = {
        "source": str(input_path),
        "source_sha256": actual_hash,
        "line_count": line_count,
        "accepted_count": accepted,
        "rejected_over_max_len": rejected,
        "max_observed_tokens": max_tokens,
        "max_model_tokens": args.max_len,
        "vocabulary_size_without_controls": len(words),
        "vocabulary_size_with_controls": len(words) + 2,
        "tokens_file": str(token_path),
        "vocabulary_file": str(vocab_path),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
