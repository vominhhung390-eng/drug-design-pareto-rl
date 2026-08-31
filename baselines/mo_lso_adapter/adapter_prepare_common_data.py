"""Parallel, sharded JT-VAE preprocessing for the shared SMILES dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def batches(path: Path, shard_size: int) -> Iterable[tuple[int, list[str]]]:
    start = 0
    batch: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            value = line.strip()
            if not value:
                continue
            batch.append(value)
            if len(batch) == shard_size:
                yield start, batch
                start += len(batch)
                batch = []
    if batch:
        yield start, batch


def process_shard(payload: tuple[int, list[str], str]) -> dict:
    start, smiles_batch, output_dir_text = payload
    from rdkit import RDLogger
    from weighted_retraining.chem.jtnn.mol_tree import MolTree

    RDLogger.DisableLog("rdApp.*")

    def tensorize(smiles: str):
        mol_tree = MolTree(smiles)
        mol_tree.recover()
        mol_tree.assemble()
        for node in mol_tree.nodes:
            if node.label not in node.cands:
                node.cands.append(node.label)
        del mol_tree.mol
        for node in mol_tree.nodes:
            del node.mol
        return mol_tree

    output_dir = Path(output_dir_text)
    end = start + len(smiles_batch)
    output_file = output_dir / f"tensors_{start:010d}-{end:010d}.pkl"
    if output_file.is_file():
        with output_file.open("rb") as handle:
            existing_trees = pickle.load(handle)
        existing_vocabulary = {
            node.smiles for tree in existing_trees for node in tree.nodes
        }
        rejected_file = output_dir / f"rejected_{start:010d}-{end:010d}.json"
        existing_rejected = (
            len(json.loads(rejected_file.read_text(encoding="utf-8")))
            if rejected_file.is_file()
            else 0
        )
        return {
            "start": start,
            "end": end,
            "accepted": len(existing_trees),
            "rejected": existing_rejected,
            "vocabulary": sorted(existing_vocabulary),
            "output": str(output_file),
            "reused": True,
        }

    trees = []
    vocabulary: set[str] = set()
    rejected = []
    for offset, smiles in enumerate(smiles_batch):
        try:
            tree = tensorize(smiles)
            trees.append(tree)
            vocabulary.update(node.smiles for node in tree.nodes)
        except Exception as exc:
            rejected.append({"index": start + offset, "smiles": smiles, "error": repr(exc)})
    with output_file.open("wb") as handle:
        pickle.dump(trees, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if rejected:
        with (output_dir / f"rejected_{start:010d}-{end:010d}.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(rejected, handle)
    return {
        "start": start,
        "end": end,
        "accepted": len(trees),
        "rejected": len(rejected),
        "vocabulary": sorted(vocabulary),
        "output": str(output_file),
        "reused": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, default=Path("../../data/train_smiles_only.txt")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../../results/baselines/mo_lso/data/tensors_train"),
    )
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--shard-size", type=int, default=5000)
    parser.add_argument("--max-shards", type=int, default=None)
    args = parser.parse_args()

    source = args.input.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_hash = "4301e7f6118839465012eb93510328681ef4b7b24642e8748c4ad40971f4a304"
    actual_hash = file_sha256(source)
    if actual_hash != expected_hash:
        raise RuntimeError(f"Unexpected training dataset hash: {actual_hash}")

    started = time.time()
    vocabulary: set[str] = set()
    accepted = 0
    rejected = 0
    submitted = 0
    futures = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for start, batch in batches(source, args.shard_size):
            if args.max_shards is not None and submitted >= args.max_shards:
                break
            futures.append(executor.submit(process_shard, (start, batch, str(output_dir))))
            submitted += 1
        for completed, future in enumerate(as_completed(futures), 1):
            result = future.result()
            vocabulary.update(result["vocabulary"])
            accepted += result["accepted"]
            rejected += result["rejected"]
            print(
                json.dumps(
                    {
                        "completed_shards": completed,
                        "total_shards": submitted,
                        "accepted": accepted,
                        "rejected": rejected,
                        "last_output": result["output"],
                        "seconds": time.time() - started,
                    }
                ),
                flush=True,
            )

    vocab_file = output_dir.parent / "vocab.txt"
    vocab_file.write_text("\n".join(sorted(vocabulary)) + "\n", encoding="utf-8")
    metadata = {
        "source": str(source),
        "source_sha256": actual_hash,
        "workers": args.workers,
        "shard_size": args.shard_size,
        "submitted_shards": submitted,
        "accepted": accepted,
        "rejected": rejected,
        "vocabulary_size": len(vocabulary),
        "vocabulary_file": str(vocab_file),
        "seconds": time.time() - started,
    }
    (output_dir.parent / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
