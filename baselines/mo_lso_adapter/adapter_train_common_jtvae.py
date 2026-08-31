"""Train the official MO-LSO JTNNVAE from random initialization.

The script consumes only the sharded MolTree representation produced from the
locked common SMILES dataset.  It deliberately imports the official model and
data tensorization code rather than substituting another molecular VAE.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = REPO_ROOT.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from weighted_retraining.chem.jtnn import JTNNVAE, MolTreeFolder, Vocab  # noqa: E402


EXPECTED_SOURCE_HASH = (
    "4301e7f6118839465012eb93510328681ef4b7b24642e8748c4ad40971f4a304"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def initialize_like_official_wrapper(model: JTNNVAE) -> None:
    """Match MO-LSO's JTVAE wrapper initialization and compatibility flags."""
    for parameter in model.parameters():
        if parameter.dim() == 1:
            nn.init.constant_(parameter, 0)
        else:
            nn.init.xavier_normal_(parameter)
    model.jtnn.GRU.tanh = False
    model.decoder.U_i_relu = False
    model._no_assm = True


def make_folder(
    tensor_dir: Path,
    vocab: Vocab,
    files: list[str],
    batch_size: int,
    workers: int,
    shuffle: bool,
) -> MolTreeFolder:
    folder = MolTreeFolder(
        str(tensor_dir),
        vocab,
        batch_size=batch_size,
        num_workers=workers,
        shuffle=shuffle,
        assm=True,
    )
    folder.data_files = list(files)
    return folder


def run_epoch(
    model: JTNNVAE,
    folder: MolTreeFolder,
    beta: float,
    optimizer: torch.optim.Optimizer | None,
    gradient_clip: float,
    max_batches: int | None,
) -> dict:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "kl": 0.0, "word_acc": 0.0, "topo_acc": 0.0, "assm_acc": 0.0}
    batches = 0
    skipped = 0
    started = time.time()

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in folder:
            if max_batches is not None and batches >= max_batches:
                break
            try:
                loss, kl, word_acc, topo_acc, assm_acc = model(batch, beta)
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite loss")
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                    optimizer.step()
                totals["loss"] += float(loss.detach().cpu())
                totals["kl"] += float(kl)
                totals["word_acc"] += float(word_acc)
                totals["topo_acc"] += float(topo_acc)
                totals["assm_acc"] += float(assm_acc)
                batches += 1
                if batches % 100 == 0:
                    print(
                        json.dumps(
                            {
                                "phase": "train" if training else "validation",
                                "batches": batches,
                                "skipped": skipped,
                                "mean_loss": totals["loss"] / batches,
                                "seconds": time.time() - started,
                            }
                        ),
                        flush=True,
                    )
            except RuntimeError as exc:
                # The upstream Lightning wrapper also treats individual JT-VAE
                # runtime failures as zero-gradient batches.  Keep an auditable
                # count instead of silently hiding them.
                skipped += 1
                if training:
                    optimizer.zero_grad(set_to_none=True)
                print(json.dumps({"phase": "skipped_batch", "error": repr(exc)}), flush=True)

    if batches == 0:
        raise RuntimeError("No successful JTNNVAE batches were produced")
    return {
        **{key: value / batches for key, value in totals.items()},
        "batches": batches,
        "skipped_batches": skipped,
        "seconds": time.time() - started,
    }


def checkpoint_payload(model: JTNNVAE, optimizer, epoch: int, config: dict, metrics: dict) -> dict:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "config": config,
        "metrics": metrics,
        "source_sha256": EXPECTED_SOURCE_HASH,
        "random_initialization": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=PROJECT_ROOT / "data/train_smiles_only.txt")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "results/baselines/mo_lso/data")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results/baselines/mo_lso/models")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=7e-4)
    parser.add_argument("--beta", type=float, default=0.005)
    parser.add_argument("--gradient-clip", type=float, default=20.0)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-validation-batches", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    source = args.source.resolve()
    data_dir = args.data_dir.resolve()
    tensor_dir = data_dir / "tensors_train"
    vocab_file = data_dir / "vocab.txt"
    metadata_file = data_dir / "metadata.json"
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if sha256(source) != EXPECTED_SOURCE_HASH:
        raise RuntimeError("The MO-LSO source dataset does not match the locked common dataset")
    if not metadata_file.is_file() or not vocab_file.is_file():
        raise RuntimeError("MO-LSO preprocessing is incomplete: metadata/vocabulary is missing")
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    if metadata.get("source_sha256") != EXPECTED_SOURCE_HASH:
        raise RuntimeError("MO-LSO preprocessing metadata has an unexpected source hash")

    all_files = sorted(path.name for path in tensor_dir.glob("*.pkl"))
    if len(all_files) != int(metadata["submitted_shards"]):
        raise RuntimeError(f"Expected {metadata['submitted_shards']} tensor shards, found {len(all_files)}")
    validation_count = max(1, round(len(all_files) * args.validation_fraction))
    train_files = all_files[:-validation_count]
    validation_files = all_files[-validation_count:]

    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    vocab = Vocab([line.strip() for line in vocab_file.read_text(encoding="utf-8").splitlines() if line.strip()])
    config = {
        "hidden_size": 450,
        "latent_size": 56,
        "latent_T_size": None,
        "depthT": 20,
        "depthG": 3,
        "learning_rate": args.learning_rate,
        "beta": args.beta,
        "batch_size": args.batch_size,
        "source": str(source),
        "source_sha256": EXPECTED_SOURCE_HASH,
        "train_shards": len(train_files),
        "validation_shards": len(validation_files),
        "accepted_molecules": metadata["accepted"],
        "rejected_molecules": metadata["rejected"],
        "vocabulary_size": vocab.size(),
        "seed": args.seed,
        "device": str(device),
        "initialization": "random Xavier/zero matching official JTVAE wrapper",
    }
    (output_dir / "training_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    model = JTNNVAE(vocab, 450, 56, 20, 3, latent_T_size=None)
    initialize_like_official_wrapper(model)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    history = []
    best_validation = float("inf")
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        random.Random(args.seed + epoch).shuffle(train_files)
        train_folder = make_folder(tensor_dir, vocab, train_files, args.batch_size, args.workers, True)
        validation_folder = make_folder(tensor_dir, vocab, validation_files, args.batch_size, args.workers, False)
        train_metrics = run_epoch(model, train_folder, args.beta, optimizer, args.gradient_clip, args.max_train_batches)
        validation_metrics = run_epoch(model, validation_folder, args.beta, None, args.gradient_clip, args.max_validation_batches)
        metrics = {"epoch": epoch, "train": train_metrics, "validation": validation_metrics}
        history.append(metrics)
        print(json.dumps(metrics), flush=True)
        torch.save(checkpoint_payload(model, optimizer, epoch, config, metrics), output_dir / f"epoch_{epoch:03d}.pt")
        if validation_metrics["loss"] < best_validation:
            best_validation = validation_metrics["loss"]
            stale_epochs = 0
            torch.save(checkpoint_payload(model, optimizer, epoch, config, metrics), output_dir / "best.pt")
        else:
            stale_epochs += 1
        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if stale_epochs >= args.patience:
            print(json.dumps({"early_stop_epoch": epoch, "patience": args.patience}), flush=True)
            break


if __name__ == "__main__":
    # Avoid severe CPU oversubscription inside each Windows data worker.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    main()
