"""Train DrugEx v2's native generator from random initialization."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset


REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
import models  # noqa: E402
import utils  # noqa: E402


class EncodedDataset(Dataset):
    def __init__(self, path: Path) -> None:
        self.array = np.load(path, mmap_mode="r")

    def __len__(self) -> int:
        return len(self.array)

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.from_numpy(np.array(self.array[index], dtype=np.int64, copy=True))


def evaluate(model, loader, max_batches: int) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if batch_index >= max_batches:
                break
            batch = batch.to(utils.dev, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = -model.likelihood(batch).mean()
            total += float(loss) * len(batch)
            count += len(batch)
    model.train()
    return total / max(1, count)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", type=Path, default=Path("../../results/baselines/drugex_v2/data")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("../../results/baselines/drugex_v2/models")
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.backends.cudnn.benchmark = True

    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = EncodedDataset(data_dir / "common_tokens.npy")
    split = int(len(dataset) * 0.95)
    train_set = Subset(dataset, range(0, split))
    valid_set = Subset(dataset, range(split, len(dataset)))
    loader_args = dict(
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_set, shuffle=True, generator=generator, **loader_args)
    valid_loader = DataLoader(valid_set, shuffle=False, **loader_args)

    voc = utils.Voc(init_from_file=str(data_dir / "common_voc.txt"))
    model = models.Generator(voc, is_lstm=True, lr=args.lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best = float("inf")
    stale = 0
    log_path = output_dir / "training.jsonl"
    with log_path.open("a", encoding="utf-8") as log:
        for epoch in range(args.epochs):
            started = time.time()
            running = 0.0
            seen = 0
            for batch in train_loader:
                batch = batch.to(utils.dev, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = -model.likelihood(batch).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                running += float(loss.detach()) * len(batch)
                seen += len(batch)

            valid_loss = evaluate(model, valid_loader, max_batches=128)
            record = {
                "epoch": epoch + 1,
                "train_loss": running / max(1, seen),
                "validation_loss": valid_loss,
                "seconds": time.time() - started,
                "seen": seen,
            }
            print(json.dumps(record), flush=True)
            log.write(json.dumps(record) + "\n")
            log.flush()
            torch.save(model.state_dict(), output_dir / f"epoch_{epoch + 1:03d}.pkg")
            if valid_loss < best:
                best = valid_loss
                stale = 0
                torch.save(model.state_dict(), output_dir / "drugex_v2_common_dataset_best.pkg")
            else:
                stale += 1
                if stale >= args.patience:
                    break


if __name__ == "__main__":
    main()
