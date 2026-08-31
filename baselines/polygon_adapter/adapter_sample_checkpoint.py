"""Load the user's non-default POLYGON VAE and sample without retraining."""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
from rdkit import RDLogger


# Invalid decoder strings are expected for an unconditional VAE sample.  They
# are recorded through the adapter's ``valid`` field, so RDKit's per-molecule
# parser diagnostics would only flood the formal-run log.
RDLogger.DisableLog("rdApp.error")
RDLogger.DisableLog("rdApp.warning")


REPO_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = REPO_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from baselines.common.oracle_bridge import DualTargetOracle  # noqa: E402
from polygon.vae.vae_model import VAE  # noqa: E402


def infer_config(state_dict: dict[str, torch.Tensor]) -> dict:
    decoder_layers = [
        int(key.split("_l")[-1])
        for key in state_dict
        if key.startswith("decoder_rnn.weight_ih_l")
    ]
    return {
        "q_bidir": any("_reverse" in key for key in state_dict),
        "q_d_h": state_dict["encoder_rnn.weight_hh_l0"].shape[1],
        "q_n_layers": 1,
        "q_cell": "gru",
        "q_dropout": 0.5,
        "d_cell": "gru",
        "d_n_layers": max(decoder_layers) + 1,
        "d_dropout": 0.2,
        "d_z": state_dict["decoder_lat.weight"].shape[1],
        "d_d_h": state_dict["decoder_rnn.weight_hh_l0"].shape[1],
        "freeze_embeddings": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "models" / "polygon_vae_best_valid_novel_stable_020.pt",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = VAE(**infer_config(state))
    model.load_state_dict(state)
    model.to(device).eval()
    with torch.inference_mode():
        smiles = model.sample(
            n_batch=args.count,
            max_len=120,
            temp=args.temperature,
            multinomial=True,
        )
    oracle = DualTargetOracle()
    results = oracle.score_many(smiles)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["smiles", "canonical_smiles", "valid", "egfr", "vegfr2"],
        )
        writer.writeheader()
        writer.writerows(result.__dict__ for result in results)
    print(
        {
            "requested": args.count,
            "written": len(results),
            "valid": sum(result.valid for result in results),
            "output": str(args.output.resolve()),
            "device": str(device),
        }
    )


if __name__ == "__main__":
    main()
