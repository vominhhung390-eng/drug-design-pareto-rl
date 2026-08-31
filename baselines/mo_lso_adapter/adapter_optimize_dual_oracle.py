"""Multi-objective latent-space optimization for the trained MO-LSO JT-VAE.

The JT-VAE runs in the baseline PyTorch environment while the repository's
official GPflow SGPR and Expected Improvement scripts run in an isolated
TensorFlow environment.  The target-specific adaptation is a Pareto-rank scalar
target for the two maximization oracles; the released GP and EI code is reused.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import pytorch_lightning as pl
from rdkit import RDLogger


REPO_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = REPO_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))
WINDOWS_SUBPROCESS_FLAGS = getattr(subprocess, "CREATE_NO_WINDOW", 0)

from adapter_train_common_jtvae import initialize_like_official_wrapper  # noqa: E402
from baselines.common.oracle_ledger import OracleLedger  # noqa: E402
from weighted_retraining.chem.jtnn import JTNNVAE, MolTree, Vocab  # noqa: E402
from weighted_retraining.chem.jtnn.datautils import tensorize  # noqa: E402
# Compatibility alias for the repository's legacy Lightning callback base.
if not hasattr(pl.callbacks, "ProgressBar"):
    pl.callbacks.ProgressBar = pl.callbacks.progress.ProgressBarBase
from weighted_retraining.utils import DataWeighter, PF_rank_generator  # noqa: E402

RDLogger.DisableLog("rdApp.error")


def hypervolume_2d(points: np.ndarray) -> float:
    if len(points) == 0:
        return 0.0
    values = np.clip(np.asarray(points, dtype=np.float64), 0.0, None)
    values = values[np.argsort(values[:, 0])]
    suffix = np.maximum.accumulate(values[::-1, 1])[::-1]
    previous = 0.0
    area = 0.0
    for index, point in enumerate(values):
        x = max(previous, float(point[0]))
        area += max(0.0, x - previous) * float(suffix[index])
        previous = x
    return area


def pareto_mask(points: np.ndarray) -> np.ndarray:
    mask = np.ones(len(points), dtype=bool)
    for index, point in enumerate(points):
        dominated = np.any(np.all(points >= point, axis=1) & np.any(points > point, axis=1))
        mask[index] = not dominated
    return mask


def squared_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.maximum(
        np.sum(left * left, axis=1, keepdims=True)
        + np.sum(right * right, axis=1)[None, :]
        - 2.0 * left @ right.T,
        0.0,
    )


class SparseRBFSurrogate:
    def __init__(self, max_inducing: int = 256, ridge: float = 1e-3, seed: int = 42):
        self.max_inducing = max_inducing
        self.ridge = ridge
        self.rng = np.random.default_rng(seed)

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        count = min(len(x), self.max_inducing)
        if len(x) <= count:
            indices = np.arange(len(x))
        else:
            balanced = np.minimum(y[:, 0], y[:, 1])
            elite = np.argsort(balanced)[-count // 2 :]
            pool = np.setdiff1d(np.arange(len(x)), elite, assume_unique=False)
            random_part = self.rng.choice(pool, count - len(elite), replace=False)
            indices = np.concatenate([elite, random_part])
        self.inducing = x[indices].astype(np.float64)
        probe = self.inducing[: min(256, len(self.inducing))]
        distances = squared_distances(probe, probe)
        positive = distances[distances > 1e-8]
        self.lengthscale2 = float(np.median(positive)) if len(positive) else 1.0
        phi = np.exp(-0.5 * squared_distances(x, self.inducing) / self.lengthscale2)
        matrix = phi.T @ phi + self.ridge * np.eye(phi.shape[1])
        self.weights = np.linalg.solve(matrix, phi.T @ y)

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        distances = squared_distances(x, self.inducing)
        phi = np.exp(-0.5 * distances / self.lengthscale2)
        mean = phi @ self.weights
        nearest = np.sqrt(np.min(distances, axis=1) / max(self.lengthscale2, 1e-8))
        uncertainty = 0.05 + 0.45 * (1.0 - np.exp(-0.5 * nearest))
        return mean, uncertainty[:, None]


def make_tree(smiles: str, vocab: Vocab):
    try:
        tree = MolTree(smiles)
        tree.recover()
        tree.assemble()
        for node in tree.nodes:
            if node.label not in node.cands:
                node.cands.append(node.label)
            vocab.get_index(node.smiles)
        return tree
    except Exception:
        return None


def encode_smiles(
    model: JTNNVAE,
    smiles: list[str],
    vocab: Vocab,
    batch_size: int = 64,
) -> tuple[np.ndarray, list[str]]:
    encoded = []
    accepted = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(smiles), batch_size):
            trees = []
            batch_smiles = []
            for item in smiles[start : start + batch_size]:
                tree = make_tree(item, vocab)
                if tree is not None:
                    trees.append(tree)
                    batch_smiles.append(item)
            if not trees:
                continue
            _, jt_holder, mpn_holder = tensorize(trees, model.vocab, assm=False)
            means, _ = model.encode_latent(jt_holder, mpn_holder)
            encoded.append(means.cpu().numpy())
            accepted.extend(batch_smiles)
            del trees, jt_holder, mpn_holder, means
    values = (
        np.concatenate(encoded, axis=0)
        if encoded
        else np.empty((0, model.latent_size), dtype=np.float32)
    )
    return values, accepted


def weighted_retrain(
    model,
    entries: list[tuple],
    beta: float,
    learning_rate: float,
    seed: int,
    rank_weight_k: float,
) -> dict:
    if len(entries) < 4:
        return {"batches": 0, "mean_loss": None}
    rng = np.random.default_rng(seed)
    properties = np.asarray([entry[1] for entry in entries], dtype=np.float64)
    weights = DataWeighter.rank_weights_pf(
        properties,
        max_flag=np.ones(properties.shape[1], dtype=np.float64),
        k_val=rank_weight_k,
    )
    sampled_indices = rng.choice(
        len(entries), size=len(entries), replace=True, p=weights / weights.sum()
    )
    weighted_smiles = [entries[index][0] for index in sampled_indices]
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    losses = []
    for start in range(0, len(weighted_smiles), 16):
        batch_trees = [
            tree
            for tree in (make_tree(item, model.vocab) for item in weighted_smiles[start : start + 16])
            if tree is not None
        ]
        if len(batch_trees) < 2:
            continue
        try:
            batch = tensorize(batch_trees, model.vocab, assm=True)
            loss, *_ = model(batch, beta)
            if not torch.isfinite(loss):
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 20.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        except RuntimeError:
            optimizer.zero_grad(set_to_none=True)
        finally:
            del batch_trees
    return {"batches": len(losses), "mean_loss": float(np.mean(losses)) if losses else None}


def official_gpflow_propose(
    observed_z: np.ndarray,
    observed_y: np.ndarray,
    output_dir: Path,
    iteration: int,
    args: argparse.Namespace,
    previous_gp_file: Path | None,
) -> tuple[np.ndarray, Path, dict]:
    """Fit and optimize the released MO-LSO GPflow surrogate in a subprocess."""
    gp_dir = output_dir / "gpflow" / f"iteration_{iteration:04d}"
    gp_dir.mkdir(parents=True, exist_ok=True)
    data_file = gp_dir / "data.npz"
    gp_file = gp_dir / "gp.npz"
    opt_file = gp_dir / "opt.npy"
    ranks = PF_rank_generator(np.asarray(observed_y, dtype=np.float64))
    targets = ranks / max(1.0, float(len(ranks) - 1))
    # Pareto fronts can contain ties.  The small term only orders tied fronts and
    # keeps Pareto rank as the dominant minimization target for Expected Improvement.
    targets += 1e-3 * (1.0 - np.asarray(observed_y, dtype=np.float64).mean(axis=1))
    np.savez_compressed(
        data_file,
        X_train=np.asarray(observed_z, dtype=np.float32),
        X_test=np.empty((0, observed_z.shape[1]), dtype=np.float32),
        y_train=targets.astype(np.float32).reshape(-1, 1),
        y_test=np.empty((0, 1), dtype=np.float32),
    )

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    n_inducing = min(args.n_inducing_points, len(observed_z))
    train_command = [
        str(args.gp_python.resolve()),
        str((REPO_ROOT / "weighted_retraining/gp_train.py").resolve()),
        f"--nZ={n_inducing}",
        f"--seed={args.seed + iteration}",
        f"--data_file={data_file}",
        f"--save_file={gp_file}",
        f"--logfile={gp_dir / 'train.log'}",
        f"--n_opt_iter={args.gp_opt_iterations}",
        "--measure_freq=100",
    ]
    if previous_gp_file is None:
        train_command.extend(["--init", "--kmeans_init"])
    else:
        train_command.extend([f"--gp_file={previous_gp_file}", "--n_perf_measure=1"])
    started = time.time()
    train_result = subprocess.run(
        train_command,
        capture_output=True,
        text=True,
        env=env,
        creationflags=WINDOWS_SUBPROCESS_FLAGS,
    )
    (gp_dir / "train_subprocess.log").write_text(
        train_result.stdout + "\n" + train_result.stderr, encoding="utf-8"
    )
    if train_result.returncode != 0:
        raise RuntimeError(f"Official GPflow training failed; see {gp_dir / 'train_subprocess.log'}")
    train_seconds = time.time() - started

    opt_command = [
        str(args.gp_python.resolve()),
        str((REPO_ROOT / "weighted_retraining/gp_opt.py").resolve()),
        f"--seed={args.seed + iteration}",
        f"--data_file={data_file}",
        f"--gp_file={gp_file}",
        f"--save_file={opt_file}",
        f"--logfile={gp_dir / 'opt.log'}",
        f"--n_out={args.batch_size}",
        f"--n_starts={max(args.gp_starts, args.batch_size)}",
        f"--workers={args.gp_workers}",
    ]
    started = time.time()
    opt_result = subprocess.run(
        opt_command,
        capture_output=True,
        text=True,
        env=env,
        creationflags=WINDOWS_SUBPROCESS_FLAGS,
    )
    (gp_dir / "opt_subprocess.log").write_text(
        opt_result.stdout + "\n" + opt_result.stderr, encoding="utf-8"
    )
    if opt_result.returncode != 0:
        raise RuntimeError(f"Official GPflow EI optimization failed; see {gp_dir / 'opt_subprocess.log'}")
    proposed = np.asarray(np.load(opt_file), dtype=np.float32).reshape(-1, observed_z.shape[1])
    return proposed, gp_file, {
        "gp_train_seconds": train_seconds,
        "gp_opt_seconds": time.time() - started,
        "gp_training_points": len(observed_z),
        "n_inducing": n_inducing,
    }


def decode_latents(model: JTNNVAE, z: np.ndarray, device) -> list[str]:
    tensor = torch.as_tensor(z, dtype=torch.float32, device=device)
    split = model.latent_T_size
    model.eval()
    with torch.inference_mode():
        decoded = model.decode(tensor[:, :split], tensor[:, split:], prob_decode=False)
    return [item or "" for item in decoded]


def latest_retrain_checkpoint(output_dir: Path, oracle_used: int) -> Path | None:
    candidates = []
    for path in output_dir.glob("retrained_*.pt"):
        try:
            used = int(path.stem.rsplit("_", 1)[1])
        except ValueError:
            continue
        if used <= oracle_used:
            candidates.append((used, path))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def save_resume_state(
    output_dir: Path,
    ledger: OracleLedger,
    observed_z: np.ndarray,
    observed_y: np.ndarray,
    entries: dict[str, np.ndarray],
    rng: np.random.Generator,
    current_gp_file: Path | None,
) -> None:
    smiles = list(entries)
    entry_y = (
        np.asarray([entries[item] for item in smiles], dtype=np.float32)
        if smiles
        else np.empty((0, 2), dtype=np.float32)
    )
    temporary = output_dir / "resume_state.tmp.npz"
    np.savez_compressed(
        temporary,
        oracle_used=np.asarray(ledger.used, dtype=np.int64),
        observed_z=np.asarray(observed_z, dtype=np.float32),
        observed_y=np.asarray(observed_y, dtype=np.float32),
        entry_smiles=np.asarray(smiles, dtype=np.str_),
        entry_y=entry_y,
        rng_state=np.asarray(json.dumps(rng.bit_generator.state)),
        current_gp_file=np.asarray(str(current_gp_file) if current_gp_file else ""),
    )
    temporary.replace(output_dir / "resume_state.npz")


def load_exact_resume_state(
    output_dir: Path,
    ledger: OracleLedger,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], Path | None] | None:
    state_file = output_dir / "resume_state.npz"
    if not state_file.exists():
        return None
    with np.load(state_file, allow_pickle=False) as state:
        if int(state["oracle_used"]) != ledger.used:
            return None
        observed_z = np.asarray(state["observed_z"], dtype=np.float32)
        observed_y = np.asarray(state["observed_y"], dtype=np.float32)
        smiles = [str(item) for item in state["entry_smiles"]]
        entry_y = np.asarray(state["entry_y"], dtype=np.float32)
        entries = {item: scores for item, scores in zip(smiles, entry_y)}
        rng.bit_generator.state = json.loads(str(state["rng_state"].item()))
        gp_text = str(state["current_gp_file"].item())
    gp_file = Path(gp_text) if gp_text and Path(gp_text).exists() else None
    return observed_z, observed_y, entries, gp_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "results/baselines/mo_lso/data")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--candidate-pool", type=int, default=4096)
    parser.add_argument("--retraining-frequency", type=int, default=1024)
    parser.add_argument("--retraining-learning-rate", type=float, default=1e-4)
    parser.add_argument("--rank-weight-k", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=0.005)
    parser.add_argument(
        "--gp-python",
        type=Path,
        default=REPO_ROOT / ".gpflow-venv/Scripts/python.exe",
    )
    parser.add_argument("--gp-warmup", type=int, default=256)
    parser.add_argument("--n-inducing-points", type=int, default=128)
    parser.add_argument("--gp-starts", type=int, default=128)
    parser.add_argument("--gp-workers", type=int, default=1)
    parser.add_argument("--gp-opt-iterations", type=int, default=100000)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if args.budget % args.batch_size != 0:
        raise ValueError("MO-LSO budget must be divisible by batch size")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger = OracleLedger(args.budget, output_dir / "generated.csv", resume=True)
    initial_oracle_used = ledger.used

    vocab_file = args.data_dir.resolve() / "vocab.txt"
    vocab = Vocab([line for line in vocab_file.read_text(encoding="utf-8").splitlines() if line])
    checkpoint = torch.load(args.model.resolve(), map_location="cpu", weights_only=False)
    model = JTNNVAE(vocab, 450, 56, 20, 3, latent_T_size=None)
    initialize_like_official_wrapper(model)
    model.load_state_dict(checkpoint["model_state_dict"])
    resumed_model = latest_retrain_checkpoint(output_dir, ledger.used)
    if resumed_model is not None:
        resumed_checkpoint = torch.load(resumed_model, map_location="cpu", weights_only=False)
        model.load_state_dict(resumed_checkpoint["model_state_dict"])
    model.to(device).eval()

    exact_state = load_exact_resume_state(output_dir, ledger, rng)
    recovered_from_ledger = ledger.used > 0 and exact_state is None
    if exact_state is not None:
        observed_z, observed_y, tree_entries, current_gp_file = exact_state
    else:
        tree_entries: dict[str, np.ndarray] = {}
        for record in ledger.records:
            canonical = record["canonical_smiles"]
            if not record["valid"] or not canonical or canonical in tree_entries:
                continue
            tree = make_tree(canonical, vocab)
            if tree is not None:
                tree_entries[canonical] = np.asarray(
                    [record["egfr_desirability"], record["vegfr2_desirability"]],
                    dtype=np.float32,
                )
            del tree
        if tree_entries:
            observed_z, accepted = encode_smiles(model, list(tree_entries), vocab)
            tree_entries = {item: tree_entries[item] for item in accepted}
            observed_y = np.asarray([tree_entries[item] for item in accepted], dtype=np.float32)
        else:
            observed_z = np.empty((0, 56), dtype=np.float32)
            observed_y = np.empty((0, 2), dtype=np.float32)
        current_gp_file = None
    metrics_file = output_dir / "iterations.json"
    metrics = json.loads(metrics_file.read_text(encoding="utf-8")) if metrics_file.exists() else []
    started = time.time()

    start_iteration = ledger.used // args.batch_size
    for iteration in range(start_iteration, args.budget // args.batch_size):
        gp_details = None
        if len(observed_z) < args.gp_warmup:
            proposed = rng.standard_normal((args.batch_size, 56), dtype=np.float32)
            acquisition = "prior_initialization"
        else:
            proposed, current_gp_file, gp_details = official_gpflow_propose(
                observed_z, observed_y, output_dir, iteration, args, current_gp_file
            )
            if len(proposed) < args.batch_size:
                padding = rng.standard_normal(
                    (args.batch_size - len(proposed), observed_z.shape[1]), dtype=np.float32
                )
                proposed = np.concatenate([proposed, padding], axis=0)
            proposed = proposed[: args.batch_size]
            acquisition = "official_gpflow_sgpr_expected_improvement_on_pareto_rank"

        smiles = decode_latents(model, proposed, device)
        results, objectives = ledger.score(smiles, phase="latent_query", iteration=iteration)
        usable_z = []
        usable_y = []
        for z, result, scores in zip(proposed, results, objectives):
            if not result.valid:
                continue
            if result.canonical_smiles not in tree_entries:
                tree = make_tree(result.canonical_smiles, vocab)
                if tree is not None:
                    tree_entries[result.canonical_smiles] = scores.copy()
                del tree
            if result.canonical_smiles in tree_entries:
                usable_z.append(z)
                usable_y.append(scores)
        if usable_z:
            observed_z = np.concatenate([observed_z, np.asarray(usable_z, dtype=np.float32)])
            observed_y = np.concatenate([observed_y, np.asarray(usable_y, dtype=np.float32)])

        retrain = None
        if ledger.used % args.retraining_frequency == 0 and tree_entries:
            retrain = weighted_retrain(
                model,
                list(tree_entries.items()),
                args.beta,
                args.retraining_learning_rate,
                args.seed + iteration,
                args.rank_weight_k,
            )
            observed_z, accepted = encode_smiles(model, list(tree_entries), vocab)
            tree_entries = {item: tree_entries[item] for item in accepted}
            observed_y = np.asarray([tree_entries[item] for item in accepted], dtype=np.float32)
            current_gp_file = None
            torch.save(
                {"model_state_dict": model.state_dict(), "iteration": iteration, "oracle_used": ledger.used},
                output_dir / f"retrained_{ledger.used:05d}.pt",
            )

        front = observed_y[pareto_mask(observed_y)] if len(observed_y) else observed_y
        row = {
            "iteration": iteration,
            "oracle_used": ledger.used,
            "valid_usable": len(usable_z),
            "surrogate_points": len(observed_z),
            "pareto_size": len(front),
            "hypervolume": hypervolume_2d(front),
            "acquisition": acquisition,
            "gpflow": gp_details,
            "retraining": retrain,
            "seconds": time.time() - started,
        }
        metrics.append(row)
        print(json.dumps(row), flush=True)
        (output_dir / "iterations.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        save_resume_state(
            output_dir,
            ledger,
            observed_z,
            observed_y,
            tree_entries,
            rng,
            current_gp_file,
        )

    torch.save({"model_state_dict": model.state_dict()}, output_dir / "mo_lso_optimized.pt")
    ledger.write_metadata(
        output_dir / "metadata.json",
        method="MO-LSO",
        seed=args.seed,
        model=str(args.model.resolve()),
        from_scratch_generator=True,
        algorithm="official GPflow SGPR/EI latent optimization with Pareto-rank weighted retraining",
        acquisition="official Expected Improvement on two-objective Pareto rank",
        batch_size=args.batch_size,
        retraining_frequency=args.retraining_frequency,
        rank_weight_k=args.rank_weight_k,
        gpflow_official_scripts=True,
        gpflow_python=str(args.gp_python.resolve()),
        gp_warmup=args.gp_warmup,
        n_inducing_points=args.n_inducing_points,
        gp_starts=max(args.gp_starts, args.batch_size),
        gp_workers=args.gp_workers,
        gp_opt_iterations=args.gp_opt_iterations,
        target_adaptation="normalized two-oracle Pareto rank; both objectives maximized",
        resumed_from_oracle_rows=initial_oracle_used,
        exact_resume_state=bool(initial_oracle_used and not recovered_from_ledger),
        recovered_compact_state_from_ledger=recovered_from_ledger,
    )


if __name__ == "__main__":
    main()
