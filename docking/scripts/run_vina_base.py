from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
HERE = PROJECT / "docking/seed_top10_two_pairs_20260830"


def resolve_vina() -> Path:
    candidates: list[Path] = []
    explicit = os.environ.get("VINA_EXECUTABLE", "").strip()
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates.append(PROJECT / "tools/autodock_vina_1.1.2/vina.exe")
    on_path = shutil.which("vina")
    if on_path:
        candidates.append(Path(on_path))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "AutoDock Vina is not bundled in the public repository. "
        "Set VINA_EXECUTABLE to a user-supplied Vina 1.1.2 binary or place it at "
        "tools/autodock_vina_1.1.2/vina.exe."
    )


VINA = resolve_vina()
MAX_PARALLEL = 6
VINA_CPU = 4
BASE_SEED = 202608300

TARGETS = {
    "pairA": {
        "EGFR": {
            "receptor": PROJECT / "docking/receptors/EGFR_VEGFR2/1M17_prepared.pdbqt",
            "center": (22.014, 0.253, 52.794),
            "size": (25.709, 22.0, 22.0),
        },
        "VEGFR2": {
            "receptor": PROJECT / "docking/receptors/EGFR_VEGFR2/4AG8_prepared.pdbqt",
            "center": (20.824, 25.535, 39.46),
            "size": (23.78, 22.0, 22.0),
        },
    },
    "pairB": {
        "PARP1": {
            "receptor": PROJECT / "docking/receptors/PARP1_BRD4/7KK4_prepared.pdbqt",
            "center": (-9.294, 6.13, 26.348),
            "size": (22.0, 24.211, 22.0),
        },
        "BRD4": {
            "receptor": PROJECT / "docking/receptors/PARP1_BRD4/3MXF_prepared.pdbqt",
            "center": (28.751, 15.826, -2.335),
            "size": (22.0, 22.0, 22.0),
        },
    },
}

AFFINITY_LINE = re.compile(r"^\s*\d+\s+(-?\d+(?:\.\d+)?)\s+[-+]?\d+(?:\.\d+)?\s+[-+]?\d+(?:\.\d+)?\s*$")


def parse_affinities(text: str) -> list[float]:
    values = []
    for line in text.splitlines():
        match = AFFINITY_LINE.match(line)
        if match:
            values.append(float(match.group(1)))
    return values


def atomic_json(path: Path, payload) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def task_record(row, target_name: str, index: int) -> dict:
    pair_key = str(row.pair_key)
    compound_id = str(row.compound_id)
    ligand = HERE / "ligands" / f"{compound_id}.pdbqt"
    out_dir = HERE / "poses" / target_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{compound_id}_out.pdbqt"
    log_file = out_dir / f"{compound_id}.log"
    stdout_file = out_dir / f"{compound_id}.stdout.txt"
    spec = TARGETS[pair_key][target_name]
    cx, cy, cz = spec["center"]
    sx, sy, sz = spec["size"]
    # Vina 1.1.2 on Windows cannot reliably create output files when the
    # command line contains the Chinese project root.  Keep manifest records
    # absolute, but pass paths relative to the ASCII-only docking directory.
    rel = lambda path: os.path.relpath(str(path), str(HERE))
    return {
        "task_key": f"{pair_key}:{target_name}:{compound_id}",
        "pair_key": pair_key,
        "target": target_name,
        "compound_id": compound_id,
        "method": str(row.method),
        "source_seed": int(row.source_seed),
        "canonical_smiles": str(row.canonical_smiles),
        "ligand": str(ligand),
        "receptor": str(spec["receptor"]),
        "out_file": str(out_file),
        "log_file": str(log_file),
        "stdout_file": str(stdout_file),
        "seed": int(BASE_SEED + index),
        "command": [
            str(VINA),
            "--receptor",
            rel(spec["receptor"]),
            "--ligand",
            rel(ligand),
            "--center_x",
            str(cx),
            "--center_y",
            str(cy),
            "--center_z",
            str(cz),
            "--size_x",
            str(sx),
            "--size_y",
            str(sy),
            "--size_z",
            str(sz),
            "--exhaustiveness",
            "32",
            "--num_modes",
            "9",
            "--energy_range",
            "5",
            "--cpu",
            str(VINA_CPU),
            "--seed",
            str(BASE_SEED + index),
            "--out",
            rel(out_file),
            "--log",
            rel(log_file),
        ],
    }


def run_one(task: dict) -> dict:
    result = dict(task)
    try:
        ligand = Path(task["ligand"])
        receptor = Path(task["receptor"])
        if not ligand.exists():
            raise FileNotFoundError(f"ligand not found: {ligand}")
        if not receptor.exists():
            raise FileNotFoundError(f"receptor not found: {receptor}")
        proc = subprocess.run(
            task["command"],
            cwd=HERE,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if proc.stdout:
            Path(task["stdout_file"]).write_text(proc.stdout, encoding="utf-8")
        if proc.stderr:
            Path(task["stdout_file"]).open("a", encoding="utf-8").write("\n[stderr]\n" + proc.stderr)
        log_text = Path(task["log_file"]).read_text(encoding="utf-8", errors="replace") if Path(task["log_file"]).exists() else proc.stdout
        affinities = parse_affinities(log_text)
        result.update(
            {
                "returncode": int(proc.returncode),
                "affinities": affinities,
                "best_affinity": min(affinities) if affinities else None,
                "status": "ok" if proc.returncode == 0 and affinities else "error",
            }
        )
        if proc.returncode != 0 and not result.get("error"):
            result["error"] = (proc.stderr or proc.stdout)[-2000:]
        if not affinities and not result.get("error"):
            result["error"] = "no affinity rows parsed from Vina log"
    except Exception as exc:
        result.update({"returncode": None, "affinities": [], "best_affinity": None, "status": "error", "error": str(exc)})
    return result


def reusable(record: dict) -> bool:
    return bool(
        record.get("status") == "ok"
        and record.get("affinities")
        and Path(record.get("out_file", "")).exists()
        and Path(record.get("log_file", "")).exists()
    )


def main() -> None:
    manifest = json.loads((HERE / "ligand_prep_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("errors"):
        raise SystemExit("ligand preparation manifest still contains errors")
    selected = pd.read_csv(HERE / "selected_compounds.csv", encoding="utf-8-sig")
    tasks = []
    idx = 0
    for row in selected.itertuples(index=False):
        for target_name in TARGETS[str(row.pair_key)]:
            tasks.append(task_record(row, target_name, idx))
            idx += 1
    partial_path = HERE / "docking_raw_results.partial.json"
    existing = {}
    if partial_path.exists():
        try:
            existing = {str(item["task_key"]): item for item in json.loads(partial_path.read_text(encoding="utf-8"))}
        except Exception:
            existing = {}
    records = dict(existing)
    pending = []
    for task in tasks:
        old = records.get(task["task_key"])
        if old and reusable(old):
            continue
        pending.append(task)
    print(f"tasks_total={len(tasks)} reusable={len(tasks)-len(pending)} pending={len(pending)} workers={MAX_PARALLEL}", flush=True)
    if pending:
        with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
            futures = {pool.submit(run_one, task): task for task in pending}
            done = 0
            for future in as_completed(futures):
                record = future.result()
                records[record["task_key"]] = record
                done += 1
                atomic_json(partial_path, [records[t["task_key"]] for t in tasks if t["task_key"] in records])
                ok = sum(1 for item in records.values() if item.get("status") == "ok")
                errors = sum(1 for item in records.values() if item.get("status") == "error")
                print(f"completed={done}/{len(pending)} total_ok={ok} errors={errors}", flush=True)
    ordered = [records[t["task_key"]] for t in tasks if t["task_key"] in records]
    atomic_json(partial_path, ordered)
    errors = [item for item in ordered if item.get("status") != "ok"]
    atomic_json(HERE / "docking_raw_results.json", ordered)
    if errors:
        atomic_json(HERE / "docking_errors.json", errors)
        raise SystemExit(f"docking completed with {len(errors)} errors")
    print(f"docking_complete={len(ordered)}/{len(tasks)}", flush=True)


if __name__ == "__main__":
    main()
