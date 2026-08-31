"""Fail-fast verification for the five-baseline experiment protocol."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "baseline_experiments.json"
REPORT_PATH = PROJECT_ROOT / "results" / "baselines" / "protocol_verification.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def line_stats(path: Path) -> tuple[int, int]:
    lines = 0
    empty = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            lines += 1
            empty += not bool(line.strip())
    return lines, empty


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    failures: list[str] = []
    dataset = PROJECT_ROOT / config["dataset"]["path"]
    actual_hash = sha256(dataset) if dataset.is_file() else None
    actual_lines, actual_empty = line_stats(dataset) if dataset.is_file() else (None, None)
    if actual_hash != config["dataset"]["sha256"]:
        failures.append("dataset SHA-256 mismatch")
    if actual_lines != config["dataset"]["line_count"]:
        failures.append("dataset line-count mismatch")
    if actual_empty != config["dataset"]["empty_line_count"]:
        failures.append("dataset empty-line mismatch")

    assets = {
        "dataset": dataset,
        "egfr_oracle": PROJECT_ROOT / config["oracles"]["egfr"],
        "vegfr2_oracle": PROJECT_ROOT / config["oracles"]["vegfr2"],
        "polygon_checkpoint": PROJECT_ROOT
        / config["baselines"]["polygon_original"]["checkpoint"],
    }
    for name, path in assets.items():
        if not path.is_file():
            failures.append(f"missing asset: {name} -> {path}")

    baseline_dirs = {
        name: PROJECT_ROOT / details["source_dir"]
        for name, details in config["baselines"].items()
    }
    for name, path in baseline_dirs.items():
        if not path.is_dir():
            failures.append(f"missing baseline source: {name} -> {path}")

    gpu = None
    try:
        gpu = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
            timeout=15,
        ).strip()
    except Exception as exc:  # pragma: no cover - hardware dependent
        failures.append(f"GPU check failed: {exc}")

    report = {
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "protocol": config["protocol_name"],
        "dataset": {
            "path": str(dataset),
            "sha256": actual_hash,
            "line_count": actual_lines,
            "empty_line_count": actual_empty,
        },
        "assets": {name: str(path) for name, path in assets.items()},
        "baseline_dirs": {name: str(path) for name, path in baseline_dirs.items()},
        "host": {
            "platform": platform.platform(),
            "logical_cpu_count": config["resources"]["logical_cpu_count"],
            "gpu": gpu,
        },
        "status": "pass" if not failures else "fail",
        "failures": failures,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
