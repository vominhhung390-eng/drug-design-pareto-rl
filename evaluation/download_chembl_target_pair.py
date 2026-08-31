#!/usr/bin/env python
"""Download versioned ChEMBL activity records for a configured target pair."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from validate_target_predictors import api_status, download_activities


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "config" / "predictor_pik3ca_mtor_20260804.json",
    )
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--page-size", type=int, default=1000)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--retries", type=int, default=6)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    base = "https://www.ebi.ac.uk/chembl/api/data"
    manifest: dict[str, object] = {
        "run_id": config["run_id"],
        "api_status": api_status(base, args.timeout, args.retries),
        "targets": {},
    }
    for target, target_id in config["target_ids"].items():
        cache_path = root / config["sources"][target]
        rows = download_activities(
            base=base,
            target_id=target_id,
            cache_path=cache_path,
            page_size=args.page_size,
            timeout=args.timeout,
            retries=args.retries,
            refresh=args.refresh,
        )
        manifest["targets"][target] = {
            "target_chembl_id": target_id,
            "rows": len(rows),
            "path": str(cache_path.resolve()),
            "sha256": sha256(cache_path),
        }

    output = root / config["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    (output / "download_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
