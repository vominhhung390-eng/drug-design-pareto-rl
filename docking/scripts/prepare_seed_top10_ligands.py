from __future__ import annotations

from pathlib import Path

import prepare_docking_ligands as base


PROJECT = Path(__file__).resolve().parents[2]
base.HERE = PROJECT / "docking/seed_top10_two_pairs_20260830"


if __name__ == "__main__":
    base.main()
