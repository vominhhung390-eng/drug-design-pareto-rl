#!/usr/bin/env python
"""Launch Chemprop with UTF-8 console output and Tensor Core matmul enabled."""
from __future__ import annotations

import os

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("RICH_FORCE_TERMINAL", "false")

import torch

torch.set_float32_matmul_precision("high")

from chemprop.cli.main import main


if __name__ == "__main__":
    main()
