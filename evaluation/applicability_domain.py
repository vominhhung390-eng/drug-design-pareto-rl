#!/usr/bin/env python
"""Disabled legacy entry point retained to prevent VAE similarity being mislabeled as AD."""

raise SystemExit(
    "applicability_domain.py is disabled: the available reference is the VAE training set, "
    "not the RF predictor training set. Use vae_reference_similarity.py for the explicitly "
    "named VAE-reference audit, or provide the exact RF training compounds before computing "
    "a predictor applicability domain."
)
