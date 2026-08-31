"""Tests for RL distance penalty in reinvent/runmodes/RL/distance_penalty.py."""

import pytest
import numpy as np
import torch

from reinvent.runmodes.RL.distance_penalty import get_distance_to_prior

# ── torch.Tensor branch ──────────────────────────────────────────────────────


def test_distance_penalty_tensor_above_threshold_is_fractional():
    ll = torch.tensor([10.0, 20.0])
    mask = get_distance_to_prior(ll, distance_threshold=5.0)
    assert mask[0] == pytest.approx(0.5, abs=1e-5)  # 5.0/10.0
    assert mask[1] == pytest.approx(0.25, abs=1e-5)  # 5.0/20.0


def test_distance_penalty_tensor_at_or_below_threshold_is_one():
    ll = torch.tensor([1.0, 3.0, 5.0])
    mask = get_distance_to_prior(ll, distance_threshold=5.0)
    assert np.all(mask == 1.0)


def test_distance_penalty_tensor_returns_numpy():
    ll = torch.tensor([8.0])
    mask = get_distance_to_prior(ll, distance_threshold=4.0)
    assert isinstance(mask, np.ndarray)


# ── numpy.ndarray branch ──────────────────────────────────────────────────────


def test_distance_penalty_ndarray_above_threshold():
    ll = np.array([10.0, 5.0])
    mask = get_distance_to_prior(ll, distance_threshold=5.0)
    assert mask[0] == pytest.approx(0.5, abs=1e-5)
    assert mask[1] == pytest.approx(1.0, abs=1e-5)  # exactly at threshold → 1


def test_distance_penalty_ndarray_below_threshold_all_ones():
    ll = np.array([1.0, 2.0])
    mask = get_distance_to_prior(ll, distance_threshold=10.0)
    assert np.all(mask == 1.0)


def test_distance_penalty_mask_values_bounded():
    ll = torch.tensor([0.1, 5.0, 100.0])
    mask = get_distance_to_prior(ll, distance_threshold=5.0)
    assert np.all(mask <= 1.0)
    assert np.all(mask > 0.0)
