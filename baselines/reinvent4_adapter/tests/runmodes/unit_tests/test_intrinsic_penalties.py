"""Tests for RL intrinsic penalty functions: Sigmoid, Tanh, Linear, Erf, Step."""

import pytest

from reinvent.runmodes.RL.memories.bucket_counter import BucketCounter
from reinvent.runmodes.RL.intrinsic_penalty.penalties import (
    SigmoidPenalty,
    TanhPenalty,
    LinearPenalty,
    ErfPenalty,
    StepPenalty,
)

ALL_PENALTY_CLASSES = [SigmoidPenalty, TanhPenalty, LinearPenalty, ErfPenalty, StepPenalty]


@pytest.mark.parametrize("PenaltyClass", ALL_PENALTY_CLASSES)
def test_penalty_is_positive_when_bucket_empty(PenaltyClass):
    """All penalty functions should return a positive value for an empty bucket."""
    bc = BucketCounter(max_size=10)
    p = PenaltyClass(bc)
    val = p.calculate_penalty("empty_scaffold")
    assert val > 0.0, f"{PenaltyClass.__name__} should give positive penalty for empty bucket"


# Sigmoid, Linear and Step are bounded to [0,1] by design; Tanh and Erf can
# exceed 1 when the bucket count is below 1 (i.e. zero).
@pytest.mark.parametrize("PenaltyClass", [SigmoidPenalty, LinearPenalty, StepPenalty])
def test_strictly_bounded_penalties_stay_in_unit_interval(PenaltyClass):
    bc_empty = BucketCounter(max_size=5)
    bc_full = BucketCounter(max_size=5, scaffold=20)
    for bc in (bc_empty, bc_full):
        p = PenaltyClass(bc)
        val = p.calculate_penalty("scaffold")
        assert 0.0 <= val <= 1.0, f"{PenaltyClass.__name__} returned {val} outside [0,1]"


@pytest.mark.parametrize("PenaltyClass", [SigmoidPenalty, TanhPenalty, LinearPenalty])
def test_penalty_decreases_as_bucket_fills(PenaltyClass):
    bc = BucketCounter(max_size=10)
    p = PenaltyClass(bc)
    penalties = []
    for i in range(12):
        penalties.append(p.calculate_penalty("scaffold"))
        bc.add("scaffold")
    # overall trend should be non-increasing
    assert (
        penalties[0] >= penalties[-1]
    ), f"{PenaltyClass.__name__}: penalty should not increase as bucket fills"


@pytest.mark.parametrize("PenaltyClass", ALL_PENALTY_CLASSES)
def test_penalty_unknown_scaffold_treated_as_empty(PenaltyClass):
    bc = BucketCounter(max_size=5, known=3)
    p = PenaltyClass(bc)
    val = p.calculate_penalty("completely_new_scaffold")
    assert val > 0.5
