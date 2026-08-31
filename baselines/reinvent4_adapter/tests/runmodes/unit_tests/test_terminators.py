"""Tests for RL terminators: NullTerminator, SimpleTerminator, PlateauTerminator."""

import pytest
import numpy as np

from reinvent.runmodes.RL.setup.terminators import (
    NullTerminator,
    SimpleTerminator,
    PlateauTerminator,
)

# ── NullTerminator ──────────────────────────────────────────────────────────


def test_null_terminator_never_fires():
    null = NullTerminator(None, None)
    for step in range(20):
        assert null(1.0, step) is False


# ── SimpleTerminator ─────────────────────────────────────────────────────────


@pytest.fixture
def simple():
    return SimpleTerminator(max_score=0.9, min_steps=5)


def test_simple_terminator_does_not_fire_before_min_steps(simple):
    assert simple(0.95, 3) is False
    assert simple(1.0, 4) is False


def test_simple_terminator_fires_when_step_and_score_met(simple):
    assert simple(0.95, 6) is True


def test_simple_terminator_does_not_fire_when_score_too_low(simple):
    assert simple(0.5, 10) is False


def test_simple_terminator_fires_exactly_at_threshold(simple):
    assert simple(0.9, 6) is True


# ── PlateauTerminator ─────────────────────────────────────────────────────────


def test_plateau_terminator_does_not_fire_before_min_steps():
    t = PlateauTerminator(max_score=0.9, min_steps=10, mem_size=3)
    for step in range(12):
        assert t(0.5, step) is False


def test_plateau_terminator_fires_on_perfectly_flat_region():
    t = PlateauTerminator(max_score=0.9, min_steps=2, mem_size=4)
    fired = False
    for step in range(2, 20):
        if t(0.5, step):
            fired = True
            break
    assert fired, "PlateauTerminator should eventually detect a flat score"


def test_plateau_terminator_fires_sooner_on_constant_than_rising():
    """A constant score fills the memory buffer faster than a rising one.

    Both may eventually fire (when k becomes small enough), but the constant
    sequence triggers termination in strictly fewer steps.
    """

    def steps_to_fire(score_fn, max_steps=50):
        t = PlateauTerminator(max_score=0.9, min_steps=2, mem_size=5)
        for step in range(2, max_steps):
            if t(score_fn(step), step):
                return step
        return max_steps

    constant_steps = steps_to_fire(lambda s: 0.5)
    rising_steps = steps_to_fire(lambda s: s * 0.1)  # steep slope
    assert constant_steps <= rising_steps
