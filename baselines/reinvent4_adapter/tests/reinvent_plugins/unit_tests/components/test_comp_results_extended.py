"""Additional tests for SmilesAssociatedComponentResults and ComponentResults."""

import numpy as np
import pytest

from reinvent_plugins.components.component_results import (
    ComponentResults,
    SmilesAssociatedComponentResults,
    SmilesResult,
)

SMILIES = ["CCO", "c1ccccc1", "CCC"]
SCORES_1 = [np.array([1.0, 2.0, 3.0])]
SCORES_2 = [np.array([4.0, 5.0, 6.0]), np.array([7.0, 8.0, 9.0])]


# ── fetch_scores(transpose=False) — default ──────────────────────────────────


def test_fetch_scores_untransposed_shape():
    cr = ComponentResults(scores=SCORES_1)
    sar = SmilesAssociatedComponentResults(component_results=cr, smiles=SMILIES)
    result = sar.fetch_scores(SMILIES, transpose=False)
    assert len(result) == len(SMILIES)
    assert result[0] == (1.0,)
    assert result[1] == (2.0,)


# ── fetch_scores(transpose=True) ─────────────────────────────────────────────


def test_fetch_scores_transposed_single_endpoint():
    cr = ComponentResults(scores=SCORES_1)
    sar = SmilesAssociatedComponentResults(component_results=cr, smiles=SMILIES)
    result = sar.fetch_scores(SMILIES, transpose=True)
    # transpose=True → first dimension is endpoints
    assert len(result) == 1
    assert tuple(result[0]) == (1.0, 2.0, 3.0)


def test_fetch_scores_transposed_two_endpoints():
    cr = ComponentResults(scores=SCORES_2)
    sar = SmilesAssociatedComponentResults(component_results=cr, smiles=SMILIES)
    result = sar.fetch_scores(SMILIES, transpose=True)
    assert len(result) == 2
    assert tuple(result[0]) == (4.0, 5.0, 6.0)
    assert tuple(result[1]) == (7.0, 8.0, 9.0)


# ── metadata per-SMILES ───────────────────────────────────────────────────────


def test_smiles_result_metadata_associated_correctly():
    metadata = {"confidence": [0.9, 0.8, 0.7]}
    cr = ComponentResults(scores=SCORES_1, metadata=metadata)
    sar = SmilesAssociatedComponentResults(component_results=cr, smiles=SMILIES)
    assert sar["CCO"].metadata == {"confidence": 0.9}
    assert sar["c1ccccc1"].metadata == {"confidence": 0.8}
    assert sar["CCC"].metadata == {"confidence": 0.7}


def test_smiles_result_score_correct():
    cr = ComponentResults(scores=SCORES_1)
    sar = SmilesAssociatedComponentResults(component_results=cr, smiles=SMILIES)
    assert sar["CCO"].score == (1.0,)
    assert sar["CCC"].score == (3.0,)


# ── __getitem__ for unknown SMILES ────────────────────────────────────────────


def test_getitem_unknown_smiles_returns_none():
    cr = ComponentResults(scores=SCORES_1)
    sar = SmilesAssociatedComponentResults(component_results=cr, smiles=SMILIES)
    assert sar["CCCCCCCCCC"] is None


# ── update_scores ─────────────────────────────────────────────────────────────


def test_update_scores_adds_new_smiles():
    cr = ComponentResults(scores=SCORES_1)
    sar = SmilesAssociatedComponentResults(component_results=cr, smiles=SMILIES)
    sar.update_scores(["CCCC"], [[99.0]])
    assert sar["CCCC"].score == (99.0,)


# ── create_from_scores ────────────────────────────────────────────────────────


def test_create_from_scores():
    sar = SmilesAssociatedComponentResults.create_from_scores(
        smiles=["CCO", "CCC"], scores=[[0.5, 0.8]]
    )
    assert sar["CCO"].score == (0.5,)
    assert sar["CCC"].score == (0.8,)


def test_create_from_scores_empty():
    sar = SmilesAssociatedComponentResults.create_from_scores(smiles=[], scores=[[]])
    assert len(sar.data) == 0
