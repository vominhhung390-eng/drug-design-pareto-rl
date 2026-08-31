"""Tests for ScaffoldSimilarity._find_similar_scaffold."""

import pytest

from reinvent.runmodes.RL.memories.scaffold_similarity import ScaffoldSimilarity

# Benzene and toluene — atom-pair Tanimoto similarity is moderate
BENZENE = "c1ccccc1"
TOLUENE = "Cc1ccccc1"
CYCLOHEXANE = "C1CCCCC1"  # very different from benzene


def _make_filter(minsimilarity):
    return ScaffoldSimilarity(
        bucket_size=5,
        minscore=0.4,
        minsimilarity=minsimilarity,
        penalty_multiplier=0.5,
        rdkit_smiles_flags={},
    )


def test_empty_scaffold_returns_scaffold_unchanged():
    sf = _make_filter(minsimilarity=0.5)
    result = sf._find_similar_scaffold("")
    assert result == ""


def test_first_scaffold_always_registered():
    sf = _make_filter(minsimilarity=0.5)
    sf._find_similar_scaffold(BENZENE)
    assert BENZENE in sf.scaffold_fingerprints


def test_identical_scaffold_does_not_add_new_entry():
    sf = _make_filter(minsimilarity=0.5)
    sf._find_similar_scaffold(BENZENE)
    sf._find_similar_scaffold(BENZENE)
    assert len(sf.scaffold_fingerprints) == 1


def test_very_similar_scaffold_merged_at_low_threshold():
    sf = _make_filter(minsimilarity=0.01)  # very permissive — almost anything merges
    sf._find_similar_scaffold(BENZENE)
    result = sf._find_similar_scaffold(TOLUENE)
    # at this threshold toluene should be merged into the benzene bucket
    assert result == BENZENE


def test_dissimilar_scaffold_gets_new_bucket_at_high_threshold():
    sf = _make_filter(minsimilarity=0.99)  # very strict — almost nothing merges
    sf._find_similar_scaffold(BENZENE)
    result = sf._find_similar_scaffold(CYCLOHEXANE)
    assert result == CYCLOHEXANE
    assert len(sf.scaffold_fingerprints) == 2


def test_find_similar_scaffold_returns_string():
    sf = _make_filter(minsimilarity=0.5)
    result = sf._find_similar_scaffold(BENZENE)
    assert isinstance(result, str)
