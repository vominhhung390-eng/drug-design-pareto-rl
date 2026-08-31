import pytest

import numpy as np

from reinvent.runmodes.RL.memories.identical_murcko_scaffold import IdenticalMurckoScaffold
from reinvent.models.model_factory.sample_batch import SmilesState


@pytest.fixture
def diversity_filter():

    diversity_filter = IdenticalMurckoScaffold(
        bucket_size=2,
        minscore=0.4,
        minsimilarity=0.4,
        penalty_multiplier=0.5,
        rdkit_smiles_flags={},
    )

    yield diversity_filter

    pass  # no tear-down


def test_filter_updates_scaffold_memory(diversity_filter):
    scores = [0.64, 0.55, 0.897, 0.737]
    smilies = ["CCc1ccccc1", "CCC=O=O", "CCCCc1ccccc1", "CCc1cnccc1"]
    states = np.array(
        [SmilesState.VALID, SmilesState.INVALID, SmilesState.VALID, SmilesState.DUPLICATE]
    )

    # DF needs all valid SMILES including duplicates
    mask = np.where(
        (states == SmilesState.VALID) | (states == SmilesState.DUPLICATE),
        True,
        False,
    )

    diversity_filter.update_score(scores, smilies, mask)

    # The two valid/duplicate SMILES above the minscore=0.4 threshold should be stored
    assert len(diversity_filter.scaffold_memory) == 2


def test_filter_invalid_smiles_not_stored(diversity_filter):
    """SMILES with INVALID state should not enter scaffold memory."""
    scores = [0.9, 0.1]
    smilies = ["CCC=O=O", "CCC=O=O"]  # both invalid SMILES string
    states = np.array([SmilesState.INVALID, SmilesState.INVALID])

    mask = np.where(
        (states == SmilesState.VALID) | (states == SmilesState.DUPLICATE),
        True,
        False,
    )

    diversity_filter.update_score(scores, smilies, mask)

    assert len(diversity_filter.scaffold_memory) == 0


def test_filter_second_call_same_scaffold_scores_zero(diversity_filter):
    """When the same scaffold fills its bucket a second call should zero the score."""
    # bucket_size=2; three additions of the same scaffold should trigger zeroing on the third
    scores_1 = [0.8, 0.9]
    smilies_1 = ["CCc1ccccc1", "CCc1ccccc1"]  # same scaffold twice (fills bucket)
    mask_1 = np.array([True, True])
    diversity_filter.update_score(scores_1, smilies_1, mask_1)

    scores_2 = [0.7]
    smilies_2 = ["CCc1ccccc1"]  # same scaffold again — bucket now full
    mask_2 = np.array([True])
    diversity_filter.update_score(scores_2, smilies_2, mask_2)

    # score should be zeroed because the SMILES was already seen
    assert scores_2[0] == 0.0
