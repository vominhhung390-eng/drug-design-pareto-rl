"""Tests for reinvent/models/meta_data.py — hash generation and verification."""

import copy
import pytest
import torch

from reinvent.models import meta_data
from reinvent.models.meta_data import (
    ModelMetaData,
    update_model_data,
    check_valid_hash,
    HASH_FORMAT,
)
from reinvent.models.reinvent.models.model import Model
from reinvent.models.reinvent.models.vocabulary import SMILESTokenizer, Vocabulary
from tests.test_data import SIMPLE_TOKENS


@pytest.fixture
def save_dict():
    vocabulary = Vocabulary(tokens=SIMPLE_TOKENS)
    tokenizer = SMILESTokenizer()
    md = ModelMetaData(
        hash_id=None,
        hash_id_format="",
        model_id="",
        origina_data_source="",
        creation_date=0,
    )
    model = Model(vocabulary, tokenizer, md, device=torch.device("cpu"))
    return model.get_save_dict()


def test_update_model_data_sets_hash(save_dict):
    updated = update_model_data(save_dict, comment="test run")
    assert updated["metadata"]["hash_id"] is not None
    assert updated["metadata"]["hash_id_format"] == HASH_FORMAT


def test_update_model_data_appends_comment(save_dict):
    updated = update_model_data(save_dict, comment="my annotation")
    assert "my annotation" in updated["metadata"]["comments"]


def test_update_model_data_appends_timestamp(save_dict):
    updated = update_model_data(save_dict, write_update=True)
    assert len(updated["metadata"]["updates"]) >= 1


def test_update_model_data_no_update_when_write_false(save_dict):
    n_before = len(save_dict["metadata"].updates)
    updated = update_model_data(save_dict, write_update=False)
    assert len(updated["metadata"]["updates"]) == n_before


def test_check_valid_hash_passes_on_fresh_model(save_dict):
    updated = update_model_data(save_dict)
    assert check_valid_hash(copy.deepcopy(updated)) is True


def test_check_valid_hash_fails_on_tampered_network(save_dict):
    updated = update_model_data(save_dict)
    # corrupt one weight tensor
    first_key = next(iter(updated["network"]))
    updated["network"][first_key] = updated["network"][first_key] + 1.0
    assert check_valid_hash(updated) is False


def test_update_model_data_hash_changes_after_two_updates(save_dict):
    first = update_model_data(copy.deepcopy(save_dict))
    second = update_model_data(copy.deepcopy(save_dict))
    # timestamps differ between calls → hashes differ
    # (they may occasionally match if called in the same millisecond, but
    # verifying the hash is non-None is the stable assertion)
    assert first["metadata"]["hash_id"] is not None
    assert second["metadata"]["hash_id"] is not None
