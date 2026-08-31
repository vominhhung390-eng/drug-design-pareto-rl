import itertools

import pytest
import torch
from rdkit import Chem
from torch import tensor

from reinvent.models.model_factory.sample_batch import SampleBatch, BatchRow, SmilesState
from reinvent.runmodes.enumeration.aa_enumerator import AAEnumerator
from reinvent.runmodes.enumeration.validation import AAEnumerationConfig


@pytest.fixture(scope="module")
def setup_mock_files(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("mock_data")
    library_file = tmp_dir / "mock_library.csv"
    peptide_file = tmp_dir / "mock_peptide.smi"

    library_file.write_text(
        "Name,SMILES\n"
        "R,N=C(N)NCCC[C@H](N)C(=O)\n"
        "H,N[C@@H](Cc1c[nH]cn1)C(=O)\n"
        "A,N[C@@H](C)C(=O)\n"
    )
    peptide_file.write_text("N[C@@H](CS)C(=O)|?|N[C@@H](C)C(=O)|?|N[C@@H](C)C(=O)O\n")

    yield library_file, peptide_file


@pytest.fixture
def setup_test_environment(setup_mock_files, tmp_path):
    library_file, peptide_file = setup_mock_files
    config = AAEnumerationConfig(
        parameters={
            "smiles_file": str(peptide_file),
            "amino_acid_library_file": str(library_file),
            "aa_names_column": "Name",
            "smiles_column": "SMILES",
            "output_csv": str(tmp_path / "output.csv"),
            "batch_size": 2,
        },
        scoring={"type": "geometric_mean", "component": []},
    )
    enumerator = AAEnumerator(config)
    return config, enumerator


def test_aa_enumerator_create_iterator(setup_test_environment):
    _, enumerator = setup_test_environment
    iterator = enumerator._create_iterator(enumerator.input_smilies)
    assert iterator is not None, "Iterator creation failed"
    assert getattr(enumerator, "num_mask", None) in (1, 2), "Unexpected number of masked positions"
    assert len(list(itertools.islice(iterator, 0, 1))) == 1, "Iterator did not yield combinations"


def test_aa_enumerator_convert_to_smiles_output(setup_test_environment):
    _, enumerator = setup_test_environment
    smiles_output = enumerator._convert_to_smiles_output(("R", "H"))
    assert isinstance(smiles_output, str), "Output is not a string"
    assert "|" in smiles_output, "Output format is incorrect"


def test_aa_enumerator_create_complete_mol(setup_test_environment):
    _, enumerator = setup_test_environment
    sample = BatchRow(
        input="N[C@@H](CS)C(=O)|?|N[C@@H](C)C(=O)O",
        output="N=C(N)NCCC[C@H](N)C(=O)|N[C@@H](Cc1c[nH]cn1)C(=O)",
        smiles="",
        nll=0.0,
        state=SmilesState(1),
    )
    mol, complete_smiles = enumerator._create_complete_mol(sample)
    assert isinstance(mol, Chem.rdchem.Mol), "Output is not an RDKit molecule"
    assert isinstance(complete_smiles, str), "Complete SMILES is not a string"
    assert "?" not in complete_smiles, "Masked token was not replaced"


def test_aa_enumerator_join_fragments(setup_test_environment):
    _, enumerator = setup_test_environment
    sample_batch = SampleBatch(
        items1=["N[C@@H](CS)C(=O)|?|N[C@@H](C)C(=O)O", "N[C@@H](CS)C(=O)|?|N[C@@H](C)C(=O)O"],
        items2=["R|H", "A|H"],
        nlls=tensor([0.0, 0.0]),
    )
    mols, joined_batch = enumerator._join_fragments(sample_batch)
    assert len(mols) == len(sample_batch.items1), "Number of molecules does not match input size"
    assert all(isinstance(mol, (Chem.rdchem.Mol, type(None))) for mol in mols), "Invalid molecule type"
    assert len(joined_batch.items1) == len(sample_batch.items1), "Joined batch size mismatch"


def test_aa_enumerator_sample(setup_test_environment):
    _, enumerator = setup_test_environment
    batch_size = 2
    sample_batch = enumerator.sample(batch_size)
    assert sample_batch is not None, "Sample batch is None"
    assert len(sample_batch.items1) == batch_size, "Sample batch size is incorrect"
    assert len(sample_batch.items2) == batch_size, "Sample batch output size is incorrect"
    assert all(isinstance(smiles, str) for smiles in sample_batch.items2), "Output SMILES are not strings"
    assert all(isinstance(smiles, str) for smiles in sample_batch.items1), "Input SMILES are not strings"
    assert isinstance(sample_batch, SampleBatch), "Returned object is not a SampleBatch"


def test_aa_enumerator_sample_simple(tmp_path):
    peptide_file = tmp_path / "peptide.smi"
    library_file = tmp_path / "library.csv"
    peptide_file.write_text("?C?\n")
    library_file.write_text("Name,SMILES\nA,N[C@@H](C)C(=O)\nB,N[C@@H](CC)C(=O)\n")

    config = AAEnumerationConfig(
        parameters={
            "smiles_file": str(peptide_file),
            "amino_acid_library_file": str(library_file),
            "aa_names_column": "Name",
            "smiles_column": "SMILES",
            "output_csv": str(tmp_path / "out.csv"),
            "batch_size": 2,
        },
        scoring={"type": "geometric_mean", "component": []},
    )
    enumerator = AAEnumerator(config)
    batch_size = 2
    result = enumerator.sample(batch_size)
    assert len(result.items1) == batch_size
    assert len(result.items2) == batch_size
    assert torch.equal(result.nlls, torch.zeros(batch_size))
