import csv
import pytest
from rdkit import Chem

from reinvent.runmodes.samplers.run_sampling import run_sampling
from reinvent.runmodes.utils.helpers import set_torch_device

# FIXME: the problem here is that we need to read this from JSON but we can use
#        json_config as a parameter to a function


def check_csv_columns_in_file(filename, num_columns):
    with open(filename, "r") as csv_file:
        reader = csv.reader(csv_file, delimiter=",")

        for i, row in enumerate(reader):
            if i == 0:
                continue

            if len(row) != num_columns:
                return False

            float(row[-1])

    return i


def read_smiles_from_csv(filename, smiles_col=0):
    smiles = []
    with open(filename) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if row:
                smiles.append(row[smiles_col])
    return smiles


@pytest.fixture
def param(request, json_config):
    params = {
        "reinvent": {
            "model_file": json_config["PRIOR_PATH"],
            "num_smiles": 20,
            "smiles_multiplier": 1,
            "smiles_file": None,
            "sample_strategy": None,
            "num_cols": 3,
        },
        "libinvent": {
            "model_file": json_config["LIBINVENT_CHEMBL_PRIOR_PATH"],
            "num_smiles": 10,
            "smiles_multiplier": 2,
            "smiles_file": json_config["LIBINVENT_SMILES_SCAFFOLDS"],
            "sample_strategy": None,
            "num_cols": 5,
        },
        "linkinvent": {
            "model_file": json_config["LINKINVENT_CHEMBL_PRIOR_PATH"],
            "num_smiles": 10,
            "smiles_multiplier": 1,
            "smiles_file": json_config["LINKINVENT_SMILES_WARHEADS"],
            "sample_strategy": "multinomial",
            "num_cols": 5,
        },
        "mol2mol-multi": {
            "model_file": json_config["MOLFORMER_PRIOR_PATH"],
            "num_smiles": 5,
            "smiles_multiplier": 3,
            "smiles_file": json_config["MOLFORMER_SMILES_SET_PATH"],
            "sample_strategy": "multinomial",
            "num_cols": 5,
            "temperature": 1,
        },
        "mol2mol-beam": {
            "model_file": json_config["MOLFORMER_PRIOR_PATH"],
            "num_smiles": 1,
            "smiles_multiplier": 3,
            "smiles_file": json_config["MOLFORMER_SMILES_SET_PATH"],
            "sample_strategy": "beamsearch",
            "num_cols": 5,
        },
    }

    return params[request.param]


IDs = ["reinvent", "libinvent", "linkinvent", "mol2mol-multi", "mol2mol-beam"]


@pytest.fixture
def setup(tmp_path, pytestconfig):
    device = pytestconfig.getoption("device")
    set_torch_device(device)

    output_filename = tmp_path / "samplers.smi"

    config = {
        "parameters": {
            "output_file": str(output_filename),
            "unique_molecules": False,
            "randomize_smiles": True,
        }
    }

    return output_filename, config


@pytest.mark.integration
@pytest.mark.parametrize("param", IDs, ids=IDs, indirect=True)
def test_run_sampling_with_likelihood(param, setup, pytestconfig):
    output_filename, config = setup
    device = pytestconfig.getoption("device")

    num_cols = param["num_cols"]

    cfg_params = config["parameters"]
    cfg_params["model_file"] = param["model_file"]
    cfg_params["num_smiles"] = param["num_smiles"]
    cfg_params["smiles_file"] = param["smiles_file"]
    cfg_params["sample_strategy"] = param.get("sample_strategy", "")

    run_sampling(config, device, None)

    num_lines = check_csv_columns_in_file(output_filename, num_columns=num_cols)
    num_smiles = param["smiles_multiplier"] * param["num_smiles"]

    assert num_lines == num_smiles


@pytest.mark.integration
@pytest.mark.parametrize("param", IDs, ids=IDs, indirect=True)
def test_run_sampling_smiles_validity(param, setup, pytestconfig):
    """At least 50 % of sampled SMILES must be chemically valid."""
    output_filename, config = setup
    device = pytestconfig.getoption("device")

    cfg_params = config["parameters"]
    cfg_params["model_file"] = param["model_file"]
    cfg_params["num_smiles"] = param["num_smiles"]
    cfg_params["smiles_file"] = param["smiles_file"]
    cfg_params["sample_strategy"] = param.get("sample_strategy", "")

    run_sampling(config, device, None)

    smiles = read_smiles_from_csv(output_filename, smiles_col=0)
    assert len(smiles) > 0

    valid = [s for s in smiles if Chem.MolFromSmiles(s) is not None]
    validity_rate = len(valid) / len(smiles)
    assert (
        validity_rate >= 0.5
    ), f"Only {validity_rate:.0%} of sampled SMILES are valid (expected ≥50%)"


@pytest.mark.integration
def test_run_sampling_reinvent_with_inception_smiles(json_config, tmp_path, pytestconfig):
    """Reinvent sampler accepts an inception SMILES file as input."""
    device = pytestconfig.getoption("device")
    set_torch_device(device)

    output_filename = tmp_path / "reinvent_inception.smi"
    num_smiles = 10

    config = {
        "parameters": {
            "model_file": json_config["PRIOR_PATH"],
            "output_file": str(output_filename),
            "num_smiles": num_smiles,
            "smiles_file": json_config["REINVENT_INCEPTION_SMI"],
            "unique_molecules": False,
            "randomize_smiles": False,
            "sample_strategy": None,
        }
    }

    run_sampling(config, device, None)

    num_lines = check_csv_columns_in_file(output_filename, num_columns=3)
    assert num_lines == num_smiles
