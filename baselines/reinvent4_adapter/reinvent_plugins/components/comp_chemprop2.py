"""Compute scores with ChemProp v2.x

Single-task model (target_column is optional):

[[component.ChemProp2.endpoint]]
name = "ChemProp Score"
weight = 0.7
param.model_path = "best.pt"
param.featurizers = ["V1RDKit2D"]  # disables scaling!
param.featurizers_options = [{}]
transform.type = "reverse_sigmoid"
transform.high = -5.0
transform.low = -35.0
transform.k = 0.4

Multi-task model (target_column required, one endpoint per target):

[[component.ChemProp2.endpoint]]
name = "pIC50"
weight = 0.7
param.model_path = "best.pt"
param.target_column = "pIC50"

[[component.ChemProp2.endpoint]]
name = "logD"
weight = 0.3
param.model_path = "best.pt"
param.target_column = "logD"
"""

from __future__ import annotations

__all__ = ["ChemProp2"]
from pathlib import Path
from typing import Any, Optional
import logging
logging.getLogger("lightning.pytorch").setLevel(logging.FATAL)  # shut up lightning!

import chemprop

if chemprop.__version__.split(".")[0] != "2":
    raise ImportError(f"only Chemprop v2 supported")

from chemprop import models, data, featurizers
from chemprop.models.utils import load_output_columns
import torch
from lightning.pytorch import Trainer
from rdkit import Chem
import numpy as np
from pydantic import Field
from pydantic.dataclasses import dataclass

from .component_results import ComponentResults
from .add_tag import add_tag
from ..normalize import normalize_smiles

logger = logging.getLogger("reinvent")


@add_tag("__parameters")
@dataclass
class Parameters:
    """Parameters for the scoring component

    Note that all parameters are always lists because components can have
    multiple endpoints and so all the parameters from each endpoint is
    collected into a list.  This is also true in cases where there is only one
    endpoint.
    """

    model_path: list[str]
    featurizers: list[list[str]]
    featurizers_options: list[list[dict[str, Any]]]
    target_column: Optional[list[str]] = Field(default_factory=lambda: [None])


@add_tag("__component")
class ChemProp2:
    def __init__(self, params: Parameters):
        self.smiles_type = "rdkit_smiles"

        model_path = Path(params.model_path[0])
        self.models = []

        if model_path.is_dir():
            pt_files = sorted(model_path.rglob("*.pt"))
            for path in pt_files:
                self.models.append(models.load_model(path))
            first_pt = pt_files[0]
        else:
            self.models = [models.load_model(model_path)]
            first_pt = model_path

        self.featurizers = params.featurizers[0]
        self.featurizers_options = params.featurizers_options[0]

        n_tasks = self.models[0].n_tasks
        output_columns = load_output_columns(first_pt)

        # output_columns may be a list[str] or None (old checkpoints)
        if isinstance(output_columns, (list, tuple)) and not isinstance(output_columns[0], str):
            output_columns = None  # MolAtomBond models not supported here

        self.column_indices = []

        for target_col in params.target_column:
            if target_col is None:
                if n_tasks == 1:
                    self.column_indices.append(0)
                else:
                    cols = output_columns if output_columns else list(range(n_tasks))
                    raise ValueError(
                        f"target_column is required for multi-task models "
                        f"(n_tasks={n_tasks}), available columns: {cols}"
                    )
            else:
                if output_columns is None:
                    raise ValueError(
                        f"target_column '{target_col}' specified but model "
                        f"checkpoint has no stored column names"
                    )
                if target_col not in output_columns:
                    raise ValueError(
                        f"target_column '{target_col}' not found in model, "
                        f"available columns: {output_columns}"
                    )
                self.column_indices.append(output_columns.index(target_col))

        self.number_of_endpoints = len(self.column_indices)

        logger.info(f"Using ChemProp version {chemprop.__version__} with "
                f"featurizers {':'.join(self.featurizers)}, options "
                f"{self.featurizers_options} for model\n{self.models}")
        if output_columns:
            selected = [output_columns[i] for i in self.column_indices]
            logger.info(f"Selected target columns: {selected} (indices: {self.column_indices})")

    @normalize_smiles
    def __call__(self, smilies: list[str]) -> ComponentResults:
        mols = [Chem.MolFromSmiles(smiles) for smiles in smilies]
        molecule_features = []
        disable_scaling = False

        for featurizer, featurizer_options in zip(self.featurizers, self.featurizers_options):
            if featurizer == "V1RDKit2D" or featurizer == "RDKit2D":
                disable_scaling = True  # assumes scaling is not needed for any feature

            molecule_featurizer_class = getattr(featurizers, f"{featurizer}Featurizer", None)

            if "Morgan" in featurizer:
                molecule_featurizer = molecule_featurizer_class(**featurizer_options)
            else:
                molecule_featurizer = molecule_featurizer_class()

            molecule_features.append([molecule_featurizer(mol) for mol in mols])

        if molecule_features:
            datapoints = []

            for mol, features in zip(mols, zip(*molecule_features)):
                datapoints.append(data.MoleculeDatapoint(mol, x_d=np.hstack(features)))
        else:
            datapoints = [data.MoleculeDatapoint(mol) for mol in mols]

        graph_featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        dataset = data.MoleculeDataset(datapoints, featurizer=graph_featurizer, n_workers=7)

        if disable_scaling:
            logger.info("scaling disabled")
            dataset.normalize_inputs(scaler=None)

        dataloader = data.build_dataloader(dataset, shuffle=False)

        all_preds = []

        with torch.inference_mode():
            for model in self.models:
                trainer = Trainer(logger=None, enable_progress_bar=False, accelerator="auto", devices=1)
                preds = trainer.predict(model, dataloader)
                # preds is a list of tensors, one per batch, each (batch_size, n_tasks)
                model_preds = np.concatenate(preds, axis=0)  # (N, n_tasks)

                if model_preds.ndim == 1:
                    model_preds = model_preds[:, np.newaxis]  # (N,) -> (N, 1)

                all_preds.append(model_preds)

        # all_preds: list of M arrays each (N, T) — average across ensemble
        scores = np.mean(all_preds, axis=0)  # (N, T)

        return ComponentResults(
            [scores[:, idx].astype(float) for idx in self.column_indices]
        )
