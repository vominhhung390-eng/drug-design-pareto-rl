#!/usr/bin/env python
"""Online EGFR/VEGFR2 V4.1 oracle for predictor-as-reward experiments.

The deployed classifiers return probabilities.  The generation code expects
the historical pActivity interval [3, 10], so probabilities are mapped with
the monotone affine transform ``3 + 7 * p``.  The existing HV normalization
therefore recovers the original probability exactly, without changing Pareto
dominance or objective balance.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdFingerprintGenerator


PROBABILITY_LOWER = 3.0
PROBABILITY_SPAN = 7.0


class V41TwoTargetObjectiveCalculator:
    """Exact online deployment of the frozen V4.1 predictor pair."""

    def __init__(self, project_root: str | Path, device: str = "cuda"):
        self.project_root = Path(project_root).resolve()
        self.device = torch.device(
            device if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu"
        )
        torch.set_float32_matmul_precision("high")

        # Chemprop is an optional dependency for the original RF experiment,
        # so it is imported only when this V4.1 oracle is selected.
        from chemprop.featurizers import MoleculeFeaturizerRegistry
        from chemprop.models.utils import load_model

        model_root = (
            self.project_root
            / "results"
            / "predictor_v41_20260802"
            / "egfr_bindingdb_external_v2"
        )
        self.dmpnn_models = [
            load_model(path, False).to(self.device).eval()
            for path in sorted((model_root / "dmpnn").rglob("best.pt"))
        ]
        self.morgan_models = [
            load_model(path, False).to(self.device).eval()
            for path in sorted((model_root / "dmpnn_morgan").rglob("best.pt"))
        ]
        if len(self.dmpnn_models) != 5 or len(self.morgan_models) != 5:
            raise RuntimeError(
                "V4.1 EGFR deployment requires five D-MPNN and five Morgan-DMPNN members"
            )
        self.morgan_descriptor = MoleculeFeaturizerRegistry["morgan_binary"]()

        reference_path = (
            self.project_root
            / "results"
            / "predictor_retraining_v3_20260731"
            / "data"
            / "egfr"
            / "single_protein_assay_ge10"
            / "development_through_2023.csv"
        )
        reference = pd.read_csv(reference_path)
        reference = reference[
            (reference.pactivity <= 5.5) | (reference.pactivity >= 7.5)
        ].reset_index(drop=True)
        self.knn_labels = (reference.pactivity >= 7.5).astype(int).to_numpy()
        self.knn_generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=1, fpSize=2048
        )
        self.knn_reference = [
            self.knn_generator.GetFingerprint(Chem.MolFromSmiles(smiles))
            for smiles in reference.smiles
        ]

        deployment_path = (
            self.project_root
            / "results"
            / "predictor_v41_20260802"
            / "deployment"
            / "vegfr2_extratrees.pkl"
        )
        if not deployment_path.exists():
            raise FileNotFoundError(
                f"Build the frozen VEGFR2 deployment model first: {deployment_path}"
            )
        self.vegfr2_model = joblib.load(deployment_path)
        # Avoid CPU oversubscription when several paired seeds run in parallel.
        self.vegfr2_model.n_jobs = 4
        self.vegfr2_generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=2, fpSize=2048, includeChirality=True
        )
        self._cache: dict[str, np.ndarray] = {}

    @staticmethod
    def probabilities_to_reward(probabilities: np.ndarray) -> np.ndarray:
        probabilities = np.clip(np.asarray(probabilities, dtype=np.float32), 0.0, 1.0)
        return PROBABILITY_LOWER + PROBABILITY_SPAN * probabilities

    def _chemprop_batch(self, smiles: Sequence[str], morgan: bool) -> np.ndarray:
        from chemprop import data
        from chemprop.cli.utils import make_dataset

        datapoints = []
        for value in smiles:
            mol = Chem.MolFromSmiles(value)
            x_d = self.morgan_descriptor(mol) if morgan else None
            datapoints.append(data.MoleculeDatapoint(mol=mol, x_d=x_d, name=value))
        dataset = make_dataset(datapoints, n_workers=0)
        loader = data.build_dataloader(
            dataset, batch_size=len(datapoints), num_workers=0, shuffle=False
        )
        batch = next(iter(loader))
        models = self.morgan_models if morgan else self.dmpnn_models
        batch = models[0].transfer_batch_to_device(batch, self.device, 0)
        members = []
        with torch.inference_mode():
            for model in models:
                members.append(
                    model.predict_step(batch, 0).detach().cpu().numpy().reshape(-1)
                )
        return np.mean(np.vstack(members), axis=0)

    def _knn_batch(self, smiles: Sequence[str]) -> np.ndarray:
        output = np.empty(len(smiles), dtype=np.float32)
        for index, value in enumerate(smiles):
            fingerprint = self.knn_generator.GetFingerprint(Chem.MolFromSmiles(value))
            similarities = np.asarray(
                DataStructs.BulkTanimotoSimilarity(fingerprint, self.knn_reference),
                dtype=float,
            )
            nearest = np.argpartition(similarities, -20)[-20:]
            weights = np.maximum(similarities[nearest], 1e-6) ** 6
            output[index] = np.average(self.knn_labels[nearest], weights=weights)
        return output

    def _vegfr2_features(self, smiles: Sequence[str]) -> np.ndarray:
        output = np.empty((len(smiles), 2058), dtype=np.float32)
        for index, value in enumerate(smiles):
            mol = Chem.MolFromSmiles(value)
            fingerprint = self.vegfr2_generator.GetFingerprint(mol)
            DataStructs.ConvertToNumpyArray(fingerprint, output[index, :2048])
            output[index, 2048:] = (
                Descriptors.MolWt(mol),
                Crippen.MolLogP(mol),
                Descriptors.TPSA(mol),
                Lipinski.NumHDonors(mol),
                Lipinski.NumHAcceptors(mol),
                Lipinski.NumRotatableBonds(mol),
                Lipinski.RingCount(mol),
                Lipinski.FractionCSP3(mol),
                Lipinski.HeavyAtomCount(mol),
                Chem.GetFormalCharge(mol),
            )
        return output

    def predict_probabilities_batch(self, smiles: Sequence[str]) -> np.ndarray:
        if not smiles:
            return np.empty((0, 2), dtype=np.float32)
        dmpnn = self._chemprop_batch(smiles, morgan=False)
        morgan = self._chemprop_batch(smiles, morgan=True)
        knn = self._knn_batch(smiles)
        egfr = 0.7 * dmpnn + 0.1 * morgan + 0.2 * knn
        vegfr2 = self.vegfr2_model.predict_proba(self._vegfr2_features(smiles))[:, 1]
        return np.column_stack([egfr, vegfr2]).astype(np.float32)

    def calculate_scores_batch(self, smiles_list: Sequence[str]) -> np.ndarray:
        if not smiles_list:
            return np.empty((0, 2), dtype=np.float32)
        missing = list(dict.fromkeys(value for value in smiles_list if value not in self._cache))
        if missing:
            rewards = self.probabilities_to_reward(
                self.predict_probabilities_batch(missing)
            )
            self._cache.update({value: score for value, score in zip(missing, rewards)})
        return np.vstack([self._cache[value] for value in smiles_list]).astype(np.float32)

    def calculate_scores(self, smiles: str) -> np.ndarray:
        return self.calculate_scores_batch([smiles])[0]
