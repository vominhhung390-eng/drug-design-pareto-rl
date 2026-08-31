from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem


PROJECT = Path(__file__).resolve().parents[2]
HERE = PROJECT / "docking/seed_top10_two_pairs_20260830"
PREP_LIGAND = Path(os.environ.get("MEEKO_PREP_LIGAND", "mk_prepare_ligand"))
EMBED_SEED = 20260829


def prepare(row) -> dict:
    sdf = HERE / "ligands" / f"{row.compound_id}.sdf"
    pdbqt = HERE / "ligands" / f"{row.compound_id}.pdbqt"
    if sdf.exists() and pdbqt.exists():
        return {"compound_id": row.compound_id, "status": "reused", "optimizer": None}
    mol = Chem.AddHs(Chem.MolFromSmiles(row.canonical_smiles))
    if mol is None:
        raise RuntimeError(f"SMILES parse failed for {row.compound_id}")
    digest = hashlib.sha256(row.canonical_smiles.encode("utf-8")).digest()
    base_seed = int((EMBED_SEED + int.from_bytes(digest[:4], "little")) % (2**31 - 1))
    embedded = False
    # Keep deterministic coordinates while allowing a chemically valid molecule
    # that is difficult to embed from the first random-coordinate initialization
    # to be retried without changing the selected candidate.
    # ETKDGv3 is the primary conformer method.  A deterministic
    # non-random-coordinate retry and the older ETDG distance geometry are
    # retained as chemically valid fallbacks for otherwise valid SMILES that
    # are occasionally difficult to embed.
    attempts = [
        (AllChem.ETKDGv3, True),
        (AllChem.ETKDGv3, True),
        (AllChem.ETKDGv3, False),
        (AllChem.ETDG, False),
        (AllChem.ETDG, True),
    ]
    for attempt, (factory, use_random) in enumerate(attempts):
        mol.RemoveAllConformers()
        params = factory()
        params.randomSeed = int((base_seed + attempt * 100003) % (2**31 - 1))
        params.useRandomCoords = use_random
        if hasattr(params, "enforceChirality"):
            params.enforceChirality = False
        if AllChem.EmbedMolecule(mol, params) == 0:
            embedded = True
            break
    if not embedded:
        raise RuntimeError(f"3D embedding failed after deterministic retries for {row.compound_id}")
    if AllChem.MMFFHasAllMoleculeParams(mol):
        AllChem.MMFFOptimizeMolecule(mol, maxIters=1000)
        optimizer = "MMFF94"
    else:
        AllChem.UFFOptimizeMolecule(mol, maxIters=1000)
        optimizer = "UFF"
    mol.SetProp("_Name", row.compound_id)
    mol.SetProp("canonical_smiles", row.canonical_smiles)
    mol.SetProp("target_pair", row.target_pair)
    mol.SetProp("source_seed", str(row.source_seed))
    mol.SetProp("conformer_optimizer", optimizer)
    sdf.parent.mkdir(parents=True, exist_ok=True)
    with sdf.open("w", encoding="utf-8") as handle:
        writer = Chem.SDWriter(handle)
        writer.write(mol)
        writer.close()
    result = subprocess.run(
        [
            str(PREP_LIGAND),
            "-i",
            str(sdf.relative_to(HERE)),
            "-o",
            str(pdbqt.relative_to(HERE)),
            "--charge_model",
            "gasteiger",
            "--rigid_macrocycles",
        ],
        cwd=HERE,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(f"Meeko preparation failed for {row.compound_id}: {result.stderr[-2000:]}")
    return {"compound_id": row.compound_id, "status": "prepared", "optimizer": optimizer}


def main() -> None:
    selected = pd.read_csv(HERE / "selected_compounds.csv", encoding="utf-8-sig")
    results = []
    errors = []
    for row in selected.itertuples(index=False):
        try:
            results.append(prepare(row))
        except Exception as exc:
            errors.append({"compound_id": row.compound_id, "error": str(exc)})
        if len(results) % 20 == 0:
            print(f"prepared={len(results)}/{len(selected)}", flush=True)
    manifest = {
        "selected_total": int(len(selected)),
        "prepared_or_reused": int(len(results)),
        "errors": errors,
        "conformer_seed_base": EMBED_SEED,
        "charge_model": "gasteiger",
        "macrocycle_policy": "rigid_macrocycles",
    }
    (HERE / "ligand_prep_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if errors:
        raise SystemExit(f"ligand preparation failed for {len(errors)} molecules")
    print(f"prepared_or_reused={len(results)}")


if __name__ == "__main__":
    main()
