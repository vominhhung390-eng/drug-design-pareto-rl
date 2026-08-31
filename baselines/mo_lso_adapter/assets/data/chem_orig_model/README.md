# Original Model Data

The upstream pretrained-model train/validation/vocabulary files are intentionally omitted.
This reproduction trains MO-LSO from random initialization on
`data/train_smiles_only.txt`; `adapter_prepare_common_data.py` regenerates the
formal vocabulary and tensors under `results/baselines/mo_lso/data/`.
