REINVENT4 input examples
========================

A set of input files explaining each run mode in REINVENT4.  These examples
are meant for demonstration only and will need to be adjusted for actual use.


Format
------

The file format is TOML; see https://toml.io/en/.  Each file can also be
written out as JSON by setting the top-level `json_out_config` key.


Run modes
---------

| TOML file                    | Description                                                    |
|------------------------------|----------------------------------------------------------------|
| sampling.toml                | Sample molecules from a prior or fine-tuned model.            |
| scoring.toml                 | Score an existing set of compounds without a model.            |
| transfer\_learning.toml      | Fine-tune a model on a set of input SMILES (transfer learning).|
| staged\_learning.toml        | Reinforcement / curriculum learning (one or more stages).      |
| enumeration.toml             | Enumerate and score peptide or molecular variants.             |
| data\_pipeline.toml          | Pre-process and filter a SMILES dataset.                       |


Supporting files
----------------

| TOML file                      | Description                                               |
|--------------------------------|-----------------------------------------------------------|
| scoring\_components\_example.toml | Extended scoring component examples.                   |
| stage1\_scoring.toml           | Scoring setup loaded externally by staged\_learning.toml. |
| stage2\_scoring.toml           | Scoring setup loaded externally by staged\_learning.toml. |


SMILES seed files
-----------------

| File            | Used by                                              |
|-----------------|------------------------------------------------------|
| scaffolds.smi   | LibInvent (1 scaffold per line with attachment points)|
| warheads.smi    | LinkInvent (2 warheads per line separated by `\|`)   |
| mol2mol.smi     | Mol2Mol (1 compound per line)                        |
| pepinvent.smi   | Pepinvent (1 peptide SMILES per line)                |
