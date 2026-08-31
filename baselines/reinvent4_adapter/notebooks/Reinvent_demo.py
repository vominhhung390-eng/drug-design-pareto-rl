# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # _De novo_ Reinforcement Learning with the classical Reinvent prior
#
# **IMPORTANT** Run this notebook in the same environment you have installed REINVENT in.
#
# ## What this tutorial teaches
#
# This notebook walks through a complete _de novo_ reinforcement learning (RL) run using
# REINVENT.  By the end you will know how to:
#
# - Write a TOML configuration file for a staged RL run
# - Run REINVENT from within a notebook
# - Interpret the TensorBoard output to judge whether training is going well
# - Extract the raw data from TensorBoard and the CSV summary file for further analysis
#
# ## Background: what is generative molecular design?
#
# Traditional drug discovery screens large libraries of existing compounds.  Generative
# molecular design turns the problem around: instead of searching a fixed library, we train
# a machine-learning model to _propose_ new molecules that satisfy a set of desired
# properties.
#
# In this tutorial, REINVENT uses a **recurrent neural network (RNN)** language model that generates molecules
# as SMILES strings — one atom or bond token at a time — much like a language model generates
# text.  The model assigns a probability to every possible next token, so we can talk about
# the **likelihood** of a complete molecule: the product of all per-token probabilities.  It
# is more convenient to work with the **negative log-likelihood (NLL)**: a _low_ NLL means
# the model considers the molecule _likely_; a _high_ NLL means the molecule is surprising
# to the model.
#
# ## Background: Reinforcement Learning in REINVENT
#
# The generative model starts as a **prior**: a model pre-trained on a large database of
# drug-like molecules (here, ChEMBL data) so it already knows how to write chemically valid
# SMILES.  During RL, a copy of the prior — called the **agent** — is optimised to generate
# molecules that score well according to a user-defined **scoring function**.  The prior is
# kept frozen and acts as a reference that prevents the agent from "forgetting" basic chemistry
# (a phenomenon sometimes called _mode collapse_).
#
# At every RL step:
# 1. The agent samples a batch of SMILES.
# 2. Each SMILES is scored by the scoring function (a value between 0 and 1).
# 3. The **DAP** (Direct Augmented Posterior) reward function combines the prior NLL, the
#    agent NLL, and the score into a training signal.
# 4. The agent's weights are updated via stochastic gradient descent (SGD).
#
# Over many steps the agent learns to generate molecules that increasingly resemble the
# target profile described by the scoring function.
#
# > **Further reading**
# > - Parameter reference: [`configs/PARAMS.md`](../configs/PARAMS.md)
# > - Scoring components reference: [`configs/SCORING.md`](../configs/SCORING.md)
# > - Full annotated example config: [`configs/staged_learning.toml`](../configs/staged_learning.toml)
# > - REINVENT 4 paper: https://doi.org/10.1186/s13321-024-00812-5 (open access)

# ## Prerequisites
#
# Make sure all required packages are installed in the active Python environment before
# running any further cells.

# +
import importlib

for pkg in ["reinvent", "tensorboard", "mols2grid", "seaborn", "ipywidgets"]:
    found = importlib.util.find_spec(pkg) is not None
    print(f"{'OK     ' if found else 'MISSING'} {pkg}")

# +
import os
import shutil

import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import reinvent
from reinvent.notebooks import load_tb_data, plot_scalars, get_image, create_mol_grid

# %load_ext tensorboard
# -

# ## Device selection
#
# REINVENT can run on an **NVIDIA/AMD/Intel/Apple GPU** (strongly recommended for transfer learning
# and model training) or on the **CPU**.  For RL runs the scoring components dominate
# runtime and often run on the CPU anyway, so a GPU is less critical here.  The cell
# below detects the available hardware automatically — you do not need to edit it.
#
# Here a table mapping GPU hardware to their device names in PyTorch.
#
# |GPU    | device name |
# --------|-------------|
# |NVIDIA | cuda        |
# |AMD    | cuda        |
# |Intel  | xpu         | 
# |Apple  | mps         |

# set to your device and remove the if clause if you do not have CUDA
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ## Set up the REINVENT run
#
# We define a **working directory** where all output files (config, CSV, TensorBoard logs,
# checkpoints) will be written.  The prior file ships with the repository.  We start the
# agent from the same prior — the agent will drift away from it during RL.

wd = "/tmp/R4_notebooks_output"

# ### Working directory
#
# The cell below creates the working directory if it does not yet exist and reuses it
# otherwise.  To start a completely fresh run, delete the directory manually first, then
# re-run all cells from the top.

os.makedirs(wd, exist_ok=True)
os.chdir(wd)

# ### Writing the TOML configuration
#
# REINVENT is configured through a plain-text **TOML** file
# (see https://toml.io/en/ for the syntax).  All run parameters live in this file, making
# runs reproducible and easy to share.  Below we build the configuration as a Python
# f-string and write it to disk.
#
# A complete reference for every parameter is in
# [`configs/PARAMS.md`](../configs/PARAMS.md).  An annotated example with all options is
# in [`configs/staged_learning.toml`](../configs/staged_learning.toml).
#
# #### Global parameters
#
# | Parameter | Value | Meaning |
# |---|---|---|
# | `run_type` | `staged_learning` | RL run (one or more stages) |
# | `device` | auto-detected | GPU or CPU |
# | `tb_logdir` | `tb_stage1` | Root name for TensorBoard output directory |
# | `json_out_config` | `_stage1.json` | Saves the resolved config in JSON for reference |
#
# #### Model and sampling parameters
#
# | Parameter | Value | Meaning |
# |---|---|---|
# | `prior_file` | `reinvent.prior` | Frozen reference model (ChEMBL-trained RNN) |
# | `agent_file` | same as prior | Starting point for training; will diverge during RL |
# | `summary_csv_prefix` | `stage1` | Output CSV file will be named `stage1_1.csv` (stage number appended) |
# | `batch_size` | 100 | Molecules sampled per RL step; also the SGD mini-batch size |
# | `use_checkpoint` | `false` | Start fresh, ignoring any diversity filter state in the agent file |
#
# **On batch size:** Larger batches give a more stable gradient estimate but slow down each
# step.  Values between 64 and 128 are a good starting point.  Changing batch size also
# changes the effective learning rate, so the two often need to be tuned together.
#
# #### Learning strategy
#
# | Parameter | Value | Meaning |
# |---|---|---|
# | `type` | `dap` | Direct Augmented Posterior — the only supported strategy |
# | `sigma` | 128 | Controls the steepness of the score-to-reward mapping.  Higher values create a sharper distinction between good and bad molecules but can make training less stable. |
# | `rate` | 0.0001 | Learning rate for the Adam optimiser.  The default is conservative; increase cautiously. |
#
# #### Scoring function
#
# The scoring function defines _what_ we want the agent to learn.  Each **component**
# returns a _raw_ value (e.g. a QED score, a count of stereocentres).  A **transformation**
# maps that raw value to the range [0, 1] so that all components are comparable.  The
# individual component scores are combined into a single **total score** by an
# **aggregation function** — here `geometric_mean`.
#
# We use the **weighted geometric mean** rather than the arithmetic mean because it is
# _multiplicative_: if _any_ component scores near zero the total score is dragged towards
# zero as well.  This enforces _all_ constraints simultaneously rather than allowing
# trade-offs between them.
#
# See [`configs/SCORING.md`](../configs/SCORING.md) for the full list of available
# components and transforms.
#
# **Components in this run:**
#
# | Component | Purpose |
# |---|---|
# | `custom_alerts` | Hard filter: any molecule matching one of the SMARTS patterns is scored **zero** |
# | `QED` | Quantitative Estimate of Drug-likeness (0–1, higher is better).  Combines eight physicochemical properties into a single score. |
# | `NumAtomStereoCenters` | Penalises stereocentres.  The Reinvent prior was trained on molecules without stereocentres, so generating them is wasteful.  A `left_step` transform at 0 maps "zero stereocentres" → 1 and "any stereocentres" → 0. |
#
# The `custom_alerts` SMARTS list encodes well-known **medicinal chemistry alerts** —
# functional groups that are reactive, toxic, or otherwise undesirable in drug candidates
# (e.g. peroxides `[#8][#8]`, disulfides `[#16][#16]`, strained large rings).
#
# #### Stage termination
#
# | Parameter | Value | Meaning |
# |---|---|---|
# | `max_steps` | 300 | Hard limit on RL steps.  The run stops when this is reached. |
# | `max_score` | 1.0 | Early-stop if the mean total score reaches this value (unlikely here). |
# | `chkpt_file` | `stage1.chkpt` | Checkpoint saved at the end; can be used as `agent_file` in a subsequent stage. |

# +
# NOTE: the priors are expected to be installed in priors/ at the top level of the repository
prior_filename = os.path.join(reinvent.__path__[0], "..", "priors", "reinvent.prior")
agent_filename = prior_filename
summary_csv_prefix = "stage1"

config = f"""
run_type = "staged_learning"
device = "{device}"
tb_logdir = "tb_stage1"
json_out_config = "_stage1.json"

[parameters]

prior_file = "{prior_filename}"
agent_file = "{agent_filename}"
summary_csv_prefix = "{summary_csv_prefix}"

batch_size = 100

use_checkpoint = false

[learning_strategy]

type = "dap"
sigma = 128
rate = 0.0001

[[stage]]

max_score = 1.0
max_steps = 300

chkpt_file = 'stage1.chkpt'

[stage.scoring]
type = "geometric_mean"

[[stage.scoring.component]]
[stage.scoring.component.custom_alerts]

[[stage.scoring.component.custom_alerts.endpoint]]
name = "Alerts"

params.smarts = [
    "[*;r{{8-17}}]",
    "[#8][#8]",
    "[#6;+]",
    "[#16][#16]",
    "[#7;!n][S;!$(S(=O)=O)]",
    "[#7;!n][#7;!n]",
    "C#C",
    "C(=[O,S])[O,S]",
    "[#7;!n][C;!$(C(=[O,N])[N,O])][#16;!s]",
    "[#7;!n][C;!$(C(=[O,N])[N,O])][#7;!n]",
    "[#7;!n][C;!$(C(=[O,N])[N,O])][#8;!o]",
    "[#8;!o][C;!$(C(=[O,N])[N,O])][#16;!s]",
    "[#8;!o][C;!$(C(=[O,N])[N,O])][#8;!o]",
    "[#16;!s][C;!$(C(=[O,N])[N,O])][#16;!s]"
]

[[stage.scoring.component]]
[stage.scoring.component.QED]

[[stage.scoring.component.QED.endpoint]]
name = "QED"
weight = 0.6


[[stage.scoring.component]]
[stage.scoring.component.NumAtomStereoCenters]

[[stage.scoring.component.NumAtomStereoCenters.endpoint]]
name = "Stereo"
weight = 0.4

transform.type = "left_step"
transform.low = 0
"""

toml_config_filename = "stage1.toml"

with open(toml_config_filename, "w") as tf:
    tf.write(config)

print(f"Config written to {os.path.join(wd, toml_config_filename)}")
# -

# ## Run Reinforcement Learning
#
# The `reinvent` command reads the TOML file and runs the RL loop.  The `%%time` magic
# reports wall-clock time when the run completes.
#
# **Expected duration:** a few minutes on a modern GPU, up to ~30 minutes on CPU, for
# 300 steps with a batch size of 100.
#
# Progress is logged to `stage1.log`.  You can watch it in a terminal with:
# ```shell
# tail -f /tmp/R4_notebooks_output/stage1.log
# ```

# %%time
# !reinvent -l stage1.log $toml_config_filename

# ## Inspect results with TensorBoard
#
# [TensorBoard](https://www.tensorflow.org/tensorboard) is a browser-based dashboard for
# monitoring training runs.  REINVENT writes a TensorBoard event file in real time so you
# can open it _while the run is ongoing_ or after it has finished.
#
# **What to look for in TensorBoard:**
#
# | Tab | What to check |
# |---|---|
# | SCALARS → `Loss` | **Agent NLL** should decrease and move away from the **Prior NLL**. This confirms the agent is learning to generate molecules different from the prior and closer to the scoring target. |
# | SCALARS → component scores | All components should trend upward.  A component stuck at zero may indicate the scoring function is too strict. |
# | SCALARS → `Valid SMILES (%)` | Should stay close to 100%.  A significant drop indicates the agent is generating chemically invalid strings — reduce the learning rate if this happens. |
# | SCALARS → `Duplicates (%)` | Some duplicates are normal; many duplicates indicate the agent has collapsed onto a narrow region of chemical space. |
# | IMAGES | Visualises the first ~30 molecules sampled at each logged step, labelled with their total score. |
#
# > **Note:** TensorBoard appends `_0` to the log directory name (to handle multiple runs).
# > The actual directory is therefore `tb_stage1_0`.

# %tensorboard --bind_all --logdir $wd/tb_stage1_0

# ## Extract data from TensorBoard programmatically
#
# After the run you can load all TensorBoard scalars directly into a Pandas DataFrame for
# custom analysis, plotting, or export.

# ### Load the TensorBoard event file

ea = load_tb_data(wd)

# ### Plot all scalar metrics
#
# `plot_scalars` skips the `(raw)` component variants (untransformed scores) and shows only
# the transformed scores together with the NLL traces and sample statistics.  The function
# also returns the data as a DataFrame that you can save or analyse further.

df = plot_scalars(ea)
df

# ### Display the last molecule grid from TensorBoard
#
# The image shows the first 30 molecules generated in the **final** RL step, each labelled
# with its total score.  This gives a quick visual impression of the chemical space the
# agent has converged to.

img = get_image(ea)
display(img)

# ## Extract data from the CSV summary file
#
# REINVENT writes a CSV file in real time during the run — one row per molecule per step.
# This file is more complete than TensorBoard: it includes every SMILES string, its
# validity state, all raw and transformed component scores, and the step number.  It is
# therefore the primary source for post-hoc analysis.
#
# **SMILES_state encoding:**
#
# | Value | Meaning |
# |---|---|
# | 0 | Invalid — could not be parsed as a SMILES by RDKit |
# | 1 | Valid |
# | 2 | Batch duplicate — the same SMILES appeared more than once in this step's batch |
#
# The filename follows the pattern `{summary_csv_prefix}_{stage_number}.csv`.  The stage
# number starts at 1 for the first `[[stage]]` block.

stage_number = 1  # matches the first [[stage]] block in the config above
csv_file = os.path.join(wd, f"{summary_csv_prefix}_{stage_number}.csv")
df = pd.read_csv(csv_file)
df

# ### Sample efficiency
#
# **Sample efficiency** measures how much of the agent's sampling budget produces useful
# molecules.  Invalid SMILES are wasted computation.  Batch duplicates are also wasteful
# because the agent computes their score but gains no new information.  A well-performing
# run should have close to 100% valid and close to 0% duplicate molecules.
#
# Note the distinction between _batch_ duplicates (the same molecule appears twice in a
# single 100-molecule batch) and _global_ duplicates (the same molecule was generated in
# any previous step).  Global duplicates are expected to accumulate as the agent converges
# — they indicate the agent has learnt a narrow but high-scoring distribution.

# +
total_smilies = len(df)

invalids = df[df["SMILES_state"] == 0]     # 0 = invalid
total_invalid_smilies = len(invalids)

duplicates = df[df["SMILES_state"] == 2]   # 2 = batch duplicate
total_batch_duplicate_smilies = len(duplicates)

all_duplicates = df[df.duplicated(subset=["SMILES"])]
total_duplicate_smilies = len(all_duplicates)

print(
    f"Total SMILES generated:          {total_smilies}\n"
    f"Invalid SMILES:                  {total_invalid_smilies} "
    f"({100 * total_invalid_smilies / total_smilies:.1f}%)\n"
    f"Batch duplicates:                {total_batch_duplicate_smilies} "
    f"({100 * total_batch_duplicate_smilies / total_smilies:.1f}%)\n"
    f"Global duplicates (all steps):   {total_duplicate_smilies} "
    f"({100 * total_duplicate_smilies / total_smilies:.1f}%)"
)
# -

# ### Display all globally duplicated molecules
#
# A large number of global duplicates suggests the agent has over-converged.  In a
# production run you would add a **Diversity Filter** (see
# [`configs/staged_learning.toml`](../configs/staged_learning.toml)) to penalise repeated
# scaffolds and encourage broader exploration.

if len(all_duplicates):
    mol_view = create_mol_grid(all_duplicates)
    display(mol_view)
else:
    print("No global duplicates — the agent explored a wide region of chemical space.")

# ### Display the molecules from the last RL step
#
# This gives a representative snapshot of what the agent has learnt to generate after
# 300 steps.  Compare these visually to what a random sample from the prior looks like —
# the molecules should appear more "drug-like" (fewer reactive groups, no large rings,
# reasonable molecular complexity).

last = df[df["step"] == max(df["step"])]
mol_view = create_mol_grid(last)
display(mol_view)

# ### Plot the NLL traces
#
# These three traces summarise the RL dynamics:
#
# | Trace | Description |
# |---|---|
# | **Prior** | NLL of each generated molecule under the frozen prior.  Should be roughly constant. |
# | **Agent** | NLL of each molecule under the trained agent.  Should **decrease** — molecules the agent has learnt to favour become more likely (lower NLL). |
# | **Target** | The _augmented NLL_ — the training target computed from the DAP reward function.  This is what the agent is actually trying to minimise. |
#
# A healthy run shows **Agent NLL decreasing** and **Agent NLL staying below Prior NLL**.
# This means the agent generates molecules that are more probable under itself than under
# the prior, i.e. the two distributions have diverged in the direction of the scoring
# target.  If Agent and Prior NLL remain overlapping, the agent has not learnt anything.

# +
grouped_df = df.groupby("step")

for label in "Agent", "Prior", "Target":
    means = grouped_df.aggregate({label: "mean"})
    sns.scatterplot(data=means, x=means.index, y=label, label=label)

plt.xlabel("RL step")
plt.ylabel("Mean NLL")
plt.title("NLL traces: agent learning dynamics")
plt.show()
# -

# ## Summary
#
# You have completed a full _de novo_ RL run with REINVENT.  Key takeaways:
#
# - The **prior** sets a chemically sensible baseline; the **agent** is optimised against
#   your scoring function while staying anchored to the prior via the DAP reward.
# - The **scoring function** is a weighted geometric mean of property components — all
#   constraints must be satisfied simultaneously.
# - **TensorBoard** provides a live view of training dynamics; the **CSV file** is the
#   complete record for post-hoc analysis.
# - Watch **valid SMILES %** (should stay near 100%) and **duplicates %** (should stay
#   low) as primary health indicators for a run.
#
# **Next steps:**
# - Try adjusting the `weight` values for QED and Stereo and observe the effect.
# - Add a Diversity Filter to reduce global duplicates (see
#   [`configs/staged_learning.toml`](../configs/staged_learning.toml)).
# - For a target-focused campaign, see the `Reinvent_TLRL` notebook, which adds a
#   Transfer Learning step and a structure-based scoring component.


