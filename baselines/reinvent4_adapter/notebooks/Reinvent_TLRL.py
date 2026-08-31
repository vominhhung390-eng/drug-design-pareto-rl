# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Focusing a _de novo_ Model: Transfer Learning followed by Reinforcement Learning
#
# ## What this tutorial teaches
#
# This notebook demonstrates a two-stage workflow that is common in real drug-discovery
# projects:
#
# 1. **Stage 1 RL** — Run a short reinforcement learning campaign starting from the
#    general-purpose Reinvent prior to produce a "drug-like" baseline agent.
# 2. **Transfer Learning (TL)** — Fine-tune that agent on a set of known binders to
#    a specific target (Tankyrase-2, TNKS2) so it learns the chemical language of
#    that target class.
# 3. **Stage 2 RL** — Run a second RL campaign starting from the focused model, now
#    including a structure-based scoring component (ChemProp free-energy model) to
#    drive the agent towards potent binders.
#
# By the end you will understand:
# - Why Transfer Learning is used to _focus_ a prior before target-specific RL
# - How to prepare experimental data for Transfer Learning
# - How to choose a TL checkpoint and what the TensorBoard diagnostics mean
# - How to add a predictive model (ChemProp) and a Diversity Filter to a second RL stage
# - How to post-process the CSV output to select candidate molecules
#
# ## Background: why focus a prior before RL?
#
# The general-purpose Reinvent prior was trained on millions of drug-like compounds from
# ChEMBL.  It can generate a vast diversity of molecules, but only a small fraction of
# that chemical space is likely to bind a specific target.  Starting RL directly from the
# prior forces the agent to explore a huge space before finding rewarding molecules —
# this is sample-inefficient.
#
# **Transfer Learning** addresses this by re-training the prior on a small set of known
# active compounds.  The result is a _focused_ model whose distribution is shifted towards
# the target's chemical neighbourhood.  Starting RL from this focused model means the
# agent begins much closer to the relevant region of chemical space, reaching good scores
# faster and with fewer wasted steps.
#
# The trade-off is that over-focusing can reduce diversity: if TL is pushed too far, the
# model essentially memorises the training set and generates only close analogues.  The
# TL diagnostics below (valid SMILES %, duplicates, internal diversity) help you find the
# sweet spot.
#
# ## Background: the DAP reward function
#
# During RL, each generated molecule receives a **total score** (0–1) from the scoring
# function.  REINVENT uses the **Direct Augmented Posterior (DAP)** algorithm to convert
# this score into a training signal.  The key formula is:
#
# ```
# augmented_NLL = prior_NLL - σ × total_score
# ```
#
# The agent is trained to minimise the gap between its own NLL and the augmented NLL.
# Molecules with a high total score have a low augmented NLL, so the agent is pushed to
# assign them a low NLL too — i.e. to generate them more often.  `σ` (sigma) controls
# how strongly the score influences training: larger values create a sharper reward
# signal but can destabilise training.
#
# > **Further reading**
# > - Parameter reference: [`configs/PARAMS.md`](../configs/PARAMS.md)
# > - Scoring components reference: [`configs/SCORING.md`](../configs/SCORING.md)
# > - Full annotated RL config: [`configs/staged_learning.toml`](../configs/staged_learning.toml)
# > - Full annotated TL config: [`configs/transfer_learning.toml`](../configs/transfer_learning.toml)
# > - REINVENT 4 paper: https://doi.org/10.1186/s13321-024-00812-5

# ## Prerequisites
#
# Make sure all required packages are installed before proceeding.

# +
import importlib

for pkg in ["reinvent", "tensorboard", "mols2grid", "seaborn", "ipywidgets"]:
    found = importlib.util.find_spec(pkg) is not None
    print(f"{'OK     ' if found else 'MISSING'} {pkg}")

# +
import os
import shutil
import glob

import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import reinvent
from reinvent.notebooks import load_tb_data, plot_scalars, get_image, create_mol_grid
from reinvent.scoring.transforms import ReverseSigmoid
from reinvent.scoring.transforms.sigmoids import Parameters as SigmoidParameters

import ipywidgets as widgets

# %load_ext tensorboard
# -

# ## Device selection
#
# GPU is strongly recommended, especially for the Transfer Learning step which requires
# many forward and backward passes through the model.  The cell auto-detects hardware.

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ## Paths

wd = "/tmp/R4_notebooks_output"
top = os.path.abspath(os.path.join(reinvent.__path__[0], ".."))
top

# ### Working directory
#
# The cell below creates the working directory if it does not yet exist and reuses it
# otherwise.  To start a completely fresh run, delete the directory manually and
# re-run all cells from the top.

os.makedirs(wd, exist_ok=True)
os.chdir(wd)
wd

# ## Stage 1: Reinforcement Learning for a drug-like baseline
#
# This stage is identical in purpose to the `Reinvent_demo` notebook — we train the
# prior to generate drug-like molecules (high QED, no structural alerts, no
# stereocentres).  The checkpoint produced here becomes the starting point for
# Transfer Learning.
#
# If you have already completed the `Reinvent_demo` notebook and the checkpoint
# `stage1.chkpt` exists in the working directory, you can skip this section.

# ### Write the stage 1 config
#
# See the `Reinvent_demo` notebook for a detailed explanation of each parameter.
# The full parameter reference is in [`configs/PARAMS.md`](../configs/PARAMS.md).

# +
prior_filename = os.path.abspath(os.path.join(reinvent.__path__[0], "..", "priors", "reinvent.prior"))
agent_filename = prior_filename

stage1_checkpoint = "stage1.chkpt"
stage1_summary_csv_prefix = "stage1"

stage1_parameters = f"""
run_type = "staged_learning"
device = "{device}"
tb_logdir = "tb_stage1"
json_out_config = "_stage1.json"

[parameters]

prior_file = "{prior_filename}"
agent_file = "{agent_filename}"
summary_csv_prefix = "{stage1_summary_csv_prefix}"

batch_size = 100

use_checkpoint = false

[learning_strategy]

type = "dap"
sigma = 128
rate = 0.0001

[[stage]]

max_score = 1.0
max_steps = 300

chkpt_file = "{stage1_checkpoint}"

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

stage1_config_filename = "stage1.toml"

with open(stage1_config_filename, "w") as tf:
    tf.write(stage1_parameters)

print(f"Config written to {os.path.join(wd, stage1_config_filename)}")
# -

# ### Run stage 1 RL
#
# This run will take several minutes on a GPU or longer on CPU.

shutil.rmtree("tb_stage1_0", ignore_errors=True)

# %%time
# !reinvent -l stage1.log $stage1_config_filename

# ### Inspect stage 1 results with TensorBoard
#
# Check that QED and Stereo scores are rising and that valid SMILES % remains high
# before moving on to Transfer Learning.

# %tensorboard --bind_all --logdir $wd/tb_stage1_0

# ## Transfer Learning: focusing the model on TNKS2 binders
#
# ### What Transfer Learning does
#
# Transfer Learning (TL) re-trains the model to assign **lower NLL** (higher probability)
# to the molecules in the training set.  Concretely, TL minimises the mean NLL of the
# input SMILES using standard supervised learning.
#
# After TL the model samples molecules that are chemically more similar to the training
# set.  This is useful when you have a set of known actives for a target: the model
# learns the "chemical language" of that target class and proposes novel analogues rather
# than sampling from the much larger and less relevant general drug-like space.
#
# TL is **not** the same as memorisation.  The goal is to shift the model's distribution,
# not to reproduce the training SMILES exactly.  If the model is over-trained it begins
# to generate only very close analogues of the input, losing the diversity needed for
# drug design.  The validation loss and the diversity metrics in TensorBoard tell you
# when this starts to happen.
#
# ### Prepare the training data
#
# We use known Tankyrase-2 (TNKS2) binders from
# [BindingDB](https://www.bindingdb.org/rwd/jsp/dbsearch/PrimarySearch_ki.jsp?tag=pol&submit=Search&target=tankyrase-2).
# TNKS2 is a validated oncology target; potent small-molecule inhibitors have been
# reported in the literature.
#
# > **A note on data quality:** the IC50 values in BindingDB come from many different
# > laboratories and assay formats.  Mixing values from different sources introduces
# > noise — see
# > [Combining IC50 or Ki Values from Different Sources Is a Source of Significant Noise](https://doi.org/10.1021/acs.jcim.4c00049).
# > The filtering below is intentionally simple and is meant only to illustrate the
# > workflow, not to represent best practice for data curation.

bdb = pd.read_csv(f"{top}/notebooks/data/tnks2.csv")
bdb

# #### Filter to "good" binders
#
# We keep only IC50 measurements (discarding Ki and Kd), remove entries where the
# reported value is a range (`<` or `>`) to avoid ambiguity, and apply a 1 µM (1000 nM)
# potency cutoff.  This is a reasonable starting point for a focused model, but in a
# real project you would want to apply additional curation steps (salt stripping,
# standardisation, activity cliff analysis, etc.).

clean = bdb[~bdb["exp (nM)"].str.match("[<>]")]
clean = clean.astype({'exp (nM)': 'float'})
print(f"Rows before filter: {len(bdb)}, after removing ranges: {len(clean)}")

good = clean[clean["exp (nM)"] < 1000]
good = good[good["exp_method"] != "EC50"]
good = good[good["exp_method"] != "Kd"]
good = good.rename(columns={"exp (nM)": "IC50"})
good = good.drop(columns=["exp_method"])

print(f"Good binders (IC50 < 1 µM): {len(good)}")
grid = create_mol_grid(good)
display(grid)

# #### Split into training and validation sets and write SMILES files
#
# We use an 80/20 random split.  The **validation set** is not used for training — its
# loss during TL indicates whether the model is generalising (validation loss decreasing)
# or overfitting (validation loss increasing while training loss continues to drop).
#
# TL reads plain tab-separated SMILES files.  The first column must contain SMILES;
# additional columns (IC50 here) are ignored during training.

# +
TL_train_filename = "tnks2_train.smi"
TL_validation_filename = "tnks2_validation.smi"

data = good.sample(frac=1)  # shuffle
n_head = int(0.8 * len(data))
n_tail = len(good) - n_head
print(f"Training set: {n_head} molecules, validation set: {n_tail} molecules")

train, validation = data.head(n_head), data.tail(n_tail)

train.to_csv(TL_train_filename, sep="\t", index=False, header=False)
validation.to_csv(TL_validation_filename, sep="\t", index=False, header=False)
# -

# ### TL configuration
#
# Key TL parameters (full reference in [`configs/PARAMS.md`](../configs/PARAMS.md) and
# [`configs/transfer_learning.toml`](../configs/transfer_learning.toml)):
#
# | Parameter | Value | Meaning |
# |---|---|---|
# | `num_epochs` | 50 | Number of full passes through the training set |
# | `save_every_n_epochs` | 2 | Write a checkpoint file every 2 epochs — gives you a range of models to choose from |
# | `batch_size` | 100 | Training mini-batch size |
# | `sample_batch_size` | 2000 | Number of molecules sampled each epoch to compute sample statistics (valid %, duplicates, diversity) |
# | `randomize_smiles` | true | Randomly permutes atom ordering in training SMILES.  Augments the dataset and improves generalisation because the same molecule can be written as many equivalent SMILES strings. |
# | `internal_diversity` | true | Reports the internal Tanimoto diversity of the sampled batch in TensorBoard — useful for detecting collapse. |
#
# The `input_model_file` is the checkpoint from stage 1.  If you did not run stage 1
# above, replace this with the path to any REINVENT checkpoint.

TL_parameters = f"""
run_type = "transfer_learning"
device = "{device}"
tb_logdir = "tb_TL"


[parameters]

num_epochs = 50
save_every_n_epochs = 2
batch_size = 100
sample_batch_size = 2000

input_model_file = "{stage1_checkpoint}"
output_model_file = "TL_reinvent.model"
smiles_file = "{TL_train_filename}"
validation_smiles_file = "{TL_validation_filename}"
standardize_smiles = true
randomize_smiles = true
randomize_all_smiles = false
internal_diversity = true
"""

# +
TL_config_filename = "transfer_learning.toml"

with open(TL_config_filename, "w") as tf:
    tf.write(TL_parameters)

print(f"Config written to {os.path.join(wd, TL_config_filename)}")
# -

# ## Run Transfer Learning
#
# TL is computationally heavier than a single RL step because every epoch processes the
# full training set.  On a GPU this typically takes a few minutes for 50 epochs on a
# dataset of this size.  Make sure that you have set the device above to the right GPU on
# your computer.

shutil.rmtree("tb_TL", ignore_errors=True)

# !reinvent -l transfer_learning.log $TL_config_filename

# ### Inspect TL results with TensorBoard
#
# **What to look for:**
#
# | Metric | What it tells you |
# |---|---|
# | Training loss | Should decrease steadily — the model is fitting the training data. |
# | Validation loss | Should decrease early, then plateau or rise slightly.  The minimum validation loss epoch is usually the best checkpoint. |
# | Valid SMILES (%) | Should stay high (> 95%).  A large drop means the model is losing its ability to generate valid chemistry. |
# | Duplicates (%) | Initially decreases as the model diversifies, then may rise again as it starts over-focusing.  Aim for low duplicates. |
# | Internal diversity | Should remain high.  A sharp drop indicates the model is collapsing onto a narrow set of structures. |
#
# The right checkpoint is a **judgement call**: you want the validation loss to be low
# (model has learnt the target chemistry) but not so low that the model has memorised
# the training set.  A useful heuristic is the epoch of minimum validation loss, but
# always check the other metrics too.

# %tensorboard --bind_all --logdir $wd/tb_TL

# ### Choose a TL checkpoint
#
# REINVENT saves a checkpoint every 2 epochs.  The cell below lists all available
# checkpoints.  Select the one that corresponds to the best epoch from TensorBoard and
# set `TL_model_filename` accordingly.
#
# **Note:** because training is stochastic, the best epoch may differ between runs.
# Do not assume epoch 30 is always optimal — always check TensorBoard first.

# +
available_checkpoints = sorted(glob.glob(os.path.join(wd, "TL_reinvent.model.*.chkpt")))
print("Available TL checkpoints:")
for c in available_checkpoints:
    print(" ", os.path.basename(c))

# Edit the line below to select a different epoch if TensorBoard suggests a better one.
TL_model_filename = available_checkpoints[-1] if available_checkpoints else os.path.join(wd, "TL_reinvent.model.30.chkpt")
print(f"\nSelected: {os.path.basename(TL_model_filename)}")
# -

# ## Stage 2: Target-focused Reinforcement Learning
#
# Now that we have a focused model, we run a second RL campaign that adds a **predictive
# scoring component** based on free-energy simulations for TNKS2.
#
# ### The ChemProp scoring component
#
# [ChemProp](https://chemprop.readthedocs.io/) is a directed message-passing neural
# network (D-MPNN) for molecular property prediction.  Here it has been trained on
# free-energy perturbation (FEP) data for TNKS2 and predicts the binding free energy ΔG
# (in kcal/mol) for each generated molecule.  More negative ΔG means stronger predicted
# binding.
#
# **Download the model** from the link in the cell below and place the file in the
# `chemprop/` directory inside the working directory.  The scoring component expects a
# directory, not a single file.  **Note**: this is still ChemProp v2.x model which is the default in REINVENT.
#
# > Model download: https://drive.google.com/file/d/1ZHy7izLjbXJw5_ZWn57-Q4mSaId_re88/view?usp=drive_link
# >
# > After downloading: place `model2.pt` into `{wd}/`

# +
chemprop_path = os.path.join(wd, "model2.pt")

if not os.path.isfile(chemprop_path):
    raise FileNotFoundError(
        f"ChemProp model not found: {chemprop_path}\n"
        f"Download model2.pt from the link above and save it as {chemprop_path}"
    )
# -

# ### Stage 2 config
#
# The stage 2 config reuses the same drug-likeness components from stage 1 (custom alerts,
# QED, no stereocentres) and adds an additional scoring function and two new elements:
#
# **ChemProp scoring component:**
#
# The raw output is ΔG in kcal/mol.  Values closer to 0 mean poor binding; more negative
# values mean stronger binding.  We want to convert this to a score between 0 and 1 where
# strong binders score close to 1.  A **reverse sigmoid** transform does this: it maps the
# `[low, high]` range onto [0, 1] with a smooth S-shaped transition, but in reverse so
# that the most negative values map to the highest scores.
#
# **Diversity Filter:**
#
# The Diversity Filter (DF) tracks the **Murcko scaffold** of every molecule that scores
# above `minscore`.  If the same scaffold is generated more than `bucket_size` times, all
# further molecules with that scaffold are scored zero.  This prevents the agent from
# collapsing onto a single scaffold and encourages it to explore diverse chemical series.
#
# **Inception memory:**
#
# Inception is a form of **experience replay**.  At each step, a random sample of
# high-scoring molecules from previous steps is drawn from the memory and included in the
# training batch alongside the freshly generated molecules.  This stabilises training by
# preventing the agent from forgetting earlier discoveries and by providing a consistent
# gradient signal from molecules that are known to score well.
#
# See [`configs/staged_learning.toml`](../configs/staged_learning.toml) for a full
# annotated example with all options.

stage2_summary_csv_prefix = "stage2"

stage2_parameters = f"""
run_type = "staged_learning"
device = "{device}"
tb_logdir = "tb_stage2"
json_out_config = "_stage2.json"

[parameters]

prior_file = "{prior_filename}"
agent_file = "{TL_model_filename}"
summary_csv_prefix = "{stage2_summary_csv_prefix}"

batch_size = 100

use_checkpoint = false

[learning_strategy]

type = "dap"
sigma = 128
rate = 0.0001

[[stage]]

max_score = 1.0
max_steps = 500

chkpt_file = "stage2.chkpt"

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

pred_model_parameters = f"""
[[stage.scoring.component]]
[stage.scoring.component.ChemProp2]

[[stage.scoring.component.ChemProp2.endpoint]]
name = "ChemProp"
weight = 0.6

params.model_path = "{chemprop_path}"
params.featurizers = ["MorganBinary"]
params.featurizers_options = [{{radius = 2, length = 2048, include_chirality = true}}]

transform.type = "reverse_sigmoid"
transform.high = 0.0
transform.low = -50.0
transform.k = 0.4
"""

# ### Preview the reverse sigmoid transform
#
# Before committing to specific `low`, `high`, and `k` values, it is useful to plot the
# transform to understand how it maps raw ΔG values to scores.
#
# - **`low`** and **`high`** define the ΔG range of interest (in kcal/mol).  Values
#   outside this range are mapped to ~0 (`high` end) or ~1 (`low` end).
# - **`k`** controls the steepness of the transition.  A larger `k` creates a sharper
#   step; a smaller `k` gives a gentler gradient.
#
# Adjust the sliders to explore different parameterisations.  The goal is a transform that
# assigns a score near 1 to molecules with a predicted ΔG around -25 to -30 kcal/mol and
# near 0 to molecules with ΔG > -10 kcal/mol.  **Note:** the model has been constructed from
# MM-PBSA MD simulation data which tends to strongly overestimate the binding free
# energy.

# +
def plot_transform(low, high, k):
    params = SigmoidParameters(type="reverse_sigmoid", high=high, low=low, k=k)
    reverse_sigmoid = ReverseSigmoid(params)
    x = np.linspace(low, high, num=25)
    vf = np.vectorize(reverse_sigmoid)

    plt.figure(figsize=(6, 3))
    ax = sns.lineplot(x=x, y=vf(x))
    ax.set(title="Reverse Sigmoid transform", xlabel="Raw ΔG score (kcal/mol)", ylabel="Transformed score [0–1]")
    plt.tight_layout()
    plt.show()

low_slider = widgets.FloatSlider(min=-70, max=-30, step=5, value=-50.0, description="low")
high_slider = widgets.FloatSlider(min=-20, max=20, step=5, value=0.0, description="high")
k_slider = widgets.FloatSlider(min=0.1, max=0.7, step=0.1, value=0.4, orientation='vertical', description="k")

# +
p = widgets.interactive(plot_transform, low=low_slider, high=high_slider, k=k_slider)

low_high_ctrl = widgets.HBox(p.children[:2], layout=widgets.Layout(flex_flow='row wrap'))
k_ctrl = p.children[2]
output = p.children[-1]
vbox = widgets.VBox([output, low_high_ctrl])

display(widgets.HBox([vbox, k_ctrl]))
# -

# If the widget above doesn't work, plot directly by changing the cell below to Code.

# + active=""
# plot_transform(-50.0, 0.0, 0.4)
# -

# ### Diversity Filter and Inception configuration
#
# See the explanations in the stage 2 config section above.
#
# A note on `minscore = 0.7`: only molecules with a total score above 0.7 are registered
# in the scaffold memory.  This avoids penalising scaffolds that appear in early, low-quality
# molecules before the agent has learnt to score well.

df_parameters = """
[diversity_filter]

type = "IdenticalMurckoScaffold"
bucket_size = 10
minscore = 0.7
"""

# Inception is seeded from the first sampled batch (no smiles_file provided).
# The memory holds the 50 highest-scoring molecules seen so far and 10 are
# sampled from it at random each step to augment the training batch.
inception_parameters = """
[inception]

memory_size = 50
sample_size = 10
"""

# +
full_stage2_parameters = stage2_parameters + pred_model_parameters + df_parameters + inception_parameters
stage2_config_filename = "stage2.toml"

with open(stage2_config_filename, "w") as tf:
    tf.write(full_stage2_parameters)

print(f"Config written to {os.path.join(wd, stage2_config_filename)}")
# -

# ## Run Stage 2 RL
#
# This run is longer than stage 1 (500 steps with an additional ChemProp scoring call
# per molecule).  Expect 30–90 minutes on a GPU depending on hardware.

# %%time
# !reinvent -l stage2.log $stage2_config_filename

# ### Inspect stage 2 results with TensorBoard
#
# **What to look for:**
#
# - All scoring component scores (QED, Stereo, ChemProp) should trend upward, confirming
#   the agent is learning to satisfy all constraints simultaneously.
# - `ChemProp (raw)` shows the predicted ΔG values becoming more negative over time —
#   the agent is learning to generate stronger binders.
# - The loss traces (`Agent NLL`, `Prior NLL`) should show the agent diverging from the
#   prior as in stage 1.
# - If you observe high duplicate rates despite the Diversity Filter, consider increasing
#   `bucket_size` or lowering `minscore` in the filter.

# %tensorboard --bind_all --logdir $wd/tb_stage2_0

# ## Post-processing: selecting candidate molecules
#
# The CSV file contains every molecule generated during stage 2 with all its scores.
# We filter to find molecules that simultaneously satisfy our quality criteria and rank
# them for further assessment.
#
# **SMILES_state encoding:**
#
# | Value | Meaning |
# |---|---|
# | 0 | Invalid — could not be parsed as SMILES |
# | 1 | Valid |
# | 2 | Batch duplicate — same SMILES generated more than once in this step |

stage_number = 1  # matches the first [[stage]] block in the stage 2 config
csv_file = os.path.join(wd, f"{stage2_summary_csv_prefix}_{stage_number}.csv")
df = pd.read_csv(csv_file)
df

# ### Apply quality filters
#
# We define "good binders" as molecules that meet all three criteria simultaneously:
# - **QED > 0.8**: highly drug-like according to the quantitative estimate of drug-likeness
# - **ΔG < -25 kcal/mol** (`ChemProp (raw) < -25`): strong predicted binding affinity
#
# The thresholds are somewhat arbitrary — adjust them based on the score distributions
# you observe in TensorBoard and the requirements of your project.

# +
good_QED = df["QED"] > 0.8
good_dG = df["ChemProp (raw)"] < -25.0  # kcal/mol

good_binders = df[good_QED & good_dG]
print(f"Molecules passing both filters: {len(good_binders)} out of {len(df)} total")
# -

# ### Remove duplicate SMILES
#
# The same molecule can appear in multiple steps if the agent has converged onto it.
# The SMILES in the CSV have been canonicalised, so simple deduplication is sufficient.
# Keeping only unique structures gives a cleaner candidate list for downstream analysis.

good_binders = good_binders.drop_duplicates(subset=['SMILES'])
print(f"Unique candidate molecules: {len(good_binders)}")

# ### Display the candidate molecules
#
# The interactive grid lets you inspect each molecule and sort by score.  Click the "i"
# icon in the top-right corner of a molecule card to see all its associated data.
#
# **What to look for:**
# - Structural diversity — are there multiple scaffolds, or has the agent converged on one?
# - Similarity to known TNKS2 binders in the training set — rediscovery is a good
#   sanity check but is not the ultimate goal; novel chemotypes are more valuable.
# - Obvious chemoinformatic issues (very large molecules, unusual functional groups).

grid = create_mol_grid(good_binders)
display(grid)

# ## Discussion and next steps
#
# This tutorial has demonstrated a complete generative design workflow:
#
# 1. **Stage 1 RL** established a drug-like baseline by training the prior to avoid
#    structural alerts and generate molecules with high QED.
# 2. **Transfer Learning** shifted the model's distribution towards known TNKS2 binders,
#    making stage 2 RL more sample-efficient.
# 3. **Stage 2 RL** combined the focused model with a structure-based predictive model
#    (ChemProp), a Diversity Filter, and Inception memory to generate novel molecules
#    predicted to bind TNKS2.
#
# **Interpreting the results:**
#
# All score components should increase over the stage 2 run, confirming the agent is
# learning the multi-objective profile.  `ChemProp (raw)` values becoming more negative
# confirms predicted binding affinity is improving.  A high duplicate rate despite the
# Diversity Filter may indicate the scoring function is too easy or the model has
# over-focused; consider adding more components or adjusting weights.
#
# **Important caveats:**
#
# - RL is stochastic — different random seeds produce different results.  Run multiple
#   independent campaigns and aggregate to get robust statistics.
# - Rediscovering known binders is a useful validation; generating only known structures
#   is not useful in practice.  Aim for a mix of novel scaffolds and analogues.
# - The ChemProp model is a surrogate for expensive FEP calculations.  All predictions
#   should ultimately be validated experimentally.
#
# **Possible extensions:**
#
# - Replace ChemProp with a docking score (Vina, Glide) via the `Maize` component —
#   see [`configs/SCORING.md`](../configs/SCORING.md).
# - Add ADMET constraints (TPSA, rotatable bonds, CYP inhibition) as additional scoring
#   components.
# - Add a `TanimotoSimilarity` component to the scoring function to bias generation towards
#   a specific reference compound while still encouraging novelty.
# - Try `Mol2Mol` or `LibInvent` priors for scaffold-restrained or R-group optimisation
#   tasks — see [`configs/staged_learning.toml`](../configs/staged_learning.toml) for
#   setup examples.


