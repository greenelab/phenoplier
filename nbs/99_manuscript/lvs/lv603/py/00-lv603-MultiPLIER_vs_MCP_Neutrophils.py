# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all,-execution,-papermill,-trusted
#     formats: ipynb,py//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Description

# %% [markdown]
# Generates a plot between the projection of a gene expression data set (systemic lupus erythematosus (SLE) whole blood (WB) from the MultiPLIER paper) into the LV603 (a neutrophil-termed latent variable) and a neutrophil count estimation.
#
# See the [MultiPLIER paper](https://doi.org/10.1016/j.cels.2019.04.003) for more details.

# %% [markdown]
# # Modules loading

# %%
# %load_ext autoreload
# %autoreload 2

# %%
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

import conf

# %%
assert (
    conf.MANUSCRIPT["BASE_DIR"] is not None
), "The manuscript directory was not configured"

display(conf.MANUSCRIPT["BASE_DIR"])

# %% [markdown]
# # Settings

# %%
LV_NUMBER_SELECTED = 603
LV_NAME_SELECTED = f"LV{LV_NUMBER_SELECTED}"
display(LV_NAME_SELECTED)

# %%
OUTPUT_FIGURES_DIR = Path(conf.MANUSCRIPT["FIGURES_DIR"], "entire_process").resolve()
display(OUTPUT_FIGURES_DIR)
OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# # Load data

# %%
input_file = conf.MULTIPLIER["BANCHEREAU_MCPCOUNTER_NEUTROPHIL_FILE"]
display(input_file)

# %%
data = pd.read_csv(input_file, sep="\t")

# %%
data.shape

# %%
data.head()

# %% [markdown]
# # Plot

# %%
with sns.plotting_context("paper", font_scale=3.0):
    g = sns.lmplot(
        data=data.rename(
            columns={
                f"recount2_LV{LV_NUMBER_SELECTED}": f"MultiPLIER LV{LV_NUMBER_SELECTED}",
                "Neutrophil_estimate": "Neutrophils estimate",
            }
        ),
        x=f"MultiPLIER LV{LV_NUMBER_SELECTED}",
        y="Neutrophils estimate",
        scatter_kws={"rasterized": True},
    )

    g.ax.set_xlabel(f"$\mathbf{{B}}_{{\mathrm{{LV}}{LV_NUMBER_SELECTED}}}$")
    g.ax.set_ylabel("Neutrophils\nestimate")
    g.set_xticklabels([])
    g.set_yticklabels([])
    plt.tight_layout()

    # save figure
    plt.savefig(
        Path(OUTPUT_FIGURES_DIR, "lv603_vs_multiplier_neutrophils.pdf"),
        dpi=300,
        bbox_inches="tight",
    )

# %%
