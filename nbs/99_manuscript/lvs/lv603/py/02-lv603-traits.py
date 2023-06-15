# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all,-execution,-papermill,-trusted
#     formats: ipynb,py//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Description

# %% [markdown]
# Generates a plot with the top traits and traits categories for LV603 (a neutrophil-termed latent variable in MultiPLIER).

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
import matplotlib.ticker as ticker
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

# %% tags=[]
input_filepath = Path(
    conf.RESULTS["PROJECTIONS_DIR"],
    "projection-smultixcan-efo_partial-mashr-zscores.pkl",
).resolve()
display(input_filepath)

assert input_filepath.exists(), "Input file does not exist"

# %% tags=[]
data = pd.read_pickle(input_filepath)

# %% tags=[]
data.shape

# %% tags=[]
data.head()

# %% [markdown]
# # Show LV traits

# %%
from entity import Trait

# %%
# Show related traits
lv_traits = data.loc[LV_NAME_SELECTED].sort_values(ascending=False)
display(lv_traits.head(20))

# %% [markdown]
# # Remove repeated traits

# %% [markdown]
# Here I remove traits that represent the same phenotype. I remove the least significant trait from duplicates.

# %%
lv_traits_to_remove = [
    "leukocyte count",
    "30140_raw-Neutrophill_count",
    "monocyte count",
]

# %%
_old_len = lv_traits.shape[0]
lv_traits = lv_traits.drop(lv_traits_to_remove)
assert lv_traits.shape[0] == _old_len - len(lv_traits_to_remove)
display(lv_traits.shape)

# %% [markdown]
# # Rename LV traits

# %%
lv_traits = lv_traits.reset_index().rename(columns={"index": "traits"})

# %%
lv_traits

# %%
lv_traits = lv_traits.replace(
    {
        "traits": {
            t: (
                Trait.get_traits_from_efo(t)[0].description
                if Trait.is_efo_label(t)
                else Trait.get_trait(full_code=t).description
            )
            for t in lv_traits["traits"].values
        }
    }
)

# %%
lv_traits.head(10)

# %%
lv_traits = lv_traits.replace(
    {
        "traits": {
            "Neutrophil Count": "Neutrophil #",
            "Monocyte count": "Monocyte #",
            "Basophill count": "Basophil #",
            "White blood cell (leukocyte) count": "Leukocyte #",
            "Basophill percentage": "Basophil %",
            "Lymphocyte percentage": "Lymphocyte %",
            "Neutrophill percentage": "Neutrophil %",
            "Mean platelet (thrombocyte) volume": "Mean platelet volume",
            "Myeloid White Cell Count": "Myeloid white cell #",
            "Sum Neutrophil Eosinophil Count": "Neutrophil+Eosinophil #",
            "Granulocyte Count": "Granulocyte #",
            #     'Platelet crit': 'PCT',
            "Sum Basophil Neutrophil Count": "Basophil+Neutrophil #",
        }
    }
)

# %%
lv_traits.head(10)

# %% [markdown]
# # Plot

# %%
with sns.plotting_context("paper", font_scale=5.0):
    g = sns.catplot(
        data=lv_traits.rename(
            columns={
                "traits": "Traits",
            }
        ).head(10),
        x="Traits",
        y=LV_NAME_SELECTED,
        kind="bar",
        aspect=2.6,
        height=6,  # ci=None, height=6, aspect=1.6, legend_out=False,
        color="orange",
    )

    g.ax.set_ylabel(f"$\mathbf{{\hat{{M}}}}_{{\mathrm{{LV}}{LV_NUMBER_SELECTED}}}$")
    g.set_xticklabels(g.ax.get_xticklabels(), rotation=45, horizontalalignment="right")
    # g.set_yticklabels([])
    g.ax.yaxis.set_major_locator(ticker.LinearLocator(4))
    g.ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.2f}"))
    g.ax.set_xlabel("")

    # save figure
    plt.savefig(Path(OUTPUT_FIGURES_DIR, "lv603_traits.pdf"), bbox_inches="tight")

# %%
