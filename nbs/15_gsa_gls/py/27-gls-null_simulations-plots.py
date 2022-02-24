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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] tags=[]
# # Description

# %% [markdown] tags=[]
# TODO
#
# - rename this file to be 27-

# %% [markdown] tags=[]
# # Modules

# %% tags=[]
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.graphics.gofplots import qqplot_2samples
import matplotlib.pyplot as plt
import seaborn as sns

import conf

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
INPUT_DIR = conf.RESULTS["GLS"]
display(INPUT_DIR)

# %% tags=[]
INPUT_FILENAME = INPUT_DIR / "gls-null_simulations.pkl"
display(INPUT_FILENAME)
assert INPUT_FILENAME.exists()

# %% tags=[]
INPUT_REAL_FILENAME = INPUT_DIR / "gls-null_simulations-real_data.pkl"
display(INPUT_REAL_FILENAME)
assert INPUT_REAL_FILENAME.exists()

# %% [markdown] tags=[]
# # Null simulations - artificial gene-trait associations

# %% [markdown]
# ## Load data

# %% tags=[]
results = pd.read_pickle(INPUT_FILENAME)

# %% tags=[]
results.shape

# %% tags=[]
results.head()

# %% [markdown]
# ## Plot

# %% tags=[]
data = results["pvalue"].to_numpy()
uniform_data = np.linspace(data.min(), data.max(), num=data.shape[0])

# %% tags=[]
display(data[:5])
display(uniform_data[:5])

# %% tags=[]
with sns.plotting_context("paper", font_scale=1.5):
    fig, ax = plt.subplots(figsize=(5, 5))

    fig = qqplot_2samples(-np.log10(uniform_data), -np.log10(data), line="45", ax=ax)

    ax.set_xlabel("$-\log_{10}$(expected pvalue)")
    ax.set_ylabel("$-\log_{10}$(observed pvalue)")
    ax.set_title("QQ-Plot")

# %% [markdown] tags=[]
# # Null simulations - real gene-trait associations

# %% [markdown]
# ## Load data

# %% tags=[]
results = pd.read_pickle(INPUT_REAL_FILENAME)

# %% tags=[]
results.shape

# %% tags=[]
results.head()

# %% [markdown]
# ## Plot

# %% tags=[]
data = results["pvalue"].to_numpy()
uniform_data = np.linspace(data.min(), data.max(), num=data.shape[0])

# %% tags=[]
display(data[:5])
display(uniform_data[:5])

# %% tags=[]
with sns.plotting_context("paper", font_scale=1.5):
    fig, ax = plt.subplots(figsize=(5, 5))

    fig = qqplot_2samples(-np.log10(uniform_data), -np.log10(data), line="45", ax=ax)

    ax.set_xlabel("$-\log_{10}$(expected pvalue)")
    ax.set_ylabel("$-\log_{10}$(observed pvalue)")
    ax.set_title("QQ-Plot")

# %% tags=[]
