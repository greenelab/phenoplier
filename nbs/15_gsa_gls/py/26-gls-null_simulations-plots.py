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

# %% [markdown] tags=[]
# # Modules

# %% tags=[]
from pathlib import Path

# import statsmodels.api as sm
import numpy as np
from scipy.stats import uniform
import pandas as pd
from statsmodels.graphics.gofplots import qqplot_2samples
import matplotlib.pyplot as plt

import conf

# from gls import GLSPhenoplier

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
INPUT_DIR = conf.RESULTS["GLS"]
display(INPUT_DIR)

# %% tags=[]
INPUT_FILENAME = INPUT_DIR / "gls-null_simulations.pkl"
display(INPUT_FILENAME)
assert INPUT_FILENAME.exists()

# %% [markdown] tags=[]
# # Load data

# %% [markdown] tags=[]
# ## Null simulations

# %% tags=[]
results = pd.read_pickle(INPUT_FILENAME)

# %% tags=[]
results.shape

# %% tags=[]
results.head()

# %% [markdown] tags=[]
# # Plots

# %% tags=[]
# data = -np.log10(results["pvalue"].to_numpy())
data = results["pvalue"].to_numpy()
# uniform_data = -np.log10(uniform.rvs(size=data.shape[0]))
uniform_data = np.linspace(data.min(), data.max(), num=data.shape[0])

# %% tags=[]
uniform_data

# %% tags=[]
fig, ax = plt.subplots(figsize=(5, 5))

fig = qqplot_2samples(-np.log10(uniform_data), -np.log10(data), line="45", ax=ax)

ax.set_xlabel("-log10(expected pvalue)")
ax.set_ylabel("-log10(observed pvalue)")
ax.set_title("QQ-Plot")

# %% tags=[]
