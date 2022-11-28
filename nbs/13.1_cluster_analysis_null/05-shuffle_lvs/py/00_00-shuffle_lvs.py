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

# %% [markdown] tags=[]
# # Description

# %% [markdown] tags=[]
# This notebook implements one of the strategies (one out of two) to generate a null distribution for clustering results.
# This strategy is referred to as "Null #2" in the manuscript.
# See notebook `../00-shuffle_genes/00_00-shuffle_genes.ipynb` for introductory details.
#
# This strategy shuffles the latent space instead of the the input data. For this, it projects the input matrix **M** (genes x traits) into the latent space, and then it shuffles LVs for each trait in the projected matrix. Finally, this projected matrix is used in the clustering pipeline (rest of the notebooks in this folder) to get the results.

# %% [markdown] tags=[]
# # Modules loading

# %% tags=[]
# %load_ext autoreload
# %autoreload 2

# %% tags=[]
from pathlib import Path

from IPython.display import display
import numpy as np
from scipy import stats
import pandas as pd
import pytest

import conf

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
NULL_DIR = conf.RESULTS["CLUSTERING_NULL_DIR"] / "shuffle_lvs"

# %% tags=[]
RESULTS_PROJ_OUTPUT_DIR = Path(conf.RESULTS["PROJECTIONS_DIR"])
RESULTS_PROJ_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

display(RESULTS_PROJ_OUTPUT_DIR)

# %% tags=[]
OUTPUT_DIR = Path(
    NULL_DIR,
    "projections",
).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

display(OUTPUT_DIR)

# %% tags=[]
rs = np.random.RandomState(0)

# %% [markdown] tags=[]
# # Load projection of S-MultiXcan into LV space

# %% tags=[]
smultixcan_results_filename = conf.PHENOMEXCAN[
    "SMULTIXCAN_EFO_PARTIAL_MASHR_ZSCORES_FILE"
]

display(smultixcan_results_filename)
assert smultixcan_results_filename.exists()

# %% tags=[]
results_filename_stem = smultixcan_results_filename.stem
display(results_filename_stem)

# %% tags=[]
input_file = Path(
    RESULTS_PROJ_OUTPUT_DIR, f"projection-{results_filename_stem}.pkl"
).resolve()

display(input_file)

# %% tags=[]
projected_data = pd.read_pickle(input_file)

# %% tags=[]
projected_data.shape

# %% tags=[]
projected_data.head()

# %% [markdown] tags=[]
# # Shuffle projected data

# %% tags=[]
shuffled_projected_data = projected_data.apply(
    lambda x: x.sample(frac=1, random_state=rs).to_numpy()
)

# %% tags=[]
shuffled_projected_data.shape

# %% tags=[]
shuffled_projected_data.head()

# %% [markdown] tags=[]
# ## Testing

# %% tags=[]
assert stats.pearsonr(projected_data.loc["LV1"], projected_data.loc["LV1"])[
    0
] == pytest.approx(1.0)
assert stats.pearsonr(
    shuffled_projected_data.loc["LV1"], shuffled_projected_data.loc["LV1"]
)[0] == pytest.approx(1.0)

# %% tags=[]
_tmp = stats.pearsonr(shuffled_projected_data.loc["LV1"], projected_data.loc["LV1"])
display(_tmp)
assert _tmp[0] == pytest.approx(0.0, rel=0, abs=0.01)

# %% tags=[]
assert stats.pearsonr(
    projected_data["100001_raw-Food_weight"], projected_data["100001_raw-Food_weight"]
)[0] == pytest.approx(1.0)
assert stats.pearsonr(
    shuffled_projected_data["100001_raw-Food_weight"],
    shuffled_projected_data["100001_raw-Food_weight"],
)[0] == pytest.approx(1.0)

# %% tags=[]
_tmp = stats.pearsonr(
    shuffled_projected_data["100001_raw-Food_weight"],
    projected_data["100001_raw-Food_weight"],
)
display(_tmp)
assert _tmp[0] == pytest.approx(0.02, rel=0, abs=0.01)

# %% [markdown] tags=[]
# # Quick analysis

# %% [markdown] tags=[]
# Ensure we broke known relationships

# %% tags=[]
(shuffled_projected_data.loc["LV603"].sort_values(ascending=False).head(20))

# %% tags=[]
(shuffled_projected_data.loc["LV136"].sort_values(ascending=False).head(20))

# %% tags=[]
(shuffled_projected_data.loc["LV844"].sort_values(ascending=False).head(20))

# %% tags=[]
(shuffled_projected_data.loc["LV246"].sort_values(ascending=False).head(20))

# %% [markdown] tags=[]
# # Save

# %% tags=[]
output_file = Path(OUTPUT_DIR, f"projection-{results_filename_stem}.pkl").resolve()

display(output_file)

# %% tags=[]
shuffled_projected_data.to_pickle(output_file)

# %% tags=[]
