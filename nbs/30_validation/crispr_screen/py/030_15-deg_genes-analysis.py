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

# %% [markdown] tags=[]
# # Description

# %% [markdown] tags=[]
# TODO

# %% [markdown] tags=[]
# # Modules loading

# %% tags=[]
# %load_ext autoreload
# %autoreload 2

# %% tags=[]
from pathlib import Path

import pandas as pd
from IPython.display import HTML

from entity import Trait
from data.cache import read_data
import conf

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
EXPERIMENT_NAME = "lv"
LIPIDS_GENE_SET = "gene_set_increase"

# %% tags=[]
# OUTPUT_DIR = Path(
#     conf.RESULTS["CRISPR_ANALYSES"]["BASE_DIR"], f"{EXPERIMENT_NAME}-{LIPIDS_GENE_SET}"
# )
# OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
# display(OUTPUT_DIR)

# %% [markdown] tags=[]
# # Data loading

# %% [markdown] tags=[]
# ## MultiPLIER Z matrix

# %%
multiplier_z = read_data(conf.MULTIPLIER["MODEL_Z_MATRIX_FILE"])

# %%
multiplier_z.shape

# %%
multiplier_z.head()

# %% [markdown] tags=[]
# ## PhenomeXcan projection

# %% tags=[]
input_filepath = Path(
    conf.RESULTS["PROJECTIONS_DIR"],
    "projection-smultixcan-efo_partial-mashr-zscores.pkl",
).resolve()
display(input_filepath)

# %% tags=[]
phenomexcan_data = pd.read_pickle(input_filepath).T

# %% tags=[]
phenomexcan_data.shape

# %% tags=[]
phenomexcan_data.head()

# %% [markdown] tags=[]
# ## LVs enrichment on DEG from CRISPR screen

# %% tags=[]
deg_enrich = pd.read_csv(
    Path(conf.RESULTS["CRISPR_ANALYSES"]["BASE_DIR"], "fgsea-all_lvs.tsv").resolve(),
    sep="\t",
)

# %% tags=[]
deg_enrich.shape

# %% tags=[]
deg_enrich.head()

# %% tags=[]
deg_enrich_max_idx = deg_enrich.groupby(["lv", "pathway"])["padj"].idxmax()

# %% tags=[]
deg_enrich = deg_enrich.loc[deg_enrich_max_idx].reset_index(drop=True)
display(deg_enrich.shape)
display(deg_enrich.head())

# %% [markdown] tags=[]
# ## MultiPLIER summary

# %% tags=[]
multiplier_model_summary = read_data(conf.MULTIPLIER["MODEL_SUMMARY_FILE"])

# %% tags=[]
multiplier_model_summary.shape

# %% tags=[]
multiplier_model_summary.head()

# %% [markdown] tags=[]
# # Analysis

# %% tags=[]
df = deg_enrich[
    deg_enrich["pathway"].isin(("gene_set_increase",)) & (deg_enrich["padj"] < 0.05)
].sort_values("padj", ascending=True)

# %% tags=[]
df.shape

# %%
df.head()

# %% tags=[]
with pd.option_context(
    "display.max_rows", None, "display.max_columns", None, "max_colwidth", None
):
    _tmp = df[df["lv"] == "LV246"]
    display(_tmp)

# %%
_tmp = multiplier_z["LV246"]
_tmp.loc["DGAT2,ACACA,CEBPA,PLIN2,FERMT2,SREBF2,MED8".split(",")]

# %%

# %%
_tmp = multiplier_z[df["lv"].values].iloc[:, 0]

# %%
_tmp[_tmp > 0.0]

# %%
multiplier_z[df["lv"].values].apply(lambda x: x[x > 0].shape[0]).mean()

# %%
