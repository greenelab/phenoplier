# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all,-execution,-papermill,-trusted
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
# This notebooks analyzes the gene bands at the top of each LV, and in particular, LV246 (associated with lipids).

# %% [markdown] tags=[]
# # Modules

# %% tags=[]
import pandas as pd

import conf
from entity import Gene
from data.recount2 import LVAnalysis

# %% [markdown] tags=[]
# # Settings

# %% tags=["parameters"]
BAND = "1p13"

# %% tags=["injected-parameters"]
# Parameters
PHENOPLIER_NOTEBOOK_FILEPATH = "nbs/15_gsa_gls/misc/explore_lv_genes_in_1p13.ipynb"


# %% [markdown] tags=[]
# # Load data

# %% [markdown] tags=[]
# ## MultiPLIER Z

# %% tags=[]
multiplier_z = pd.read_pickle(conf.MULTIPLIER["MODEL_Z_MATRIX_FILE"])

# %% tags=[]
multiplier_z.shape

# %% tags=[]
multiplier_z.head()

# %% [markdown] tags=[]
# # Gene gene info

# %% tags=[]
gene_objs = [
    Gene(name=gene_name)
    for gene_name in multiplier_z.index
    if gene_name in Gene.GENE_NAME_TO_ID_MAP
]

# %% tags=[]
len(gene_objs)

# %% tags=[]
gene_bands = [g.band for g in gene_objs]

# %% tags=[]
gene_bands[:5]

# %% tags=[]
gene_df = (
    pd.DataFrame(
        {
            "symbol": [g.name for g in gene_objs],
            "band": [g.band for g in gene_objs],
        }
    )
    .set_index("symbol")
    .squeeze()
)

# %% tags=[]
gene_df.shape

# %% tags=[]
gene_df.isna().any()

# %% tags=[]
gene_df = gene_df.dropna()

# %% tags=[]
gene_df.shape

# %% tags=[]
gene_df.head()


# %% [markdown] tags=[]
# # Create LV-band dataframe

# %% tags=[]
def get_lv_genes(lv_code: str):
    """
    Given an LV code (such as LV123), it returns a dataframe with gene symbols
    as index values, and two columns: the LV name with weights and "gene_band".
    The dataframe is sorted (in descending order) according to the gene weight.
    """
    lv_genes = multiplier_z[lv_code].sort_values(ascending=False)
    lv_obj = LVAnalysis(lv_code)
    return lv_obj.lv_genes.set_index("gene_name").loc[lv_genes.index]


# %% tags=[]
get_lv_genes("LV246")

# %% tags=[]
lv_gene_bands = {
    lv_code: get_lv_genes(lv_code).rename(columns={lv_code: "lv"})
    for lv_code in multiplier_z.columns
}

# %% tags=[]
lv_gene_bands["LV1"]

# %% [markdown] tags=[]
# # Summarize LV-band

# %% tags=[]
_tmp = lv_gene_bands["LV246"]

# %% tags=[]
_tmp2 = _tmp.head(70)["gene_band"].value_counts()

# %% tags=[]
_tmp2

# %% tags=[]
_tmp2[_tmp2.index.str.startswith("1p13")]


# %% tags=[]
def count_band(lv_gene_bands, band):
    """
    It takes the top 70 genes (around 1%) in the LV data (given by lv_gene_bands)
    and counts how many genes' bands starts with the value given by band.
    For instance, if band="1p13", it would count all genes in bands "1p13.1", "1p13.2"
    and all "1p13*".
    """
    top_bands = lv_gene_bands.head(70)["gene_band"].value_counts()
    return top_bands[top_bands.index.str.startswith(band)].sum()


# %% tags=[]
count_band(lv_gene_bands["LV1"], "1p13")

# %% tags=[]
BAND

# %% tags=[]
lv_band_summary = {k: count_band(v, BAND) for k, v in lv_gene_bands.items()}

# %% tags=[]
lv_band_summary_df = pd.Series(lv_band_summary).sort_values(ascending=False)

# %% tags=[]
lv_band_summary_df.shape

# %% tags=[]
lv_band_summary_df.head()

# %% [markdown] tags=[]
# # LV246

# %% tags=[]
# count how many top genes in LV246 are in 1p13
lv_band_summary_df["LV246"]

# %% tags=[]
# now, take a look at what are the top bands in LV246
lv_gene_bands["LV246"].head(20)

# %% tags=[]
lv_gene_bands["LV246"].head(70)["gene_band"].value_counts().head(20)

# %% tags=[]
