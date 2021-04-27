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
# This notebook separates all SNPs in prediction models into "genes" (SNPs are grouped according to whethere they are predictors for a gene's expresssion).

# %% [markdown]
# # Modules

# %% tags=[]
# %load_ext autoreload
# %autoreload 2

# %%
import numpy as np
import pandas as pd
import sqlite3

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

import conf

# %%
load_rda = ro.r["load"]

# %% [markdown]
# # Load data

# %% [markdown]
# ## LD blocks

# %%
input_file = str(conf.PHENOMEXCAN["LD_BLOCKS"]["LD_BLOCKS_FILE"])
display(input_file)

# %%
load_rda(input_file)

# %%
ro.r["ld_block_pickrell_eur_b38"]

# %% [markdown]
# ### Show metadata

# %%
ro.r["ld_block_pickrell_eur_b38"][1]

# %%
np.array(ro.r["ld_block_pickrell_eur_b38"][1][0])

# %%
np.array(ro.r["ld_block_pickrell_eur_b38"][1][1])

# %% [markdown]
# ### Load LD blocks

# %%
ld_block_r = ro.r["ld_block_pickrell_eur_b38"][0]

# %%
ld_block_r.rownames

# %%
ld_block_r.colnames

# %%
with localconverter(ro.default_converter + pandas2ri.converter):
    ld_block_df = ro.conversion.rpy2py(ld_block_r)

# %%
ld_block_df

# %%
ld_block_df.dtypes

# %% [markdown]
# ### Save in tsv

# %%
output_file = conf.PHENOMEXCAN["LD_BLOCKS"]["LD_BLOCKS_FILE"].parent / (
    conf.PHENOMEXCAN["LD_BLOCKS"]["LD_BLOCKS_FILE"].stem + ".tsv"
)
display(output_file)

# %%
ld_block_df.to_csv(output_file, sep="\t", index=False)

# %% [markdown]
# ## SNPs in predictions models

# %%
mashr_models_db_files = list(
    conf.PHENOMEXCAN["PREDICTION_MODELS"]["MASHR"].glob("*.db")
)

# %%
assert len(mashr_models_db_files) == 49

# %%
all_variants_ids = []

for m in mashr_models_db_files:
    print(f"Processing {m.name}")
    tissue = m.name.split("mashr_")[1].split(".db")[0]

    with sqlite3.connect(m) as conn:
        df = pd.read_sql("select gene, varID from weights", conn)
        df["gene"] = df["gene"].apply(lambda x: x.split(".")[0])
        df = df.assign(tissue=tissue)

        all_variants_ids.append(df)

# %%
all_gene_snps = pd.concat(all_variants_ids, ignore_index=True)

# %%
all_gene_snps.shape

# %%
all_gene_snps.head()

# %% [markdown]
# # Create variant block dataframe

# %% [markdown]
# ## Preprocess SNPs data

# %%
variants_ld_block_df = all_gene_snps[["varID", "gene"]]

# %%
variants_ld_block_df.shape

# %%
variants_info = variants_ld_block_df["varID"].str.split("_", expand=True)

# %%
variants_info.shape

# %%
assert variants_ld_block_df.shape[0] == variants_info.shape[0]

# %%
variants_ld_block_df = variants_ld_block_df.join(variants_info)[
    ["varID", 0, 1, 2, 3, "gene"]
]

# %%
assert variants_ld_block_df.shape[0] == variants_info.shape[0]

# %%
variants_ld_block_df.head()

# %%
variants_ld_block_df = variants_ld_block_df.rename(
    columns={
        0: "chr",
        1: "position",
        2: "ref_allele",
        3: "eff_allele",
    }
)

# %%
variants_ld_block_df["chr"] = variants_ld_block_df["chr"].apply(lambda x: int(x[3:]))

# %%
variants_ld_block_df["position"] = variants_ld_block_df["position"].astype(int)

# %%
variants_ld_block_df.shape

# %%
variants_ld_block_df.head()

# %%
variants_ld_block_df.dtypes

# %% [markdown]
# # Testing

# %%
_unique_chr_per_ld_block = variants_ld_block_df.groupby("gene").apply(
    lambda x: x["chr"].unique().shape[0]
)
display(_unique_chr_per_ld_block)

# %%
display(_unique_chr_per_ld_block.unique())
assert _unique_chr_per_ld_block.unique().shape[0] == 1
assert _unique_chr_per_ld_block.unique()[0] == 1

# %% [markdown]
# # Save

# %%
variants_ld_block_df.head()

# %%
output_file = conf.PHENOMEXCAN["LD_BLOCKS"]["BASE_DIR"] / "mashr_snps_gene_blocks.pkl"
display(output_file)

# %%
variants_ld_block_df.to_pickle(output_file)

# %%
