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
# This notebook computes the covariance inside each LD block

# %% [markdown]
# # Modules

# %% tags=[]
# %load_ext autoreload
# %autoreload 2

# %%
import numpy as np
import pandas as pd
from tqdm import tqdm

# import rpy2.robjects as ro
# from rpy2.robjects import pandas2ri
# from rpy2.robjects.conversion import localconverter

import conf

# %% [markdown]
# # Load data

# %% [markdown]
# ## SNPs per LD block data

# %%
input_file = conf.PHENOMEXCAN["LD_BLOCKS"]["BASE_DIR"] / "mashr_snps_ld_blocks.pkl"
display(input_file)

# %%
variants_ld_block_df = pd.read_pickle(input_file)

# %%
variants_ld_block_df.shape

# %%
variants_ld_block_df.head()

# %% [markdown]
# ## 1000G variants metadata

# %%
input_file = (
    conf.PHENOMEXCAN["LD_BLOCKS"]["1000G_GENOTYPE_DIR"] / "variant_metadata.parquet"
)
display(input_file)

# %%
variants_metadata = pd.read_parquet(input_file, columns=["id"])

# %%
variants_metadata.shape

# %%
variants_metadata.head()

# %%
variants_ids_with_genotype = set(variants_metadata["id"])

# %%
len(variants_ids_with_genotype)

# %%
list(variants_ids_with_genotype)[:10]

# %%
del variants_metadata


# %% [markdown]
# # Compute covariance for each LD block

# %%
def compute_snps_cov(snps_df):
    assert snps_df["chr"].unique().shape[0]
    chromosome = snps_df["chr"].unique()[0]

    # keep variants only present in genotype
    snps_ids = list(set(snps_df.index).intersection(variants_ids_with_genotype))

    chromosome_file = (
        conf.PHENOMEXCAN["LD_BLOCKS"]["1000G_GENOTYPE_DIR"]
        / f"chr{chromosome}.variants.parquet"
    )
    snps_genotypes = pd.read_parquet(chromosome_file, columns=snps_ids)

    return snps_genotypes.cov()


# %%
output_file = conf.PHENOMEXCAN["LD_BLOCKS"]["BASE_DIR"] / "mashr_snps_ld_blocks_cov.h5"
display(output_file)

# %%
with pd.HDFStore(output_file, mode="w", complevel=4) as store:
    pbar = tqdm(
        variants_ld_block_df.groupby("ld_block"),
        ncols=100,
        total=variants_ld_block_df["ld_block"].unique().shape[0],
    )

    store["metadata"] = variants_ld_block_df

    for grp_name, grp_data in pbar:
        pbar.set_description(f"{grp_name} {grp_data.shape}")
        snps_cov = compute_snps_cov(grp_data).astype(np.float32)
        assert not snps_cov.isna().any().any()
        store[grp_name] = snps_cov

# %% [markdown]
# # Testing

# %%
_tmp = variants_ld_block_df[variants_ld_block_df["ld_block"] == "chr10_10"]

# %%
_tmp.shape

# %%
n_expected = len(set(_tmp.index).intersection(variants_ids_with_genotype))
display(n_expected)

# %%
with pd.HDFStore(output_file, mode="r") as store:
    df = store["chr10_10"]
    assert df.shape == (n_expected, n_expected)
    assert not df.isna().any().any()

# %%
