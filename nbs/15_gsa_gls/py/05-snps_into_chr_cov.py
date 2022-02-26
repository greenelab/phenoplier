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
# This notebook computes the covariance of SNPs for each chr.

# %% [markdown] tags=[]
# # Modules

# %% tags=[]
# %load_ext autoreload
# %autoreload 2

# %% tags=[]
import gc
import sqlite3

import numpy as np
import pandas as pd
from tqdm import tqdm

import conf
from entity import Gene

# %% [markdown]
# # Settings

# %%
COV_DTYPE = np.float32

# %% tags=["parameters"]
# mashr
EQTL_MODEL = "MASHR"
EQTL_MODEL_FILES_PREFIX = "mashr_"

# # elastic net
# EQTL_MODEL = "ELASTIC_NET"
# EQTL_MODEL_FILES_PREFIX = "en_"

# make it read the prefix from conf.py
EQTL_MODEL_FILES_PREFIX = None

# %%
if EQTL_MODEL_FILES_PREFIX is None:
    EQTL_MODEL_FILES_PREFIX = conf.PHENOMEXCAN["PREDICTION_MODELS"][
        f"{EQTL_MODEL}_PREFIX"
    ]

# %%
display(f"Using eQTL model: {EQTL_MODEL} / {EQTL_MODEL_FILES_PREFIX}")

# %% [markdown] tags=[]
# # Load data

# %% [markdown] tags=[]
# ## SNPs in predictions models

# %% tags=[]
mashr_models_db_files = list(
    conf.PHENOMEXCAN["PREDICTION_MODELS"][EQTL_MODEL].glob("*.db")
)

# %% tags=[]
assert len(mashr_models_db_files) == 49

# %% tags=[]
all_variants_ids = []

for m in mashr_models_db_files:
    print(f"Processing {m.name}")
    tissue = m.name.split(EQTL_MODEL_FILES_PREFIX)[1].split(".db")[0]

    with sqlite3.connect(m) as conn:
        df = pd.read_sql("select gene, varID from weights", conn)
        df["gene"] = df["gene"].apply(lambda x: x.split(".")[0])
        df = df.assign(tissue=tissue)

        all_variants_ids.append(df)

# %% tags=[]
all_gene_snps = pd.concat(all_variants_ids, ignore_index=True)

# %% tags=[]
all_gene_snps.shape

# %% tags=[]
all_gene_snps.head()

# %% tags=[]
all_snps_in_models = set(all_gene_snps["varID"].unique())

# %% [markdown] tags=[]
# ## MultiPLIER Z

# %% tags=[]
multiplier_z = pd.read_pickle(conf.MULTIPLIER["MODEL_Z_MATRIX_FILE"])

# %% tags=[]
multiplier_z.shape

# %% tags=[]
multiplier_z.head()

# %% [markdown] tags=[]
# ## 1000G variants metadata

# %% tags=[]
input_file = (
    conf.PHENOMEXCAN["LD_BLOCKS"]["1000G_GENOTYPE_DIR"] / "variant_metadata.parquet"
)
display(input_file)

# %% tags=[]
variants_metadata = pd.read_parquet(input_file, columns=["id"])

# %% tags=[]
variants_metadata.shape

# %% tags=[]
variants_metadata.head()

# %% tags=[]
variants_ids_with_genotype = set(variants_metadata["id"])

# %% tags=[]
len(variants_ids_with_genotype)

# %% tags=[]
list(variants_ids_with_genotype)[:10]

# %% tags=[]
del variants_metadata

# %% [markdown] tags=[]
# # How many variants in predictions models are present in 1000G?

# %% tags=[]
n_snps_in_models = len(all_snps_in_models)
display(n_snps_in_models)

# %% tags=[]
n_snps_in_1000g = len(all_snps_in_models.intersection(variants_ids_with_genotype))
display(n_snps_in_1000g)

# %% tags=[]
n_snps_in_1000g / n_snps_in_models

# %% [markdown] tags=[]
# # Get final list of genes in MultiPLIER

# %% tags=[]
genes_in_z = [
    Gene(name=gene_name).ensembl_id
    for gene_name in multiplier_z.index
    if gene_name in Gene.GENE_NAME_TO_ID_MAP
]

# %% tags=[]
len(genes_in_z)

# %% tags=[]
genes_in_z[:5]

# %% tags=[]
genes_in_z = set(genes_in_z)

# %% tags=[]
len(genes_in_z)

# %% tags=[]
# keep genes in MultiPLIER only
display(all_gene_snps.shape)

all_gene_snps = all_gene_snps[all_gene_snps["gene"].isin(genes_in_z)]

display(all_gene_snps.shape)

# %% [markdown] tags=[]
# # (For MultiPLIER genes): How many variants in predictions models are present in 1000G?

# %% tags=[]
all_snps_in_models_multiplier = set(all_gene_snps["varID"])

n_snps_in_models = len(all_snps_in_models_multiplier)
display(n_snps_in_models)

# %% tags=[]
n_snps_in_1000g = len(
    all_snps_in_models_multiplier.intersection(variants_ids_with_genotype)
)
display(n_snps_in_1000g)

# %% tags=[]
n_snps_in_1000g / n_snps_in_models

# %% [markdown] tags=[]
# ## Preprocess SNPs data

# %% tags=[]
variants_ld_block_df = all_gene_snps[["varID"]].drop_duplicates()

# %% tags=[]
variants_ld_block_df.shape

# %% tags=[]
variants_ld_block_df.head()

# %% tags=[]
variants_info = variants_ld_block_df["varID"].str.split("_", expand=True)

# %% tags=[]
variants_info.shape

# %% tags=[]
assert variants_ld_block_df.shape[0] == variants_info.shape[0]

# %% tags=[]
variants_ld_block_df = variants_ld_block_df.join(variants_info)[["varID", 0, 1, 2, 3]]

# %% tags=[]
assert variants_ld_block_df.shape[0] == variants_info.shape[0]

# %% tags=[]
variants_ld_block_df.head()

# %% tags=[]
variants_ld_block_df = variants_ld_block_df.rename(
    columns={
        0: "chr",
        1: "position",
        2: "ref_allele",
        3: "eff_allele",
    }
)

# %% tags=[]
variants_ld_block_df["chr"] = variants_ld_block_df["chr"].apply(lambda x: int(x[3:]))

# %% tags=[]
variants_ld_block_df["position"] = variants_ld_block_df["position"].astype(int)

# %% tags=[]
variants_ld_block_df.shape

# %% tags=[]
variants_ld_block_df.head()

# %% tags=[]
variants_ld_block_df.dtypes


# %% [markdown] tags=[]
# # Covariance for each chromosome block

# %% [markdown] tags=[]
# ## Functions

# %%
def covariance(df, dtype):
    n = df.shape[0]
    df = df.sub(df.mean(), axis=1).astype(dtype)
    return df.T.dot(df) / (n - 1)


# %%
# testing
rs = np.random.RandomState(0)

_test_data = pd.DataFrame(rs.normal(size=(50, 5)), columns=[f"c{i}" for i in range(5)])

# float64
pd.testing.assert_frame_equal(
    covariance(_test_data, np.float64),
    _test_data.cov(),
    rtol=1e-10,
    atol=1e-10,
    check_dtype=True,
)

# float32
pd.testing.assert_frame_equal(
    covariance(_test_data, np.float32),
    _test_data.cov(),
    rtol=1e-5,
    atol=1e-8,
    check_dtype=False,
)

del _test_data


# %% tags=[]
def compute_snps_cov(snps_df):
    assert snps_df["chr"].unique().shape[0] == 1
    chromosome = snps_df["chr"].unique()[0]

    # keep variants only present in genotype
    snps_ids = list(set(snps_df["varID"]).intersection(variants_ids_with_genotype))

    chromosome_file = (
        conf.PHENOMEXCAN["LD_BLOCKS"]["1000G_GENOTYPE_DIR"]
        / f"chr{chromosome}.variants.parquet"
    )
    snps_genotypes = pd.read_parquet(chromosome_file, columns=snps_ids)

    return covariance(snps_genotypes, COV_DTYPE)


# %% tags=[]
# testing
_tmp_snps = variants_ld_block_df[variants_ld_block_df["chr"] == 22]
assert _tmp_snps.shape[0] > 0

# %% tags=[]
_tmp_snps.shape

# %% tags=[]
n_expected = len(set(_tmp_snps["varID"]).intersection(variants_ids_with_genotype))
display(n_expected)

# %% tags=[]
_tmp = compute_snps_cov(_tmp_snps)

# %% tags=[]
assert _tmp.shape == (n_expected, n_expected)
assert not _tmp.isna().any().any()

# %%
del _tmp_snps, _tmp

# %% [markdown] tags=[]
# ## Compute covariance and save

# %% tags=[]
output_file = conf.PHENOMEXCAN["LD_BLOCKS"][EQTL_MODEL]["SNPS_COVARIANCE_FILE"]
display(output_file)

# %% tags=[]
with pd.HDFStore(output_file, mode="w", complevel=4) as store:
    pbar = tqdm(
        variants_ld_block_df.groupby("chr"),
        ncols=100,
        total=variants_ld_block_df["chr"].unique().shape[0],
    )

    store["metadata"] = variants_ld_block_df

    for grp_name, grp_data in pbar:
        pbar.set_description(f"{grp_name} {grp_data.shape}")
        snps_cov = compute_snps_cov(grp_data)  # .astype(COV_DTYPE)
        assert not snps_cov.isna().any().any()
        store[f"chr{grp_name}"] = snps_cov

        del snps_cov
        store.flush()

        gc.collect()

# %% [markdown] tags=[]
# # Testing

# %% tags=[]
_tmp = variants_ld_block_df[variants_ld_block_df["chr"] == 1]

# %% tags=[]
_tmp.shape

# %% tags=[]
assert _tmp.shape[0] > 0

# %% tags=[]
n_expected = len(set(_tmp["varID"]).intersection(variants_ids_with_genotype))
display(n_expected)
assert n_expected > 0

# %% tags=[]
with pd.HDFStore(output_file, mode="r") as store:
    df = store["chr1"]
    assert df.shape == (n_expected, n_expected)
    assert not df.isna().any().any()

# %% tags=[]
