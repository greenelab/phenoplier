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
# **TODO UPDATE**
#
# (Please, take a look at the README.md file in this directory for instructions on how to run this notebook)
#
# This notebook reads all gene correlations across all chromosomes and computes a single correlation matrix by assembling a big correlation matrix with all genes.

# %% [markdown] tags=[]
# # Modules

# %% tags=[]
# %load_ext autoreload
# %autoreload 2

# %% tags=[]
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.spatial.distance import squareform
from scipy import sparse
import pandas as pd
from tqdm import tqdm

import conf
from utils import chunker
from entity import Gene

# %% [markdown] tags=[]
# # Settings

# %% tags=["parameters"]
# reference panel
# REFERENCE_PANEL = "GTEX_V8"
REFERENCE_PANEL = "1000G"

# prediction models
## mashr
EQTL_MODEL = "MASHR"

# This is one S-MultiXcan result file on the same target cohort
# Genes will be read from here to align the correlation matrices
SMULTIXCAN_RESULTS_TEMPLATE = (
    conf.RESULTS["GLS_NULL_SIMS"]
    / "twas"
    / "smultixcan"
    / "random.pheno0-gtex_v8-mashr-smultixcan.txt"
)

# %% tags=["injected-parameters"]
# Parameters
REFERENCE_PANEL = "1000G"
EQTL_MODEL = "MASHR"


# %% tags=[]
assert (SMULTIXCAN_RESULTS_TEMPLATE is not None) and (
    SMULTIXCAN_RESULTS_TEMPLATE.exists()
), "You have to provide the path to a S-MultiXcan results file"

# %% tags=[]
display(f"Using eQTL model: {EQTL_MODEL}")

# %% tags=[]
REFERENCE_PANEL_DIR = conf.PHENOMEXCAN["LD_BLOCKS"][f"{REFERENCE_PANEL}_GENOTYPE_DIR"]

# %% tags=[]
display(f"Using reference panel folder: {str(REFERENCE_PANEL_DIR)}")

# %% tags=[]
OUTPUT_DIR_BASE = (
    conf.PHENOMEXCAN["LD_BLOCKS"][f"GENE_CORRS_DIR"]
    / REFERENCE_PANEL.lower()
    / EQTL_MODEL.lower()
)
display(OUTPUT_DIR_BASE)
OUTPUT_DIR_BASE.mkdir(parents=True, exist_ok=True)

# %% [markdown] tags=[]
# # Load data

# %% [markdown] tags=[]
# ## S-MultiXcan genes

# %% tags=[]
smultixcan_df = pd.read_csv(SMULTIXCAN_RESULTS_TEMPLATE, sep="\t")

# %% tags=[]
smultixcan_df.shape

# %% tags=[]
smultixcan_df.head()

# %% tags=[]
assert not smultixcan_df.isin([np.inf, -np.inf]).any().any()

# %% tags=[]
# remove NaNs
smultixcan_df = smultixcan_df.dropna(subset=["pvalue"])
display(smultixcan_df.shape)

# %% tags=[]
smultixcan_genes = set(smultixcan_df["gene_name"].tolist())

# %% tags=[]
len(smultixcan_genes)

# %% tags=[]
list(smultixcan_genes)[:5]

# %% [markdown] tags=[]
# ## Gene correlations

# %% tags=[]
input_file_name_template = conf.PHENOMEXCAN["LD_BLOCKS"][
    "GENE_CORRS_FILE_NAME_TEMPLATES"
]["GENE_CORR_AVG"]

input_file = OUTPUT_DIR_BASE / input_file_name_template.format(
    prefix="",
    suffix=f"-gene_symbols",
)

# %% tags=[]
# load correlation matrix
gene_corrs = pd.read_pickle(input_file)

# %% tags=[]
gene_corrs.shape

# %% tags=[]
gene_corrs.head()

# %% [markdown] tags=[]
# ## Define output dir (based on gene correlation's file)

# %% tags=[]
# output file (hdf5)
output_dir = Path(input_file).with_suffix(".per_lv")
output_dir.mkdir(parents=True, exist_ok=True)

display(output_dir)

# %% [markdown] tags=[]
# ## MultiPLIER Z

# %% tags=[]
multiplier_z = pd.read_pickle(conf.MULTIPLIER["MODEL_Z_MATRIX_FILE"])

# %% tags=[]
multiplier_z.shape

# %% tags=[]
multiplier_z.head()

# %% [markdown] tags=[]
# ## Common genes

# %% tags=[]
common_genes = sorted(
    list(
        smultixcan_genes.intersection(multiplier_z.index).intersection(gene_corrs.index)
    )
)

# %% tags=[]
len(common_genes)

# %% tags=[]
common_genes[:5]


# %% [markdown] tags=[]
# # Compute inverse correlation matrix for each LV

# %% tags=[]
def store_df(nparray, base_filename):
    if base_filename in ("metadata", "gene_names"):
        np.savez_compressed(output_dir / (base_filename + ".npz"), data=nparray)
    else:
        sparse.save_npz(
            output_dir / (base_filename + ".npz"),
            sparse.csc_matrix(nparray),
            compressed=False,
        )


# %% tags=[]
def compute_chol_inv(lv_codes):
    for lv_code in lv_codes:
        corr_mat_sub = pd.DataFrame(
            np.identity(len(common_genes)),
            index=common_genes.copy(),
            columns=common_genes.copy(),
        )

        lv_data = multiplier_z[lv_code]
        lv_nonzero_genes = lv_data[lv_data > 0].index
        lv_nonzero_genes = lv_nonzero_genes.intersection(corr_mat_sub.index)

        corr_mat_sub.loc[lv_nonzero_genes, lv_nonzero_genes] = gene_corrs.loc[
            lv_nonzero_genes, lv_nonzero_genes
        ]

        chol_mat = np.linalg.cholesky(corr_mat_sub)
        chol_inv = np.linalg.inv(chol_mat)

        store_df(chol_inv, lv_code)


# %% tags=[]
# divide LVs in chunks for parallel processing
lvs_chunks = list(chunker(list(multiplier_z.columns), 50))

# %% tags=[]
# metadata
metadata = np.array([REFERENCE_PANEL, EQTL_MODEL])
store_df(metadata, "metadata")

# gene names
gene_names = np.array(common_genes)
store_df(gene_names, "gene_names")

# pbar = tqdm(total=multiplier_z.columns.shape[0])

with ProcessPoolExecutor(max_workers=conf.GENERAL["N_JOBS"]) as executor, tqdm(
    total=len(lvs_chunks), ncols=100
) as pbar:
    tasks = [executor.submit(compute_chol_inv, chunk) for chunk in lvs_chunks]
    for future in as_completed(tasks):
        res = future.result()
        pbar.update(1)


# %% [markdown] tags=[]
# ## Some checks

# %% tags=[]
def load_df(base_filename):
    full_filepath = output_dir / (base_filename + ".npz")

    if base_filename in ("metadata", "gene_names"):
        return np.load(full_filepath)["data"]
    else:
        return sparse.load_npz(full_filepath).toarray()


# %% tags=[]
_genes = load_df("gene_names")

# %% tags=[]
display(len(_genes))
assert len(_genes) == len(common_genes)

# %% tags=[]
_metadata = load_df("metadata")

# %% tags=[]
display(_metadata)
assert _metadata[0] == REFERENCE_PANEL
assert _metadata[1] == EQTL_MODEL

# %% tags=[]
lv1_inv = load_df("LV1")

# %% tags=[]
lv2_inv = load_df("LV2")

# %% tags=[]
lv_last_inv = load_df("LV987")

# %% tags=[]
assert lv1_inv.shape == lv2_inv.shape

# %% tags=[]
assert not np.allclose(lv1_inv, lv2_inv)

# %% tags=[]
assert not np.allclose(lv1_inv, lv_last_inv)

# %% tags=[]
assert not np.allclose(lv2_inv, lv_last_inv)

# %% tags=[]
