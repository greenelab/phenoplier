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
# a cohort name (it could be something like UK_BIOBANK, etc)
COHORT_NAME = None

# reference panel such as 1000G or GTEX_V8
REFERENCE_PANEL = None

# predictions models such as MASHR or ELASTIC_NET
EQTL_MODEL = None

# This is one S-MultiXcan result file on the same target cohort
# Genes will be read from here to align the correlation matrices
SMULTIXCAN_FILE = None

LV_CODE = None

# a number from 0.0 to 1.0 indicating a percentile of the genes in the LV to keep
# if zero, then all nonzero weighted genes in the LV will be kept
LV_PERCENTILE = None

# %% tags=[]
assert COHORT_NAME is not None and len(COHORT_NAME) > 0, "A cohort name must be given"

COHORT_NAME = COHORT_NAME.lower()
display(f"Cohort name: {COHORT_NAME}")

# %% tags=[]
assert (
    REFERENCE_PANEL is not None and len(REFERENCE_PANEL) > 0
), "A reference panel must be given"

display(f"Reference panel: {REFERENCE_PANEL}")

# %% tags=[]
assert (
    EQTL_MODEL is not None and len(EQTL_MODEL) > 0
), "A prediction/eQTL model must be given"

EQTL_MODEL_FILES_PREFIX = conf.PHENOMEXCAN["PREDICTION_MODELS"][f"{EQTL_MODEL}_PREFIX"]
display(f"eQTL model: {EQTL_MODEL} / {EQTL_MODEL_FILES_PREFIX}")

# %% tags=[]
assert (
    SMULTIXCAN_FILE is not None and len(SMULTIXCAN_FILE) > 0
), "An S-MultiXcan result file path must be given"
SMULTIXCAN_FILE = Path(SMULTIXCAN_FILE).resolve()
assert SMULTIXCAN_FILE.exists(), "S-MultiXcan result file does not exist"

display(f"S-MultiXcan file path: {str(SMULTIXCAN_FILE)}")

# %% tags=[]
assert LV_CODE is not None and len(LV_CODE) > 0, "An LV code must be given"

display(f"LV code: {LV_CODE}")

# %% tags=[]
if LV_PERCENTILE is not None:
    LV_PERCENTILE = float(LV_PERCENTILE)

display(f"LV percentile: {LV_PERCENTILE}")

# %% tags=[]
OUTPUT_DIR_BASE = (
    conf.RESULTS["GLS"]
    / "gene_corrs"
    / "cohorts"
    / COHORT_NAME.lower()
    / REFERENCE_PANEL.lower()
    / EQTL_MODEL.lower()
)
OUTPUT_DIR_BASE.mkdir(parents=True, exist_ok=True)

display(f"Using output dir base: {OUTPUT_DIR_BASE}")

# %% [markdown] tags=[]
# # Load data

# %% [markdown] tags=[]
# ## S-MultiXcan genes

# %% tags=[]
smultixcan_df = pd.read_csv(SMULTIXCAN_FILE, sep="\t")

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
sorted(list(smultixcan_genes))[:5]

# %% [markdown] tags=[]
# ## Gene correlations

# %% tags=[]
input_file = OUTPUT_DIR_BASE / "gene_corrs-symbols.pkl"
display(input_file)
assert input_file.exists()

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
def exists_df(base_filename):
    full_filepath = output_dir / (base_filename + ".npz")

    return full_filepath.exists()


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


# %%
def get_sub_mat(corr_matrix, lv_data, lv_perc=None):
    sub_mat = pd.DataFrame(
        np.diag(np.diag(corr_matrix)),
        index=corr_matrix.index.copy(),
        columns=corr_matrix.columns.copy(),
    )

    lv_thres = 0.0
    if lv_perc is not None and lv_perc > 0.0:
        lv_thres = lv_data[lv_data > 0.0].quantile(lv_perc)

    lv_selected_genes = lv_data[lv_data > lv_thres].index
    lv_selected_genes = lv_selected_genes.intersection(corr_matrix.index)

    sub_mat.loc[lv_selected_genes, lv_selected_genes] = corr_matrix.loc[
        lv_selected_genes, lv_selected_genes
    ]
    return sub_mat


# %% tags=[]
def compute_chol_inv(lv_codes):
    for lv_code in lv_codes:
        lv_data = multiplier_z[lv_code]

        corr_mat_sub = get_sub_mat(gene_corrs, lv_data, LV_PERCENTILE)
        store_df(corr_mat_sub.to_numpy(), f"{lv_code}_corr_mat")

        chol_mat = np.linalg.cholesky(corr_mat_sub)
        chol_inv = np.linalg.inv(chol_mat)

        store_df(chol_inv, lv_code)


# %% tags=[]
# divide LVs in chunks for parallel processing
# lvs_chunks = list(chunker(list(multiplier_z.columns), 50))
lvs_chunks = [[LV_CODE]]

# %% tags=[]
# metadata
if not exists_df("metadata"):
    metadata = np.array([REFERENCE_PANEL, EQTL_MODEL])
    store_df(metadata, "metadata")
else:
    display("Metadata file already exists")

# gene names
if not exists_df("gene_names"):
    gene_names = np.array(common_genes)
    store_df(gene_names, "gene_names")
else:
    display("Gene names file already exists")

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
# lv1_inv = load_df("LV1")

# %% tags=[]
# lv2_inv = load_df("LV2")

# %% tags=[]
# lv_last_inv = load_df("LV987")
lv_last_inv = load_df(LV_CODE)
display(lv_last_inv)

# %% tags=[]
# assert lv1_inv.shape == lv2_inv.shape

# %% tags=[]
# assert not np.allclose(lv1_inv, lv2_inv)

# %% tags=[]
# assert not np.allclose(lv1_inv, lv_last_inv)

# %% tags=[]
# assert not np.allclose(lv2_inv, lv_last_inv)

# %% tags=[]
