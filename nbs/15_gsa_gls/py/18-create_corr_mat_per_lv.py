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
from gls import GLSPhenoplier

# %% [markdown] tags=[]
# # Settings

# %% tags=["parameters"]
# a cohort name (it could be something like UK_BIOBANK, etc)
COHORT_NAME = None

# reference panel such as 1000G or GTEX_V8
REFERENCE_PANEL = None

# predictions models such as MASHR or ELASTIC_NET
EQTL_MODEL = None

LV_CODE = None

# A number from 0.0 to 1.0 indicating the top percentile of the genes in the LV to keep.
# A value of 0.01 would take the top 1% of the genes in the LV.
# If zero or None, then all nonzero weighted genes in the LV will be kept.
LV_PERCENTILE = None

# %%
N_JOBS = 1

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
# ## Gene correlations

# %% tags=[]
input_files = list(OUTPUT_DIR_BASE.glob("gene_corrs-symbols*.pkl"))
display(input_files)
assert len(input_files) > 0, "No input correlation files"

# %% tags=[]
# load correlation matrix
gene_corrs_dict = {f.name: pd.read_pickle(f) for f in input_files}

# %% tags=[]
gene_corrs_dict[f.name].shape

# %% tags=[]
gene_corrs_dict[f.name].head()

# %%
current_index = gene_corrs_dict[f.name]
assert all(
    [current_index.equals(gc.index) for k, gc in gene_corrs_dict.items()]
), "Correlation matrices are not compatible"

# %% [markdown] tags=[]
# ## Define output dir (based on gene correlation's file)

# %% tags=[]
# # output file (hdf5)
# output_dir = Path(input_file).with_suffix(".per_lv")
# output_dir.mkdir(parents=True, exist_ok=True)

# display(output_dir)

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
# common_genes = gene_corrs_dict[f.name].index.tolist()

# %% tags=[]
# len(common_genes)

# %% tags=[]
# common_genes[:5]

# %% [markdown] tags=[]
# # Compute inverse correlation matrix for each LV

# %% tags=[]
def exists_df(output_dir, base_filename):
    full_filepath = output_dir / (base_filename + ".npz")

    return full_filepath.exists()


# %% tags=[]
def store_df(output_dir, nparray, base_filename):
    if base_filename in ("metadata", "gene_names"):
        np.savez_compressed(output_dir / (base_filename + ".npz"), data=nparray)
    else:
        sparse.save_npz(
            output_dir / (base_filename + ".npz"),
            sparse.csc_matrix(nparray),
            compressed=False,
        )


# %%
def get_output_dir(gene_corr_filename):
    path = OUTPUT_DIR_BASE / gene_corr_filename
    assert path.exists()
    return path.with_suffix(".per_lv")


# %% tags=[]
def compute_chol_inv(lv_codes):
    for gene_corr_filename, gene_corrs in gene_corrs_dict.items():
        output_dir = get_output_dir(gene_corr_filename)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output dir: {str(output_dir)}", flush=True)

        # save LV chol inverse
        for lv_code in lv_codes:
            lv_data = multiplier_z[lv_code]

            corr_mat_sub = GLSPhenoplier.get_sub_mat(gene_corrs, lv_data, LV_PERCENTILE)
            store_df(output_dir, corr_mat_sub.to_numpy(), f"{lv_code}_corr_mat")

            chol_mat = np.linalg.cholesky(corr_mat_sub)
            chol_inv = np.linalg.inv(chol_mat)

            store_df(output_dir, chol_inv, lv_code)

        # save metadata
        if not exists_df("metadata"):
            metadata = np.array([REFERENCE_PANEL, EQTL_MODEL])
            store_df(output_dir, metadata, "metadata")
        else:
            display("Metadata file already exists")

        # save gene names
        if not exists_df("gene_names"):
            gene_names = np.array(gene_corrs.index.tolist())
            store_df(output_dir, gene_names, "gene_names")
        else:
            display("Gene names file already exists")


# %% tags=[]
# divide LVs in chunks for parallel processing
# lvs_chunks = list(chunker(list(multiplier_z.columns), 50))
lvs_chunks = [[LV_CODE]]

# %% tags=[]
with ProcessPoolExecutor(max_workers=N_JOBS) as executor, tqdm(
    total=len(lvs_chunks), ncols=100
) as pbar:
    tasks = [executor.submit(compute_chol_inv, chunk) for chunk in lvs_chunks]
    for future in as_completed(tasks):
        res = future.result()
        pbar.update(1)


# %% [markdown] tags=[]
# ## Some checks

# %% tags=[]
def load_df(output_dir, base_filename):
    full_filepath = output_dir / (base_filename + ".npz")

    if base_filename in ("metadata", "gene_names"):
        return np.load(full_filepath)["data"]
    else:
        return sparse.load_npz(full_filepath).toarray()


# %% tags=[]
_genes = load_df("gene_names")

# %% tags=[]
display(len(_genes))
assert len(_genes) == gene_corrs_dict[f.name].index.shape[0]

# %% tags=[]
_metadata = load_df("metadata")

# %% tags=[]
display(_metadata)
assert _metadata[0] == REFERENCE_PANEL
assert _metadata[1] == EQTL_MODEL

# %% tags=[]
all_lvs_inv = {}
lv_prev = None

for gene_corr_filename, _ in gene_corrs_dict.items():
    output_dir = get_output_dir(gene_corr_filename)

    lv_data = load_df(output_dir, LV_CODE)
    display(lv_data)

    if lv_prev is not None:
        assert lv_data.shape == lv_prev.shape
        assert not np.allclose(lv_data, lv_prev)

    lv_prev = lv_data

# %% tags=[]
