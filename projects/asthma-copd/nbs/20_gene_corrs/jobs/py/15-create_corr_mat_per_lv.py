# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all,-execution,-papermill,-trusted
#     formats: ipynb,py//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] tags=[]
# # Description

# %% [markdown] tags=[]
# It computes an LV-specific correlation matrix by using the top genes in that LV only.
#
# It has specicfic parameters for papermill (see under `Settings` below).
#
# This notebook should not be directly run. It is used by other notebooks.

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
REFERENCE_PANEL = "GTEX_V8"

# predictions models such as MASHR or ELASTIC_NET
EQTL_MODEL = "MASHR"

# A range of LVs in the format X-Y, such as 1-50 (from LV1 to LV50).
# If None, all LVs will be processed.
LV_RANGE = None

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
if LV_PERCENTILE is not None:
    LV_PERCENTILE = float(LV_PERCENTILE)

display(f"LV percentile: {LV_PERCENTILE}")

# %% tags=[]
assert (
    OUTPUT_DIR_BASE is not None and len(OUTPUT_DIR_BASE) > 0
), "Output directory path must be given"

OUTPUT_DIR_BASE = (Path(OUTPUT_DIR_BASE) / "gene_corrs" / COHORT_NAME).resolve()

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

# %%
orig_corr_name = "gene_corrs-symbols.pkl"

# %% tags=[]
gene_corrs_dict[orig_corr_name].shape

# %% tags=[]
gene_corrs_dict[orig_corr_name].head()

# %%
current_index = gene_corrs_dict[orig_corr_name].index
assert all(
    [current_index.equals(gc.index) for k, gc in gene_corrs_dict.items()]
), "Correlation matrices are not compatible"

# %% [markdown] tags=[]
# ## MultiPLIER Z

# %% tags=[]
multiplier_z = pd.read_pickle(conf.MULTIPLIER["MODEL_Z_MATRIX_FILE"])

# %% tags=[]
multiplier_z.shape

# %% tags=[]
multiplier_z.head()


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
        display(f"Output dir: {str(output_dir)}")

        # save LV chol inverse
        for lv_code in lv_codes:
            lv_data = multiplier_z[lv_code]

            corr_mat_sub = GLSPhenoplier.get_sub_mat(gene_corrs, lv_data, LV_PERCENTILE)
            store_df(output_dir, corr_mat_sub.to_numpy(), f"{lv_code}_corr_mat")

            chol_mat = np.linalg.cholesky(corr_mat_sub)
            chol_inv = np.linalg.inv(chol_mat)

            store_df(output_dir, chol_inv, lv_code)

        # save metadata
        if not exists_df(output_dir, "metadata"):
            metadata = np.array([REFERENCE_PANEL, EQTL_MODEL])
            store_df(output_dir, metadata, "metadata")
        else:
            display("Metadata file already exists")

        # save gene names
        if not exists_df(output_dir, "gene_names"):
            gene_names = np.array(gene_corrs.index.tolist())
            store_df(output_dir, gene_names, "gene_names")
        else:
            display("Gene names file already exists")


# %% tags=[]
if LV_RANGE is None:
    # divide LVs in chunks for parallel processing
    display("LV_RANGE was not given")

    lvs_chunks = list(chunker(list(multiplier_z.columns), 50))
else:
    display("LV_RANGE was given")

    assert "-" in LV_RANGE, "LV_RANGE has no '-'"
    lv_min, lv_max = LV_RANGE.split("-")
    lv_min, lv_max = int(lv_min), int(lv_max)
    assert lv_min <= lv_max, "LV_RANGE is incorrect"

    # create a single chunk in this case
    lvs_chunks = [[f"LV{i}" for i in range(lv_min, lv_max + 1)]]

# %%
display(f"# of chunks: {len(lvs_chunks)}")
display(f"# of LVs in each chunk: {len(lvs_chunks[0])}")

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
_genes = load_df(get_output_dir(orig_corr_name), "gene_names")

# %% tags=[]
display(len(_genes))
assert len(_genes) == gene_corrs_dict[orig_corr_name].index.shape[0]

# %% tags=[]
_metadata = load_df(get_output_dir(orig_corr_name), "metadata")

# %% tags=[]
display(_metadata)
assert _metadata[0] == REFERENCE_PANEL
assert _metadata[1] == EQTL_MODEL

# %% tags=[]
