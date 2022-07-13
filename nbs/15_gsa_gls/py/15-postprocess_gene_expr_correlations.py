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
# (Please, take a look at the README.md file in this directory for instructions on how to run this notebook)
#
# This notebook reads all gene correlations across all chromosomes and computes a single correlation matrix by assembling a big correlation matrix with all genes.

# %% [markdown] tags=[]
# # Modules

# %% tags=[]
# %load_ext autoreload
# %autoreload 2

# %% tags=[]
import numpy as np
from scipy.spatial.distance import squareform
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import conf
from entity import Gene

# %% [markdown] tags=[]
# # Settings

# %% tags=["parameters"]
# reference panel
REFERENCE_PANEL = "GTEX_V8"
# REFERENCE_PANEL = "1000G"

# prediction models
## mashr
EQTL_MODEL = "MASHR"
EQTL_MODEL_FILES_PREFIX = "mashr_"

# ## elastic net
# EQTL_MODEL = "ELASTIC_NET"
# EQTL_MODEL_FILES_PREFIX = "en_"

# make it read the prefix from conf.py
EQTL_MODEL_FILES_PREFIX = None

# check that final correlation matrix works with statsmodels.GLS
COMPATIBLE_WITH_STATSMODELS_GLS = True

# %%
if EQTL_MODEL_FILES_PREFIX is None:
    EQTL_MODEL_FILES_PREFIX = conf.PHENOMEXCAN["PREDICTION_MODELS"][
        f"{EQTL_MODEL}_PREFIX"
    ]

# %%
display(f"Using eQTL model: {EQTL_MODEL} / {EQTL_MODEL_FILES_PREFIX}")

# %%
REFERENCE_PANEL_DIR = conf.PHENOMEXCAN["LD_BLOCKS"][f"{REFERENCE_PANEL}_GENOTYPE_DIR"]

# %%
display(f"Using reference panel folder: {str(REFERENCE_PANEL_DIR)}")

# %%
OUTPUT_DIR_BASE = (
    conf.PHENOMEXCAN["LD_BLOCKS"][f"GENE_CORRS_DIR"]
    / REFERENCE_PANEL.lower()
    / EQTL_MODEL.lower()
)
display(OUTPUT_DIR_BASE)
OUTPUT_DIR_BASE.mkdir(parents=True, exist_ok=True)

# %%
display(f"Using output dir base: {OUTPUT_DIR_BASE}")

# %%
INPUT_DIR = OUTPUT_DIR_BASE / "by_chr"

if COMPATIBLE_WITH_STATSMODELS_GLS:
    INPUT_DIR = INPUT_DIR / "corrected_positive_definite"

display(INPUT_DIR)
assert INPUT_DIR.exists()

# %% [markdown] tags=[]
# # Load data

# %% [markdown] tags=[]
# ## Gene correlations

# %% tags=[]
all_gene_corr_files = list(INPUT_DIR.glob("gene_corrs-chr*.pkl"))

# %%
# sort by chromosome
all_gene_corr_files = sorted(
    all_gene_corr_files, key=lambda x: int(x.name.split("-chr")[1].split(".pkl")[0])
)

# %% tags=[]
len(all_gene_corr_files)

# %% tags=[]
all_gene_corr_files

# %% tags=[]
assert len(all_gene_corr_files) == 22

# %% [markdown] tags=[]
# ## MultiPLIER Z

# %% tags=[]
multiplier_z_genes = pd.read_pickle(
    conf.MULTIPLIER["MODEL_Z_MATRIX_FILE"]
).index.tolist()

# %% tags=[]
len(multiplier_z_genes)

# %% tags=[]
multiplier_z_genes[:10]

# %% [markdown] tags=[]
# ## Get gene objects

# %% tags=[]
multiplier_gene_obj = {
    gene_name: Gene(name=gene_name)
    for gene_name in multiplier_z_genes
    if gene_name in Gene.GENE_NAME_TO_ID_MAP
}

# %% tags=[]
len(multiplier_gene_obj)

# %% tags=[]
multiplier_gene_obj["GAS6"].ensembl_id

# %% tags=[]
_gene_obj = list(multiplier_gene_obj.values())

genes_info = pd.DataFrame(
    {
        "name": [g.name for g in _gene_obj],
        "id": [g.ensembl_id for g in _gene_obj],
        "chr": [g.chromosome for g in _gene_obj],
        "start_position": [g.get_attribute("start_position") for g in _gene_obj],
    }
).dropna()

# %%
assert not genes_info.isna().any().any()

# %%
genes_info.dtypes

# %%
genes_info["chr"] = genes_info["chr"].apply(pd.to_numeric, downcast="integer")
genes_info["start_position"] = genes_info["start_position"].astype(
    int
)  # .apply(pd.to_numeric, downcast="signed")

# %%
genes_info.dtypes

# %% tags=[]
genes_info.shape

# %% tags=[]
genes_info.head()

# %%
assert not genes_info.isna().any().any()

# %% [markdown] tags=[]
# # Create full correlation matrix

# %%
genes_info = genes_info.sort_values(["chr", "start_position"])

# %%
genes_info

# %%
full_corr_matrix = pd.DataFrame(
    np.zeros((genes_info.shape[0], genes_info.shape[0])),
    index=genes_info["id"].tolist(),
    columns=genes_info["id"].tolist(),
)

# %%
assert full_corr_matrix.index.is_unique & full_corr_matrix.columns.is_unique

# %%
for chr_corr_file in all_gene_corr_files:
    print(chr_corr_file.name, flush=True)

    corr_data = pd.read_pickle(chr_corr_file)
    full_corr_matrix.loc[corr_data.index, corr_data.columns] = corr_data

# %%
full_corr_matrix.shape

# %%
# make sure all elements in the diagonal are ones/1.0
# maybe it's not a good idea to modify the matrix after being adjusted for positive definiteness
# full_corr_matrix[full_corr_matrix > 1.0] = 1.0
# np.fill_diagonal(full_corr_matrix.values, 1.0)

# %%
full_corr_matrix

# %% [markdown] tags=[]
# ## Some checks

# %%
# assert np.all(full_corr_matrix.to_numpy().diagonal() == 1.0)

# %%
# check that all genes have a value
assert not full_corr_matrix.isna().any().any()

# %%
_min_val = full_corr_matrix.min().min()
display(_min_val)
# sometimes, if using statsmodels.GLS and after adjusting correlation matrices,
# correlations are lower than zero
assert _min_val >= -1e-3

# %%
_max_val = full_corr_matrix.max().max()  # this will capture the 1.0 in the diagonal
display(_max_val)
assert _max_val <= 1.01

# %%
# Check that matrix is invertible
inv_mat = np.linalg.inv(full_corr_matrix)

# %%
assert not np.isnan(inv_mat).any()
assert not np.isinf(inv_mat).any()
assert not np.iscomplex(inv_mat).any()

# %%
# print negative eigenvalues
eigs = np.linalg.eigvals(full_corr_matrix.to_numpy())
display(len(eigs[eigs < 0]))
display(eigs[eigs < 0])

# %%
if COMPATIBLE_WITH_STATSMODELS_GLS:
    # A Cholesky decomposition must not fail for statsmodels.GLS to work
    np.linalg.cholesky(np.linalg.inv(full_corr_matrix))

    import statsmodels.api as sm

    np.random.seed(0)

    y = np.random.rand(full_corr_matrix.shape[0])
    X = np.random.rand(full_corr_matrix.shape[0], 2)
    X[:, 0] = 1

    # this should not throw an exception: LinAlgError("Matrix is not positive definite")
    _gls_model = sm.GLS(y, X, sigma=full_corr_matrix)

    _gls_results = _gls_model.fit()

    display(_gls_results.summary())

# %% [markdown] tags=[]
# ## Stats

# %% tags=[]
full_corr_matrix_flat = full_corr_matrix.mask(
    np.triu(np.ones(full_corr_matrix.shape)).astype(bool)
).stack()

# %%
display(full_corr_matrix_flat.shape)
assert full_corr_matrix_flat.shape[0] == int(
    full_corr_matrix.shape[0] * (full_corr_matrix.shape[0] - 1) / 2
)

# %%
full_corr_matrix_flat[full_corr_matrix_flat == 1.0]

# %%
full_corr_matrix_flat.head()

# %% tags=[]
full_corr_matrix_flat.describe().apply(str)

# %%
full_corr_matrix_flat_quantiles = full_corr_matrix_flat.quantile(np.arange(0, 1, 0.05))
display(full_corr_matrix_flat_quantiles)

# %% [markdown]
# ## Plot: distribution

# %%
with sns.plotting_context("paper", font_scale=1.5):
    g = sns.displot(full_corr_matrix_flat, kde=True, height=7)
    g.ax.set_title("Distribution of gene correlation values in all chromosomes")

# %% [markdown]
# ## Plot: heatmap

# %%
vmin_val = min(-0.05, full_corr_matrix_flat_quantiles[0.10])
vmax_val = max(0.05, full_corr_matrix_flat_quantiles[0.90])
display(f"{vmin_val} / {vmax_val}")

# %%
f, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(
    full_corr_matrix,
    xticklabels=False,
    yticklabels=False,
    square=True,
    vmin=vmin_val,
    vmax=vmax_val,
    cmap="YlGnBu",
    ax=ax,
)
ax.set_title("Gene correlations in all chromosomes")

# %% [markdown] tags=[]
# # Save

# %% [markdown] tags=[]
# ## With ensemble ids

# %%
# output_file_name_template = conf.PHENOMEXCAN["LD_BLOCKS"][
#     "GENE_CORRS_FILE_NAME_TEMPLATES"
# ]["GENE_CORR_AVG"]

# output_file = OUTPUT_DIR_BASE / output_file_name_template.format(
#     prefix="",
#     suffix=f"-ssm_corrs-gene_ensembl_ids",
# )
# display(output_file)

# %% tags=[]
# full_corr_matrix.to_pickle(output_file)

# %% [markdown] tags=[]
# ## With gene symbols

# %%
output_file_name_template = conf.PHENOMEXCAN["LD_BLOCKS"][
    "GENE_CORRS_FILE_NAME_TEMPLATES"
]["GENE_CORR_AVG"]

output_file = OUTPUT_DIR_BASE / output_file_name_template.format(
    prefix="",
    suffix=f"-gene_symbols",
)
display(output_file)

# %% tags=[]
full_corr_matrix_gene_symbols = full_corr_matrix.rename(
    index=Gene.GENE_ID_TO_NAME_MAP, columns=Gene.GENE_ID_TO_NAME_MAP
)

# %% tags=[]
assert full_corr_matrix_gene_symbols.index.is_unique

# %% tags=[]
assert full_corr_matrix_gene_symbols.columns.is_unique

# %% tags=[]
full_corr_matrix_gene_symbols.shape

# %% tags=[]
full_corr_matrix_gene_symbols.head()

# %% tags=[]
full_corr_matrix_gene_symbols.to_pickle(output_file)

# %% tags=[]
