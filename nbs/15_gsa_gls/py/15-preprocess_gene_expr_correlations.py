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
# This notebook reads all gene correlations across all tissues and computes a single correlation matrix.

# %% [markdown] tags=[]
# # Modules

# %% tags=[]
# %load_ext autoreload
# %autoreload 2

# %% tags=[]
import numpy as np
from scipy.spatial.distance import squareform
import pandas as pd

# from tqdm import tqdm

import conf
from entity import Gene

# %% [markdown] tags=[]
# # Settings

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
    EQTL_MODEL_FILES_PREFIX = conf.PHENOMEXCAN["PREDICTION_MODELS"][f"{EQTL_MODEL}_PREFIX"]

# %%
display(f"Using eQTL model: {EQTL_MODEL} / {EQTL_MODEL_FILES_PREFIX}")

# %%
INPUT_DIR = (
    conf.PHENOMEXCAN["LD_BLOCKS"]["BASE_DIR"] / f"{EQTL_MODEL_FILES_PREFIX}gene_corrs"
)
display(INPUT_DIR)

# %% [markdown] tags=[]
# # Load data

# %% [markdown] tags=[]
# ## Gene correlations

# %% tags=[]
all_gene_corr_files = list(INPUT_DIR.rglob("*.pkl"))

# %% tags=[]
len(all_gene_corr_files)

# %% tags=[]
all_gene_corr_files[:5]

# %% tags=[]
assert len(all_gene_corr_files) == 22 * 49

# %% tags=[]
all_gene_corr_files_df = pd.DataFrame({"corr_file": [f for f in all_gene_corr_files]})

# %% tags=[]
all_gene_corr_files_df = all_gene_corr_files_df.assign(
    file_name=all_gene_corr_files_df["corr_file"].apply(lambda x: x.name)
)

# %% tags=[]
all_gene_corr_files_df = all_gene_corr_files_df.assign(
    tissue=all_gene_corr_files_df["file_name"].apply(
        lambda x: x.split("-chr")[0].split("gene_corrs-")[1]
    )
)

# %% tags=[]
all_gene_corr_files_df = all_gene_corr_files_df.assign(
    chromosome=all_gene_corr_files_df["file_name"].apply(
        lambda x: int(x.split("-chr")[1].split(".")[0])
    )
)

# %% tags=[]
assert all_gene_corr_files_df["tissue"].unique().shape[0] == 49

# %% tags=[]
assert all_gene_corr_files_df["chromosome"].unique().shape[0] == 22
assert set(all_gene_corr_files_df["chromosome"]) == set(range(1, 23))

# %% tags=[]
all_gene_corr_files_df.shape

# %% tags=[]
all_gene_corr_files_df.head()

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
    }
).dropna()

# %% tags=[]
genes_info.shape

# %% tags=[]
genes_info.head()

# %% [markdown] tags=[]
# ## Get tissues names

# %% tags=[]
db_files = list(conf.PHENOMEXCAN["PREDICTION_MODELS"][EQTL_MODEL].glob("*.db"))

# %% tags=[]
assert len(db_files) == 49

# %% tags=[]
tissues = [str(f).split(EQTL_MODEL_FILES_PREFIX)[1].split(".db")[0] for f in db_files]

# %% tags=[]
tissues[:5]

# %% [markdown] tags=[]
# # Average correlations per chromosome

# %% tags=[]
corrs_per_chr = {}

for chr_num in range(1, 23):
    print(f"Chromosome: {chr_num}", flush=True)

    chr_files = all_gene_corr_files_df[all_gene_corr_files_df["chromosome"] == chr_num]
    print(f"Number of corrs files: {chr_files.shape}")

    multiplier_genes_in_chr = genes_info[genes_info["chr"] == str(chr_num)]
    print(f"Number of MultiPLIER genes: {multiplier_genes_in_chr.shape}")

    # create final dataframe with corrs for this chr
    chr_df = pd.DataFrame(
        data=0.0,
        index=multiplier_genes_in_chr["id"],
        columns=multiplier_genes_in_chr["id"],
    )

    print("Reading corrs per tissue", flush=True)
    for idx, tissue_corrs in chr_files.iterrows():
        tissue_corrs_df = pd.read_pickle(tissue_corrs["corr_file"])

        nan_values = tissue_corrs_df.isna()
        if nan_values.any().any():
            print(
                f"  WARNING ({tissue_corrs['tissue']}): has NaN values ({nan_values.sum().sum()})"
            )
            tissue_corrs_df = tissue_corrs_df.fillna(0.0)

        # align
        tissue_corrs_df = tissue_corrs_df.loc[chr_df.index, chr_df.columns]

        chr_df = chr_df + tissue_corrs_df
    #         chr_df = chr_df.where(chr_df.abs() > tissue_corrs_df.abs(), tissue_corrs_df).fillna(chr_df)

    chr_df = chr_df / float(chr_files.shape[0])
    chr_df_flat = pd.Series(squareform(chr_df.values, checks=False))
    display(chr_df_flat.describe())

    corrs_per_chr[chr_num] = chr_df

    print("\n")

# %% [markdown] tags=[]
# # Create full gene correlation matrix

# %% [markdown] tags=[]
# This matrix has all genes in MultiPLIER Z

# %% tags=[]
gene_corrs_df = pd.DataFrame(data=0.0, index=genes_info["id"], columns=genes_info["id"])

# %% tags=[]
gene_corrs_df.shape

# %% tags=[]
gene_corrs_df.head()

# %% tags=[]
for chr_num, chr_data in corrs_per_chr.items():
    chr_data = chr_data.reindex(
        index=gene_corrs_df.index, columns=gene_corrs_df.columns
    )
    gene_corrs_df = gene_corrs_df + chr_data.fillna(0.0)

# %% tags=[]
gene_corrs_df = gene_corrs_df.astype(np.float32)

# %% tags=[]
gene_corrs_df.head()

# %% tags=[]
assert np.all(gene_corrs_df.values.diagonal() == 1.0)

# %% [markdown] tags=[]
# ## Stats

# %% tags=[]
_gene_corrs_flat = squareform(gene_corrs_df.values, checks=False)

# %% tags=[]
pd.Series(_gene_corrs_flat).describe()

# %% [markdown] tags=[]
# # Plot

# %% tags=[]
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# %% tags=[]
genes_order = genes_info.sort_values("chr")["id"].tolist()

# %% tags=[]
cmap = ListedColormap(["w", "r"])

# %% tags=[]
fig, ax = plt.subplots(figsize=(10, 10))
ax.matshow(
    gene_corrs_df.loc[genes_order, genes_order].values, vmin=-0.05, vmax=0.05, cmap=cmap
)
ax.set_xlabel("Genes")
ax.set_ylabel("Genes")
ax.set_xticks([])
ax.set_yticks([])

# %% [markdown] tags=[]
# # Testing

# %% tags=[]
# COL4A1 and COL4A2
gene1 = "ENSG00000187498"
gene2 = "ENSG00000134871"

gene_corrs_df.loc[gene1, gene2]

# %% tags=[]
_genes_files = all_gene_corr_files_df[all_gene_corr_files_df["chromosome"] == 13][
    "corr_file"
].tolist()
assert len(_genes_files) == 49

# %% tags=[]
_gene_values = []
for f in _genes_files:
    gene1_gene2_corr = pd.read_pickle(f).loc[gene1, gene2]
    _gene_values.append(gene1_gene2_corr)

# %% tags=[]
_gene_values = np.array(_gene_values)
assert _gene_values.shape[0] == 49

# %% tags=[]
display(_gene_values.mean())
assert gene_corrs_df.loc[gene1, gene2] == _gene_values.mean()

# %% [markdown] tags=[]
# # Save

# %% [markdown] tags=[]
# ## With ensemble ids

# %% tags=[]
output_file = conf.PHENOMEXCAN["LD_BLOCKS"][EQTL_MODEL]["GENE_IDS_CORR_AVG"]
display(output_file)

# %% tags=[]
gene_corrs_df.to_pickle(output_file)

# %% [markdown] tags=[]
# ## With gene symbols

# %% tags=[]
output_file = conf.PHENOMEXCAN["LD_BLOCKS"][EQTL_MODEL]["GENE_NAMES_CORR_AVG"]
display(output_file)

# %% tags=[]
gene_corrs_gene_names_df = gene_corrs_df.rename(
    index=Gene.GENE_ID_TO_NAME_MAP, columns=Gene.GENE_ID_TO_NAME_MAP
)

# %% tags=[]
assert gene_corrs_gene_names_df.index.is_unique

# %% tags=[]
assert gene_corrs_gene_names_df.columns.is_unique

# %% tags=[]
gene_corrs_gene_names_df.shape

# %% tags=[]
gene_corrs_gene_names_df.head()

# %% tags=[]
gene_corrs_gene_names_df.to_pickle(output_file)

# %% tags=[]
