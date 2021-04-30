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
INPUT_DIR = conf.PHENOMEXCAN["LD_BLOCKS"]["BASE_DIR"] / "gene_corrs"
display(INPUT_DIR)

# %% [markdown] tags=[]
# # Load data

# %% [markdown] tags=[]
# ## Gene correlations

# %%
all_gene_corr_files = list(INPUT_DIR.rglob("*.pkl"))

# %%
len(all_gene_corr_files)

# %%
all_gene_corr_files[:5]

# %%
assert len(all_gene_corr_files) == 22 * 49

# %%
all_gene_corr_files_df = pd.DataFrame({"corr_file": [f for f in all_gene_corr_files]})

# %%
all_gene_corr_files_df = all_gene_corr_files_df.assign(
    file_name=all_gene_corr_files_df["corr_file"].apply(lambda x: x.name)
)

# %%
all_gene_corr_files_df = all_gene_corr_files_df.assign(
    tissue=all_gene_corr_files_df["file_name"].apply(
        lambda x: x.split("-chr")[0].split("gene_corrs-")[1]
    )
)

# %%
all_gene_corr_files_df = all_gene_corr_files_df.assign(
    chromosome=all_gene_corr_files_df["file_name"].apply(
        lambda x: int(x.split("-chr")[1].split(".")[0])
    )
)

# %%
assert all_gene_corr_files_df["tissue"].unique().shape[0] == 49

# %%
assert all_gene_corr_files_df["chromosome"].unique().shape[0] == 22
assert set(all_gene_corr_files_df["chromosome"]) == set(range(1, 23))

# %%
all_gene_corr_files_df.shape

# %%
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
db_files = list(conf.PHENOMEXCAN["PREDICTION_MODELS"]["MASHR"].glob("*.db"))

# %% tags=[]
assert len(db_files) == 49

# %% tags=[]
tissues = [str(f).split("mashr_")[1].split(".db")[0] for f in db_files]

# %% tags=[]
tissues[:5]

# %% [markdown] tags=[]
# # Average correlations per chromosome

# %%
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

# %% [markdown]
# This matrix has all genes in MultiPLIER Z

# %%
gene_corrs_df = pd.DataFrame(data=0.0, index=genes_info["id"], columns=genes_info["id"])

# %%
gene_corrs_df.shape

# %%
gene_corrs_df.head()

# %%
for chr_num, chr_data in corrs_per_chr.items():
    chr_data = chr_data.reindex(
        index=gene_corrs_df.index, columns=gene_corrs_df.columns
    )
    gene_corrs_df = gene_corrs_df + chr_data.fillna(0.0)

# %%
gene_corrs_df = gene_corrs_df.astype(np.float32)

# %%
gene_corrs_df.head()

# %%
assert np.all(gene_corrs_df.values.diagonal() == 1.0)

# %% [markdown]
# ## Stats

# %%
_gene_corrs_flat = squareform(gene_corrs_df.values, checks=False)

# %%
pd.Series(_gene_corrs_flat).describe()

# %% [markdown]
# # Plot

# %%
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# %%
genes_order = genes_info.sort_values("chr")["id"].tolist()

# %%
cmap = ListedColormap(["w", "r"])

# %%
fig, ax = plt.subplots(figsize=(10, 10))
ax.matshow(
    gene_corrs_df.loc[genes_order, genes_order].values, vmin=-0.05, vmax=0.05, cmap=cmap
)
ax.set_xlabel("Genes")
ax.set_ylabel("Genes")
ax.set_xticks([])
ax.set_yticks([])

# %% [markdown]
# # Testing

# %%
# COL4A1 and COL4A2
gene1 = "ENSG00000187498"
gene2 = "ENSG00000134871"

gene_corrs_df.loc[gene1, gene2]

# %%
_genes_files = all_gene_corr_files_df[all_gene_corr_files_df["chromosome"] == 13][
    "corr_file"
].tolist()
assert len(_genes_files) == 49

# %%
_gene_values = []
for f in _genes_files:
    gene1_gene2_corr = pd.read_pickle(f).loc[gene1, gene2]
    _gene_values.append(gene1_gene2_corr)

# %%
_gene_values = np.array(_gene_values)
assert _gene_values.shape[0] == 49

# %%
display(_gene_values.mean())
assert gene_corrs_df.loc[gene1, gene2] == _gene_values.mean()

# %% [markdown]
# # Save

# %% [markdown]
# ## With ensemble ids

# %%
output_file = (
    conf.PHENOMEXCAN["LD_BLOCKS"]["BASE_DIR"]
    / "multiplier_genes-pred_expression_corr_avg.pkl"
)
display(output_file)

# %%
gene_corrs_df.to_pickle(output_file)

# %% [markdown]
# ## With gene symbols

# %%
output_file = (
    conf.PHENOMEXCAN["LD_BLOCKS"]["BASE_DIR"]
    / "multiplier_genes-pred_expression_corr_avg-gene_names.pkl"
)
display(output_file)

# %%
gene_corrs_gene_names_df = gene_corrs_df.rename(
    index=Gene.GENE_ID_TO_NAME_MAP, columns=Gene.GENE_ID_TO_NAME_MAP
)

# %%
assert gene_corrs_gene_names_df.index.is_unique

# %%
assert gene_corrs_gene_names_df.columns.is_unique

# %%
gene_corrs_gene_names_df.shape

# %%
gene_corrs_gene_names_df.head()

# %%
gene_corrs_gene_names_df.to_pickle(output_file)

# %%
