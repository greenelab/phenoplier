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
# This notebook computes predicted expression correlations between all genes in the MultiPLIER models.

# %% [markdown]
# # Modules

# %% tags=[]
# %load_ext autoreload
# %autoreload 2

# %%
import numpy as np
import pandas as pd
from tqdm import tqdm

import conf
from entity import Gene

# %% [markdown]
# # Load data

# %% [markdown]
# ## SNPs covariance

# %%
snps_covar_file = (
    conf.PHENOMEXCAN["LD_BLOCKS"]["BASE_DIR"] / "mashr_snps_chr_blocks_cov.h5"
)
display(snps_covar_file)

# %%
with pd.HDFStore(snps_covar_file, mode="r") as store:
    snps_covar_metadata = (
        store["metadata"].drop_duplicates(subset=["varID"]).set_index("varID")
    )
    assert snps_covar_metadata.index.is_unique

# %%
snps_covar_metadata.shape

# %%
snps_covar_metadata.head()

# %% [markdown]
# ## MultiPLIER Z

# %%
multiplier_z_genes = pd.read_pickle(
    conf.MULTIPLIER["MODEL_Z_MATRIX_FILE"]
).index.tolist()

# %%
len(multiplier_z_genes)

# %%
multiplier_z_genes[:10]

# %% [markdown]
# ## Get gene objects

# %%
multiplier_gene_obj = {
    gene_name: Gene(name=gene_name)
    for gene_name in multiplier_z_genes
    if gene_name in Gene.GENE_NAME_TO_ID_MAP
}

# %%
len(multiplier_gene_obj)

# %%
multiplier_gene_obj["GAS6"].ensembl_id

# %%
_gene_obj = list(multiplier_gene_obj.values())

genes_info = pd.DataFrame(
    {
        "name": [g.name for g in _gene_obj],
        "id": [g.ensembl_id for g in _gene_obj],
        "chr": [g.chromosome for g in _gene_obj],
    }
)

# %%
genes_info.shape

# %%
genes_info.head()

# %% [markdown]
# ## Get tissues names

# %%
db_files = list(conf.PHENOMEXCAN["PREDICTION_MODELS"]["MASHR"].glob("*.db"))

# %%
assert len(db_files) == 49

# %%
tissues = [str(f).split("mashr_")[1].split(".db")[0] for f in db_files]

# %%
tissues[:5]

# %% [markdown]
# # Test: compute correlation in one chromosome

# %%
genes_chr = genes_info[genes_info["chr"] == "1"]

# %%
genes_chr.shape

# %%
genes_chr.head()

# %%
gene_chr_objs = [Gene(ensembl_id=gene_id) for gene_id in genes_chr["id"]]

# %%
len(gene_chr_objs)

# %%
gene_chr_objs[0].name, gene_chr_objs[1].name

# %%
tissues[0]

# %%
Gene("ENSG00000134686").get_pred_expression_variance(tissues[0])

# %%
Gene("ENSG00000163221").get_pred_expression_variance(tissues[0])

# %%
gene_corrs = []

n = len(gene_chr_objs)
n_comb = int(n * (n - 1) / 2.0)
display(n_comb)
pbar = tqdm(ncols=100, total=n_comb)

i = 0
for gene_idx1 in range(0, len(gene_chr_objs) - 1):
    gene_obj1 = gene_chr_objs[gene_idx1]

    for gene_idx2 in range(gene_idx1 + 1, len(gene_chr_objs)):
        gene_obj2 = gene_chr_objs[gene_idx2]

        try:
            gene_corrs.append(
                gene_obj1.get_expression_correlation(gene_obj2, tissues[0])
            )
            #             i = i + 1
            pbar.update(1)
        except TypeError:
            print((gene_obj1.ensembl_id, gene_obj2.ensembl_id))

pbar.close()

# %%
