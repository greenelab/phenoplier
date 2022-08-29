# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all,-execution,-papermill,-trusted
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

# %% [markdown]
# # Description

# %% [markdown]
# It groups GWASs acording to the number of variants in each one.

# %% [markdown]
# # Modules

# %%
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import conf

# %% [markdown]
# # Settings

# %%
GWAS_PARSING_BASE_DIR = conf.PHENOMEXCAN["BASE_DIR"] / "gwas_parsing"
display(GWAS_PARSING_BASE_DIR)
GWAS_PARSING_BASE_DIR.mkdir(exist_ok=True, parents=True)

# %%
GWAS_PARSING_N_LINES_DIR = GWAS_PARSING_BASE_DIR / "gwas_parsing_n_lines"
display(GWAS_PARSING_N_LINES_DIR)
GWAS_PARSING_N_LINES_DIR.mkdir(exist_ok=True, parents=True)

# %%
GWAS_PARSING_INPUT_DIR = GWAS_PARSING_BASE_DIR / "full"
display(GWAS_PARSING_INPUT_DIR)
assert GWAS_PARSING_INPUT_DIR.exists()

# %% [markdown]
# # Load data

# %% [markdown]
# ## Phenotype info from PhenomeXcan

# %%
pheno_info = pd.read_csv(conf.PHENOMEXCAN["UNIFIED_PHENO_INFO_FILE"], sep="\t")

# %%
pheno_info.shape

# %%
pheno_info.head()

# %%
pheno_info["source"].value_counts()

# %% [markdown]
# ## GWAS number of variants

# %%
gwas_n_variants = pd.read_pickle(GWAS_PARSING_BASE_DIR / "gwas_n_variants.pkl")
display(gwas_n_variants)

# %%
# keep only those phenotypes present in PhenomeXcan
_common_pheno_codes = gwas_n_variants.index.intersection(set(pheno_info["short_code"]))
assert len(_common_pheno_codes) == pheno_info.shape[0]

# %%
gwas_n_variants = gwas_n_variants.loc[_common_pheno_codes]

# %%
gwas_n_variants.shape


# %% [markdown]
# # Functions

# %%
def read_gwas_variants(pheno_code):
    gwas_data = pd.read_csv(
        GWAS_PARSING_INPUT_DIR / f"{pheno_code}.txt.gz",
        sep="\t",
        usecols=["panel_variant_id"],
    ).squeeze()

    return set(gwas_data.tolist())


# %%
def get_upper_triag(similarity_matrix, k: int = 1):
    """
    It returns the upper triangular matrix of a dataframe representing a
    similarity matrix between n elements.
    Args:
        similarity_matrix: a squared dataframe with a pairwise similarity
          matrix. That means the matrix is equal to its transposed version.
        k: argument given to numpy.triu function. It indicates the that the
          elements of the k-th diagonal to be zeroed.
    Returns:
        A dataframe with non-selected elements as NaNs.
    """
    mask = np.triu(np.ones(similarity_matrix.shape), k=k).astype(bool)
    return similarity_matrix.where(mask)


# %% [markdown]
# # Identify groups of cohorts

# %%
group_variants = {}

# %%
df_counts = gwas_n_variants.value_counts()
display(df_counts.shape)
display(df_counts.head())

# %% [markdown]
# There are 31 groups of GWASs with different number of SNPs.
#
# Most GWAS (4049) seem to have the same set of SNPs/variants, which are part of the Rapid GWAS project (UK Biobank).

# %% [markdown]
# ## Rapid GWAS project

# %%
_tmp = gwas_n_variants[gwas_n_variants == 8496089]
display(_tmp.shape)
display(_tmp)

# %%
group_snps = read_gwas_variants(_tmp.index[0])

# %%
# make sure a random sample of phenotypes in this group really have the same set of SNPs
assert group_snps == read_gwas_variants("K08") == read_gwas_variants("40001_J841")

# %%
group_variants["rapid"] = group_snps

# %%
list(group_variants["rapid"])[:5]

# %% [markdown]
# ## GTEx GWASs

# %% [markdown]
# ### Astle GWASs

# %%
_tmp = gwas_n_variants[gwas_n_variants == 8871980]
display(_tmp.shape)
display(_tmp)

# %%
group_snps = read_gwas_variants(_tmp.index[0])

# %%
# make sure a random sample of phenotypes really have the same set of SNPs
assert (
    group_snps
    == read_gwas_variants("Astle_et_al_2016_Sum_eosinophil_basophil_counts")
    == read_gwas_variants("Astle_et_al_2016_Red_blood_cell_count")
)

# %%
# and are different from the other sets
assert group_snps != group_variants["rapid"]

# %%
group_variants["astle"] = group_snps

# %% [markdown]
# ### Others

# %%
_tmp = gwas_n_variants[gwas_n_variants.isin(df_counts[df_counts == 1].index)]
display(_tmp.shape)
display(_tmp)

# %%
for pheno in tqdm(_tmp.index, ncols=100):
    group_snps = read_gwas_variants(pheno)
    assert group_snps != group_variants["rapid"]
    assert group_snps != group_variants["astle"]
    group_variants[pheno] = group_snps

# %% [markdown]
# # Others: merge GWAS snps with SNPs in models

# %% [markdown]
# Here I want to see whether groups of SNPs are really different in terms of their intersection with model SNPs.
#
# I only consider groups that are not "rapid" nor "astle".

# %% [markdown] tags=[]
# ## Load SNPs in predictions models

# %% tags=[]
mashr_models_db_files = list(
    conf.PHENOMEXCAN["PREDICTION_MODELS"]["MASHR"].glob("*.db")
)

# %% tags=[]
assert len(mashr_models_db_files) == 49

# %% tags=[]
all_variants_ids = []

for m in mashr_models_db_files:
    print(f"Processing {m.name}")
    tissue = m.name.split("mashr_")[1].split(".db")[0]

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
display(len(all_snps_in_models))

# %% [markdown]
# ## Compute intersections with SNPs in models

# %%
group_eqtls = {}

# %%
for group_name, group_gwas_snps in group_variants.items():
    # consider only phenotype that are not part of rapid nor astle groups
    if group_name in ("rapid", "astle"):
        continue

    group_eqtls[group_name] = all_snps_in_models.intersection(group_gwas_snps)
    print(f"{group_name}: {len(group_eqtls[group_name])}")

# %%
group_eqtl_intersections = np.eye(len(group_eqtls))
group_names = list(group_eqtls.keys())

for group1_idx in range(len(group_names) - 1):
    group1_name = group_names[group1_idx]
    group1_eqtls = group_eqtls[group1_name]
    group1_size = len(group1_eqtls)

    for group2_idx in range(group1_idx + 1, len(group_names)):
        group2_name = group_names[group2_idx]
        group2_eqtls = group_eqtls[group2_name]
        group2_size = len(group2_eqtls)

        max_size = max(group1_size, group2_size)

        n_intersections = len(group1_eqtls.intersection(group2_eqtls))
        group_eqtl_intersections[group1_idx, group2_idx] = n_intersections / max_size
        group_eqtl_intersections[group2_idx, group1_idx] = n_intersections / max_size

# %%
df = pd.DataFrame(
    data=group_eqtl_intersections,
    index=group_names,
    columns=group_names,
)

# %%
df.shape

# %%
df_flat = get_upper_triag(df).stack().sort_index()

# %%
df_flat

# %%
df_stats = df_flat.describe()
display(df_stats)
assert df_stats["min"] > 0.99

# %%
df_flat.sort_values()

# %% [markdown]
# # Clustering of traits

# %%
# X_std = (df - df.min(axis=0)) / (df.max(axis=0) - df.min(axis=0))

# %%
# import numpy as np

# from matplotlib import pyplot as plt
# from scipy.cluster.hierarchy import dendrogram
# from sklearn.cluster import AgglomerativeClustering


# def plot_dendrogram(model, **kwargs):
#     # Create linkage matrix and then plot the dendrogram

#     # create the counts of samples under each node
#     counts = np.zeros(model.children_.shape[0])
#     n_samples = len(model.labels_)
#     for i, merge in enumerate(model.children_):
#         current_count = 0
#         for child_idx in merge:
#             if child_idx < n_samples:
#                 current_count += 1  # leaf node
#             else:
#                 current_count += counts[child_idx - n_samples]
#         counts[i] = current_count

#     linkage_matrix = np.column_stack(
#         [model.children_, model.distances_, counts]
#     ).astype(float)

#     # Plot the corresponding dendrogram
#     dendrogram(linkage_matrix, **kwargs)


# X = X_std

# # setting distance_threshold=0 ensures we compute the full tree.
# model = AgglomerativeClustering(distance_threshold=0, n_clusters=None, affinity='precomputed', linkage="complete")

# model = model.fit(X)
# plt.title("Hierarchical Clustering Dendrogram")
# # plot the top three levels of the dendrogram
# plot_dendrogram(model, truncate_mode="level", p=3)
# plt.xlabel("Number of points in node (or index of point if no parenthesis).")
# plt.show()

# %%
# model = AgglomerativeClustering(n_clusters=4, affinity='precomputed', linkage="complete")
# labels = model.fit_predict(X)

# %%
# np.unique(labels)

# %%
# df.index[labels == 3]

# %% [markdown]
# # Conclusion

# %% [markdown]
# We need to compute three (3) gene correlation matrices for three groups of phenotypes:
#
# * One for "rapid"
# * One for "astle"
# * One for the rest of the phenotypes
#
# All phenotypes in the first two ("rapid" and "astle") have exactly the same SNPs, so the correlation matrix will be accurate for all those phenotypes.
#
# For the third group, the correlation matrix will not be exactly accurate, but since they share more than 99% of the eQTLs, we assume it's enough.

# %%
