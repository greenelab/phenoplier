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
# It profiles some functions to compute the correlation between predicted gene expression. Each of these notebooks is supposed to be run in a particular changeset.
#
# **Before running this notebook**, make sure you are in this changeset:
# ```bash
# git co 6149a6f90f41534d0979b434cd16d17cc28d2c5f
# ```

# %%
# %load_ext line_profiler

# %% [markdown] tags=[]
# # Modules

# %% tags=[]
from entity import Gene


# %% [markdown]
# # Functions

# %%
def compute_ssm_correlation(all_genes):
    res = []
    for g1_idx, g1 in enumerate(all_genes[:-1]):
        for g2 in all_genes[g1_idx:]:
            c = g1.get_ssm_correlation(
                g2,
                reference_panel="1000G",
                model_type="MASHR",
                use_within_distance=False,
            )
            res.append(c)
    return res


# %% [markdown]
# # Test case

# %%
gene1 = Gene(ensembl_id="ENSG00000180596")
gene2 = Gene(ensembl_id="ENSG00000180573")
gene3 = Gene(ensembl_id="ENSG00000274641")
gene4 = Gene(ensembl_id="ENSG00000277224")

all_genes = [gene1, gene2, gene3, gene4]

# %%
assert len(set([g.chromosome for g in all_genes])) == 1

# %% [markdown]
# # Run timeit

# %%
# %timeit compute_ssm_correlation(all_genes)

# %% [markdown]
# # Profile

# %%
# %prun -l 20 -s cumulative compute_ssm_correlation(all_genes)

# %% [markdown]
# # Profile by line

# %% [markdown]
# ## Function `get_expression_correlation`

# %%
# %lprun -f Gene.get_expression_correlation compute_ssm_correlation(all_genes)

# %% [markdown]
# ## Function `_get_snps_cov`

# %%
# %lprun -f Gene._get_snps_cov compute_ssm_correlation(all_genes)

# %%
