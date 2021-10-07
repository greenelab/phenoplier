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
# TODO

# %% [markdown]
# # Modules

# %%
from pathlib import Path
import tempfile

import numpy as np
from scipy import stats
import pandas as pd

from entity import Trait, Gene
import conf

# %% [markdown]
# # Load gene module-gene membership matrix (matrix Z)

# %%
matrix_z = pd.read_pickle(conf.MULTIPLIER["MODEL_Z_MATRIX_FILE"])

# %%
matrix_z.shape

# %%
matrix_z.head()

# %% [markdown]
# Matrix Z contains the membership value for each gene across all LVs (or gene modules).
# A value of zero means that the gene does not belong to that LV, whereas a larger value represents how strongly that gene belongs to the LV.

# %% [markdown]
# # Load information about LV alignment with pathways

# %%
lv_metadata = pd.read_pickle(conf.MULTIPLIER["MODEL_SUMMARY_FILE"])

# %%
lv_metadata.shape

# %%
lv_metadata.head()

# %% [markdown]
# # Load GWAS information

# %%
gwas_info = pd.read_csv(conf.PHENOMEXCAN["RAPID_GWAS_PHENO_INFO_FILE"], sep="\t")

# %%
gwas_info.head()

# %% [markdown]
# # Load gene associations

# %%
# FIXME: LINK PHENOMEXCAN PAPER AND EXPLAIN BRIEFLY WHAT ARE THESE RESULTS
# MENTION TWAS AND LINK MULTIXCAN PAPER

# %%
phenomexcan_df = pd.read_pickle(conf.PHENOMEXCAN["SMULTIXCAN_MASHR_PVALUES_FILE"])

# %%
phenomexcan_df.shape

# %%
phenomexcan_df.head()

# %% [markdown]
# Let's convert gene Ensembl IDs to symbols:

# %%
phenomexcan_df = phenomexcan_df.rename(index=Gene.GENE_ID_TO_NAME_MAP)

# %%
phenomexcan_df = phenomexcan_df.loc[~phenomexcan_df.index.duplicated()]

# %%
phenomexcan_df.head()

# %% [markdown]
# Let's keep genes present in our matrix Z only

# %%
common_genes = phenomexcan_df.index.intersection(matrix_z.index)
display(common_genes)

# %%
phenomexcan_df = phenomexcan_df.loc[common_genes]

# %%
phenomexcan_df.shape

# %%
matrix_z = matrix_z.loc[common_genes]

# %%
matrix_z.shape

# %% [markdown]
# # Take a look at genes associated with neutrophil counts

# %%
phenomexcan_df.columns[phenomexcan_df.columns.str.lower().str.contains("neutrophil")]

# %%
# trait_code = "30140_raw-Neutrophill_count"
# t = Trait.get_trait(full_code=trait_code)
# display(f"{trait_code} - sample size: {t.n}")

# %%
trait_code = "Astle_et_al_2016_Neutrophil_count"
t = Trait.get_trait(full_code=trait_code)
display(f"{trait_code} - sample size: {t.n}")

# %%
# traits_df = phenomexcan_df[["30140_raw-Neutrophill_count", "Astle_et_al_2016_Neutrophil_count"]].dropna()

# %%
traits_df = phenomexcan_df[["Astle_et_al_2016_Neutrophil_count"]].dropna()

# %%
traits_df.shape

# %%
traits_df.head()

# %%
traits_df.apply(lambda x: -np.log10(x)).describe()

# %%
# traits_df.apply(lambda x: -np.log10(x)).corr()

# %%
assert not traits_df.isna().any().any()

# %% [markdown]
# # Convert p-values to z-scores

# %% [markdown]
# This converts a p-value to a scalar > 0, where higher positive values mean stronger association, and lower values close to zero mean weaker associations.

# %%
traits_zscores = pd.DataFrame(
    data=np.abs(stats.norm.ppf(traits_df / 2)),
    index=traits_df.index.copy(),
    columns=traits_df.columns.copy(),
)

# %%
assert not traits_zscores.isna().any().any()

# %%
traits_zscores.head()

# %%
traits_zscores.describe()

# %% [markdown]
# # Analysis of an neutrophil-termed LV

# %% [markdown]
# Let's take as an example an LV that was previously analyzed in another study [here](https://doi.org/10.1016/j.cels.2019.04.003), which we identify as `LV603`. This LV aligns well with pathways related to neutrophils, as you can see below.

# %%
lv_metadata[lv_metadata["LV index"] == "603"].sort_values("FDR")

# %% [markdown]
# Let's see which genes more strongly belong to LV603:

# %%
lv603_top_genes = matrix_z["LV603"].sort_values(ascending=False)
display(lv603_top_genes.head(20))

# %% [markdown]
# Are these top genes associated with our neutrophil?

# %%
traits_zscores.loc[lv603_top_genes.index].head(20)

# %% [markdown]
# It seems so. But what about the rest of the genes? They might be also strongly associated.
# Let's take a random sample:

# %%
traits_zscores.sample(n=20, random_state=0)

# %% [markdown]
# They do not seem as high as those with the top gene in LV603. If we compute the correlation between LV603 gene weights (`lv603_top_genes`) and gene associations for neutrophil counts (`traits_zscores`) we get this:

# %%
stats.pearsonr(
    traits_zscores.loc[lv603_top_genes.index].iloc[:, 0].to_numpy(),
    lv603_top_genes.to_numpy(),
)

# %% [markdown]
# Although the correlation is significant (`8.16e-10`) and the slope positive (we are interested only in genes at the top of the LV), we need to account for correlated predicted expression from the TWAS models (for example, if the expression of two genes at the top of the LV is correlated, results will not be as significant after accounting for this). We have a class implemented in Python that computes this, as shown below.

# %% [markdown]
# # Association between an LV and a trait

# %% [markdown]
# You can use our class `gls.GLSPhenoplier` to compute an association between an LV and a trait, in a similar way we did it in the previous cells by correlating LV603 gene weight's and neutrophil count gene associations (transformed to z-scores). However, `gls.GLSPhenoplier` takes into account correlations between the predicted expression of genes by using a generalized least squares (GLS) model.

# %%
from gls import GLSPhenoplier

# %% [markdown]
# We need to save our trait gene association to a pickle file before:

# %%
with tempfile.NamedTemporaryFile(suffix=".pkl") as f:
    output_filepath = f.name
    display(output_filepath)

traits_zscores.to_pickle(output_filepath)

# %% [markdown]
# Now select an LV id and the trait id, as shown below:

# %%
lv_code = "LV603"

phenotype_code = traits_zscores.columns[0]
display(phenotype_code)

# %% [markdown]
# Run the model:

# %%
gls_model = GLSPhenoplier(
    smultixcan_result_set_filepath=output_filepath,
)

gls_model.fit_named(lv_code, phenotype_code)

# %% [markdown]
# You can show the entire table of results from the statsmodels model:

# %%
res = gls_model.results

# %%
print(res.summary())

# %% [markdown]
# This p-value (`3.76e-08`) is a two-sided test on the LV coefficient (`0.0698`).
# You can see that the p-value is slightly less significant than the p-value from the Pearson correlation that we computed before.

# %% [markdown]
# This is how you can access the model's estimated parameters.
#
# These are the coefficients:

# %%
res.params

# %% [markdown]
# Two-sided p-values:

# %%
res.pvalues

# %% [markdown]
# One-sided p-values

# %%
res.pvalues_onesided

# %% [markdown]
# # Conclusions

# %% [markdown]
# - summary of what we did
# - next steps
