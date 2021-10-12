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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Description

# %% [markdown]
# This notebooks will show the structure of the main data matrices in PhenoPLIER, and will guide you in analyzing gene associations for a particular trait: neutrophil counts, which is presented in the [manuscript](https://greenelab.github.io/phenoplier_manuscript/#phenoplier-an-integration-framework-based-on-gene-co-expression-patterns) in Figure 1c.

# %% [markdown]
# # Modules

# %%
import tempfile

import numpy as np
from scipy import stats
import pandas as pd

from entity import Trait, Gene
import conf

# %% [markdown]
# # Load gene module-gene membership matrix (matrix Z)

# %% [markdown]
# Here we load the gene module-gene membership matri, or "latent variables loadings matrix" (from the terminology of the [MultiPLIER article](https://doi.org/10.1016/j.cels.2019.04.003)).

# %%
matrix_z = pd.read_pickle(conf.MULTIPLIER["MODEL_Z_MATRIX_FILE"])

# %%
matrix_z.shape

# %%
matrix_z.head()

# %% [markdown]
# As you can see, this matrix Z contains the membership value for each gene across all LVs (or gene modules).
# A value of zero means that the gene does not belong to that LV, whereas a larger value represents how strongly that gene belongs to the LV.
# A group of genes that belong to the same LV represent a gene-set that has a similar expression profile across a set of tissues or cell types.
# We'll cover this in more detail in the next notebook (`02-LV_cell_types-...`).

# %% [markdown]
# # Load information about LV alignment with pathways

# %% [markdown]
# LV in matrix Z can represent a group of genes that align well with prior pathways (or prior knowledge) or be "novel" in the sense that the combination of genes do not represent a known unit but was found the PLIER when factorizing the recount2 data (see the MultiPLIER article for more details).
#
# Here we load that information, where for each LV and pathway, we have a p-value and area under the curve (AUC) that indicate how well the LV aligns to that pathway.

# %%
lv_metadata = pd.read_pickle(conf.MULTIPLIER["MODEL_SUMMARY_FILE"])

# %%
lv_metadata.shape

# %%
lv_metadata.head()

# %% [markdown]
# # Load gene associations from PhenomeXcan

# %% [markdown]
# Now we load the gene association across ~4,000 traits from [PhenomeXcan](https://doi.org/10.1126/sciadv.aba2083).
# The file we load here are the Summary-MultiXcan (or S-MultiXcan) results, essentially a p-value for each gene-trait pair.
# In the notebook I refer to these results generically as "TWAS results", meaning that we have gene-trait associations.
# All these TWAS results were derived solely from GWAS summary stats, so you can also generate yours relatively easily by using [S-MultiXcan](https://doi.org/10.1371/journal.pgen.1007889).

# %%
phenomexcan_df = pd.read_pickle(conf.PHENOMEXCAN["SMULTIXCAN_MASHR_PVALUES_FILE"])

# %%
phenomexcan_df.shape

# %%
phenomexcan_df.head()

# %% [markdown]
# Convert gene Ensembl IDs to symbols:

# %%
phenomexcan_df = phenomexcan_df.rename(index=Gene.GENE_ID_TO_NAME_MAP)

# %%
phenomexcan_df = phenomexcan_df.loc[~phenomexcan_df.index.duplicated()]

# %%
phenomexcan_df.head()

# %% [markdown]
# Keep genes present in our matrix Z only:

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

# %% [markdown]
# Below I search the PhenomeXcan results to find traits related to "neutrophils".

# %%
phenomexcan_df.columns[phenomexcan_df.columns.str.lower().str.contains("neutrophil")]

# %% [markdown]
# For this demo, I select the the "neutrophil count" TWAS derived from the GWAS performed by [Astle et. al](https://doi.org/10.1016/j.cell.2016.10.042). Below you can see the sample size:

# %%
trait_code = "Astle_et_al_2016_Neutrophil_count"
t = Trait.get_trait(full_code=trait_code)
display(f"{trait_code} - sample size: {t.n}")

# %%
traits_df = phenomexcan_df[[trait_code]].dropna()

# %%
traits_df.shape

# %%
traits_df.head()

# %% [markdown]
# Here I quickly show the data summary for this trait's gene associations:

# %%
traits_df.apply(lambda x: -np.log10(x)).describe()

# %% [markdown]
# Make sure we don't have missing values or NaN.

# %%
assert not traits_df.isna().any().any()
assert not np.isinf(traits_df).any().any()

# %% [markdown]
# # Convert p-values to z-scores

# %% [markdown]
# This converts a p-value to a scalar > 0, where higher positive values mean stronger association, and lower values close to zero mean weaker associations. Check our manuscript for more details on this.

# %%
traits_zscores = pd.DataFrame(
    data=np.abs(stats.norm.ppf(traits_df / 2)),
    index=traits_df.index.copy(),
    columns=traits_df.columns.copy(),
)

# %%
assert not traits_zscores.isna().any().any()
assert not np.isinf(traits_zscores).any().any()

# %%
traits_zscores.head()

# %%
traits_zscores.describe()

# %% [markdown]
# # Analysis of a neutrophil-termed LV

# %% [markdown]
# Let's take as an example an LV that was previously analyzed in the MultiPLIER study, which we identify as `LV603`. This LV aligns well with pathways related to neutrophils, as you can see below.

# %%
lv_metadata[
    (lv_metadata["LV index"] == "603") & (lv_metadata["FDR"] < 0.05)
].sort_values("FDR")

# %% [markdown]
# Let's see which genes more strongly belong to LV603 (the numbers are the gene weights in this LV):

# %%
lv603_top_genes = matrix_z["LV603"].sort_values(ascending=False)
display(lv603_top_genes.head(20))

# %% [markdown]
# Are these top genes associated with our neutrophil count?

# %%
traits_zscores.loc[lv603_top_genes.index].head(20)

# %% [markdown]
# It seems so. But what about the rest of the genes? They might be also strongly associated.
# Let's take a random sample:

# %%
traits_zscores.sample(n=20, random_state=0)

# %% [markdown]
# They do not seem as high as those within the top genes in LV603.
# If we compute the correlation between LV603 gene weights (`lv603_top_genes`) and gene associations for neutrophil counts (`traits_zscores`) we get this:

# %%
stats.pearsonr(
    traits_zscores.loc[lv603_top_genes.index].iloc[:, 0].to_numpy(),
    lv603_top_genes.to_numpy(),
)

# %% [markdown]
# Although the correlation is significant (`8.16e-10`) and the slope positive (we are interested only in genes at the top of the LV), we need to account for correlated predicted expression from the TWAS models (for example, if the expression of two genes at the top of the LV is correlated that would invalidate our test).
# We have a class implemented in Python that computes this, as shown below.

# %% [markdown]
# # Association between an LV and a trait

# %% [markdown]
# You can use our class `gls.GLSPhenoplier` to compute an association between an LV and a trait, in a similar way we did it in the previous cells by correlating LV603 gene weight's and neutrophil count gene associations (transformed to z-scores). However, `gls.GLSPhenoplier` takes into account correlations between the predicted expression of genes by using a generalized least squares (GLS) model.

# %%
from gls import GLSPhenoplier

# %% [markdown]
# We need to save our trait gene association to a pickle file before moving on. Here I generate a temporary file path and save:

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
# It will take a few seconds. You can show the entire table of results from the [statsmodels's GLS model](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.GLS.html):

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
# The one-sided p-values (the ones we used in our manuscript, since we are only interested in the top genes of an LV):

# %%
res.pvalues_onesided

# %% [markdown]
# And the two-sided p-values (in case you need them or are interested in a different hypothesis):

# %%
res.pvalues

# %% [markdown]
# # Conclusions

# %% [markdown]
# Hopefully, now have a more clear idea of the main data matrixes involved in PhenoPLIER (matrix Z, PhenomeXcan gene-trait associations, etc).
# We also see how to compute a p-value between an LV (group of genes or gene module) and a trait of interest.
# To do this with your own data, you only need to compute the S-MultiXcan TWAS results (gene-based) from your GWAS summary stats.
#
# In the next notebook (`02-LV_cell_types-...`), we'll see how to check in which tissues or cell types are our LV603'genes expressed.

# %%
