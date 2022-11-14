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

# %% [markdown]
# # Description

# %% [markdown]
# This notebook will show the structure of the main data matrices in PhenoPLIER, and will guide you in analyzing gene associations for a particular trait: basophill percentage, which is presented in the [manuscript](https://greenelab.github.io/phenoplier_manuscript/#phenoplier-an-integration-framework-based-on-gene-co-expression-patterns) in Figure 1c.

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
# Now we load gene-trait association from [PhenomeXcan](https://doi.org/10.1126/sciadv.aba2083).
# PhenomeXcan provides TWAS results (using [Summary-MultiXcan](https://doi.org/10.1371/journal.pgen.1007889) and [Summary-PrediXcan](https://doi.org/10.1038/s41467-018-03621-1)) across ~4,000 traits.
# If you are interested in PhenomeXcan you can also check out the [Github repo](https://github.com/hakyimlab/phenomexcan) to know how to download results.
#
# For this demo, we'll load a file that contains Summary-MultiXcan (or S-MultiXcan) results for basophill percentage.
# This file contains a list of p-values for ~22k genes, where a significant p-value means that the gene's predicted expression (across different tissues) is associated with basophill percentage.
# In the notebook I refer to these results generically as "TWAS results", meaning that we have gene-trait associations.
# All these TWAS results were derived solely from GWAS summary stats, so you can also generate yours relatively easily by using [S-MultiXcan](https://doi.org/10.1371/journal.pgen.1007889).

# %% language="bash"
# # download S-MultiXcan results for basophill percentage
# wget https://uchicago.box.com/shared/static/g70nq1c6wjvado242t9yg05jrhvdykrv.gz -O /tmp/smultixcan_30220_raw_ccn30.tsv.gz

# %%
df = pd.read_csv("/tmp/smultixcan_30220_raw_ccn30.tsv.gz", sep="\t")

# %%
df.shape

# %%
df.head()

# %% [markdown]
# # Take a look at genes associated with basophill percentage

# %% [markdown]
# Show the sample size for this trait.

# %%
trait_code = "30220_raw-Basophill_percentage"
t = Trait.get_trait(full_code=trait_code)
display(f"{trait_code} - sample size: {t.n}")

# %% [markdown]
# Below I list the top associated genes for basophill percentage.

# %%
traits_df = df[["gene_name", "pvalue"]].dropna().set_index("gene_name")

# remove duplicated gene names
traits_df = traits_df.loc[~traits_df.index.duplicated()]

# %%
traits_df.shape

# %%
traits_df.head()

# %% [markdown]
# Here I quickly show the data summary for this trait's gene associations:

# %%
traits_df["pvalue"].apply(lambda x: -np.log10(x)).describe()

# %% [markdown]
# # Get a set of common genes between TWAS and LVs

# %%
common_genes = traits_df.index.intersection(matrix_z.index)

# %%
common_genes

# %%
# keep only the genes in common
traits_df = traits_df.loc[common_genes]

# %%
traits_df.shape

# %%
matrix_z = matrix_z.loc[common_genes]

# %%
matrix_z.shape

# %% [markdown]
# # Analysis of a neutrophil-termed LV

# %% [markdown]
# Let's take as an example an LV that was previously analyzed in the MultiPLIER study, which we identify as `LV603`. This LV aligns well with pathways related to neutrophils, as you can see below. In the next notebook (`02-LV_cell_types...`) we'll see that this LV is expressed in neutrophils and other granulocytes.

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
# Are these top genes associated with our trait of interest?

# %%
traits_df.loc[lv603_top_genes.index].head(20)

# %% [markdown]
# It seems so. But what about the rest of the genes? They might be also strongly associated.
# Let's take a random sample:

# %%
traits_df.sample(n=20, random_state=0)

# %% [markdown]
# They do not seem as significant as those within the top genes in LV603.
#
# If we compute the correlation between LV603 gene weights (`lv603_top_genes`) and gene associations for basophill percentage (`traits_df`) we get this:

# %%
lv603_top_genes

# %%
stats.pearsonr(
    traits_df["pvalue"]
    .apply(lambda x: -np.log10(x))
    .loc[lv603_top_genes.index]
    .to_numpy(),
    lv603_top_genes.to_numpy(),
)

# %% [markdown]
# Although the correlation is significant (`2.94e-27`) and the slope positive (we are interested only in genes at the top of the LV), we need to account for correlated predicted expression from the TWAS models (for example, if the expression of two genes at the top of the LV is correlated that would invalidate our test).
# We provide a class, `GLSPhenoplier` (implemented in Python) that computes this. We also provide a command-line tool, `gls_cli.py`, that performs several preprocessing steps, and below we show how to use it for our example.

# %% [markdown]
# # `gls_cli.py`: association between an LV and a trait

# %% [markdown]
# The `gls_cli.py` command-line tool needs as input the S-MultiXcan TWAS results and a gene correlation matrix (which is specific to the TWAS results, see below).

# %% tags=[] scrolled=true language="bash"
# # remove previously computed results (if exist)
# rm /tmp/gls_phenoplier-basophill_percentage.tsv.gz
#
# # print full path of gls_cli.py tool
# echo ${PHENOPLIER_CODE_DIR}/libs/gls_cli.py
#
# # print full path of gene correlations file (which is trait-specific!)
# COHORT_NAME="phenomexcan_rapid_gwas"
# REFERENCE_PANEL="gtex_v8"
# GENE_CORR_DIR="${PHENOPLIER_RESULTS_GLS}/gene_corrs/cohorts/${COHORT_NAME}/${REFERENCE_PANEL}/mashr/gene_corrs-symbols-within_distance_5mb.per_lv/"
# echo ${GENE_CORR_DIR}
#
# python ${PHENOPLIER_CODE_DIR}/libs/gls_cli.py \
#   --input-file /tmp/smultixcan_30220_raw_ccn30.tsv.gz \
#   --duplicated-genes-action keep-first \
#   --gene-corr-file ${GENE_CORR_DIR} \
#   --debug-use-sub-gene-corr \
#   --covars gene_size gene_size_log gene_density gene_density_log \
#   --output-file /tmp/gls_phenoplier-basophill_percentage.tsv.gz

# %% [markdown]
# As you can see from the output, the tool performs some preprocessing of the input TWAS file, and then computes an association for all LVs in the model (987 in our case) and the gene p-values from TWAS. The output is finally written to the path specified.
#
# **IMPORTANT:** keep in mind that you have to use a gene correlation matrix that is specific to your TWAS results. This is because gene correlations depend on the variants present in the original GWAS used. Check out [the notebooks here](https://github.com/greenelab/phenoplier/tree/main/nbs/15_gsa_gls) to see how to compute a gene correlation matrix specific for your trait of interest.

# %% [markdown]
# # Load LV-trait results

# %%
lv_df = pd.read_csv("/tmp/gls_phenoplier-basophill_percentage.tsv.gz", sep="\t")

# %%
lv_df

# %% [markdown]
# As you can see, LV603 is at the top of the LVs associations for basophill percentage.
# However, the onesided p-value here (`5.32e-15`) is larger than a simple correlation (`2.94e-27`), suggesting that we have correlated genes at the top of the LV.

# %% [markdown]
# # Conclusions

# %% [markdown]
# Hopefully, now have a more clear idea of the main data matrixes involved in PhenoPLIER (matrix Z, PhenomeXcan gene-trait associations, etc).
# We also see how to compute a p-value between an LV (group of genes or gene module) and a trait of interest.
# To do this with your own data, you need to compute the S-MultiXcan TWAS results (gene-based) from your GWAS summary stats and generate your own gene correlation matrix.
#
# In the next notebook (`02-LV_cell_types-...`), we'll see how to check in which tissues or cell types are our LV603'genes expressed.

# %%
