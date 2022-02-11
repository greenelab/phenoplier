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
# TODO

# %% [markdown] tags=[]
# # Environment variables

# %% tags=[]
# %load_ext autoreload
# %autoreload 2

# %% tags=[]
import conf

# %% tags=[]
N_JOBS = conf.GENERAL["N_JOBS"]
display(N_JOBS)

# %% tags=[]
# %env MKL_NUM_THREADS=$N_JOBS
# %env OPEN_BLAS_NUM_THREADS=$N_JOBS
# %env NUMEXPR_NUM_THREADS=$N_JOBS
# %env OMP_NUM_THREADS=$N_JOBS

# %% [markdown] tags=[]
# # Modules

# %% tags=[]
import warnings
from pathlib import Path

# import statsmodels.api as sm
# from statsmodels.stats.multitest import multipletests
import numpy as np
import pandas as pd

# from sklearn.preprocessing import scale
# from tqdm import tqdm

# from gls import GLSPhenoplier

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
OUTPUT_DIR = conf.RESULTS["GLS"]
display(OUTPUT_DIR)

assert OUTPUT_DIR.exists()

# %% [markdown] tags=[]
# # Get results files

# %% tags=[]
INPUT_FILES_PER_COHORT = {
    "phenomexcan": list(OUTPUT_DIR.glob("gls_phenotypes*combined*phenomexcan*.pkl")),
    "emerge": list(OUTPUT_DIR.glob("gls_phenotypes*combined*emerge*.pkl")),
}

# %% tags=[]
display(INPUT_FILES_PER_COHORT)

# %% [markdown] tags=[]
# # PhenomeXcan

# %% tags=[]
phenomexcan_df = pd.read_pickle(INPUT_FILES_PER_COHORT["phenomexcan"][0]).drop(
    columns=["pvalue_twosided", "part_k", "cluster_id"]
)

# %%
phenomexcan_df.shape

# %%
phenomexcan_df = phenomexcan_df.replace(
    {
        "phenotype": {
            "6152_100-Blood_clot_DVT_bronchitis_emphysema_asthma_rhinitis_eczema_allergy_diagnosed_by_doctor_None_of_the_above": "Asthma/allergic rhinitis/atopic dermatitis",
            "6152_8-Blood_clot_DVT_bronchitis_emphysema_asthma_rhinitis_eczema_allergy_diagnosed_by_doctor_Asthma": "Doctor diagnosed asthma",
            "6152_9-Blood_clot_DVT_bronchitis_emphysema_asthma_rhinitis_eczema_allergy_diagnosed_by_doctor_Hayfever_allergic_rhinitis_or_eczema": "Doctor diagnosed allergic rhinitis or atopic dermatitis",
        }
    }
)

# %%
with pd.option_context(
    "display.max_rows", None, "display.max_columns", None, "display.max_colwidth", None
):
    tmp = phenomexcan_df.sort_values("fdr").head(20)
    display(tmp)

# %% [markdown] tags=[]
# # eMERGE

# %% [markdown]
# ## Trait info

# %%
input_filepath = conf.EMERGE["DESC_FILE_WITH_SAMPLE_SIZE"]
display(input_filepath)

# %%
emerge_traits_df = pd.read_csv(
    input_filepath,
    sep="\t",
    dtype={"phecode": str},
    usecols=["phecode", "phenotype", "category"],
)

# %%
emerge_traits_df = emerge_traits_df.rename(
    columns={
        "phenotype": "phecode_phenotype",
        "category": "phecode_category",
    }
)

# %%
assert emerge_traits_df["phecode"].is_unique
emerge_traits_df = emerge_traits_df.set_index("phecode")

# %%
emerge_traits_df.shape

# %%
emerge_traits_df.head()

# %%
emerge_code_desc_map = emerge_traits_df["phecode_phenotype"].to_dict()

# %%
emerge_code_category_map = emerge_traits_df["phecode_category"].to_dict()

# %% [markdown]
# ## Association results

# %% tags=[]
emerge_df = pd.read_pickle(INPUT_FILES_PER_COHORT["emerge"][0]).drop(
    columns=["pvalue_twosided"]
)

# %%
emerge_df.insert(
    1, "phenotype_desc", emerge_df["phenotype"].apply(lambda x: emerge_code_desc_map[x])
)
emerge_df.insert(
    2,
    "phenotype_category",
    emerge_df["phenotype"].apply(lambda x: emerge_code_category_map[x]),
)

# %%
emerge_df.shape

# %%
emerge_df.head()

# %%
# phenomexcan_df = phenomexcan_df.replace(
#     {
#         "phenotype": {
#             "6152_100-Blood_clot_DVT_bronchitis_emphysema_asthma_rhinitis_eczema_allergy_diagnosed_by_doctor_None_of_the_above": "Asthma/allergic rhinitis/atopic dermatitis",
#             "6152_8-Blood_clot_DVT_bronchitis_emphysema_asthma_rhinitis_eczema_allergy_diagnosed_by_doctor_Asthma": "Doctor diagnosed asthma",
#             "6152_9-Blood_clot_DVT_bronchitis_emphysema_asthma_rhinitis_eczema_allergy_diagnosed_by_doctor_Hayfever_allergic_rhinitis_or_eczema": "Doctor diagnosed allergic rhinitis or atopic dermatitis",
#         }
#     }
# )

# %%
with pd.option_context(
    "display.max_rows", None, "display.max_columns", None, "display.max_colwidth", None
):
    tmp = emerge_df[emerge_df.fdr < 0.10]
    tmp = tmp.sort_values("fdr").head(20)
    display(tmp)

# %%
