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
# # Modules loading

# %% tags=[]
# %load_ext autoreload
# %autoreload 2

# %% tags=[]
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import HTML
from tqdm import tqdm

from entity import Trait, Gene

import conf

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
EXPERIMENT_NAME = "lv"
LIPIDS_GENE_SET = "gene_set_increase"

# %%
RANDOM_SEED = 0
N_PERMUTATIONS = 1000
N_TOP_TRAITS = 25

# %% [markdown] tags=[]
# # Data loading

# %% [markdown] tags=[]
# ## PhenomeXcan data (S-MultiXcan)

# %% tags=[]
smultixcan_results_filename = conf.PHENOMEXCAN[
    "SMULTIXCAN_EFO_PARTIAL_MASHR_ZSCORES_FILE"
]

display(smultixcan_results_filename)

# %% tags=[]
results_filename_stem = smultixcan_results_filename.stem
display(results_filename_stem)

# %% tags=[]
smultixcan_results = pd.read_pickle(smultixcan_results_filename)

# %% tags=[]
smultixcan_results.shape

# %% tags=[]
smultixcan_results.head()

# %% [markdown] tags=[]
# ### Gene IDs to Gene names

# %% tags=[]
smultixcan_results = smultixcan_results.rename(index=Gene.GENE_ID_TO_NAME_MAP)

# %% tags=[]
smultixcan_results.shape

# %% tags=[]
smultixcan_results.head()

# %% [markdown] tags=[]
# ### Remove duplicated gene entries

# %% tags=[]
smultixcan_results.index[smultixcan_results.index.duplicated(keep="first")]

# %% tags=[]
smultixcan_results = smultixcan_results.loc[
    ~smultixcan_results.index.duplicated(keep="first")
]

# %% tags=[]
smultixcan_results.shape

# %% [markdown] tags=[]
# ### Some checks

# %% tags=[]
# the data should have no NaN values
assert smultixcan_results.shape == smultixcan_results.dropna(how="any").shape

# %% [markdown] tags=[]
# ### Standardize S-MultiXcan results

# %% tags=[]
_tmp = smultixcan_results.apply(lambda x: x / x.sum())

# %% tags=[]
_tmp.shape

# %% tags=[]
assert _tmp.shape == smultixcan_results.shape

# %% tags=[]
smultixcan_results = _tmp

# %% tags=[]
smultixcan_results.head()

# %% [markdown] tags=[]
# ## MultiPLIER Z matrix

# %%
multiplier_z = pd.read_pickle(conf.MULTIPLIER["MODEL_Z_MATRIX_FILE"])

# %%
multiplier_z.shape

# %%
multiplier_z.head()

# %% [markdown] tags=[]
# ## PhenomeXcan projection

# %% tags=[]
input_filepath = Path(
    conf.RESULTS["PROJECTIONS_DIR"],
    "projection-smultixcan-efo_partial-mashr-zscores.pkl",
).resolve()
display(input_filepath)

# %% tags=[]
phenomexcan_data = pd.read_pickle(input_filepath).T

# %% tags=[]
phenomexcan_data.shape

# %% tags=[]
phenomexcan_data.head()

# %% [markdown] tags=[]
# ## LVs enrichment on DEG from CRISPR screen

# %% tags=[]
deg_enrich = pd.read_csv(
    Path(conf.RESULTS["CRISPR_ANALYSES"]["BASE_DIR"], "fgsea-all_lvs.tsv").resolve(),
    sep="\t",
)

# %% tags=[]
deg_enrich.shape

# %% tags=[]
deg_enrich.head()

# %% [markdown] tags=[]
# ## MultiPLIER summary

# %% tags=[]
multiplier_model_summary = pd.read_pickle(conf.MULTIPLIER["MODEL_SUMMARY_FILE"])

# %% tags=[]
multiplier_model_summary.shape

# %% tags=[]
multiplier_model_summary.head()

# %% [markdown] tags=[]
# # Get gene modules under analysis

# %% tags=[]
df = deg_enrich[
    deg_enrich["pathway"].isin(("gene_set_decrease",)) & (deg_enrich["padj"] < 0.05)
].sort_values("padj", ascending=True)

# %% tags=[]
df.shape

# %%
df.head()

# %% tags=[]
important_lvs = df["lv"].unique()

# %% tags=[]
display(important_lvs.shape)
assert important_lvs.shape[0] == 24

# %% tags=[]
important_lvs

# %% [markdown]
# # How many genes with nonzero weight on average in these modules?

# %%
multiplier_z[df["lv"].values].apply(lambda x: x[x > 0].shape[0]).describe()

# %% [markdown]
# # How likely is to get relevant traits at the top of these 24 modules?

# %%
from multiplier import MultiplierProjection

# %%
lipids_related_traits = {
    "celiac disease",
    "4079_raw-Diastolic_blood_pressure_automated_reading",
    "6150_100-Vascularheart_problems_diagnosed_by_doctor_None_of_the_above",
    "I9_UAP-Unstable_angina_pectoris",
    "6150_4-Vascularheart_problems_diagnosed_by_doctor_High_blood_pressure",
    "malabsorption syndrome",
    "K11_COELIAC-Coeliac_disease",
    "hypertension",
    "atherosclerosis",
    "4080_raw-Systolic_blood_pressure_automated_reading",
}

# %%
np.random.seed(RANDOM_SEED)

# %% tags=[]
z = multiplier_z  # .copy()

permutation_results = []

for i in tqdm(range(N_PERMUTATIONS)):
    traits = []

    sub_z = z[important_lvs]  # .copy()

    # shuffle index
    sub_z_index = sub_z.index.tolist()
    np.random.shuffle(sub_z_index)
    sub_z.index = sub_z_index

    new_model_z = z.drop(columns=important_lvs).join(sub_z)
    new_model_z = new_model_z.loc[z.index.tolist(), z.columns.tolist()]
    assert new_model_z.shape == multiplier_z.shape

    mproj = MultiplierProjection()
    new_proj = mproj.transform(smultixcan_results, multiplier_model_z=new_model_z)
    new_proj = new_proj.T

    for lv in important_lvs:
        _tmp = new_proj[lv]
        _tmp = _tmp[_tmp > 0.0].sort_values(ascending=False)
        traits.append(_tmp)

    traits_df = (
        pd.concat(traits)
        .reset_index()
        .groupby("index")
        .sum()
        .sort_values(0, ascending=False)
        .reset_index()
    ).rename(columns={"index": "trait", 0: "value"})

    top_traits = traits_df.head(N_TOP_TRAITS)
    permutation_results.append(lipids_related_traits.intersection(top_traits["trait"]))

# %% [markdown] tags=[]
# ## Calculate p-value

# %%
# in this case we are permisive to compute the p-value, and count cases where at least half of the important traits are among the top
pval = (
    sum(
        [
            (i / len(lipids_related_traits)) > 0.50
            for i in list(map(len, permutation_results))
        ]
    )
    + 1
) / (len(permutation_results) + 1)
display(pval)

# %%
# what we claim in the manuscript
assert pval < 0.001

# %%
