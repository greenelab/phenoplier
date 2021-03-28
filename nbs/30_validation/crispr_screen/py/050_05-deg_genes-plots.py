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

import pandas as pd
from IPython.display import HTML
import matplotlib.pyplot as plt
import matplotlib_venn as mv
import seaborn as sns

from entity import Trait, Gene
from data.cache import read_data
import conf

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
# EXPERIMENT_NAME = "single_gene"

# LIPIDS_GENE_SET = "gene_set_increase"
# LIPIDS_GENE_SET_QUERY = "(rank == 3) | (rank == 2)"

# %% tags=[]
OUTPUT_DIR = Path(
    conf.RESULTS["CRISPR_ANALYSES"]["BASE_DIR"], f"analyses"
)
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
display(OUTPUT_DIR)

# %% [markdown] tags=[]
# # Data loading

# %% [markdown]
# ## LV

# %%
lv_traits = pd.read_pickle(Path(conf.RESULTS["CRISPR_ANALYSES"]["BASE_DIR"], "lv-gene_set_decrease", "traits.pkl").resolve())

# %%
lv_traits.shape

# %%
lv_traits.head()

# %% tags=[]
lv_traits_categories = (
    lv_traits.groupby("category")
    .mean()
    .sort_values("value", ascending=False)
    .reset_index()
)

# %%
lv_traits_categories.shape

# %%
lv_traits_categories.head()

# %% [markdown]
# ## Single gene

# %%
sg_traits = pd.read_pickle(Path(conf.RESULTS["CRISPR_ANALYSES"]["BASE_DIR"], "single_gene-gene_set_decrease", "traits.pkl").resolve())

# %%
sg_traits.shape

# %%
sg_traits.head()

# %% tags=[]
sg_traits_categories = (
    sg_traits.groupby("category")
    .mean()
    .sort_values("value", ascending=False)
    .reset_index()
)

# %%
sg_traits_categories.shape

# %%
sg_traits_categories.head()

# %% [markdown] tags=[]
# # Categorize traits

# %%
trait_categories_map = {
#     "body height": "Body size measures",
    
    "6150_4-Vascularheart_problems_diagnosed_by_doctor_High_blood_pressure": "Hypertension",
    "hypertension": "Hypertension",
    
    "atherosclerosis": "Cardiovascular diseases",
    "DM_PERIPHATHERO-Peripheral_atherosclerosis": "Cardiovascular diseases",
    
    "I9_UAP-Unstable_angina_pectoris": "Heart diseases",
    "bundle branch block": "Heart diseases", # also CVD
    "I9_AVBLOCK-AVblock": "Heart diseases", # also CVD
    "I9_CONDUCTIO-Conduction_disorders": "Heart diseases", # also CVD
    
    "H7_EPIPHORA-Epiphora": "Eye problems",
    
    "C3_SKIN-Malignant_neoplasm_of_skin": "Skin cancers",
    "C_SKIN": "Skin cancers",
    "C_OTHER_SKIN-Other_malignant_neoplasms_of_skin": "Skin cancers",
    
    "ASTHMA_EOSINOPHIL_SUGG-Suggestive_for_eosinophilic_asthma": "Respiratory diseases",
    "J20-Diagnoses_main_ICD10_J20_Acute_bronchitis": "Respiratory diseases",
    "pleural empyema": "Respiratory diseases",
    
    "RHEUMA_SEROPOS-Seropositive_rheumatoid_arthritis": "Autoimmune diseases",
    "RHEUMA_SEROPOS_OTH-Otherunspecified_seropositiverheumatoid_arthritis": "Autoimmune diseases",
    "celiac disease": "Autoimmune diseases",
    "K11_COELIAC-Coeliac_disease": "Autoimmune diseases",
    "malabsorption syndrome": "Autoimmune diseases", # this one comes from ICD10 K00, which is mainly "celiac disease"
    
    "22507_raw-Age_of_stopping_smoking": "Smoking",
    "22506_112-Tobacco_smoking_Occasionally": "Smoking",
    
    "diabetes mellitus": "Diseases (endocrine/diabetes)",
    
    "20090_360-Type_of_fatoil_used_in_cooking_Spreadable_butter": "Diet",
    "103060-Poultry_intake": "Diet",
    "104460-Banana_intake": "Diet",
    "100920_2105-Type_milk_consumed_soya_with_calcium": "Diet",
}

categories_map = {
    "Blood": "Blood count",
    "Diseases (cardiovascular)": "Cardiovascular diseases",
    "Diseases (respiratory/ent)": "Respiratory diseases",
    "Diseases (FinnGen)": "Other diseases/disorders",
    "Diseases (ICD10 main)": "Other diseases/disorders",
    "Employment history": "Employment",
    "Medication": "Medications",
}

def _assign_category(x):
    trait_name = x["trait"]
    if trait_name in trait_categories_map:
        trait_categ = trait_categories_map[trait_name]
    else:
        trait_categ = x["category"]
    
    if trait_categ in categories_map:
        return categories_map[trait_categ]
    
    return trait_categ


# %%
with pd.option_context(
    "display.max_rows", None, "display.max_columns", None, "max_colwidth", None
):
    _tmp = lv_traits.assign(category=lv_traits.apply(_assign_category, axis=1))
    display(_tmp.head(100))

# %%
_n_top = 50

_lv_tmp = lv_traits.head(_n_top).assign(category=lv_traits.apply(_assign_category, axis=1)).assign(group="lv")
_lv_tmp["value"] = _lv_tmp["value"] / _lv_tmp["value"].max()

_sg_tmp = sg_traits.head(_n_top).assign(category=sg_traits.apply(_assign_category, axis=1)).assign(group="single-gene")
_sg_tmp["value"] = _sg_tmp["value"] / _sg_tmp["value"].max()

# %%
_all_tmp = pd.concat((_lv_tmp, _sg_tmp), ignore_index=True)

# %%
_all_tmp.shape

# %%
_all_tmp.head()

# %%
_cat_order = _all_tmp.groupby(["group", "category"])["value"].max().sort_values(ascending=False)

# %%
_cat_order.loc["lv"].head(10)

# %%
_cat_order.loc["single-gene"].head(10)

# %%
_final_cat_order = pd.concat((
    _cat_order.loc["lv"].head(10),
    _cat_order.loc["single-gene"].head(10),
))

# %%
_final_cat_order.sort_values(ascending=False)

# %%
with sns.plotting_context("paper"):
    f, ax = plt.subplots(figsize=(8, 8))  # (figsize=(8, 8))
    ax = sns.boxplot(
        x="category",
        y="value",
        hue="group",
        order=_final_cat_order.index[:10].tolist(),
        data=_all_tmp,
        palette="Set3",
        ax=ax
    )
    ax.set_xticklabels(ax.get_xticklabels(),rotation=30, horizontalalignment="right")

# %%
with sns.plotting_context("paper"):
    f, ax = plt.subplots(figsize=(8, 8))  # (figsize=(8, 8))
    ax = sns.countplot(
        x="category",
#         y="value",
        hue="group",
        order=_final_cat_order.index.drop_duplicates(keep="first").tolist(),
        data=_all_tmp,
        palette="Set3",
        ax=ax
    )
    ax.set_xticklabels(ax.get_xticklabels(),rotation=30, horizontalalignment="right")

# %% [markdown] tags=[]
# # Venn diagram

# %% tags=[]
plt.figure()

_tmp = sns.color_palette("colorblind", 2)

_category = "Blood pressure"

_top_lv_traits = lv_traits.head(100)
_top_lv_traits = _top_lv_traits[_top_lv_traits["category"] == _category]["trait"]

_top_sg_traits = sg_traits.head(100)
_top_sg_traits = _top_sg_traits[_top_sg_traits["category"] == _category]["trait"]

# s1 = set(lv_traits_categories.head(10)["category"])
# s2 = set(sg_traits_categories.head(10)["category"])

s1 = set(_top_lv_traits)
s2 = set(_top_sg_traits)

v = mv.venn2([s1, s2], ["LV","SG"], set_colors=_tmp)
v.get_label_by_id('10').set_text('a, b, c, d')
v.get_label_by_id('11').set_text('a, b, c, d')
v.get_label_by_id('01').set_text('a, b, c, d')

# %%
_top_lv_traits.tolist()

# %%
_top_sg_traits.tolist()

# %%
_tmp = sns.color_palette("hls", 2)

# %%
