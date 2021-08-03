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
# This notebook analyzes the LVs driving the association of Niacin with some cardiovascular traits. Then it writes a table in markdown with the results.

# %% [markdown] tags=[]
# # Modules loading

# %% tags=[]
# %load_ext autoreload
# %autoreload 2

# %% tags=[]
from pathlib import Path
import re

import numpy as np
import pandas as pd

from entity import Gene
import conf

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
QUANTILE = 0.95

# %% [markdown] tags=[]
# # Paths

# %% tags=[]
OUTPUT_DIR = conf.RESULTS["DRUG_DISEASE_ANALYSES"] / "lincs" / "analyses"
display(OUTPUT_DIR)
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# %%
INPUT_DIR = conf.RESULTS["DRUG_DISEASE_ANALYSES"] / "lincs" / "predictions"
# display(OUTPUT_DIR)
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
input_predictions_by_tissue_file = INPUT_DIR / "full_predictions_by_tissue-rank.h5"
display(input_predictions_by_tissue_file)

# %%
assert "CONTENT_DIR" in conf.MANUSCRIPT

OUTPUT_FILE_PATH = conf.MANUSCRIPT["CONTENT_DIR"] / "04.15.drug_disease_prediction.md"
display(OUTPUT_FILE_PATH)
assert OUTPUT_FILE_PATH.exists()

# %% [markdown] tags=[]
# # Data loading

# %% [markdown] tags=[]
# ## PharmacotherapyDB: load gold standard

# %% [markdown]
# ### Final

# %%
gold_standard = pd.read_pickle(
    Path(conf.RESULTS["DRUG_DISEASE_ANALYSES"], "gold_standard.pkl"),
)

# %%
gold_standard.shape

# %%
gold_standard.head()

# %% [markdown]
# ### Info

# %% tags=[]
input_file = conf.PHARMACOTHERAPYDB["INDICATIONS_FILE"]
display(input_file)

# %%
gold_standard_info = pd.read_csv(input_file, sep="\t")

# %%
gold_standard_info = gold_standard_info.rename(columns={"drug": "drug_name"})

# %% tags=[]
gold_standard_info.shape

# %% tags=[]
gold_standard_info.head()

# %%
gold_standard_info = (
    gold_standard.set_index(["trait", "drug"])
    .join(
        gold_standard_info.rename(
            columns={"doid_id": "trait", "drugbank_id": "drug"}
        ).set_index(["trait", "drug"])
    )
    .reset_index()
)

# %% tags=[]
gold_standard_info.shape

# %% tags=[]
gold_standard_info.head()

# %% [markdown] tags=[]
# ## LINCS data

# %% tags=[]
input_file = Path(
    conf.RESULTS["DRUG_DISEASE_ANALYSES"], "lincs", "lincs-data.pkl"
).resolve()

display(input_file)

# %% tags=[]
lincs_data = pd.read_pickle(input_file).T.rename(columns=Gene.GENE_ID_TO_NAME_MAP)

# %% tags=[]
display(lincs_data.shape)

# %% tags=[]
display(lincs_data.head())

# %% [markdown] tags=[]
# ## LINCS projection

# %% tags=[]
input_file = Path(
    conf.RESULTS["DRUG_DISEASE_ANALYSES"], "lincs", "lincs-projection.pkl"
).resolve()

display(input_file)

# %% tags=[]
lincs_projection = pd.read_pickle(input_file).T

# %% tags=[]
display(lincs_projection.shape)

# %% tags=[]
display(lincs_projection.head())

# %% [markdown]
# # Niacin and cardiovascular diseases

# %%
from entity import Trait

# %%
Trait.get_traits_from_efo("atherosclerosis")

# %%
Trait.get_traits_from_efo("coronary artery disease")

# %%
_phenomexcan_traits = [
    "I70-Diagnoses_main_ICD10_I70_Atherosclerosis",
    "CARDIoGRAM_C4D_CAD_ADDITIVE",
    "I25-Diagnoses_main_ICD10_I25_Chronic_ischaemic_heart_disease",
    "20002_1473-Noncancer_illness_code_selfreported_high_cholesterol",
    "6150_100-Vascularheart_problems_diagnosed_by_doctor_None_of_the_above",
    "6150_1-Vascularheart_problems_diagnosed_by_doctor_Heart_attack",
    "I9_CHD-Major_coronary_heart_disease_event",
    "I9_CORATHER-Coronary_atherosclerosis",
    "I9_IHD-Ischaemic_heart_disease_wide_definition",
    "I9_MI-Myocardial_infarction",
    "I21-Diagnoses_main_ICD10_I21_Acute_myocardial_infarction",
    "20002_1075-Noncancer_illness_code_selfreported_heart_attackmyocardial_infarction",
]

_drug_id = "DB00627"
_drug_name = "Niacin"

# %%
for p in _phenomexcan_traits:
    print(p)
    d = Trait.get_trait(full_code=p)
    print((d.n, d.n_cases))

    print("\n")

# %% [markdown]
# ## Get best tissue results for Niacin

# %%
drugs_tissue_df = {}

with pd.HDFStore(input_predictions_by_tissue_file, mode="r") as store:
    for tk in store.keys():
        df = store[tk][_drug_id]

        drugs_tissue_df[tk[1:]] = df

# %%
_tmp = pd.DataFrame(drugs_tissue_df)
display(_tmp.shape)
display(_tmp.head())

# %%
# show top tissue models (from TWAS) for each trait
traits_best_tissues_df = (
    pd.DataFrame(drugs_tissue_df).loc[_phenomexcan_traits].idxmax(1)
)
display(traits_best_tissues_df)

# %%
# pick the tissue with the maximum score for each trait
drug_df = pd.DataFrame(drugs_tissue_df).max(1)

# %%
drug_df.shape

# %%
drug_df.head()

# %%
drug_df.loc[_phenomexcan_traits].sort_values()

# %%
drug_df.describe()

# %%
drug_mean, drug_std = drug_df.mean(), drug_df.std()
display((drug_mean, drug_std))

# %%
drug_df_std = (drug_df - drug_mean) / drug_std
drug_df_stats = drug_df_std.describe()
display(drug_df_stats)

# %%
drug_df_std.quantile([0.80, 0.85, 0.90, 0.95])

# %%
drug_df = (drug_df.loc[_phenomexcan_traits] - drug_mean) / drug_std

# %%
drug_df.shape

# %%
drug_df.sort_values()

# %% [markdown]
# All predictions of Niacin for these traits are high (above the mean and a standard deviation away)

# %%
# select traits for which niacin has a high prediction
selected_traits = drug_df[drug_df > drug_df_stats["75%"]].index.tolist()

# %%
selected_traits


# %% [markdown]
# ## Gene module-based - LVs driving association

# %%
def find_best_tissue(trait_id):
    return traits_best_tissues_df.loc[trait_id]


# %%
_tmp_res = find_best_tissue("I9_CORATHER-Coronary_atherosclerosis")
display(_tmp_res)

# %%
# available_doids = set(predictions_by_tissue["trait"].unique())
traits_lv_data = []

for trait in selected_traits:
    best_module_tissue = find_best_tissue(trait)
    display(best_module_tissue)

    best_module_tissue_data = pd.read_pickle(
        conf.RESULTS["DRUG_DISEASE_ANALYSES"]
        / "spredixcan"
        / "proj"
        / f"spredixcan-mashr-zscores-{best_module_tissue}-projection.pkl"
    )[trait]

    traits_lv_data.append(best_module_tissue_data)

# %%
module_tissue_data = pd.DataFrame(traits_lv_data).T

# %%
module_tissue_data.shape

# %%
module_tissue_data.head()

# %%
drug_data = lincs_projection.loc[_drug_id]

# %%
drug_data.head()

# %%
_tmp = (-1.0 * drug_data.dot(module_tissue_data)).sort_values(ascending=False)
display(_tmp)

# %%
drug_trait_predictions = pd.DataFrame(
    -1.0 * (drug_data.to_frame().values * module_tissue_data.values),
    columns=module_tissue_data.columns.copy(),
    index=drug_data.index.copy(),
)

# %%
drug_trait_predictions.shape

# %%
drug_trait_predictions.head()

# %%
common_lvs = []

for c in drug_trait_predictions.columns:
    d = Trait.get_trait(full_code=c)
    display(f"Name: {d.description}")
    display(f"Sample size: {(d.n, d.n_cases)}")

    _tmp = drug_trait_predictions[c]

    _tmp = _tmp[_tmp > 0.0]
    q = _tmp.quantile(QUANTILE)
    _tmp = _tmp[_tmp > q]
    display(f"Number of LVs: {_tmp.shape[0]}")

    _tmp = (
        _tmp.sort_values(ascending=False)
        .rename("lv_diff")
        .reset_index()
        .rename(columns={"index": "lv"})
    )
    _tmp = _tmp.assign(trait=c)
    common_lvs.append(_tmp)

    display(_tmp.head(20))
    print()

# %% [markdown]
# # Get common LVs

# %%
common_lvs_df = pd.concat(common_lvs)  # .rename(columns={"index": "lv", 0: "value"})

# %%
common_lvs_df.shape

# %%
common_lvs_df.head()

# %%
lvs_by_count = (
    common_lvs_df.groupby("lv")["lv_diff"]
    .count()
    .squeeze()
    .sort_values(ascending=False)
)
display(lvs_by_count.head(25))

# %%
lvs_sel = []

# %%
with pd.option_context(
    "display.max_rows", None, "display.max_columns", None, "display.max_colwidth", None
):
    lv_df = common_lvs_df[common_lvs_df["lv"] == "LV116"].sort_values(
        "lv_diff", ascending=False
    )
    display(lv_df)
    lvs_sel.append(lv_df)

# %%
lv_df = common_lvs_df[common_lvs_df["lv"] == "LV931"].sort_values(
    "lv_diff", ascending=False
)
display(lv_df)
lvs_sel.append(lv_df)

# %%
lv_df = common_lvs_df[common_lvs_df["lv"] == "LV246"].sort_values(
    "lv_diff", ascending=False
)
display(lv_df)
lvs_sel.append(lv_df)

# %%
lv_df = pd.concat(lvs_sel, ignore_index=True)
display(lv_df.head())

# %%
from traits import SHORT_TRAIT_NAMES


# %%
def get_trait_objs(phenotype_full_code):
    if Trait.is_efo_label(phenotype_full_code):
        traits = Trait.get_traits_from_efo(phenotype_full_code)
    else:
        traits = [Trait.get_trait(full_code=phenotype_full_code)]

    # sort by sample size
    return sorted(traits, key=lambda x: x.n_cases / x.n, reverse=True)


def get_trait_description(phenotype_full_code):
    traits = get_trait_objs(phenotype_full_code)

    desc = traits[0].description
    if desc in SHORT_TRAIT_NAMES:
        return SHORT_TRAIT_NAMES[desc]

    return desc


def get_trait_n(phenotype_full_code):
    traits = get_trait_objs(phenotype_full_code)

    return traits[0].n


def get_trait_n_cases(phenotype_full_code):
    traits = get_trait_objs(phenotype_full_code)

    return traits[0].n_cases


def num_to_int_str(num):
    if pd.isnull(num):
        return ""

    return f"{num:,.0f}"


def get_part_clust(row):
    return f"{row.part_k} / {row.cluster_id}"


# %%
lv_df = lv_df.assign(trait_desc=lv_df["trait"].apply(get_trait_description))

# %%
lv_df = lv_df.assign(n=lv_df["trait"].apply(get_trait_n))

# %%
lv_df = lv_df.assign(n_cases=lv_df["trait"].apply(get_trait_n_cases))

# %%
lv_df = lv_df.assign(n=lv_df["n"].apply(num_to_int_str))

# %%
lv_df = lv_df.assign(n_cases=lv_df["n_cases"].apply(num_to_int_str))

# %%
CELL_TYPES_LVS = {
    "LV246": "Adipose tissue, liver",
    "LV116": "Immune cells, skin",
    "LV931": "Immune cells",
}

# %%
lv_df["Cell type"] = lv_df["lv"].apply(lambda x: CELL_TYPES_LVS[x])

# %%
lv_df["Niacin effect"] = lv_df["lv"].apply(
    lambda x: "-" if drug_data.loc[x] < 0 else "+"
)

# %%
lv_df = lv_df.rename(
    columns={
        "lv": "LV",
        "trait_desc": "Disease",
        "n": "Sample size",
        "n_cases": "Cases",
    }
)

# %%
lv_df[["LV", "Cell type", "Disease", "Sample size", "Cases"]]

# %%
lv_df = lv_df[["LV", "Cell type", "Disease"]]

# %%
lv_df = (
    lv_df.sort_values(["LV", "Disease"])
    .set_index("LV")
    .loc[["LV116", "LV931", "LV246"]]
    .reset_index()
)

# %%
lv_df.loc[[1, 2, 3, 5, 6, 8, 9], ["LV", "Cell type"]] = ""

# %%
with pd.option_context(
    "display.max_rows", None, "display.max_columns", None, "display.max_colwidth", None
):
    display(lv_df)

# %% [markdown]
# ## Save table

# %%
# result_set is either phenomexcan or emerge
LV_FILE_MARK_TEMPLATE = "<!-- niacin:cardiovascular:top_lvs:{position} -->"

# %%
# start
lv_file_mark_start = LV_FILE_MARK_TEMPLATE.format(position="start")
display(lv_file_mark_start)

# end
lv_file_mark_end = LV_FILE_MARK_TEMPLATE.format(position="end")
display(lv_file_mark_end)

# %%
new_content = lv_df.to_markdown(index=False, disable_numparse=True)

# %%
full_new_content = (
    lv_file_mark_start + "\n" + new_content.strip() + "\n" + lv_file_mark_end
)

# %%
with open(OUTPUT_FILE_PATH, "r", encoding="utf8") as f:
    file_content = f.read()

# %%
new_file_content = re.sub(
    lv_file_mark_start + ".*?" + lv_file_mark_end,
    full_new_content,
    file_content,
    flags=re.DOTALL,
)

# %%
with open(OUTPUT_FILE_PATH, "w", encoding="utf8") as f:
    f.write(new_file_content)  # .replace("\beta", r"\beta"))

# %% [markdown]
# # Niacin top LVs

# %%
drug_data.abs().sort_values(ascending=False).head(30)

# %%
drug_data.sort_values(ascending=False).head(30)

# %%
drug_data.sort_values(ascending=True).head(30)

# %%
