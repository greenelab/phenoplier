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
# This notebook analyzes the LVs driving the association of Niacin with some cardiovascular traits.

# %% [markdown] tags=[]
# # Modules loading

# %% tags=[]
# %load_ext autoreload
# %autoreload 2

# %% tags=[]
from pathlib import Path

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
# TODO hardcoded
input_file = Path(
    conf.DATA_DIR, "hetionet/pharmacotherapydb-v1.0", "indications.tsv"
).resolve()
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

# %% [markdown] tags=[]
# ## MultiPLIER Z

# %%
# multiplier_z = pd.read_pickle(conf.MULTIPLIER["MODEL_Z_MATRIX_FILE"])

# %%
# multiplier_z.shape

# %%
# multiplier_z.head()

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
drug_df_stats = ((drug_df - drug_mean) / drug_std).describe()
display(drug_df_stats)

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
    _tmp = drug_trait_predictions[c]

    _tmp = _tmp[_tmp > 0.0]
    q = _tmp.quantile(QUANTILE)
    _tmp = _tmp[_tmp > q]
    display(f"Number of LVs: {_tmp.shape[0]}")

    _tmp = _tmp.sort_values(ascending=False)
    common_lvs.append(_tmp)

    display(_tmp.head(20))
    print()

# %% [markdown]
# # Niacin top LVs

# %%
drug_data.abs().sort_values(ascending=False).head(30)

# %%
drug_data.sort_values(ascending=False).head(15)

# %%
drug_data.sort_values(ascending=True).head(15)

# %% [markdown]
# # Get common LVs

# %%
common_lvs_df = (
    pd.concat(common_lvs).reset_index().rename(columns={"index": "lv", 0: "value"})
)

# %%
common_lvs_df.shape

# %%
common_lvs_df.head()

# %%
lvs_by_sum = common_lvs_df.groupby("lv").sum().squeeze().sort_values(ascending=False)
display(lvs_by_sum.head(25))

# %%
lvs_by_count = (
    common_lvs_df.groupby("lv").count().squeeze().sort_values(ascending=False)
)
display(lvs_by_count.head(25))

# %% [markdown]
# # Save

# %%
output_file = OUTPUT_DIR / "cardiovascular-niacin.h5"
display(output_file)

# %%
with pd.HDFStore(output_file, mode="w", complevel=4) as store:
    store.put("traits_module_tissue_data", module_tissue_data, format="fixed")
    store.put("drug_data", drug_data, format="fixed")
    store.put("drug_trait_predictions", drug_trait_predictions, format="fixed")
    store.put("common_lvs", common_lvs_df, format="fixed")

# %%
