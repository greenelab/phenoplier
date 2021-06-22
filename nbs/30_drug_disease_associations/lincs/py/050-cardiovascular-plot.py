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
# **TODO**

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
# OUTPUT_DIR = conf.RESULTS["DRUG_DISEASE_ANALYSES"]
# display(OUTPUT_DIR)

# assert OUTPUT_DIR.exists()
# # OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %% tags=[]
# INPUT_DATA_DIR = Path(
#     conf.RESULTS["DRUG_DISEASE_ANALYSES"],
#     "data",
# )
# display(INPUT_DATA_DIR)

# %% tags=[]
# OUTPUT_PREDICTIONS_DIR = Path(
#     conf.RESULTS["DRUG_DISEASE_ANALYSES"], "predictions", "dotprod_neg"
# )
# display(OUTPUT_PREDICTIONS_DIR)

# %% [markdown] tags=[]
# # Data loading

# %% [markdown] tags=[]
# ## S-MultiXcan projection

# %% tags=[]
# input_file = Path(
#     conf.RESULTS["PROJECTIONS_DIR"],
#     "projection-smultixcan-efo_partial-mashr-zscores.pkl",
# ).resolve()
# display(input_file)

# %%
# smultixcan_proj = pd.read_pickle(input_file)

# %%
# smultixcan_proj.shape

# %%
# smultixcan_proj.head()

# %% [markdown] tags=[]
# ## S-MultiXcan

# %%
# # smultixcan_zscores = pd.read_pickle(conf.PHENOMEXCAN["SMULTIXCAN_MASHR_ZSCORES_FILE"])
# smultixcan_zscores = pd.read_pickle(
#     conf.PHENOMEXCAN["SMULTIXCAN_EFO_PARTIAL_MASHR_ZSCORES_FILE"]
# )

# %%
# smultixcan_zscores = smultixcan_zscores.rename(index=Gene.GENE_ID_TO_NAME_MAP)

# %%
# smultixcan_zscores = smultixcan_zscores[~smultixcan_zscores.index.duplicated()]

# %%
# smultixcan_zscores = smultixcan_zscores.dropna(how="any")

# %%
# smultixcan_zscores.shape

# %%
# smultixcan_zscores.head()

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

# %%
lincs_data.index.is_unique

# %%
lincs_data.columns.is_unique

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
multiplier_z = pd.read_pickle(conf.MULTIPLIER["MODEL_Z_MATRIX_FILE"])

# %%
multiplier_z.shape

# %%
multiplier_z.head()

# %% [markdown] tags=[]
# ## Prediction results

# %% [markdown]
# ### Full

# %%
output_file = Path(
    conf.RESULTS["DRUG_DISEASE_ANALYSES"],
    "lincs",
    "predictions",
    "predictions_results.pkl",
).resolve()
display(output_file)

# %%
predictions = pd.read_pickle(output_file)

# %%
predictions.shape

# %%
predictions.head()


# %%
def _reduce_mean(x):
    return pd.Series(
        {"score": x["score"].mean(), "true_class": x["true_class"].unique()[0]}
    )


# %%
predictions_by_tissue = (
    predictions.groupby(["trait", "drug", "method", "tissue"])
    .apply(_reduce_mean)
    .dropna()
    .sort_index()
    .reset_index()
)

# %%
predictions_by_tissue.head()

# %% [markdown] tags=[]
# ### Aggregated

# %%
output_file = Path(
    conf.RESULTS["DRUG_DISEASE_ANALYSES"],
    "lincs",
    "predictions",
    "predictions_results_aggregated.pkl",
).resolve()
display(output_file)

# %%
predictions_avg = pd.read_pickle(output_file)

# %%
predictions_avg.shape

# %%
predictions_avg.head()

# %% [markdown]
# ### Merge

# %%
pharmadb_predictions = pd.merge(
    gold_standard_info,
    predictions_avg,
    #     left_on=["doid_id", "drugbank_id"],
    on=["trait", "drug"],
    how="inner",
)

# %%
pharmadb_predictions

# %%
pharmadb_predictions = pharmadb_predictions[
    ["trait", "drug", "disease", "drug_name", "method", "score", "true_class_x"]
].rename(columns={"true_class_x": "true_class", "drug_x": "drug"})

# %%
display(pharmadb_predictions.shape)
assert pharmadb_predictions.shape[0] == predictions_avg.shape[0]

# %%
pharmadb_predictions.head()

# %% tags=[]
pharmadb_predictions["trait"].unique().shape

# %% tags=[]
pharmadb_predictions["drug"].unique().shape

# %%
data_stats = pharmadb_predictions.groupby("method")["score"].describe()
display(data_stats)


# %%
# Standardize scores by method
def _standardize(x):
    return (x["score"] - data_stats.loc[x["method"], "mean"]) / data_stats.loc[
        x["method"], "std"
    ]


# %%
pharmadb_predictions = pharmadb_predictions.assign(
    score_std=pharmadb_predictions.apply(_standardize, axis=1)
)

# %%
pharmadb_predictions

# %% [markdown]
# ### Testing

# %%
_tmp = pharmadb_predictions.groupby("method")[["score", "score_std"]].describe()
display(_tmp)

# %%
_tmp0 = pharmadb_predictions[(pharmadb_predictions["method"] == "Gene-based")][
    ["score", "score_std"]
]

# %%
assert all(_tmp0.corr() > 0.99999)

# %%
_tmp0 = pharmadb_predictions[(pharmadb_predictions["method"] == "Module-based")][
    ["score", "score_std"]
]

# %%
assert all(_tmp0.corr() > 0.99999)

# %% [markdown]
# # List diseases

# %%
pharmadb_predictions["disease"].unique()

# %% [markdown]
# # Looks for differences in scores of both methods

# %%
np.all(pharmadb_predictions.groupby(["trait", "drug"]).count() == 2)


# %%
def _compare(x):
    assert x.shape[0] == 2
    x_sign = np.sign(x["score_std"].values)
    x0 = x.iloc[0]["score_std"]
    x1 = x.iloc[1]["score_std"]

    return pd.Series(
        {"different_sign": x_sign[0] != x_sign[1], "score_difference": np.abs(x0 - x1)}
    )


# %%
pharmadb_predictions = pharmadb_predictions.set_index(["trait", "drug"]).join(
    pharmadb_predictions.groupby(["trait", "drug"]).apply(_compare)
)

# %%
pharmadb_predictions.head()


# %%
def find_differences(trait_name):
    with pd.option_context(
        "display.max_rows", None, "display.max_columns", None, "max_colwidth", None
    ):
        _tmp = pharmadb_predictions[
            (pharmadb_predictions["disease"] == trait_name)
            & (pharmadb_predictions["different_sign"])
        ].sort_values(
            ["score_difference", "drug_name", "method"], ascending=[False, False, False]
        )
        display(_tmp)


# %% [markdown]
# ## any disease

# %%
with pd.option_context(
    "display.max_rows", None, "display.max_columns", None, "max_colwidth", None
):
    _tmp = pharmadb_predictions[
        (pharmadb_predictions["different_sign"])
        & (~pharmadb_predictions["disease"].str.contains("cancer"))  # avoid cancer
    ].sort_values(
        ["score_difference", "drug_name", "method"], ascending=[False, False, False]
    )
    display(_tmp.head(50))

# %% [markdown]
# ## coronary artery disease

# %%
find_differences("coronary artery disease")

# %% [markdown]
# ## atherosclerosis

# %%
find_differences("atherosclerosis")

# %% [markdown]
# # Niacin and Atherosclerosis

# %%
from entity import Trait

# %%
Trait.get_traits_from_efo("atherosclerosis")

# %%
d = Trait.get_trait(full_code="I70-Diagnoses_main_ICD10_I70_Atherosclerosis")

# %%
d.n, d.n_cases

# %%
Trait.get_traits_from_efo("coronary artery disease")

# %%
d = Trait.get_trait(full_code="CARDIoGRAM_C4D_CAD_ADDITIVE")
# d = Trait.get_trait(full_code="I25-Diagnoses_main_ICD10_I25_Chronic_ischaemic_heart_disease")

# %%
d.n, d.n_cases

# %%
_doid = "DOID:1936"
_phenomexcan_traits = ["I70-Diagnoses_main_ICD10_I70_Atherosclerosis", "CARDIoGRAM_C4D_CAD_ADDITIVE"]

# _doid = "DOID:3393"
# _phenomexcan_traits = [
#     "I25-Diagnoses_main_ICD10_I25_Chronic_ischaemic_heart_disease",
#     "CARDIoGRAM_C4D_CAD_ADDITIVE"
# ]

_drug_id = "DB00627"
_drug_name = "Niacin"

# %%
pharmadb_predictions[pharmadb_predictions["drug_name"] == _drug_name].sort_values(
    ["disease", "method"]
)

# %%
predictions_avg[
    (predictions_avg["trait"] == _doid) & (predictions_avg["drug"] == _drug_id)
]

# %%
predictions_by_tissue

# %%
Trait.get_trait("CARDIoGRAM_C4D_CAD_ADDITIVE").get_do_info()


# %%
def find_best_tissue(trait_id, drug_id):
    THERE IS A BUG IN THIS FUNCTION, NOT USING PARAMETERS
    SEE NOTEBOOK 040 FOR THE FIX
    
    _tmp = (
        predictions_by_tissue[
            (predictions_by_tissue["trait"] == _doid)
            & (predictions_by_tissue["drug"] == _drug_id)
        ]
        .groupby("method")["score"]
        .idxmax()
    )
    
    _tmp = predictions_by_tissue.loc[_tmp.values][["method", "tissue"]].set_index(
        "method"
    )
    
    _gene_based_tissue = _tmp.loc["Gene-based", "tissue"]
    _module_tissue = _tmp.loc["Module-based", "tissue"]
    
    return _module_tissue, _gene_based_tissue


# %%
_tmp_res = find_best_tissue(_doid, _drug_id)
display("Module-based, gene_based:")
display(_tmp_res)

# %%
# atherosclerosis
_tmp_res = find_best_tissue("DOID:1936", _drug_id)
display("Module-based, gene_based:")
display(_tmp_res)

# %%
# coronary artery disease
_tmp_res = find_best_tissue("DOID:3393", _drug_id)
display("Module-based, gene_based:")
display(_tmp_res)

# %% [markdown]
# ## Gene module-based - LVs driving association

# %%
available_doids = set(predictions_by_tissue["trait"].unique())
traits_lv_data = []

for trait in _phenomexcan_traits:
    t = Trait.get_trait(full_code=trait)
    t_doid = t.get_do_info().id
    
    # select available doid
    t_doid = [x for x in t_doid if x in available_doids]
    assert len(t_doid) == 1
    t_doid = t_doid[0]
    display(t_doid)
    
    best_module_tissue = find_best_tissue(t_doid, _drug_id)[0]
    display(best_module_tissue)
    best_module_tissue_data = pd.read_pickle(
        conf.RESULTS["DRUG_DISEASE_ANALYSES"]
        / "spredixcan"
        / "proj"
        / f"spredixcan-mashr-zscores-{best_module_tissue}-projection.pkl"
    )[trait]
#     display(best_module_tissue_data.head())
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
drug_data

# %%

# %%

# %%
module_tissue_data.iloc[:,0].sort_values().head(20)

# %%
module_tissue_data.iloc[:,1].sort_values().head(20)

# %%
_tmp = (-1.0 * drug_data.dot(module_tissue_data)).sort_values(
    ascending=False
)
display(_tmp)

# %%
_tmp = pd.DataFrame(
    drug_data.to_frame().values * module_tissue_data.values,
    columns=module_tissue_data.columns.copy(),
    index=drug_data.index.copy()
)

# %%
_tmp.sort_values("I70-Diagnoses_main_ICD10_I70_Atherosclerosis", ascending=True).head(20)

# %%
_tmp.sort_values("CARDIoGRAM_C4D_CAD_ADDITIVE", ascending=True).head(20)

# %%
ME QUEDE MIRANDO LOS DATAFRAMES DE ARRIBA Y DIJE "TENGO QUE SEGUIR CON EL K99"

# %%
top_lvs = _tmp.head(20)
display(top_lvs)

# %%
pd.DataFrame({
    "lv": top_lvs.index,
    "lv_score": top_lvs.values,
    "trait_name": ""
})

# %%

# %%

# %%

# %%
import seaborn as sns
sns.set_theme(style="ticks")
exercise = sns.load_dataset("exercise")

# %%

# %%
_idx = np.random.randint(0, 90, 10)
exercise.loc[_idx, "pulse"] = exercise.loc[_idx, "pulse"] * -1

# %%
g = sns.catplot(x="time", y="pulse", hue="kind", data=exercise)

# %%
