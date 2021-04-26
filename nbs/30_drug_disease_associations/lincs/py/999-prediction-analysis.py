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
OUTPUT_DIR = conf.RESULTS["DRUG_DISEASE_ANALYSES"]
display(OUTPUT_DIR)

assert OUTPUT_DIR.exists()
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %% tags=[]
INPUT_DATA_DIR = Path(
    conf.RESULTS["DRUG_DISEASE_ANALYSES"],
    "data",
)
display(INPUT_DATA_DIR)

# %% tags=[]
OUTPUT_PREDICTIONS_DIR = Path(
    conf.RESULTS["DRUG_DISEASE_ANALYSES"], "predictions", "dotprod_neg"
)
display(OUTPUT_PREDICTIONS_DIR)

# %% [markdown] tags=[]
# # S-MultiXcan projection

# %% tags=[]
input_file = Path(
    conf.RESULTS["PROJECTIONS_DIR"],
    "projection-smultixcan-efo_partial-mashr-zscores.pkl",
).resolve()
display(input_file)

# %%
smultixcan_proj = pd.read_pickle(input_file)

# %%
smultixcan_proj.shape

# %%
smultixcan_proj.head()

# %% [markdown] tags=[]
# # S-MultiXcan

# %%
# smultixcan_zscores = pd.read_pickle(conf.PHENOMEXCAN["SMULTIXCAN_MASHR_ZSCORES_FILE"])
smultixcan_zscores = pd.read_pickle(
    conf.PHENOMEXCAN["SMULTIXCAN_EFO_PARTIAL_MASHR_ZSCORES_FILE"]
)

# %%
smultixcan_zscores = smultixcan_zscores.rename(index=Gene.GENE_ID_TO_NAME_MAP)

# %%
smultixcan_zscores = smultixcan_zscores[~smultixcan_zscores.index.duplicated()]

# %%
smultixcan_zscores = smultixcan_zscores.dropna(how="any")

# %%
smultixcan_zscores.shape

# %%
smultixcan_zscores.head()

# %% [markdown] tags=[]
# # PharmacotherapyDB: load gold standard

# %% [markdown]
# ## Final

# %%
gold_standard = pd.read_pickle(
    Path(conf.RESULTS["DRUG_DISEASE_ANALYSES"], "gold_standard.pkl"),
)

# %%
gold_standard.shape

# %%
gold_standard.head()

# %%
gold_standard["trait"].unique().shape

# %%
gold_standard["drug"].unique().shape

# %%
gold_standard["true_class"].value_counts()

# %%
gold_standard["true_class"].value_counts(normalize=True)

# %% [markdown]
# ## Info

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
# # LINCS data

# %% tags=[]
input_file = Path(INPUT_DATA_DIR, "raw", "lincs-data.pkl").resolve()

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
# # LINCS projection

# %% tags=[]
input_file = Path(INPUT_DATA_DIR, "proj", "lincs-projection.pkl").resolve()

display(input_file)

# %% tags=[]
lincs_projection = pd.read_pickle(input_file).T

# %% tags=[]
display(lincs_projection.shape)

# %% tags=[]
display(lincs_projection.head())

# %% [markdown] tags=[]
# # MultiPLIER Z

# %%
multiplier_z = pd.read_pickle(conf.MULTIPLIER["MODEL_Z_MATRIX_FILE"])

# %%
multiplier_z.shape

# %%
multiplier_z.head()

# %% [markdown] tags=[]
# # Prediction results

# %% [markdown]
# ## Full

# %%
output_file = Path(OUTPUT_DIR, "predictions", "predictions_results.pkl").resolve()
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
        {
            "score": x["score"].mean(),
            "true_class": x["true_class"].unique()[0]
            #             if x["true_class"].unique().shape[0] == 1
            #             else None,
        }
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
# ## Aggregated

# %%
output_file = Path(
    OUTPUT_DIR, "predictions", "predictions_results_aggregated.pkl"
).resolve()
display(output_file)

# %%
predictions_avg = pd.read_pickle(output_file)

# %%
predictions_avg.shape

# %%
predictions_avg.head()

# %% [markdown]
# # Merge

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
    #     return (x["score"] - data_stats.loc[x["method"], "min"]) / (
    #         data_stats.loc[x["method"], "max"] - data_stats.loc[x["method"], "min"]
    #     )
    return (x["score"] - data_stats.loc[x["method"], "mean"]) / data_stats.loc[
        x["method"], "std"
    ]


# %%
pharmadb_predictions = pharmadb_predictions.assign(
    score_std=pharmadb_predictions.apply(_standardize, axis=1)
)

# %%
pharmadb_predictions

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
# # Testing magma-like models

# %% [markdown]
# ```
# LV38    -0.006532
# LV881   -0.004783
# LV116   -0.004641
# LV163   -0.002454
# LV74    -0.002268
# LV1     -0.002113
# LV509   -0.001988
# LV824   -0.001976
# LV840   -0.001944
# LV189   -0.001757
# LV604   -0.001690
# LV854   -0.001651
# LV583   -0.001562
# LV170   -0.001536
# LV608   -0.001452
# LV926   -0.001390
# LV420   -0.001304
# LV272   -0.001180
# LV60    -0.001168
# LV905   -0.001164
# LV667   -0.001161
# LV395   -0.001144
# LV205   -0.001127
# LV59    -0.001080
# LV187   -0.001064
# LV21    -0.001057
# LV136   -0.001048
# LV924   -0.001008
# ```

# %% [markdown]
# ```
# LV881    0.154953
# LV116    0.097558
# LV743    0.097299
# LV38     0.088232
# LV841    0.086833
# LV163    0.080580
# LV161    0.070249
# LV931    0.069832
# LV9      0.069249
# LV536    0.068436
# LV50     0.067657
# LV707    0.067467
# LV142    0.064834
# LV95     0.061820
# LV97     0.061123
# LV74     0.056454
# LV66     0.055342
# LV974    0.053342
# LV678    0.051296
# LV879    0.050419
# Name: DB00627, dtype: float64
# ```

# %%
_tmp = multiplier_z["LV246"].rename("lv")
_tmp.index.rename("gene", inplace=True)
_tmp.sort_values(ascending=False).to_csv("/tmp/gene_set.txt")

# %%

# %%
_pheno_name = "coronary artery disease"
# _pheno_name = "hypercholesterolemia"

_tmp = smultixcan_zscores[_pheno_name].rename("pheno")
_tmp.index.rename("gene", inplace=True)
_tmp.sort_values(ascending=False).to_csv("/tmp/gene_zscores_pheno.txt")

# %%

# %%
_tmp = lincs_data.loc["DB00627"].rename("pheno")
_tmp.index.rename("gene", inplace=True)
_tmp.sort_values(ascending=False).to_csv("/tmp/gene_zscores_drug.txt")

# %%

# %%

# %%
from multiplier import MultiplierProjection

# %%
mproj = MultiplierProjection()

# %%
_tmp = pd.DataFrame(
    {
        "trait1": [10, 20, 30, 40, 50],
        "trait2": [-10, -20, -30, -40, -50],
        "trait3": [-10, 20, -30, 40, -50],
    },
    index=["SCD", "ACSS2", "GPAM", "DGAT2", "ACLY"],
)

# %%
_tmp

# %%
_tmp_proj = mproj.transform(_tmp)

# %%
_tmp_proj.loc["LV246"]

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
    display(_tmp.head(100))

# %% [markdown]
# ## coronary artery disease

# %%
find_differences("coronary artery disease")

# %% [markdown]
# ## atherosclerosis

# %%
find_differences("atherosclerosis")

# %% [markdown]
# # Niacin and vascular diseases

# %%
from entity import Trait

# %%
d = Trait.get_trait(
    full_code="I25-Diagnoses_main_ICD10_I25_Chronic_ischaemic_heart_disease"
)

# %%
d.get_efo_info()

# %%
d.get_do_info()

# %%
dd = Trait.get_trait(full_code="CARDIoGRAM_C4D_CAD_ADDITIVE")

# %%
dd.get_do_info()

# %%

# %%
_common_genes = multiplier_z.index.intersection(smultixcan_zscores.index)
display(_common_genes.shape)

# %%
z_genes = multiplier_z["LV246"].sort_values(ascending=False)
display(z_genes.head(10))

# %%
gene_zscores = smultixcan_zscores[
    "I70-Diagnoses_main_ICD10_I70_Atherosclerosis"
].sort_values(ascending=False)
display(gene_zscores.head(10))

z_genes.loc[_common_genes].dot(gene_zscores.loc[_common_genes])

# %%
display(gene_zscores.name)
z_genes.loc[_common_genes].dot(gene_zscores.loc[_common_genes])

# %%
display(gene_zscores.name)
z_genes.loc[_common_genes].dot(gene_zscores.loc[_common_genes])

# %%
lincs_data.loc["DB00627", "LDLR"]

# %%
lincs_data.loc["DB00627", "LPL"]

# %%

# %%

# %%
smultixcan_proj.loc["LV246", "coronary artery disease"]

# %%
smultixcan_proj["coronary artery disease"].describe()

# %%
smultixcan_proj["coronary artery disease"].sort_values(ascending=False).head(40)

# %%
smultixcan_proj.loc["LV246", "atherosclerosis"]

# %%
smultixcan_proj["atherosclerosis"].describe()

# %%
smultixcan_proj["atherosclerosis"].sort_values(ascending=False).head(40)

# %%

# %%

# %%
_doid = "DOID:1936"
_phenomexcan_traits = ["I70-Diagnoses_main_ICD10_I70_Atherosclerosis"]

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
with pd.option_context(
    "display.max_rows", None, "display.max_columns", None, "max_colwidth", None
):
    _tmp = (
        predictions_by_tissue[
            (predictions_by_tissue["trait"] == _doid)
            & (predictions_by_tissue["drug"] == _drug_id)
        ]
        .groupby("method")["score"]
        .idxmax()
    )
    display(predictions_by_tissue.loc[_tmp.values])

    _tmp = predictions_by_tissue.loc[_tmp.values][["method", "tissue"]].set_index(
        "method"
    )
    _gene_based_tissue = _tmp.loc["Gene-based", "tissue"]
    display(_gene_based_tissue)

    _module_tissue = _tmp.loc["Module-based", "tissue"]
    display(_module_tissue)

# %%
# predictions_by_tissue[
#     (predictions_by_tissue["trait"] == _doid)
#     & (predictions_by_tissue["drug"] == _drug_id)
# ].sort_values("score", ascending=False)

# %%
# _module_tissue = "Adipose_Subcutaneous"

# %% [markdown]
# ### Gene module-based - LVs driving association

# %%
_module_tissue_data_raw = pd.read_pickle(
    INPUT_DATA_DIR / "raw" / f"spredixcan-mashr-zscores-{_module_tissue}-data.pkl"
)
_module_tissue_data_raw = _module_tissue_data_raw.rename(index=Gene.GENE_ID_TO_NAME_MAP)

display(_module_tissue_data_raw.shape)
display(_module_tissue_data_raw.head())

# %%
_module_tissue_data_raw.index.is_unique

# %% tags=[]
_module_tissue_data_raw.index[_module_tissue_data_raw.index.duplicated(keep="first")]

# %% tags=[]
_module_tissue_data_raw = _module_tissue_data_raw.loc[
    ~_module_tissue_data_raw.index.duplicated(keep="first")
]

# %% tags=[]
_module_tissue_data_raw.shape

# %%
_module_tissue_data_raw.index.is_unique

# %%

# %%
_module_tissue_data = pd.read_pickle(
    INPUT_DATA_DIR
    / "proj"
    / f"spredixcan-mashr-zscores-{_module_tissue}-projection.pkl"
)
display(_module_tissue_data.head())

# %%

# %%
_all_dfs = []

for ntc in (None, 5, 10, 25, 50):
    _prefix = "all_genes"
    if ntc is not None:
        _prefix = f"top_{ntc}_genes"

    _df = pd.read_hdf(
        OUTPUT_PREDICTIONS_DIR
        / f"spredixcan-mashr-zscores-{_module_tissue}-projection-{_prefix}-prediction_scores.h5",
        key="prediction",
    )

    _score_col_name = f"score_{_prefix}"
    _df = (
        pd.merge(_df, gold_standard, on=["trait", "drug"], how="inner")
        .drop(columns=["true_class"])
        .set_index(["trait", "drug"])
        .rename(columns={"score": _score_col_name})
    )

    _df[_score_col_name] = _df[_score_col_name].rank()

    _all_dfs.append(_df)


# %%
_predictions_tissue = pd.concat(_all_dfs, axis=1, join="inner")  # .mean(axis=1)

# %%
_predictions_tissue.shape

# %%
_predictions_tissue.head()

# %%
_predictions_tissue = _predictions_tissue.mean(axis=1)

# %%
display(_predictions_tissue.shape)
display(_predictions_tissue.head())

assert _df.shape[0] == _predictions_tissue.shape[0]
assert not _predictions_tissue.isna().any().any()

# %%
_predictions_tissue.loc[
    _doid, _drug_id
]  # .query(f"(trait == '{_doid}') and (drug == '{_drug_id}')")

# %%
# FIXME: add assert here to make sure this number above is the same in predictions_avg

# %%

# %%
# _traits_proj = pd.read_pickle(INPUT_DATA_DIR / "proj" / f"spredixcan-mashr-zscores-{_module_tissue}-projection.pkl")

_module_tissue_data = _module_tissue_data[
    _phenomexcan_traits[0]
    #     "C62-Diagnoses_main_ICD10_C62_Malignant_neoplasm_of_testis",
    #     "I25-Diagnoses_main_ICD10_I25_Chronic_ischaemic_heart_disease",
].to_frame()

# _traits_proj = _traits_proj.apply(
#     lambda x: _zero_nontop_genes(x, 50, use_abs=True)
# )

# %%
_module_tissue_data.head()

# %%
_module_tissue_data.squeeze().sort_values()

# %%
lincs_projection.T[_drug_id].sort_values()

# %%
_tmp = (-1.0 * lincs_projection.loc[_drug_id].dot(_module_tissue_data)).sort_values(
    ascending=False
)
display(_tmp)

_top_phenomexcan_trait = _tmp.index[0]
display(_top_phenomexcan_trait)

# %%
_tmp = (
    lincs_projection.loc[_drug_id]
    .multiply(_module_tissue_data[_top_phenomexcan_trait])
    .sort_values(ascending=True)
)
display(_tmp[_tmp < 0.0].head(60))

# %%
# I should try all LVs associated with lipids
_tmp.loc["LV246"]

# %%
",".join(_tmp.head(20).index)

# %%
_tmp2 = _tmp[_tmp < 0.0].sort_values(ascending=True)
display(_tmp2)
_tmp2.to_pickle("/tmp/lv_list.pkl")

# %%
_lv_name = "LV246"

# %%
_lv_genes = multiplier_z[_lv_name].sort_values(ascending=False).head(40)
display(_lv_genes)

# %%
_module_tissue_data.loc[_lv_name, _top_phenomexcan_trait]

# %%
lincs_projection.loc[_drug_id, _lv_name]  # .sort_values()

# %%
# FIXME: here, if I want to see per gene, I should multiply by the gene weight in the LV
_genes_assoc_data = _module_tissue_data_raw.reindex(_lv_genes.index)[
    _top_phenomexcan_trait
]

display(_genes_assoc_data)

# %%
_lincs_assoc_data = lincs_data.T.reindex(_lv_genes.index)[_drug_id]

display(_lincs_assoc_data)

# %%
_genes_assoc_data.multiply(_lincs_assoc_data).sort_values(ascending=True)

# %%
# lincs_projection.loc[_drug_id].sort_values()

# %%
# _module_tissue_data.squeeze().sort_values().head(10)

# %%
_tmp = lincs_projection.T[_drug_id].abs()

_tmp2 = _tmp[_tmp > 0.0].sort_values(ascending=False)
display(_tmp2)
_tmp2.to_pickle("/tmp/niacin_lv_list.pkl")

# %%
_tmp2.head(20)

# %%
",".join(_tmp2.head(20).index)

# %%

# %%

# %% [markdown]
# ### Gene-based - Genes driving association

# %%
_gene_based_tissue_data = pd.read_pickle(
    INPUT_DATA_DIR / "raw" / f"spredixcan-mashr-zscores-{_gene_based_tissue}-data.pkl"
)
_gene_based_tissue_data = _gene_based_tissue_data.rename(index=Gene.GENE_ID_TO_NAME_MAP)

display(_gene_based_tissue_data.shape)
display(_gene_based_tissue_data.head())

# %%
_gene_based_tissue_data.index.is_unique

# %% tags=[]
_gene_based_tissue_data.index[_gene_based_tissue_data.index.duplicated(keep="first")]

# %% tags=[]
_gene_based_tissue_data = _gene_based_tissue_data.loc[
    ~_gene_based_tissue_data.index.duplicated(keep="first")
]

# %% tags=[]
_gene_based_tissue_data.shape

# %%
_gene_based_tissue_data.index.is_unique

# %%

# %%
# from drug_disease import _zero_nontop_genes

# %%
OUTPUT_PREDICTIONS_DIR

# %%
# _prefix = "top_5_genes"

_all_dfs = []

for ntc in (None, 50, 100, 250, 500):
    _prefix = "all_genes"
    if ntc is not None:
        _prefix = f"top_{ntc}_genes"

    _df = pd.read_hdf(
        OUTPUT_PREDICTIONS_DIR
        / f"spredixcan-mashr-zscores-{_gene_based_tissue}-data-{_prefix}-prediction_scores.h5",
        key="prediction",
    )

    _score_col_name = f"score_{_prefix}"
    _df = (
        pd.merge(_df, gold_standard, on=["trait", "drug"], how="inner")
        .drop(columns=["true_class"])
        .set_index(["trait", "drug"])
        .rename(columns={"score": _score_col_name})
    )

    _df[_score_col_name] = _df[_score_col_name].rank()

    _all_dfs.append(_df)

# %%
_df

# %%
_raw_predictions_tissue = pd.concat(_all_dfs, axis=1, join="inner")  # .mean(axis=1)

# %%
_raw_predictions_tissue.shape

# %%
_raw_predictions_tissue.head()

# %%
_raw_predictions_tissue = _raw_predictions_tissue.mean(axis=1)

# %%
display(_raw_predictions_tissue.shape)
display(_raw_predictions_tissue.head())

assert _df.shape[0] == _raw_predictions_tissue.shape[0]
assert not _raw_predictions_tissue.isna().any().any()

# %%
_raw_predictions_tissue.loc[
    _doid, _drug_id
]  # .query(f"(trait == '{_doid}') and (drug == '{_drug_id}')")

# %%
# FIXME: add assert here to make sure this number above is the same in predictions_avg

# %%

# %%

# %%

# %%
# _traits_proj = pd.read_pickle(INPUT_DATA_DIR / "proj" / f"spredixcan-mashr-zscores-{_module_tissue}-projection.pkl")

_gene_based_tissue_data = _gene_based_tissue_data[
    _phenomexcan_traits
    #     "C62-Diagnoses_main_ICD10_C62_Malignant_neoplasm_of_testis",
    #     "I25-Diagnoses_main_ICD10_I25_Chronic_ischaemic_heart_disease",
]

# _traits_proj = _traits_proj.apply(
#     lambda x: _zero_nontop_genes(x, 50, use_abs=True)
# )

# %%
_gene_based_tissue_data.head()

# %%
_gene_based_tissue_data.squeeze().sort_values()

# %%
lincs_data.T[_drug_id].sort_values()

# %%
# get common genes with lincs
common_genes = _gene_based_tissue_data.index.intersection(lincs_data.columns)
_gene_data = _gene_based_tissue_data.loc[common_genes]
_lincs_data = lincs_data[common_genes]

# %%
_gene_data.shape

# %%
_lincs_data.shape

# %%
_tmp = (-1.0 * _lincs_data.loc[_drug_id].dot(_gene_data)).sort_values(ascending=False)
display(_tmp)

_top_phenomexcan_trait = _tmp.index[0]
display(_top_phenomexcan_trait)

# %%
_tmp = (
    _lincs_data.loc[_drug_id]
    .multiply(_gene_data[_top_phenomexcan_trait])
    .sort_values(ascending=True)
)
display(_tmp.head(20))

# %%
",".join(_tmp.head(20).index)

# %%
_tmp[_tmp < 0.0].shape

# %%
_lincs_data.loc[_drug_id].sort_values()

# %%
_lincs_data.loc[_drug_id, "NT5DC2"]  # .sort_values()

# %%
_gene_data.squeeze().sort_values()

# %%
_gene_data.squeeze().loc["NT5DC2"]

# %%

# %%

# %%

# %%

# %% [markdown]
# ## hypertension

# %%
find_differences("hypertension")

# %% [markdown]
# ## allergic rhinitis

# %%
find_differences("allergic rhinitis")

# %%

# %%

# %%

# %%

# %% [markdown]
# # Alzheimer's disease

# %%
with pd.option_context(
    "display.max_rows", None, "display.max_columns", None, "max_colwidth", None
):
    _tmp = (
        pharmadb_predictions[
            pharmadb_predictions["disease"].isin(["allergic rhinitis"])
        ]
        #         .query("method == 'Module-based'")
        .sort_values(["score_zscore"], ascending=[False])
    )
    display(_tmp)

# %%
