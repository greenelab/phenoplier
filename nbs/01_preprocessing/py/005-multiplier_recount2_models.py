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
# This notebook reads all matrices from the MultiPLIER model (https://github.com/greenelab/multi-plier) trained in recount2, like gene loadings (Z) or the
# latent space (B), and saves them into a Python friendly format (Pandas DataFrames in pickle format for faster/easier loading, and tsv for universal access).

# %% [markdown] tags=[]
# # Modules loading

# %% tags=[]
# %load_ext autoreload
# %autoreload 2

# %% tags=[]
import pickle

import numpy as np
import pandas as pd
from IPython.display import display

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

import conf

# %% tags=[]
readRDS = ro.r["readRDS"]

# %% tags=[]
saveRDS = ro.r["saveRDS"]

# %% [markdown] tags=[]
# # Read MultiPLIER model

# %% tags=[]
conf.MULTIPLIER["RECOUNT2_MODEL_FILE"]

# %% tags=[]
multiplier_full_model = readRDS(str(conf.MULTIPLIER["RECOUNT2_MODEL_FILE"]))

# %% [markdown] tags=[]
# # Summary matrix

# %% tags=[]
multiplier_model_matrix = multiplier_full_model.rx2("summary")

# %% tags=[]
multiplier_model_matrix

# %% tags=[]
multiplier_model_matrix.rownames

# %% tags=[]
multiplier_model_matrix.colnames

# %% tags=[]
with localconverter(ro.default_converter + pandas2ri.converter):
    multiplier_model_matrix_values = ro.conversion.rpy2py(multiplier_model_matrix)

# %% tags=[]
multiplier_model_matrix_df = pd.DataFrame(
    data=multiplier_model_matrix_values,
    #     index=multiplier_model_matrix.rownames,
    columns=multiplier_model_matrix.colnames,
)

# %% tags=[]
display(multiplier_model_matrix_df.shape)
assert multiplier_model_matrix_df.shape == (2157, 5)

# %% tags=[]
multiplier_model_matrix_df.head()

# %% tags=[]
def to_scinum(x):
    return np.format_float_scientific(x, 6)


# make sure I'm seeing the same when loaded with R
_tmp = multiplier_model_matrix_df.set_index(["pathway", "LV index"])

assert _tmp.loc[("KEGG_LYSINE_DEGRADATION", "1")]["AUC"].round(7) == 0.3880591
assert (
    to_scinum(_tmp.loc[("KEGG_LYSINE_DEGRADATION", "1")]["p-value"]) == "8.660782e-01"
)
assert _tmp.loc[("KEGG_LYSINE_DEGRADATION", "1")]["FDR"].round(10) == 0.9560054810

assert _tmp.loc[("IRIS_Monocyte-Day0", "2")]["AUC"].round(7) == 0.8900356
assert to_scinum(_tmp.loc[("IRIS_Monocyte-Day0", "2")]["p-value"]) == "4.315812e-25"
assert to_scinum(_tmp.loc[("IRIS_Monocyte-Day0", "2")]["FDR"]) == "1.329887e-22"

# %% [markdown] tags=[]
# ## Save

# %% [markdown] tags=[]
# ### Pickle format

# %% tags=[]
output_file = conf.MULTIPLIER["MODEL_SUMMARY_FILE"]
display(output_file)

# %% tags=[]
multiplier_model_matrix_df.to_pickle(output_file)

# %% [markdown] tags=[]
# ### RDS format

# %% tags=[]
output_rds_file = output_file.with_suffix(".rds")
display(output_rds_file)

# %% tags=[]
saveRDS(multiplier_model_matrix, str(output_rds_file))

# %% [markdown] tags=[]
# ### Text format

# %% tags=[]
# tsv format
output_text_file = output_file.with_suffix(".tsv.gz")
display(output_text_file)

# %% tags=[]
multiplier_model_matrix_df.to_csv(
    output_text_file, sep="\t", index=False, float_format="%.5e"
)

# %% [markdown] tags=[]
# # Matrix Z (loadings; genes x LVs)

# %% tags=[]
multiplier_model_matrix = multiplier_full_model.rx2("Z")

# %% tags=[]
multiplier_model_matrix

# %% tags=[]
multiplier_model_matrix.rownames

# %% tags=[]
multiplier_model_matrix.colnames

# %% tags=[]
with localconverter(ro.default_converter + pandas2ri.converter):
    multiplier_model_matrix_values = ro.conversion.rpy2py(multiplier_model_matrix)

# %% tags=[]
multiplier_model_matrix_df = pd.DataFrame(
    data=multiplier_model_matrix_values,
    index=multiplier_model_matrix.rownames,
    columns=[f"LV{i}" for i in range(1, multiplier_model_matrix.ncol + 1)],
)

# %% tags=[]
display(multiplier_model_matrix_df.shape)
assert multiplier_model_matrix_df.shape == (6750, 987)

# %% tags=[]
multiplier_model_matrix_df.head()

# %% tags=[]
# make sure I'm seeing the same when loaded with R
assert multiplier_model_matrix_df.loc["GAS6", "LV2"] == 0
assert multiplier_model_matrix_df.loc["GAS6", "LV3"] == 0.039437739697954444
assert multiplier_model_matrix_df.loc["GAS6", "LV984"] == 0.3473620915326928
assert multiplier_model_matrix_df.loc["GAS6", "LV987"] == 0

assert multiplier_model_matrix_df.loc["SPARC", "LV981"] == 0
assert multiplier_model_matrix_df.loc["SPARC", "LV986"].round(8) == 0.12241734

# %% [markdown] tags=[]
# ## Save

# %% [markdown] tags=[]
# ### Pickle format

# %% tags=[]
output_file = conf.MULTIPLIER["MODEL_Z_MATRIX_FILE"]
display(output_file)

# %% tags=[]
multiplier_model_matrix_df.to_pickle(output_file)

# %% [markdown] tags=[]
# ### RDS format

# %% tags=[]
output_rds_file = output_file.with_suffix(".rds")
display(output_rds_file)

# %% tags=[]
saveRDS(multiplier_model_matrix, str(output_rds_file))

# %% [markdown] tags=[]
# ### Text format

# %% tags=[]
# tsv format
output_text_file = output_file.with_suffix(".tsv.gz")
display(output_text_file)

# %% tags=[]
multiplier_model_matrix_df.to_csv(
    output_text_file, sep="\t", index=True, float_format="%.5e"
)

# %% [markdown] tags=[]
# # Matrix B (latent space; LVs x samples)

# %% tags=[]
multiplier_model_matrix = multiplier_full_model.rx2("B")

# %% tags=[]
multiplier_model_matrix

# %% tags=[]
multiplier_model_matrix.rownames

# %% tags=[]
multiplier_model_matrix.colnames

# %% tags=[]
with localconverter(ro.default_converter + pandas2ri.converter):
    multiplier_model_matrix_values = ro.conversion.rpy2py(multiplier_model_matrix)

# %% tags=[]
multiplier_model_matrix_df = pd.DataFrame(
    data=multiplier_model_matrix_values,
    #    Look like the rows have a special meaning, so no overriding it.
    #    index=[f'LV{i}' for i in range(1, multiplier_model_matrix.nrow + 1)],
    index=multiplier_model_matrix.rownames,
    columns=multiplier_model_matrix.colnames,
)

# %% tags=[]
display(multiplier_model_matrix_df.shape)
assert multiplier_model_matrix_df.shape == (987, 37032)

# %% tags=[]
multiplier_model_matrix_df.head()

# %% tags=[]
# make sure I'm seeing the same when loaded with R
assert (
    multiplier_model_matrix_df.loc[
        "1,REACTOME_MRNA_SPLICING", "SRP000599.SRR013549"
    ].round(9)
    == -0.059296689
)
assert (
    multiplier_model_matrix_df.loc[
        "1,REACTOME_MRNA_SPLICING", "SRP000599.SRR013553"
    ].round(9)
    == -0.036394186
)

assert (
    multiplier_model_matrix_df.loc["2,SVM Monocytes", "SRP000599.SRR013549"].round(9)
    == 0.006212678
)
assert (
    multiplier_model_matrix_df.loc["2,SVM Monocytes", "SRP004637.SRR073776"].round(9)
    == -0.008800153
)

assert (
    multiplier_model_matrix_df.loc["LV 9", "SRP004637.SRR073774"].round(9)
    == 0.092318955
)
assert (
    multiplier_model_matrix_df.loc["LV 9", "SRP004637.SRR073776"].round(9)
    == 0.100114294
)

# %% [markdown] tags=[]
# ## Make sure no GTEx samples are included

# %% tags=[]
# Test search string first
_tmp = multiplier_model_matrix_df.columns.str.contains("SRP000599.", regex=False)
assert _tmp[0]
assert _tmp[1]
assert not _tmp[-1]

# %% tags=[]
GTEX_ACCESSION_CODE = "SRP012682"

# %% tags=[]
_tmp = multiplier_model_matrix_df.columns.str.contains(GTEX_ACCESSION_CODE, regex=False)
assert not _tmp.any()

# %% [markdown] tags=[]
# ## Save

# %% [markdown] tags=[]
# ### Pickle format

# %% tags=[]
output_file = conf.MULTIPLIER["MODEL_B_MATRIX_FILE"]
display(output_file)

# %% tags=[]
multiplier_model_matrix_df.to_pickle(output_file)

# %% [markdown] tags=[]
# ### RDS format

# %% tags=[]
output_rds_file = output_file.with_suffix(".rds")
display(output_rds_file)

# %% tags=[]
saveRDS(multiplier_model_matrix, str(output_rds_file))

# %% [markdown] tags=[]
# ### Text format

# %% tags=[]
# tsv format
output_text_file = output_file.with_suffix(".tsv.gz")
display(output_text_file)

# %% tags=[]
multiplier_model_matrix_df.to_csv(
    output_text_file, sep="\t", index=True, float_format="%.5e"
)

# %% [markdown] tags=[]
# # Matrix U (gene sets x LVs)

# %% tags=[]
multiplier_model_matrix = multiplier_full_model.rx2("U")

# %% tags=[]
multiplier_model_matrix

# %% tags=[]
multiplier_model_matrix.rownames

# %% tags=[]
multiplier_model_matrix.colnames

# %% tags=[]
with localconverter(ro.default_converter + pandas2ri.converter):
    multiplier_model_matrix_values = ro.conversion.rpy2py(multiplier_model_matrix)

# %% tags=[]
multiplier_model_matrix_df = pd.DataFrame(
    data=multiplier_model_matrix_values,
    index=multiplier_model_matrix.rownames,
    columns=multiplier_model_matrix.colnames,
)

# %% tags=[]
display(multiplier_model_matrix_df.shape)
assert multiplier_model_matrix_df.shape == (628, 987)

# %% tags=[]
multiplier_model_matrix_df.head()

# %% tags=[]
# make sure I'm seeing the same when loaded with R
assert multiplier_model_matrix_df.loc["IRIS_Bcell-Memory_IgG_IgA", "LV1"] == 0
assert (
    multiplier_model_matrix_df.loc["IRIS_Bcell-Memory_IgG_IgA", "LV898"].round(7)
    == 0.5327689
)
assert (
    multiplier_model_matrix_df.loc["IRIS_Bcell-Memory_IgG_IgA", "LV977"].round(7)
    == 0.1000158
)
assert multiplier_model_matrix_df.loc["IRIS_Bcell-Memory_IgG_IgA", "LV986"] == 0
assert multiplier_model_matrix_df.loc["IRIS_Bcell-Memory_IgG_IgA", "LV987"] == 0

assert (
    multiplier_model_matrix_df.loc["IRIS_Bcell-naive", "LV851"].round(8) == 0.01330388
)
assert multiplier_model_matrix_df.loc["IRIS_Bcell-naive", "LV977"].round(7) == 0.3966446

# %% [markdown] tags=[]
# ## Save

# %% [markdown] tags=[]
# ### Pickle format

# %% tags=[]
output_file = conf.MULTIPLIER["MODEL_U_MATRIX_FILE"]
display(output_file)

# %% tags=[]
multiplier_model_matrix_df.to_pickle(output_file)

# %% [markdown] tags=[]
# ### RDS format

# %% tags=[]
output_rds_file = output_file.with_suffix(".rds")
display(output_rds_file)

# %% tags=[]
saveRDS(multiplier_model_matrix, str(output_rds_file))

# %% [markdown] tags=[]
# ### Text format

# %% tags=[]
# tsv format
output_text_file = output_file.with_suffix(".tsv.gz")
display(output_text_file)

# %% tags=[]
multiplier_model_matrix_df.to_csv(
    output_text_file, sep="\t", index=True, float_format="%.5e"
)

# %% [markdown] tags=[]
# # Matrix U - AUC

# %% tags=[]
multiplier_model_matrix = multiplier_full_model.rx2("Uauc")

# %% tags=[]
multiplier_model_matrix

# %% tags=[]
multiplier_model_matrix.rownames

# %% tags=[]
multiplier_model_matrix.colnames

# %% tags=[]
with localconverter(ro.default_converter + pandas2ri.converter):
    multiplier_model_matrix_values = ro.conversion.rpy2py(multiplier_model_matrix)

# %% tags=[]
multiplier_model_matrix_df = pd.DataFrame(
    data=multiplier_model_matrix_values,
    index=multiplier_model_matrix.rownames,
    columns=multiplier_model_matrix.colnames,
)

# %% tags=[]
display(multiplier_model_matrix_df.shape)
assert multiplier_model_matrix_df.shape == (628, 987)

# %% tags=[]
multiplier_model_matrix_df.head()

# %% tags=[]
# make sure I'm seeing the same when loaded with R
assert multiplier_model_matrix_df.loc["PID_FASPATHWAY", "LV136"] == 0
assert (
    multiplier_model_matrix_df.loc["PID_INTEGRIN1_PATHWAY", "LV136"].round(7)
    == 0.8832853
)
assert (
    multiplier_model_matrix_df.loc["REACTOME_COLLAGEN_FORMATION", "LV136"].round(7)
    == 0.8707412
)

assert multiplier_model_matrix_df.loc["PID_FASPATHWAY", "LV603"] == 0
assert (
    multiplier_model_matrix_df.loc["IRIS_Neutrophil-Resting", "LV603"].round(7)
    == 0.9057506
)
assert multiplier_model_matrix_df.loc["SVM Neutrophils", "LV603"].round(7) == 0.9797889

# %% [markdown] tags=[]
# ## Save

# %% [markdown] tags=[]
# ### Pickle format

# %% tags=[]
output_file = conf.MULTIPLIER["MODEL_U_AUC_MATRIX_FILE"]
display(output_file)

# %% tags=[]
multiplier_model_matrix_df.to_pickle(output_file)

# %% [markdown] tags=[]
# ### RDS format

# %% tags=[]
output_rds_file = output_file.with_suffix(".rds")
display(output_rds_file)

# %% tags=[]
saveRDS(multiplier_model_matrix, str(output_rds_file))

# %% [markdown] tags=[]
# ### Text format

# %% tags=[]
# tsv format
output_text_file = output_file.with_suffix(".tsv.gz")
display(output_text_file)

# %% tags=[]
multiplier_model_matrix_df.to_csv(
    output_text_file, sep="\t", index=True, float_format="%.5e"
)

# %% [markdown] tags=[]
# # Model metadata

# %% tags=[]
model_names = list(multiplier_full_model.names)
display(model_names)

# %% tags=[]
with localconverter(ro.default_converter + pandas2ri.converter):
    model_metadata = {
        k: ro.conversion.rpy2py(multiplier_full_model.rx2(k))[0]
        for k in ("L1", "L2", "L3")
    }

# %% tags=[]
model_metadata

# %% tags=[]
assert len(model_metadata) == 3

# %% tags=[]
assert model_metadata["L2"] == 241.1321740143624

# %% [markdown] tags=[]
# ## Save

# %% [markdown] tags=[]
# ### Pickle format

# %% tags=[]
output_file = conf.MULTIPLIER["MODEL_METADATA_FILE"]
display(output_file)

# %% tags=[]
with open(output_file, "wb") as handle:
    pickle.dump(model_metadata, handle, protocol=pickle.HIGHEST_PROTOCOL)

# %% [markdown] tags=[]
# ### RDS format

# %% tags=[]
output_rds_file = output_file.with_suffix(".rds")
display(output_rds_file)

# %% tags=[]
# convert numpy.float64 to standard float objects
rds_list = ro.ListVector({k: float(v) for k, v in model_metadata.items()})

# %% tags=[]
rds_list

# %% tags=[]
saveRDS(rds_list, str(output_rds_file))

# %% [markdown] tags=[]
# ### Text format

# %% tags=[]
multiplier_model_matrix_df = (
    pd.Series(model_metadata)
    .reset_index()
    .rename(columns={"index": "parameter", 0: "value"})
)

# %% tags=[]
multiplier_model_matrix_df

# %% tags=[]
# tsv format
output_text_file = output_file.with_suffix(".tsv.gz")
display(output_text_file)

# %% tags=[]
multiplier_model_matrix_df.to_csv(
    output_text_file,
    sep="\t",
    index=False,
)

# %% tags=[]
