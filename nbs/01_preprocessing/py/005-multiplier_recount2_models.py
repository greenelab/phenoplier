# ---
# jupyter:
#   jupytext:
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

# %% [markdown] papermill={"duration": 0.035202, "end_time": "2020-12-11T19:38:57.032018", "exception": false, "start_time": "2020-12-11T19:38:56.996816", "status": "completed"} tags=[]
# # Description

# %% [markdown] papermill={"duration": 0.033614, "end_time": "2020-12-11T19:38:57.094708", "exception": false, "start_time": "2020-12-11T19:38:57.061094", "status": "completed"} tags=[]
# This notebook reads all matrices from the MultiPLIER model (https://github.com/greenelab/multi-plier) trained in recount2, like gene loadings (Z) or the
# latent space (B), and saves them into a Python friendly format (Pandas DataFrames in pickle format for faster/easier loading, and tsv for universal access).

# %% [markdown] papermill={"duration": 0.034075, "end_time": "2020-12-11T19:38:57.163995", "exception": false, "start_time": "2020-12-11T19:38:57.129920", "status": "completed"} tags=[]
# # Modules loading

# %% papermill={"duration": 0.047071, "end_time": "2020-12-11T19:38:57.245671", "exception": false, "start_time": "2020-12-11T19:38:57.198600", "status": "completed"} tags=[]
# %load_ext autoreload
# %autoreload 2

# %% papermill={"duration": 0.440539, "end_time": "2020-12-11T19:38:57.718601", "exception": false, "start_time": "2020-12-11T19:38:57.278062", "status": "completed"} tags=[]
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

import conf

# %% papermill={"duration": 0.04964, "end_time": "2020-12-11T19:38:57.801819", "exception": false, "start_time": "2020-12-11T19:38:57.752179", "status": "completed"} tags=[]
readRDS = ro.r["readRDS"]

# %% [markdown] papermill={"duration": 0.036232, "end_time": "2020-12-11T19:38:57.872759", "exception": false, "start_time": "2020-12-11T19:38:57.836527", "status": "completed"} tags=[]
# # Read MultiPLIER model

# %% papermill={"duration": 0.052637, "end_time": "2020-12-11T19:38:57.961238", "exception": false, "start_time": "2020-12-11T19:38:57.908601", "status": "completed"} tags=[]
conf.MULTIPLIER["RECOUNT2_MODEL_FILE"]

# %% papermill={"duration": 16.382883, "end_time": "2020-12-11T19:39:14.377420", "exception": false, "start_time": "2020-12-11T19:38:57.994537", "status": "completed"} tags=[]
multiplier_full_model = readRDS(str(conf.MULTIPLIER["RECOUNT2_MODEL_FILE"]))

# %% [markdown] papermill={"duration": 0.03215, "end_time": "2020-12-11T19:39:14.442532", "exception": false, "start_time": "2020-12-11T19:39:14.410382", "status": "completed"} tags=[]
# # Summary matrix

# %% papermill={"duration": 0.04465, "end_time": "2020-12-11T19:39:14.519475", "exception": false, "start_time": "2020-12-11T19:39:14.474825", "status": "completed"} tags=[]
multiplier_model_matrix = multiplier_full_model.rx2("summary")

# %% papermill={"duration": 0.049884, "end_time": "2020-12-11T19:39:14.604830", "exception": false, "start_time": "2020-12-11T19:39:14.554946", "status": "completed"} tags=[]
multiplier_model_matrix

# %% papermill={"duration": 0.049572, "end_time": "2020-12-11T19:39:14.688033", "exception": false, "start_time": "2020-12-11T19:39:14.638461", "status": "completed"} tags=[]
multiplier_model_matrix.rownames

# %% papermill={"duration": 0.049, "end_time": "2020-12-11T19:39:14.772568", "exception": false, "start_time": "2020-12-11T19:39:14.723568", "status": "completed"} tags=[]
multiplier_model_matrix.colnames

# %% papermill={"duration": 0.098289, "end_time": "2020-12-11T19:39:14.903824", "exception": false, "start_time": "2020-12-11T19:39:14.805535", "status": "completed"} tags=[]
with localconverter(ro.default_converter + pandas2ri.converter):
    multiplier_model_matrix_values = ro.conversion.rpy2py(multiplier_model_matrix)

# %% papermill={"duration": 0.04842, "end_time": "2020-12-11T19:39:14.985572", "exception": false, "start_time": "2020-12-11T19:39:14.937152", "status": "completed"} tags=[]
multiplier_model_matrix_df = pd.DataFrame(
    data=multiplier_model_matrix_values,
    #     index=multiplier_model_matrix.rownames,
    columns=multiplier_model_matrix.colnames,
)

# %% papermill={"duration": 0.044433, "end_time": "2020-12-11T19:39:15.063265", "exception": false, "start_time": "2020-12-11T19:39:15.018832", "status": "completed"} tags=[]
display(multiplier_model_matrix_df.shape)
assert multiplier_model_matrix_df.shape == (2157, 5)

# %% papermill={"duration": 0.052815, "end_time": "2020-12-11T19:39:15.149429", "exception": false, "start_time": "2020-12-11T19:39:15.096614", "status": "completed"} tags=[]
multiplier_model_matrix_df.head()

# %% papermill={"duration": 0.054884, "end_time": "2020-12-11T19:39:15.241390", "exception": false, "start_time": "2020-12-11T19:39:15.186506", "status": "completed"} tags=[]
to_scinum = lambda x: np.format_float_scientific(x, 6)

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

# %% [markdown] papermill={"duration": 0.038565, "end_time": "2020-12-11T19:39:15.318329", "exception": false, "start_time": "2020-12-11T19:39:15.279764", "status": "completed"} tags=[]
# ## Save

# %% [markdown] papermill={"duration": 0.033841, "end_time": "2020-12-11T19:39:15.387129", "exception": false, "start_time": "2020-12-11T19:39:15.353288", "status": "completed"} tags=[]
# ### Pickle format

# %% papermill={"duration": 0.045747, "end_time": "2020-12-11T19:39:15.466572", "exception": false, "start_time": "2020-12-11T19:39:15.420825", "status": "completed"} tags=[]
output_file = conf.MULTIPLIER["MODEL_SUMMARY_FILE"]
display(output_file)

# %% papermill={"duration": 0.048945, "end_time": "2020-12-11T19:39:15.551161", "exception": false, "start_time": "2020-12-11T19:39:15.502216", "status": "completed"} tags=[]
multiplier_model_matrix_df.to_pickle(output_file)

# %% [markdown] papermill={"duration": 0.037667, "end_time": "2020-12-11T19:39:15.624858", "exception": false, "start_time": "2020-12-11T19:39:15.587191", "status": "completed"} tags=[]
# ### Text format

# %% papermill={"duration": 0.049166, "end_time": "2020-12-11T19:39:15.711232", "exception": false, "start_time": "2020-12-11T19:39:15.662066", "status": "completed"} tags=[]
# tsv format
output_text_file = output_file.with_suffix(".tsv.gz")
display(output_text_file)

# %% papermill={"duration": 0.080274, "end_time": "2020-12-11T19:39:15.830199", "exception": false, "start_time": "2020-12-11T19:39:15.749925", "status": "completed"} tags=[]
multiplier_model_matrix_df.to_csv(
    output_text_file, sep="\t", index=False, float_format="%.5e"
)

# %% [markdown] papermill={"duration": 0.032908, "end_time": "2020-12-11T19:39:15.900357", "exception": false, "start_time": "2020-12-11T19:39:15.867449", "status": "completed"} tags=[]
# # Matrix Z (loadings; genes x LVs)

# %% papermill={"duration": 0.045421, "end_time": "2020-12-11T19:39:15.981091", "exception": false, "start_time": "2020-12-11T19:39:15.935670", "status": "completed"} tags=[]
multiplier_model_matrix = multiplier_full_model.rx2("Z")

# %% papermill={"duration": 0.048899, "end_time": "2020-12-11T19:39:16.065493", "exception": false, "start_time": "2020-12-11T19:39:16.016594", "status": "completed"} tags=[]
multiplier_model_matrix

# %% papermill={"duration": 0.047816, "end_time": "2020-12-11T19:39:16.148638", "exception": false, "start_time": "2020-12-11T19:39:16.100822", "status": "completed"} tags=[]
multiplier_model_matrix.rownames

# %% papermill={"duration": 0.050111, "end_time": "2020-12-11T19:39:16.237498", "exception": false, "start_time": "2020-12-11T19:39:16.187387", "status": "completed"} tags=[]
multiplier_model_matrix.colnames

# %% papermill={"duration": 0.062776, "end_time": "2020-12-11T19:39:16.337311", "exception": false, "start_time": "2020-12-11T19:39:16.274535", "status": "completed"} tags=[]
with localconverter(ro.default_converter + pandas2ri.converter):
    multiplier_model_matrix_values = ro.conversion.rpy2py(multiplier_model_matrix)

# %% papermill={"duration": 0.071621, "end_time": "2020-12-11T19:39:16.445935", "exception": false, "start_time": "2020-12-11T19:39:16.374314", "status": "completed"} tags=[]
multiplier_model_matrix_df = pd.DataFrame(
    data=multiplier_model_matrix_values,
    index=multiplier_model_matrix.rownames,
    columns=[f"LV{i}" for i in range(1, multiplier_model_matrix.ncol + 1)],
)

# %% papermill={"duration": 0.049779, "end_time": "2020-12-11T19:39:16.536144", "exception": false, "start_time": "2020-12-11T19:39:16.486365", "status": "completed"} tags=[]
display(multiplier_model_matrix_df.shape)
assert multiplier_model_matrix_df.shape == (6750, 987)

# %% papermill={"duration": 0.065021, "end_time": "2020-12-11T19:39:16.638433", "exception": false, "start_time": "2020-12-11T19:39:16.573412", "status": "completed"} tags=[]
multiplier_model_matrix_df.head()

# %% papermill={"duration": 0.048534, "end_time": "2020-12-11T19:39:16.728965", "exception": false, "start_time": "2020-12-11T19:39:16.680431", "status": "completed"} tags=[]
# make sure I'm seeing the same when loaded with R
assert multiplier_model_matrix_df.loc["GAS6", "LV2"] == 0
assert multiplier_model_matrix_df.loc["GAS6", "LV3"] == 0.039437739697954444
assert multiplier_model_matrix_df.loc["GAS6", "LV984"] == 0.3473620915326928
assert multiplier_model_matrix_df.loc["GAS6", "LV987"] == 0

assert multiplier_model_matrix_df.loc["SPARC", "LV981"] == 0
assert multiplier_model_matrix_df.loc["SPARC", "LV986"].round(8) == 0.12241734

# %% [markdown] papermill={"duration": 0.040581, "end_time": "2020-12-11T19:39:16.811043", "exception": false, "start_time": "2020-12-11T19:39:16.770462", "status": "completed"} tags=[]
# ## Save

# %% [markdown] papermill={"duration": 0.036342, "end_time": "2020-12-11T19:39:16.884763", "exception": false, "start_time": "2020-12-11T19:39:16.848421", "status": "completed"} tags=[]
# ### Pickle format

# %% papermill={"duration": 0.049306, "end_time": "2020-12-11T19:39:16.990168", "exception": false, "start_time": "2020-12-11T19:39:16.940862", "status": "completed"} tags=[]
output_file = conf.MULTIPLIER["MODEL_Z_MATRIX_FILE"]
display(output_file)

# %% papermill={"duration": 0.395266, "end_time": "2020-12-11T19:39:17.423883", "exception": false, "start_time": "2020-12-11T19:39:17.028617", "status": "completed"} tags=[]
multiplier_model_matrix_df.to_pickle(output_file)

# %% [markdown] papermill={"duration": 0.040015, "end_time": "2020-12-11T19:39:17.534793", "exception": false, "start_time": "2020-12-11T19:39:17.494778", "status": "completed"} tags=[]
# ### Text format

# %% papermill={"duration": 0.052925, "end_time": "2020-12-11T19:39:17.628391", "exception": false, "start_time": "2020-12-11T19:39:17.575466", "status": "completed"} tags=[]
# tsv format
output_text_file = output_file.with_suffix(".tsv.gz")
display(output_text_file)

# %% papermill={"duration": 23.032027, "end_time": "2020-12-11T19:39:40.699238", "exception": false, "start_time": "2020-12-11T19:39:17.667211", "status": "completed"} tags=[]
multiplier_model_matrix_df.to_csv(
    output_text_file, sep="\t", index=True, float_format="%.5e"
)

# %% [markdown] papermill={"duration": 0.039187, "end_time": "2020-12-11T19:39:40.776211", "exception": false, "start_time": "2020-12-11T19:39:40.737024", "status": "completed"} tags=[]
# # Matrix B (latent space; LVs x samples)

# %% papermill={"duration": 0.052229, "end_time": "2020-12-11T19:39:40.868641", "exception": false, "start_time": "2020-12-11T19:39:40.816412", "status": "completed"} tags=[]
multiplier_model_matrix = multiplier_full_model.rx2("B")

# %% papermill={"duration": 0.050927, "end_time": "2020-12-11T19:39:40.961508", "exception": false, "start_time": "2020-12-11T19:39:40.910581", "status": "completed"} tags=[]
multiplier_model_matrix

# %% papermill={"duration": 0.050953, "end_time": "2020-12-11T19:39:41.050675", "exception": false, "start_time": "2020-12-11T19:39:40.999722", "status": "completed"} tags=[]
multiplier_model_matrix.rownames

# %% papermill={"duration": 0.050586, "end_time": "2020-12-11T19:39:41.142068", "exception": false, "start_time": "2020-12-11T19:39:41.091482", "status": "completed"} tags=[]
multiplier_model_matrix.colnames

# %% papermill={"duration": 0.105374, "end_time": "2020-12-11T19:39:41.286844", "exception": false, "start_time": "2020-12-11T19:39:41.181470", "status": "completed"} tags=[]
with localconverter(ro.default_converter + pandas2ri.converter):
    multiplier_model_matrix_values = ro.conversion.rpy2py(multiplier_model_matrix)

# %% papermill={"duration": 0.152769, "end_time": "2020-12-11T19:39:41.481247", "exception": false, "start_time": "2020-12-11T19:39:41.328478", "status": "completed"} tags=[]
multiplier_model_matrix_df = pd.DataFrame(
    data=multiplier_model_matrix_values,
    #    Look like the rows have a special meaning, so no overriding it.
    #    index=[f'LV{i}' for i in range(1, multiplier_model_matrix.nrow + 1)],
    index=multiplier_model_matrix.rownames,
    columns=multiplier_model_matrix.colnames,
)

# %% papermill={"duration": 0.054712, "end_time": "2020-12-11T19:39:41.578015", "exception": false, "start_time": "2020-12-11T19:39:41.523303", "status": "completed"} tags=[]
display(multiplier_model_matrix_df.shape)
assert multiplier_model_matrix_df.shape == (987, 37032)

# %% papermill={"duration": 0.068808, "end_time": "2020-12-11T19:39:41.687376", "exception": false, "start_time": "2020-12-11T19:39:41.618568", "status": "completed"} tags=[]
multiplier_model_matrix_df.head()

# %% papermill={"duration": 0.056089, "end_time": "2020-12-11T19:39:41.784010", "exception": false, "start_time": "2020-12-11T19:39:41.727921", "status": "completed"} tags=[]
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

# %% [markdown] papermill={"duration": 0.042631, "end_time": "2020-12-11T19:39:41.870012", "exception": false, "start_time": "2020-12-11T19:39:41.827381", "status": "completed"} tags=[]
# ## Make sure no GTEx samples are included

# %% papermill={"duration": 0.061794, "end_time": "2020-12-11T19:39:41.971980", "exception": false, "start_time": "2020-12-11T19:39:41.910186", "status": "completed"} tags=[]
# Test search string first
_tmp = multiplier_model_matrix_df.columns.str.contains("SRP000599.", regex=False)
assert _tmp[0]
assert _tmp[1]
assert not _tmp[-1]

# %% papermill={"duration": 0.057108, "end_time": "2020-12-11T19:39:42.070428", "exception": false, "start_time": "2020-12-11T19:39:42.013320", "status": "completed"} tags=[]
GTEX_ACCESSION_CODE = "SRP012682"

# %% papermill={"duration": 0.05974, "end_time": "2020-12-11T19:39:42.171583", "exception": false, "start_time": "2020-12-11T19:39:42.111843", "status": "completed"} tags=[]
_tmp = multiplier_model_matrix_df.columns.str.contains(GTEX_ACCESSION_CODE, regex=False)
assert not _tmp.any()

# %% [markdown] papermill={"duration": 0.044868, "end_time": "2020-12-11T19:39:42.258789", "exception": false, "start_time": "2020-12-11T19:39:42.213921", "status": "completed"} tags=[]
# ## Save

# %% [markdown] papermill={"duration": 0.039846, "end_time": "2020-12-11T19:39:42.342131", "exception": false, "start_time": "2020-12-11T19:39:42.302285", "status": "completed"} tags=[]
# ### Pickle format

# %% papermill={"duration": 0.052194, "end_time": "2020-12-11T19:39:42.434666", "exception": false, "start_time": "2020-12-11T19:39:42.382472", "status": "completed"} tags=[]
output_file = conf.MULTIPLIER["MODEL_B_MATRIX_FILE"]
display(output_file)

# %% papermill={"duration": 1.759511, "end_time": "2020-12-11T19:39:44.235917", "exception": false, "start_time": "2020-12-11T19:39:42.476406", "status": "completed"} tags=[]
multiplier_model_matrix_df.to_pickle(output_file)

# %% [markdown] papermill={"duration": 0.042269, "end_time": "2020-12-11T19:39:44.320075", "exception": false, "start_time": "2020-12-11T19:39:44.277806", "status": "completed"} tags=[]
# ### Text format

# %% papermill={"duration": 0.054972, "end_time": "2020-12-11T19:39:44.420416", "exception": false, "start_time": "2020-12-11T19:39:44.365444", "status": "completed"} tags=[]
# tsv format
output_text_file = output_file.with_suffix(".tsv.gz")
display(output_text_file)

# %% papermill={"duration": 256.964658, "end_time": "2020-12-11T19:44:01.427507", "exception": false, "start_time": "2020-12-11T19:39:44.462849", "status": "completed"} tags=[]
multiplier_model_matrix_df.to_csv(
    output_text_file, sep="\t", index=True, float_format="%.5e"
)

# %% [markdown] papermill={"duration": 0.041108, "end_time": "2020-12-11T19:44:01.515007", "exception": false, "start_time": "2020-12-11T19:44:01.473899", "status": "completed"} tags=[]
# # Matrix U (gene sets x LVs)

# %% papermill={"duration": 0.07025, "end_time": "2020-12-11T19:44:01.626859", "exception": false, "start_time": "2020-12-11T19:44:01.556609", "status": "completed"} tags=[]
multiplier_model_matrix = multiplier_full_model.rx2("U")

# %% papermill={"duration": 0.054521, "end_time": "2020-12-11T19:44:01.724500", "exception": false, "start_time": "2020-12-11T19:44:01.669979", "status": "completed"} tags=[]
multiplier_model_matrix

# %% papermill={"duration": 0.055965, "end_time": "2020-12-11T19:44:01.822749", "exception": false, "start_time": "2020-12-11T19:44:01.766784", "status": "completed"} tags=[]
multiplier_model_matrix.rownames

# %% papermill={"duration": 0.055982, "end_time": "2020-12-11T19:44:01.921861", "exception": false, "start_time": "2020-12-11T19:44:01.865879", "status": "completed"} tags=[]
multiplier_model_matrix.colnames

# %% papermill={"duration": 0.057576, "end_time": "2020-12-11T19:44:02.023402", "exception": false, "start_time": "2020-12-11T19:44:01.965826", "status": "completed"} tags=[]
with localconverter(ro.default_converter + pandas2ri.converter):
    multiplier_model_matrix_values = ro.conversion.rpy2py(multiplier_model_matrix)

# %% papermill={"duration": 0.059161, "end_time": "2020-12-11T19:44:02.127735", "exception": false, "start_time": "2020-12-11T19:44:02.068574", "status": "completed"} tags=[]
multiplier_model_matrix_df = pd.DataFrame(
    data=multiplier_model_matrix_values,
    index=multiplier_model_matrix.rownames,
    columns=multiplier_model_matrix.colnames,
)

# %% papermill={"duration": 0.056394, "end_time": "2020-12-11T19:44:02.227248", "exception": false, "start_time": "2020-12-11T19:44:02.170854", "status": "completed"} tags=[]
display(multiplier_model_matrix_df.shape)
assert multiplier_model_matrix_df.shape == (628, 987)

# %% papermill={"duration": 0.072031, "end_time": "2020-12-11T19:44:02.343648", "exception": false, "start_time": "2020-12-11T19:44:02.271617", "status": "completed"} tags=[]
multiplier_model_matrix_df.head()

# %% papermill={"duration": 0.058652, "end_time": "2020-12-11T19:44:02.447173", "exception": false, "start_time": "2020-12-11T19:44:02.388521", "status": "completed"} tags=[]
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

# %% [markdown] papermill={"duration": 0.045336, "end_time": "2020-12-11T19:44:02.535247", "exception": false, "start_time": "2020-12-11T19:44:02.489911", "status": "completed"} tags=[]
# ## Save

# %% [markdown] papermill={"duration": 0.042505, "end_time": "2020-12-11T19:44:02.623123", "exception": false, "start_time": "2020-12-11T19:44:02.580618", "status": "completed"} tags=[]
# ### Pickle format

# %% papermill={"duration": 0.055742, "end_time": "2020-12-11T19:44:02.725677", "exception": false, "start_time": "2020-12-11T19:44:02.669935", "status": "completed"} tags=[]
output_file = conf.MULTIPLIER["MODEL_U_MATRIX_FILE"]
display(output_file)

# %% papermill={"duration": 0.060419, "end_time": "2020-12-11T19:44:02.830063", "exception": false, "start_time": "2020-12-11T19:44:02.769644", "status": "completed"} tags=[]
multiplier_model_matrix_df.to_pickle(output_file)

# %% [markdown] papermill={"duration": 0.044823, "end_time": "2020-12-11T19:44:02.917961", "exception": false, "start_time": "2020-12-11T19:44:02.873138", "status": "completed"} tags=[]
# ### Text format

# %% papermill={"duration": 0.05739, "end_time": "2020-12-11T19:44:03.018476", "exception": false, "start_time": "2020-12-11T19:44:02.961086", "status": "completed"} tags=[]
# tsv format
output_text_file = output_file.with_suffix(".tsv.gz")
display(output_text_file)

# %% papermill={"duration": 0.991982, "end_time": "2020-12-11T19:44:04.054046", "exception": false, "start_time": "2020-12-11T19:44:03.062064", "status": "completed"} tags=[]
multiplier_model_matrix_df.to_csv(
    output_text_file, sep="\t", index=True, float_format="%.5e"
)

# %% [markdown] papermill={"duration": 0.043725, "end_time": "2020-12-11T19:44:04.142180", "exception": false, "start_time": "2020-12-11T19:44:04.098455", "status": "completed"} tags=[]
# # Matrix U - AUC

# %% papermill={"duration": 0.057623, "end_time": "2020-12-11T19:44:04.245775", "exception": false, "start_time": "2020-12-11T19:44:04.188152", "status": "completed"} tags=[]
multiplier_model_matrix = multiplier_full_model.rx2("Uauc")

# %% papermill={"duration": 0.055288, "end_time": "2020-12-11T19:44:04.346342", "exception": false, "start_time": "2020-12-11T19:44:04.291054", "status": "completed"} tags=[]
multiplier_model_matrix

# %% papermill={"duration": 0.061386, "end_time": "2020-12-11T19:44:04.456681", "exception": false, "start_time": "2020-12-11T19:44:04.395295", "status": "completed"} tags=[]
multiplier_model_matrix.rownames

# %% papermill={"duration": 0.063247, "end_time": "2020-12-11T19:44:04.564569", "exception": false, "start_time": "2020-12-11T19:44:04.501322", "status": "completed"} tags=[]
multiplier_model_matrix.colnames

# %% papermill={"duration": 0.067938, "end_time": "2020-12-11T19:44:04.681014", "exception": false, "start_time": "2020-12-11T19:44:04.613076", "status": "completed"} tags=[]
with localconverter(ro.default_converter + pandas2ri.converter):
    multiplier_model_matrix_values = ro.conversion.rpy2py(multiplier_model_matrix)

# %% papermill={"duration": 0.063649, "end_time": "2020-12-11T19:44:04.791405", "exception": false, "start_time": "2020-12-11T19:44:04.727756", "status": "completed"} tags=[]
multiplier_model_matrix_df = pd.DataFrame(
    data=multiplier_model_matrix_values,
    index=multiplier_model_matrix.rownames,
    columns=multiplier_model_matrix.colnames,
)

# %% papermill={"duration": 0.059872, "end_time": "2020-12-11T19:44:04.896863", "exception": false, "start_time": "2020-12-11T19:44:04.836991", "status": "completed"} tags=[]
display(multiplier_model_matrix_df.shape)
assert multiplier_model_matrix_df.shape == (628, 987)

# %% papermill={"duration": 0.072738, "end_time": "2020-12-11T19:44:05.014535", "exception": false, "start_time": "2020-12-11T19:44:04.941797", "status": "completed"} tags=[]
multiplier_model_matrix_df.head()

# %% papermill={"duration": 0.062455, "end_time": "2020-12-11T19:44:05.125527", "exception": false, "start_time": "2020-12-11T19:44:05.063072", "status": "completed"} tags=[]
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

# %% [markdown] papermill={"duration": 0.048929, "end_time": "2020-12-11T19:44:05.223281", "exception": false, "start_time": "2020-12-11T19:44:05.174352", "status": "completed"} tags=[]
# ## Save

# %% [markdown] papermill={"duration": 0.048077, "end_time": "2020-12-11T19:44:05.320598", "exception": false, "start_time": "2020-12-11T19:44:05.272521", "status": "completed"} tags=[]
# ### Pickle format

# %% papermill={"duration": 0.059755, "end_time": "2020-12-11T19:44:05.427984", "exception": false, "start_time": "2020-12-11T19:44:05.368229", "status": "completed"} tags=[]
output_file = conf.MULTIPLIER["MODEL_U_AUC_MATRIX_FILE"]
display(output_file)

# %% papermill={"duration": 0.09867, "end_time": "2020-12-11T19:44:05.574910", "exception": false, "start_time": "2020-12-11T19:44:05.476240", "status": "completed"} tags=[]
multiplier_model_matrix_df.to_pickle(output_file)

# %% [markdown] papermill={"duration": 0.049531, "end_time": "2020-12-11T19:44:05.683625", "exception": false, "start_time": "2020-12-11T19:44:05.634094", "status": "completed"} tags=[]
# ### Text format

# %% papermill={"duration": 0.058633, "end_time": "2020-12-11T19:44:05.791317", "exception": false, "start_time": "2020-12-11T19:44:05.732684", "status": "completed"} tags=[]
# tsv format
output_text_file = output_file.with_suffix(".tsv.gz")
display(output_text_file)

# %% papermill={"duration": 0.986528, "end_time": "2020-12-11T19:44:06.833127", "exception": false, "start_time": "2020-12-11T19:44:05.846599", "status": "completed"} tags=[]
multiplier_model_matrix_df.to_csv(
    output_text_file, sep="\t", index=True, float_format="%.5e"
)

# %% [markdown] papermill={"duration": 0.047323, "end_time": "2020-12-11T19:44:06.926908", "exception": false, "start_time": "2020-12-11T19:44:06.879585", "status": "completed"} tags=[]
# # Model metadata

# %% papermill={"duration": 0.061014, "end_time": "2020-12-11T19:44:07.030653", "exception": false, "start_time": "2020-12-11T19:44:06.969639", "status": "completed"} tags=[]
model_names = list(multiplier_full_model.names)
display(model_names)

# %% papermill={"duration": 0.06223, "end_time": "2020-12-11T19:44:07.139355", "exception": false, "start_time": "2020-12-11T19:44:07.077125", "status": "completed"} tags=[]
with localconverter(ro.default_converter + pandas2ri.converter):
    model_metadata = {
        k: ro.conversion.rpy2py(multiplier_full_model.rx2(k))[0]
        for k in ("L1", "L2", "L3")
    }

# %% papermill={"duration": 0.060263, "end_time": "2020-12-11T19:44:07.249957", "exception": false, "start_time": "2020-12-11T19:44:07.189694", "status": "completed"} tags=[]
model_metadata

# %% papermill={"duration": 0.057641, "end_time": "2020-12-11T19:44:07.353936", "exception": false, "start_time": "2020-12-11T19:44:07.296295", "status": "completed"} tags=[]
assert len(model_metadata) == 3

# %% papermill={"duration": 0.062973, "end_time": "2020-12-11T19:44:07.466573", "exception": false, "start_time": "2020-12-11T19:44:07.403600", "status": "completed"} tags=[]
assert model_metadata["L2"] == 241.1321740143624

# %% [markdown] papermill={"duration": 0.047722, "end_time": "2020-12-11T19:44:07.560905", "exception": false, "start_time": "2020-12-11T19:44:07.513183", "status": "completed"} tags=[]
# ## Save

# %% [markdown] papermill={"duration": 0.052588, "end_time": "2020-12-11T19:44:07.665959", "exception": false, "start_time": "2020-12-11T19:44:07.613371", "status": "completed"} tags=[]
# ### Pickle format

# %% papermill={"duration": 0.061729, "end_time": "2020-12-11T19:44:07.777316", "exception": false, "start_time": "2020-12-11T19:44:07.715587", "status": "completed"} tags=[]
output_file = conf.MULTIPLIER["MODEL_METADATA_FILE"]
display(output_file)

# %% papermill={"duration": 0.075861, "end_time": "2020-12-11T19:44:07.903833", "exception": false, "start_time": "2020-12-11T19:44:07.827972", "status": "completed"} tags=[]
with open(output_file, "wb") as handle:
    pickle.dump(model_metadata, handle, protocol=pickle.HIGHEST_PROTOCOL)

# %% [markdown] papermill={"duration": 0.04649, "end_time": "2020-12-11T19:44:07.998581", "exception": false, "start_time": "2020-12-11T19:44:07.952091", "status": "completed"} tags=[]
# ### Text format

# %% papermill={"duration": 0.06331, "end_time": "2020-12-11T19:44:08.110655", "exception": false, "start_time": "2020-12-11T19:44:08.047345", "status": "completed"} tags=[]
multiplier_model_matrix_df = (
    pd.Series(model_metadata)
    .reset_index()
    .rename(columns={"index": "parameter", 0: "value"})
)

# %% papermill={"duration": 0.061877, "end_time": "2020-12-11T19:44:08.221207", "exception": false, "start_time": "2020-12-11T19:44:08.159330", "status": "completed"} tags=[]
multiplier_model_matrix_df

# %% papermill={"duration": 0.059519, "end_time": "2020-12-11T19:44:08.333470", "exception": false, "start_time": "2020-12-11T19:44:08.273951", "status": "completed"} tags=[]
# tsv format
output_text_file = output_file.with_suffix(".tsv.gz")
display(output_text_file)

# %% papermill={"duration": 0.062168, "end_time": "2020-12-11T19:44:08.443778", "exception": false, "start_time": "2020-12-11T19:44:08.381610", "status": "completed"} tags=[]
multiplier_model_matrix_df.to_csv(
    output_text_file,
    sep="\t",
    index=False,
    #     float_format="%.5e"
)

# %% papermill={"duration": 0.053112, "end_time": "2020-12-11T19:44:08.548750", "exception": false, "start_time": "2020-12-11T19:44:08.495638", "status": "completed"} tags=[]
