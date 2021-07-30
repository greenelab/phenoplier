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
# This notebook reads 1) the normalized gene expression and 2) pathways from the data processed by
# MultiPLIER scripts (https://github.com/greenelab/multi-plier) and saves it into a more friendly Python
# format (Pandas DataFrames as pickle files).

# %% [markdown] tags=[]
# # Modules loading

# %% tags=[]
# %load_ext autoreload
# %autoreload 2

# %% tags=[]
from pathlib import Path

from IPython.display import display
import pandas as pd

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

import conf

# %% tags=[]
readRDS = ro.r["readRDS"]

# %% [markdown] tags=[]
# # Read entire recount data prep file

# %% tags=[]
conf.RECOUNT2["PREPROCESSED_GENE_EXPRESSION_FILE"]

# %% tags=[]
recount_data_prep = readRDS(str(conf.RECOUNT2["PREPROCESSED_GENE_EXPRESSION_FILE"]))

# %% [markdown] tags=[]
# # Read recount2 gene expression data

# %% tags=[]
recount2_rpkl_cm = recount_data_prep.rx2("rpkm.cm")

# %% tags=[]
recount2_rpkl_cm

# %% tags=[]
recount2_rpkl_cm.rownames

# %% tags=[]
recount2_rpkl_cm.colnames

# %% tags=[]
with localconverter(ro.default_converter + pandas2ri.converter):
    recount2_rpkl_cm = ro.conversion.rpy2py(recount2_rpkl_cm)

# %% tags=[]
assert recount2_rpkl_cm.shape == (6750, 37032)

# %% tags=[]
recount2_rpkl_cm.shape

# %% tags=[]
recount2_rpkl_cm.head()

# %% [markdown] tags=[]
# ## Testing

# %% [markdown] tags=[]
# Test whether what I load from a plain R session is the same as in here.

# %% tags=[]
recount2_rpkl_cm.loc["GAS6", "SRP000599.SRR013549"]

# %% tags=[]
assert recount2_rpkl_cm.loc["GAS6", "SRP000599.SRR013549"].round(4) == -0.3125

# %% tags=[]
assert recount2_rpkl_cm.loc["GAS6", "SRP045352.SRR1539229"].round(7) == -0.2843801

# %% tags=[]
assert recount2_rpkl_cm.loc["CFL2", "SRP056840.SRR1951636"].round(7) == -0.3412832

# %% tags=[]
recount2_rpkl_cm.iloc[9, 16]

# %% tags=[]
assert recount2_rpkl_cm.iloc[9, 16].round(7) == -0.4938852

# %% [markdown] tags=[]
# ## Save

# %% [markdown] tags=[]
# ### Pickle format (binary)

# %% tags=[]
output_filename = Path(
    conf.RECOUNT2["BASE_DIR"], "recount_data_prep_PLIER.pkl"
).resolve()

display(output_filename)

# %% tags=[]
recount2_rpkl_cm.to_pickle(output_filename)

# %% [markdown] tags=[]
# ### HDF5 format (binary)

# %% [markdown] tags=[]
# This code is now commented out, but it might be helpful in the future if we wanted to save this data in HDF5 for more efficient access.

# %% tags=[]
# from utils.hdf5 import simplify_string_for_hdf5

# %% tags=[]
# output_filename = os.path.join(conf.DATA_DIR, 'recount_data_prep_PLIER.h5')
# display(output_filename)

# %% tags=[]
# with pd.HDFStore(output_filename, mode='w', complevel=1) as store:
#     for idx, gene in enumerate(recount2_rpkl_cm.index):
#         if idx % 100:
#             print(f'', flush=True, end='')

#         clean_gene = simplify_string_for_hdf5(gene)
#         store[clean_gene] = recount2_rpkl_cm.loc[gene]

# %% tags=[]
# delete the object to save memory
del recount2_rpkl_cm

# %% [markdown] tags=[]
# # Read recount2 pathways

# %% tags=[]
recount2_all_paths_cm = recount_data_prep.rx2("all.paths.cm")

# %% tags=[]
recount2_all_paths_cm

# %% tags=[]
recount2_all_paths_cm.rownames

# %% tags=[]
recount2_all_paths_cm.colnames

# %% tags=[]
with localconverter(ro.default_converter + pandas2ri.converter):
    recount2_all_paths_cm_values = ro.conversion.rpy2py(recount2_all_paths_cm)

# %% tags=[]
recount2_all_paths_cm_values

# %% tags=[]
recount2_all_paths_cm = pd.DataFrame(
    data=recount2_all_paths_cm_values,
    index=recount2_all_paths_cm.rownames,
    columns=recount2_all_paths_cm.colnames,
    dtype=bool,
)

# %% tags=[]
assert recount2_all_paths_cm.shape == (6750, 628)

# %% tags=[]
recount2_all_paths_cm.shape

# %% tags=[]
recount2_all_paths_cm.dtypes.unique()

# %% tags=[]
recount2_all_paths_cm.head()

# %% [markdown] tags=[]
# ## Testing

# %% tags=[]
recount2_all_paths_cm.loc["CTSD", "REACTOME_SCFSKP2_MEDIATED_DEGRADATION_OF_P27_P21"]

# %% tags=[]
assert not recount2_all_paths_cm.loc[
    "CTSD", "REACTOME_SCFSKP2_MEDIATED_DEGRADATION_OF_P27_P21"
]

# %% tags=[]
assert recount2_all_paths_cm.loc["CTSD", "PID_P53DOWNSTREAMPATHWAY"]

# %% tags=[]
assert recount2_all_paths_cm.loc["MMP14", "PID_HIF2PATHWAY"]

# %% [markdown] tags=[]
# ## Save

# %% [markdown] tags=[]
# ### Pickle format

# %% tags=[]
output_filename = Path(conf.RECOUNT2["BASE_DIR"], "recount_all_paths_cm.pkl").resolve()
display(output_filename)

# %% tags=[]
recount2_all_paths_cm.to_pickle(output_filename)

# %% [markdown] tags=[]
# ### Text format

# %% tags=[]
# tsv format
output_text_file = output_filename.with_suffix(".tsv.gz")
display(output_text_file)

# %%
recount2_all_paths_cm.astype("int").head()

# %% tags=[]
recount2_all_paths_cm.astype("int").to_csv(
    output_text_file, sep="\t", index=True, float_format="%.5e"
)

# %% tags=[]
