# ---
# jupyter:
#   jupytext:
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

# %% [markdown] papermill={"duration": 0.022218, "end_time": "2020-12-11T19:38:19.360502", "exception": false, "start_time": "2020-12-11T19:38:19.338284", "status": "completed"} tags=[]
# # Description

# %% [markdown] papermill={"duration": 0.01972, "end_time": "2020-12-11T19:38:19.401864", "exception": false, "start_time": "2020-12-11T19:38:19.382144", "status": "completed"} tags=[]
# This notebook reads 1) the normalized gene expression and 2) pathways from the data processed by
# MultiPLIER scripts (https://github.com/greenelab/multi-plier) and saves it into a more friendly Python
# format (Pandas DataFrames as pickle files).

# %% [markdown] papermill={"duration": 0.021045, "end_time": "2020-12-11T19:38:19.444093", "exception": false, "start_time": "2020-12-11T19:38:19.423048", "status": "completed"} tags=[]
# # Modules loading

# %% papermill={"duration": 0.03402, "end_time": "2020-12-11T19:38:19.499293", "exception": false, "start_time": "2020-12-11T19:38:19.465273", "status": "completed"} tags=[]
# %load_ext autoreload
# %autoreload 2

# %% papermill={"duration": 0.425798, "end_time": "2020-12-11T19:38:19.946829", "exception": false, "start_time": "2020-12-11T19:38:19.521031", "status": "completed"} tags=[]
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

import conf

# %% papermill={"duration": 0.033184, "end_time": "2020-12-11T19:38:20.000236", "exception": false, "start_time": "2020-12-11T19:38:19.967052", "status": "completed"} tags=[]
readRDS = ro.r['readRDS']

# %% [markdown] papermill={"duration": 0.021169, "end_time": "2020-12-11T19:38:20.042593", "exception": false, "start_time": "2020-12-11T19:38:20.021424", "status": "completed"} tags=[]
# # Read entire recount data prep file

# %% papermill={"duration": 0.036725, "end_time": "2020-12-11T19:38:20.099487", "exception": false, "start_time": "2020-12-11T19:38:20.062762", "status": "completed"} tags=[]
conf.RECOUNT2['PREPROCESSED_GENE_EXPRESSION_FILE']

# %% papermill={"duration": 14.604403, "end_time": "2020-12-11T19:38:34.723232", "exception": false, "start_time": "2020-12-11T19:38:20.118829", "status": "completed"} tags=[]
recount_data_prep = readRDS(str(
    conf.RECOUNT2['PREPROCESSED_GENE_EXPRESSION_FILE']
))

# %% [markdown] papermill={"duration": 0.021429, "end_time": "2020-12-11T19:38:34.765353", "exception": false, "start_time": "2020-12-11T19:38:34.743924", "status": "completed"} tags=[]
# # Read recount2 gene expression data

# %% papermill={"duration": 0.031272, "end_time": "2020-12-11T19:38:34.817440", "exception": false, "start_time": "2020-12-11T19:38:34.786168", "status": "completed"} tags=[]
recount2_rpkl_cm = recount_data_prep.rx2('rpkm.cm')

# %% papermill={"duration": 0.037961, "end_time": "2020-12-11T19:38:34.875512", "exception": false, "start_time": "2020-12-11T19:38:34.837551", "status": "completed"} tags=[]
recount2_rpkl_cm

# %% papermill={"duration": 0.029763, "end_time": "2020-12-11T19:38:34.925395", "exception": false, "start_time": "2020-12-11T19:38:34.895632", "status": "completed"} tags=[]
recount2_rpkl_cm.rownames

# %% papermill={"duration": 0.032618, "end_time": "2020-12-11T19:38:34.977783", "exception": false, "start_time": "2020-12-11T19:38:34.945165", "status": "completed"} tags=[]
recount2_rpkl_cm.colnames

# %% papermill={"duration": 4.924062, "end_time": "2020-12-11T19:38:39.924121", "exception": false, "start_time": "2020-12-11T19:38:35.000059", "status": "completed"} tags=[]
with localconverter(ro.default_converter + pandas2ri.converter):
  recount2_rpkl_cm = ro.conversion.rpy2py(recount2_rpkl_cm)

# %% papermill={"duration": 0.029254, "end_time": "2020-12-11T19:38:39.976592", "exception": false, "start_time": "2020-12-11T19:38:39.947338", "status": "completed"} tags=[]
# recount2_rpkl_cm = pd.DataFrame(
#     data=pandas2ri.ri2py(recount2_rpkl_cm).values,
#     index=recount2_rpkl_cm.rownames,
#     columns=recount2_rpkl_cm.colnames,
# )

# %% papermill={"duration": 0.031644, "end_time": "2020-12-11T19:38:40.027865", "exception": false, "start_time": "2020-12-11T19:38:39.996221", "status": "completed"} tags=[]
assert recount2_rpkl_cm.shape == (6750, 37032)

# %% papermill={"duration": 0.032357, "end_time": "2020-12-11T19:38:40.082991", "exception": false, "start_time": "2020-12-11T19:38:40.050634", "status": "completed"} tags=[]
recount2_rpkl_cm.shape

# %% papermill={"duration": 0.053342, "end_time": "2020-12-11T19:38:40.158696", "exception": false, "start_time": "2020-12-11T19:38:40.105354", "status": "completed"} tags=[]
recount2_rpkl_cm.head()

# %% [markdown] papermill={"duration": 0.022322, "end_time": "2020-12-11T19:38:40.202899", "exception": false, "start_time": "2020-12-11T19:38:40.180577", "status": "completed"} tags=[]
# ## Testing

# %% [markdown] papermill={"duration": 0.022454, "end_time": "2020-12-11T19:38:40.248543", "exception": false, "start_time": "2020-12-11T19:38:40.226089", "status": "completed"} tags=[]
# Test whether what I load from a plain R session is the same as in here.

# %% papermill={"duration": 0.036367, "end_time": "2020-12-11T19:38:40.307711", "exception": false, "start_time": "2020-12-11T19:38:40.271344", "status": "completed"} tags=[]
recount2_rpkl_cm.loc['GAS6', 'SRP000599.SRR013549']

# %% papermill={"duration": 0.037225, "end_time": "2020-12-11T19:38:40.366492", "exception": false, "start_time": "2020-12-11T19:38:40.329267", "status": "completed"} tags=[]
assert recount2_rpkl_cm.loc['GAS6', 'SRP000599.SRR013549'].round(4) == -0.3125

# %% papermill={"duration": 0.033094, "end_time": "2020-12-11T19:38:40.423206", "exception": false, "start_time": "2020-12-11T19:38:40.390112", "status": "completed"} tags=[]
assert recount2_rpkl_cm.loc['GAS6', 'SRP045352.SRR1539229'].round(7) == -0.2843801

# %% papermill={"duration": 0.034978, "end_time": "2020-12-11T19:38:40.481111", "exception": false, "start_time": "2020-12-11T19:38:40.446133", "status": "completed"} tags=[]
assert recount2_rpkl_cm.loc['CFL2', 'SRP056840.SRR1951636'].round(7) == -0.3412832

# %% papermill={"duration": 0.033706, "end_time": "2020-12-11T19:38:40.536369", "exception": false, "start_time": "2020-12-11T19:38:40.502663", "status": "completed"} tags=[]
recount2_rpkl_cm.iloc[9, 16]

# %% papermill={"duration": 0.033955, "end_time": "2020-12-11T19:38:40.593874", "exception": false, "start_time": "2020-12-11T19:38:40.559919", "status": "completed"} tags=[]
assert recount2_rpkl_cm.iloc[9, 16].round(7) == -0.4938852

# %% [markdown] papermill={"duration": 0.02355, "end_time": "2020-12-11T19:38:40.639322", "exception": false, "start_time": "2020-12-11T19:38:40.615772", "status": "completed"} tags=[]
# ## Save

# %% papermill={"duration": 0.033325, "end_time": "2020-12-11T19:38:40.695236", "exception": false, "start_time": "2020-12-11T19:38:40.661911", "status": "completed"} tags=[]
output_filename = Path(
    conf.RECOUNT2['BASE_DIR'],
    'recount_data_prep_PLIER.pkl'
).resolve()

display(output_filename)

# %% papermill={"duration": 12.10829, "end_time": "2020-12-11T19:38:52.825435", "exception": false, "start_time": "2020-12-11T19:38:40.717145", "status": "completed"} tags=[]
recount2_rpkl_cm.to_pickle(output_filename)

# %% papermill={"duration": 0.033803, "end_time": "2020-12-11T19:38:52.881567", "exception": false, "start_time": "2020-12-11T19:38:52.847764", "status": "completed"} tags=[]
# from utils.hdf5 import simplify_string_for_hdf5

# %% papermill={"duration": 0.034516, "end_time": "2020-12-11T19:38:52.940283", "exception": false, "start_time": "2020-12-11T19:38:52.905767", "status": "completed"} tags=[]
# output_filename = os.path.join(conf.DATA_DIR, 'recount_data_prep_PLIER.h5')
# display(output_filename)

# %% papermill={"duration": 0.035502, "end_time": "2020-12-11T19:38:53.000315", "exception": false, "start_time": "2020-12-11T19:38:52.964813", "status": "completed"} tags=[]
# with pd.HDFStore(output_filename, mode='w', complevel=1) as store:
#     for idx, gene in enumerate(recount2_rpkl_cm.index):
#         if idx % 100:
#             print(f'', flush=True, end='')
        
#         clean_gene = simplify_string_for_hdf5(gene)
#         store[clean_gene] = recount2_rpkl_cm.loc[gene]

# %% papermill={"duration": 0.035287, "end_time": "2020-12-11T19:38:53.059907", "exception": false, "start_time": "2020-12-11T19:38:53.024620", "status": "completed"} tags=[]
del recount2_rpkl_cm

# %% [markdown] papermill={"duration": 0.02293, "end_time": "2020-12-11T19:38:53.105200", "exception": false, "start_time": "2020-12-11T19:38:53.082270", "status": "completed"} tags=[]
# # Read recount2 pathways

# %% papermill={"duration": 0.033127, "end_time": "2020-12-11T19:38:53.161545", "exception": false, "start_time": "2020-12-11T19:38:53.128418", "status": "completed"} tags=[]
recount2_all_paths_cm = recount_data_prep.rx2('all.paths.cm')

# %% papermill={"duration": 0.034734, "end_time": "2020-12-11T19:38:53.216452", "exception": false, "start_time": "2020-12-11T19:38:53.181718", "status": "completed"} tags=[]
recount2_all_paths_cm

# %% papermill={"duration": 0.036663, "end_time": "2020-12-11T19:38:53.277759", "exception": false, "start_time": "2020-12-11T19:38:53.241096", "status": "completed"} tags=[]
recount2_all_paths_cm.rownames

# %% papermill={"duration": 0.034523, "end_time": "2020-12-11T19:38:53.334975", "exception": false, "start_time": "2020-12-11T19:38:53.300452", "status": "completed"} tags=[]
recount2_all_paths_cm.colnames

# %% papermill={"duration": 0.033812, "end_time": "2020-12-11T19:38:53.392628", "exception": false, "start_time": "2020-12-11T19:38:53.358816", "status": "completed"} tags=[]
with localconverter(ro.default_converter + pandas2ri.converter):
  recount2_all_paths_cm_values = ro.conversion.rpy2py(recount2_all_paths_cm)

# %% papermill={"duration": 0.03371, "end_time": "2020-12-11T19:38:53.449644", "exception": false, "start_time": "2020-12-11T19:38:53.415934", "status": "completed"} tags=[]
recount2_all_paths_cm_values

# %% papermill={"duration": 0.057297, "end_time": "2020-12-11T19:38:53.530196", "exception": false, "start_time": "2020-12-11T19:38:53.472899", "status": "completed"} tags=[]
recount2_all_paths_cm = pd.DataFrame(
    data=recount2_all_paths_cm_values,
    index=recount2_all_paths_cm.rownames,
    columns=recount2_all_paths_cm.colnames,
    dtype=bool,
)

# %% papermill={"duration": 0.033083, "end_time": "2020-12-11T19:38:53.585809", "exception": false, "start_time": "2020-12-11T19:38:53.552726", "status": "completed"} tags=[]
assert recount2_all_paths_cm.shape == (6750, 628)

# %% papermill={"duration": 0.037111, "end_time": "2020-12-11T19:38:53.648423", "exception": false, "start_time": "2020-12-11T19:38:53.611312", "status": "completed"} tags=[]
recount2_all_paths_cm.shape

# %% papermill={"duration": 0.035042, "end_time": "2020-12-11T19:38:53.706016", "exception": false, "start_time": "2020-12-11T19:38:53.670974", "status": "completed"} tags=[]
recount2_all_paths_cm.dtypes.unique()

# %% papermill={"duration": 0.045969, "end_time": "2020-12-11T19:38:53.776152", "exception": false, "start_time": "2020-12-11T19:38:53.730183", "status": "completed"} tags=[]
recount2_all_paths_cm.head()

# %% [markdown] papermill={"duration": 0.024673, "end_time": "2020-12-11T19:38:53.824854", "exception": false, "start_time": "2020-12-11T19:38:53.800181", "status": "completed"} tags=[]
# ## Testing

# %% papermill={"duration": 0.038699, "end_time": "2020-12-11T19:38:53.888908", "exception": false, "start_time": "2020-12-11T19:38:53.850209", "status": "completed"} tags=[]
recount2_all_paths_cm.loc['CTSD', 'REACTOME_SCFSKP2_MEDIATED_DEGRADATION_OF_P27_P21']

# %% papermill={"duration": 0.033672, "end_time": "2020-12-11T19:38:53.946226", "exception": false, "start_time": "2020-12-11T19:38:53.912554", "status": "completed"} tags=[]
assert not recount2_all_paths_cm.loc['CTSD', 'REACTOME_SCFSKP2_MEDIATED_DEGRADATION_OF_P27_P21']

# %% papermill={"duration": 0.036356, "end_time": "2020-12-11T19:38:54.007293", "exception": false, "start_time": "2020-12-11T19:38:53.970937", "status": "completed"} tags=[]
assert recount2_all_paths_cm.loc['CTSD', 'PID_P53DOWNSTREAMPATHWAY']

# %% papermill={"duration": 0.03624, "end_time": "2020-12-11T19:38:54.068482", "exception": false, "start_time": "2020-12-11T19:38:54.032242", "status": "completed"} tags=[]
assert recount2_all_paths_cm.loc['MMP14', 'PID_HIF2PATHWAY']

# %% [markdown] papermill={"duration": 0.023676, "end_time": "2020-12-11T19:38:54.117344", "exception": false, "start_time": "2020-12-11T19:38:54.093668", "status": "completed"} tags=[]
# ## Save

# %% papermill={"duration": 0.036, "end_time": "2020-12-11T19:38:54.195313", "exception": false, "start_time": "2020-12-11T19:38:54.159313", "status": "completed"} tags=[]
output_filename = Path(
    conf.RECOUNT2['BASE_DIR'],
    'recount_all_paths_cm.pkl'
).resolve()

display(output_filename)

# %% papermill={"duration": 0.042406, "end_time": "2020-12-11T19:38:54.260937", "exception": false, "start_time": "2020-12-11T19:38:54.218531", "status": "completed"} tags=[]
recount2_all_paths_cm.to_pickle(output_filename)

# %% papermill={"duration": 0.032979, "end_time": "2020-12-11T19:38:54.318417", "exception": false, "start_time": "2020-12-11T19:38:54.285438", "status": "completed"} tags=[]
del recount2_all_paths_cm

# %% papermill={"duration": 0.025328, "end_time": "2020-12-11T19:38:54.368143", "exception": false, "start_time": "2020-12-11T19:38:54.342815", "status": "completed"} tags=[]
