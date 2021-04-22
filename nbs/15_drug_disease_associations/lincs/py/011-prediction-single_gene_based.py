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
# In this notebook we predict drug-disease associations using S-PrediXcan results and LINCS expression profiles.
#
# See more details in notebook `011-prediction-gene_module_based.ipynb`.

# %% [markdown] tags=[]
# # Modules loading

# %% tags=[]
# %load_ext autoreload
# %autoreload 2

# %% tags=[]
from IPython.display import display
from pathlib import Path

import pandas as pd

import conf

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
# if True, then it doesn't check if result files already exist and runs everything again
FORCE_RUN = True

# %% tags=[]
PREDICTION_METHOD = "Gene-based"

# %% tags=[]
LINCS_DATA_DIR = conf.RESULTS["DRUG_DISEASE_ANALYSES"] / "lincs"
display(LINCS_DATA_DIR)
assert LINCS_DATA_DIR.exists()

# %% tags=[]
SPREDIXCAN_DATA_DIR = conf.RESULTS["DRUG_DISEASE_ANALYSES"] / "spredixcan" / "raw"
display(SPREDIXCAN_DATA_DIR)
assert SPREDIXCAN_DATA_DIR.exists()

# %% tags=[]
OUTPUT_PREDICTIONS_DIR = Path(LINCS_DATA_DIR, "predictions")
display(OUTPUT_PREDICTIONS_DIR)
OUTPUT_PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown] tags=[]
# # Load PharmacotherapyDB gold standard

# %% tags=[]
gold_standard = pd.read_pickle(
    Path(conf.RESULTS["DRUG_DISEASE_ANALYSES"], "gold_standard.pkl"),
)

# %% tags=[]
display(gold_standard.shape)

# %% tags=[]
display(gold_standard.head())

# %% tags=[]
doids_in_gold_standard = set(gold_standard["trait"])

# %% [markdown] tags=[]
# # Load LINCS

# %% [markdown] tags=[]
# ## Raw data

# %% tags=[]
input_file = Path(LINCS_DATA_DIR, "lincs-data.pkl").resolve()

display(input_file)

# %% tags=[]
lincs_projection = pd.read_pickle(input_file)

# %% tags=[]
display(lincs_projection.shape)

# %% tags=[]
display(lincs_projection.head())

# %% [markdown] tags=[]
# # Load S-PrediXcan

# %% tags=[]
phenomexcan_input_file_list = [
    f for f in SPREDIXCAN_DATA_DIR.glob("*.pkl") if f.name.startswith("spredixcan-")
]

# %% tags=[]
display(len(phenomexcan_input_file_list))

# %% tags=[]
pd.read_pickle(phenomexcan_input_file_list[0]).head()

# %% [markdown] tags=[]
# # Predict drug-disease associations

# %% tags=[]
from drug_disease import predict_dotprod_neg

# %% tags=[]
methods_to_run = [predict_dotprod_neg]

# %% tags=[]
for phenomexcan_input_file in phenomexcan_input_file_list:
    print(phenomexcan_input_file.name)

    # read phenomexcan data
    phenomexcan_projection = pd.read_pickle(phenomexcan_input_file)

    # get common genes with lincs
    common_genes = phenomexcan_projection.index.intersection(lincs_projection.index)
    phenomexcan_projection = phenomexcan_projection.loc[common_genes]
    lincs_projection = lincs_projection.loc[common_genes]

    print(f"  shape: {phenomexcan_projection.shape}")

    for prediction_method in methods_to_run:
        # here I use different thresholds to select the genes: the top 50, 100, 250 and 500, or all (None)
        for ntc in (None, 50, 100, 250, 500):
            prediction_method(
                lincs_projection,
                phenomexcan_input_file,
                phenomexcan_projection,
                OUTPUT_PREDICTIONS_DIR,
                PREDICTION_METHOD,
                doids_in_gold_standard,
                FORCE_RUN,
                n_top_conditions=ntc,
                use_abs=True,
            )

            print("\n")

# %% tags=[]
