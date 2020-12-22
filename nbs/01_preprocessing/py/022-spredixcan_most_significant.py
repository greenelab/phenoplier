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

# %% [markdown] papermill={"duration": 0.044577, "end_time": "2020-12-18T22:38:21.345879", "exception": false, "start_time": "2020-12-18T22:38:21.301302", "status": "completed"} tags=[]
# # Description

# %% [markdown] papermill={"duration": 0.011764, "end_time": "2020-12-18T22:38:21.398073", "exception": false, "start_time": "2020-12-18T22:38:21.386309", "status": "completed"} tags=[]
# It computes a single matrix with all traits and genes in PhenomeXcan using S-PrediXcan results, which have direction of effect (in contrast to S-MultiXcan results). For each gene-trait pair, it takes the most significant result across all tissues.

# %% [markdown] papermill={"duration": 0.011799, "end_time": "2020-12-18T22:38:21.422136", "exception": false, "start_time": "2020-12-18T22:38:21.410337", "status": "completed"} tags=[]
# # Modules loading

# %% papermill={"duration": 0.022467, "end_time": "2020-12-18T22:38:21.456126", "exception": false, "start_time": "2020-12-18T22:38:21.433659", "status": "completed"} tags=[]
# %load_ext autoreload
# %autoreload 2

# %% papermill={"duration": 0.192251, "end_time": "2020-12-18T22:38:21.659996", "exception": false, "start_time": "2020-12-18T22:38:21.467745", "status": "completed"} tags=[]
from pathlib import Path

import numpy as np
import pandas as pd

import conf
from data.cache import read_data
from data.hdf5 import simplify_trait_fullcode, HDF5_FILE_PATTERN

# %% [markdown] papermill={"duration": 0.011416, "end_time": "2020-12-18T22:38:21.683356", "exception": false, "start_time": "2020-12-18T22:38:21.671940", "status": "completed"} tags=[]
# # Settings

# %% papermill={"duration": 0.021367, "end_time": "2020-12-18T22:38:21.716134", "exception": false, "start_time": "2020-12-18T22:38:21.694767", "status": "completed"} tags=[]
SPREDIXCAN_H5_FOLDER = Path(
    conf.PHENOMEXCAN["SPREDIXCAN_MASHR_ZSCORES_FOLDER"],
    "hdf5",
)
assert SPREDIXCAN_H5_FOLDER.is_dir(), "The folder does not exist"

# %% [markdown] papermill={"duration": 0.012101, "end_time": "2020-12-18T22:38:21.740637", "exception": false, "start_time": "2020-12-18T22:38:21.728536", "status": "completed"} tags=[]
# # Get all PhenomeXcan traits

# %% [markdown] papermill={"duration": 0.011156, "end_time": "2020-12-18T22:38:21.763754", "exception": false, "start_time": "2020-12-18T22:38:21.752598", "status": "completed"} tags=[]
# ## Get all PhenomeXcan trait full codes

# %% papermill={"duration": 0.198378, "end_time": "2020-12-18T22:38:21.973975", "exception": false, "start_time": "2020-12-18T22:38:21.775597", "status": "completed"} tags=[]
from entity import Trait

# %% papermill={"duration": 0.313021, "end_time": "2020-12-18T22:38:22.299984", "exception": false, "start_time": "2020-12-18T22:38:21.986963", "status": "completed"} tags=[]
all_phenomexcan_traits = [
    trait_fullcode
    for trait_fullcode in read_data(
        conf.PHENOMEXCAN["SMULTIXCAN_MASHR_ZSCORES_FILE"]
    ).columns
]

# %% papermill={"duration": 0.026883, "end_time": "2020-12-18T22:38:22.340853", "exception": false, "start_time": "2020-12-18T22:38:22.313970", "status": "completed"} tags=[]
_tmp = set(all_phenomexcan_traits)
display(len(_tmp))
assert len(_tmp) == 4091

# %% [markdown] papermill={"duration": 0.012324, "end_time": "2020-12-18T22:38:22.366058", "exception": false, "start_time": "2020-12-18T22:38:22.353734", "status": "completed"} tags=[]
# # Get list of files

# %% papermill={"duration": 0.021104, "end_time": "2020-12-18T22:38:22.399027", "exception": false, "start_time": "2020-12-18T22:38:22.377923", "status": "completed"} tags=[]
from glob import glob

# %% papermill={"duration": 0.022083, "end_time": "2020-12-18T22:38:22.433139", "exception": false, "start_time": "2020-12-18T22:38:22.411056", "status": "completed"} tags=[]
spredixcan_files = list(SPREDIXCAN_H5_FOLDER.glob("*.h5"))

# %% papermill={"duration": 0.022217, "end_time": "2020-12-18T22:38:22.467771", "exception": false, "start_time": "2020-12-18T22:38:22.445554", "status": "completed"} tags=[]
display(spredixcan_files[:5])
assert len(spredixcan_files) == 49

# %% [markdown] papermill={"duration": 0.012604, "end_time": "2020-12-18T22:38:22.493434", "exception": false, "start_time": "2020-12-18T22:38:22.480830", "status": "completed"} tags=[]
# # Get all tissues

# %% papermill={"duration": 0.022165, "end_time": "2020-12-18T22:38:22.527982", "exception": false, "start_time": "2020-12-18T22:38:22.505817", "status": "completed"} tags=[]
import re

# %% papermill={"duration": 0.022132, "end_time": "2020-12-18T22:38:22.562942", "exception": false, "start_time": "2020-12-18T22:38:22.540810", "status": "completed"} tags=[]
all_tissues = [
    re.search(HDF5_FILE_PATTERN, file.name).group("tissue") for file in spredixcan_files
]

# %% papermill={"duration": 0.022778, "end_time": "2020-12-18T22:38:22.598466", "exception": false, "start_time": "2020-12-18T22:38:22.575688", "status": "completed"} tags=[]
display(all_tissues[:5])
assert len(all_tissues) == len(spredixcan_files)

# %% [markdown] papermill={"duration": 0.012836, "end_time": "2020-12-18T22:38:22.624455", "exception": false, "start_time": "2020-12-18T22:38:22.611619", "status": "completed"} tags=[]
# # Function to get most significant results

# %% papermill={"duration": 0.022389, "end_time": "2020-12-18T22:38:22.659677", "exception": false, "start_time": "2020-12-18T22:38:22.637288", "status": "completed"} tags=[]
from data.hdf5 import read_spredixcan


# %% papermill={"duration": 0.023524, "end_time": "2020-12-18T22:38:22.696406", "exception": false, "start_time": "2020-12-18T22:38:22.672882", "status": "completed"} tags=[]
def get_most_signif(trait_fullcode):
    """
    TODO: describe
    """
    trait_data = (
        pd.concat(
            [
                read_spredixcan(SPREDIXCAN_H5_FOLDER, trait_fullcode, t)
                for t in all_tissues
            ],
            keys=all_tissues,
            axis=0,
        )
        .reset_index()
        .drop(columns=["level_0"])[["gene_id", trait_fullcode]]
        .dropna()
    )

    return (
        trait_data.loc[
            trait_data[trait_fullcode].abs().groupby(trait_data["gene_id"]).idxmax()
        ]
        .set_index("gene_id")
        .squeeze()
    )


# %% papermill={"duration": 2.729167, "end_time": "2020-12-18T22:38:25.439075", "exception": false, "start_time": "2020-12-18T22:38:22.709908", "status": "completed"} tags=[]
get_most_signif("100002_raw-Energy")

# %% [markdown] papermill={"duration": 0.014314, "end_time": "2020-12-18T22:38:25.468260", "exception": false, "start_time": "2020-12-18T22:38:25.453946", "status": "completed"} tags=[]
# ## Testing

# %% papermill={"duration": 0.77672, "end_time": "2020-12-18T22:38:26.258172", "exception": false, "start_time": "2020-12-18T22:38:25.481452", "status": "completed"} tags=[]
_trait = "100001_raw-Food_weight"

_trait_data = pd.DataFrame(
    {t: read_spredixcan(SPREDIXCAN_H5_FOLDER, _trait, t) for t in all_tissues}
)

# %% papermill={"duration": 0.027746, "end_time": "2020-12-18T22:38:26.301695", "exception": false, "start_time": "2020-12-18T22:38:26.273949", "status": "completed"} tags=[]
_trait_data.loc["ENSG00000225595"].dropna().sort_values()

# %% papermill={"duration": 2.495059, "end_time": "2020-12-18T22:38:28.812023", "exception": false, "start_time": "2020-12-18T22:38:26.316964", "status": "completed"} tags=[]
_trait = "pgc.scz2"
_data_ms = get_most_signif(_trait)

# %% papermill={"duration": 0.026775, "end_time": "2020-12-18T22:38:28.853240", "exception": false, "start_time": "2020-12-18T22:38:28.826465", "status": "completed"} tags=[]
_gene_id = "ENSG00000158691"
assert _data_ms.loc[_gene_id].round(3) == 11.067

_gene_id = "ENSG00000204713"
assert _data_ms.loc[_gene_id].round(3) == 10.825

_gene_id = "ENSG00000225595"
assert _data_ms.loc[_gene_id].round(3) == -10.956

# %% papermill={"duration": 2.489753, "end_time": "2020-12-18T22:38:31.357609", "exception": false, "start_time": "2020-12-18T22:38:28.867856", "status": "completed"} tags=[]
_trait = "100001_raw-Food_weight"
_data_ms = get_most_signif(_trait)

# %% papermill={"duration": 0.027249, "end_time": "2020-12-18T22:38:31.399788", "exception": false, "start_time": "2020-12-18T22:38:31.372539", "status": "completed"} tags=[]
_gene_id = "ENSG00000225595"
assert _data_ms.loc[_gene_id].round(3) == -0.561

_gene_id = "ENSG00000183323"
assert _data_ms.loc[_gene_id].round(3) == 4.444

_gene_id = "ENSG00000182901"
assert _data_ms.loc[_gene_id].round(3) == -4.369

# %% [markdown] papermill={"duration": 0.014074, "end_time": "2020-12-18T22:38:31.429110", "exception": false, "start_time": "2020-12-18T22:38:31.415036", "status": "completed"} tags=[]
# # Compute most significant for all traits

# %% papermill={"duration": 0.028758, "end_time": "2020-12-18T22:38:31.471828", "exception": false, "start_time": "2020-12-18T22:38:31.443070", "status": "completed"} tags=[]
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


# %% papermill={"duration": 0.024492, "end_time": "2020-12-18T22:38:31.510760", "exception": false, "start_time": "2020-12-18T22:38:31.486268", "status": "completed"} tags=[]
def _run(trait):
    return {trait: get_most_signif(trait)}


# %% papermill={"duration": 4286.610718, "end_time": "2020-12-18T23:49:58.135553", "exception": false, "start_time": "2020-12-18T22:38:31.524835", "status": "completed"} tags=[]
all_results = {}
with ProcessPoolExecutor(max_workers=conf.GENERAL["N_JOBS"]) as executor:
    tasks = [executor.submit(_run, trait) for trait in all_phenomexcan_traits]
    for future in tqdm(as_completed(tasks), total=len(all_phenomexcan_traits)):
        res = future.result()
        all_results.update(res)

# %% papermill={"duration": 26.054277, "end_time": "2020-12-18T23:50:24.777402", "exception": false, "start_time": "2020-12-18T23:49:58.723125", "status": "completed"} tags=[]
data_most_signif = pd.DataFrame(all_results)

# %% papermill={"duration": 0.571589, "end_time": "2020-12-18T23:50:25.917225", "exception": false, "start_time": "2020-12-18T23:50:25.345636", "status": "completed"} tags=[]
data_most_signif.shape

# %% papermill={"duration": 0.574449, "end_time": "2020-12-18T23:50:27.040414", "exception": false, "start_time": "2020-12-18T23:50:26.465965", "status": "completed"} tags=[]
data_most_signif.head()

# %% [markdown] papermill={"duration": 0.556128, "end_time": "2020-12-18T23:50:28.184126", "exception": false, "start_time": "2020-12-18T23:50:27.627998", "status": "completed"} tags=[]
# # Save

# %% papermill={"duration": 0.572249, "end_time": "2020-12-18T23:50:29.312505", "exception": false, "start_time": "2020-12-18T23:50:28.740256", "status": "completed"} tags=[]
output_folder = Path(
    conf.PHENOMEXCAN["SPREDIXCAN_MASHR_ZSCORES_FOLDER"],
    "most_signif",
).resolve()
output_folder.mkdir(exist_ok=True)

# %% papermill={"duration": 0.566996, "end_time": "2020-12-18T23:50:30.433450", "exception": false, "start_time": "2020-12-18T23:50:29.866454", "status": "completed"} tags=[]
output_file = Path(output_folder, "spredixcan-most_signif.pkl").resolve()
display(output_file)

# %% papermill={"duration": 1.296338, "end_time": "2020-12-18T23:50:32.300840", "exception": false, "start_time": "2020-12-18T23:50:31.004502", "status": "completed"} tags=[]
data_most_signif.to_pickle(output_file)
