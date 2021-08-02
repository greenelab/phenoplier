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
# Generates manubot tables for pathways enriched (from the MultiPLIER models) given an LV name (in Settings below).

# %% [markdown] tags=[]
# # Modules loading

# %% tags=[]
# %load_ext autoreload
# %autoreload 2

# %% tags=[]
import re
from pathlib import Path

import pandas as pd

from entity import Trait
import conf

# %% [markdown] tags=[]
# # Settings

# %% tags=["parameters"]
LV_NAME = "LV57"

# %%
assert (
    conf.MANUSCRIPT["BASE_DIR"] is not None
), "The manuscript directory was not configured"

OUTPUT_FILE_PATH = conf.MANUSCRIPT["CONTENT_DIR"] / "50.00.supplementary_material.md"
display(OUTPUT_FILE_PATH)
assert OUTPUT_FILE_PATH.exists()

# %% [markdown] tags=[]
# # Load MultiPLIER summary

# %% tags=[]
multiplier_model_summary = pd.read_pickle(conf.MULTIPLIER["MODEL_SUMMARY_FILE"])

# %% tags=[]
multiplier_model_summary.shape

# %% tags=[]
multiplier_model_summary.head()

# %% [markdown]
# # LV pathways

# %%
lv_pathways = multiplier_model_summary[
    multiplier_model_summary["LV index"].isin((LV_NAME[2:],))
    & (
        (multiplier_model_summary["FDR"] < 0.05)
        #         | (multiplier_model_summary["AUC"] >= 0.75)
    )
]

# %%
lv_pathways.shape

# %%
lv_pathways = lv_pathways[["pathway", "AUC", "FDR"]].sort_values("FDR")

# %%
lv_pathways = lv_pathways.assign(AUC=lv_pathways["AUC"].apply(lambda x: f"{x:.2f}"))

# %%
lv_pathways = lv_pathways.assign(FDR=lv_pathways["FDR"].apply(lambda x: f"{x:.2e}"))

# %%
lv_pathways = lv_pathways.rename(
    columns={
        "pathway": "Pathway",
    }
)

# %%
lv_pathways.head()

# %% [markdown]
# ## Split names

# %%
lv_pathways["Pathway"] = lv_pathways["Pathway"].apply(lambda x: " ".join(x.split("_")))

# %%
lv_pathways.head()

# %% [markdown]
# ## Fill empty

# %%
if lv_pathways.shape[0] == 0:
    lv_pathways.loc[0, "Pathway"] = "No pathways significantly enriched"
    lv_pathways = lv_pathways.fillna("")

# %% [markdown]
# ## Save

# %%
# result_set is either phenomexcan or emerge
LV_FILE_MARK_TEMPLATE = "<!-- {lv}:multiplier_pathways:{position} -->"

# %%
TABLE_CAPTION = (
    "Table: Pathways aligned to {lv_name} from the MultiPLIER models. {table_id}"
)

# %%
TABLE_CAPTION_ID = "#tbl:sup:multiplier_pathways:{lv_name_lower_case}"

# %%
# start
lv_file_mark_start = LV_FILE_MARK_TEMPLATE.format(lv=LV_NAME, position="start")
display(lv_file_mark_start)

# end
lv_file_mark_end = LV_FILE_MARK_TEMPLATE.format(lv=LV_NAME, position="end")
display(lv_file_mark_end)

# %%
new_content = lv_pathways.to_markdown(index=False, disable_numparse=True)

# %%
# add table caption
table_caption = TABLE_CAPTION.format(
    lv_name=LV_NAME,
    table_id="{" + TABLE_CAPTION_ID.format(lv_name_lower_case=LV_NAME.lower()) + "}",
)
display(table_caption)

# %%
new_content += "\n\n" + table_caption

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

# %%
