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
# This notebooks writes a markdown table (in the manuscript) with the LVs that are enriched with the lipids-altering gene sets.

# %% [markdown] tags=[]
# # Modules loading

# %% tags=[]
# %load_ext autoreload
# %autoreload 2

# %% tags=[]
import re
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import HTML
from statsmodels.stats.multitest import multipletests

from entity import Trait
from data.cache import read_data
import conf

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
FGSEA_INPUT_FILEPATH = Path(
    conf.RESULTS["CRISPR_ANALYSES"]["BASE_DIR"], "fgsea-hi_conf-all_lvs.tsv"
).resolve()

display(FGSEA_INPUT_FILEPATH)

# %%
assert "CONTENT_DIR" in conf.MANUSCRIPT

OUTPUT_FILE_PATH = conf.MANUSCRIPT["CONTENT_DIR"] / "50.00.supplementary_material.md"
display(OUTPUT_FILE_PATH)
assert OUTPUT_FILE_PATH.exists()

# %%
PVAL_THRESHOLD = 0.01

# %% [markdown] tags=[]
# # Data loading

# %% [markdown] tags=[]
# ## MultiPLIER summary

# %% tags=[]
multiplier_model_summary = pd.read_pickle(conf.MULTIPLIER["MODEL_SUMMARY_FILE"])

# %% tags=[]
multiplier_model_summary.shape

# %% tags=[]
multiplier_model_summary.head()

# %% tags=[]
well_aligned_lvs = multiplier_model_summary[
    (
        multiplier_model_summary["FDR"] < 0.05
    )  # & (multiplier_model_summary["AUC"] >= 0.75)
]

display(well_aligned_lvs.shape)
display(well_aligned_lvs.head())

# %% tags=[]
well_aligned_lv_codes = set([f"LV{lvi}" for lvi in well_aligned_lvs["LV index"]])

# %% tags=[]
len(well_aligned_lv_codes)

# %% tags=[]
list(well_aligned_lv_codes)[:5]

# %% [markdown] tags=[]
# ## LVs enrichment on DEG from CRISPR screen

# %% tags=[]
deg_enrich = pd.read_csv(
    FGSEA_INPUT_FILEPATH,
    sep="\t",
)

# %% tags=[]
deg_enrich.shape

# %%
deg_enrich = deg_enrich.assign(
    lv_aligned=deg_enrich["lv"].apply(lambda x: x in well_aligned_lv_codes)
)

# %% tags=[]
deg_enrich.head()

# %% [markdown] tags=[]
# # Get significantly enriched modules

# %% tags=[]
df = deg_enrich[(deg_enrich["pval"] < PVAL_THRESHOLD)].sort_values(
    "pval", ascending=True
)

# %% tags=[]
# df = deg_enrich[(deg_enrich["pval"] < PVAL_THRESHOLD) & (deg_enrich["lv_aligned"])].sort_values(
#     "pval", ascending=True
# )

# %% tags=[]
# df = deg_enrich[(deg_enrich["pval"] < PVAL_THRESHOLD)].sort_values(
#     "pval", ascending=True
# )

# %% tags=[]
df.shape

# %% tags=[]
df.sort_values("fdr")

# %% [markdown]
# # Prepare table

# %%
df = df.assign(pval=df["pval"].apply(lambda x: f"{x:.4f}"))

# %%
df = df.assign(fdr=df["fdr"].apply(lambda x: f"{x:.2e}"))

# %%
df = df.rename(
    columns={
        "pathway": "Lipids gene-set",
        "pval": "p-value",
        "leadingEdge": "Leading edge",
        #         "fdr": "FDR",
        "lv": "Gene module",
    }
)

# %%
df = df.replace(
    {
        "Lipids gene-set": {
            "gene_set_decrease": "decrease",
            "gene_set_increase": "increase",
        },
    }
)

# %%
df = df.replace(
    {
        "Leading edge": {
            "([A-Z\d]+)": "*\\1*",
        }
    },
    regex=True,
)

# %%
df["Gene module"] = df.apply(
    lambda x: f"**{x['Gene module']}**" if x["lv_aligned"] else x["Gene module"], axis=1
)

# %%
df = df[["Gene module", "Lipids gene-set", "Leading edge", "p-value"]]

# %%
df

# %% [markdown]
# # Save lipids-increasing

# %%
gene_set_name = "increase"

# %%
# result_set is either phenomexcan or emerge
LV_FILE_MARK_TEMPLATE = (
    "<!-- lipids_gene_sets:modules_enriched_{gene_set}:{position} -->"
)

# %%
TABLE_CAPTION = f"Table: Gene modules (LVs) nominally enriched for the lipids-increasing gene-set from the CRISPR-screen (*P* < {PVAL_THRESHOLD}). LVs significantly aligned with pathways (FDR < 0.05) from the MultiPLIER models are shown in boldface. {{table_id}}"

# %%
TABLE_CAPTION_ID = "#tbl:sup:lipids_crispr:modules_enriched_{gene_set}"

# %%
# start
lv_file_mark_start = LV_FILE_MARK_TEMPLATE.format(
    gene_set=gene_set_name, position="start"
)
display(lv_file_mark_start)

# end
lv_file_mark_end = LV_FILE_MARK_TEMPLATE.format(gene_set=gene_set_name, position="end")
display(lv_file_mark_end)

# table caption
TABLE_CAPTION_ID = TABLE_CAPTION_ID.format(gene_set=gene_set_name)
display(TABLE_CAPTION_ID)

# %%
new_content = df[df["Lipids gene-set"] == gene_set_name].to_markdown(
    index=False, disable_numparse=True
)

# %%
# add table caption
table_caption = TABLE_CAPTION.format(
    table_id="{" + TABLE_CAPTION_ID + "}",
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

# %% [markdown]
# # Save lipids-decreasing

# %%
gene_set_name = "decrease"

# %%
# result_set is either phenomexcan or emerge
LV_FILE_MARK_TEMPLATE = (
    "<!-- lipids_gene_sets:modules_enriched_{gene_set}:{position} -->"
)

# %%
TABLE_CAPTION = f"Table: Gene modules (LVs) nominally enriched for the lipids-decreasing gene-set from the CRISPR-screen (*P* < {PVAL_THRESHOLD}). LVs significantly aligned with pathways (FDR < 0.05) from the MultiPLIER models are shown in boldface. {{table_id}}"

# %%
TABLE_CAPTION_ID = "#tbl:sup:lipids_crispr:modules_enriched_{gene_set}"

# %%
# start
lv_file_mark_start = LV_FILE_MARK_TEMPLATE.format(
    gene_set=gene_set_name, position="start"
)
display(lv_file_mark_start)

# end
lv_file_mark_end = LV_FILE_MARK_TEMPLATE.format(gene_set=gene_set_name, position="end")
display(lv_file_mark_end)

# table caption
TABLE_CAPTION_ID = TABLE_CAPTION_ID.format(gene_set=gene_set_name)
display(TABLE_CAPTION_ID)

# %%
new_content = df[df["Lipids gene-set"] == gene_set_name].to_markdown(
    index=False, disable_numparse=True
)

# %%
# add table caption
table_caption = TABLE_CAPTION.format(
    table_id="{" + TABLE_CAPTION_ID + "}",
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

# %% tags=[]
