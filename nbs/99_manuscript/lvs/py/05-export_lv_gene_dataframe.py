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

# %% [markdown]
# # Description

# %% [markdown]
# This notebook generates more end-user-friendly Excel files of some of the data generated in PhenoPLIER, such as the LV-gene matrix and LV-pathways.

# %% [markdown]
# # Modules loading

# %%
import pandas as pd
import openpyxl
from openpyxl.utils import get_column_letter

import conf
from utils import get_git_repository_path

# %% [markdown]
# # Settings

# %% tags=[]
DELIVERABLES_BASE_DIR = get_git_repository_path() / "data"
display(DELIVERABLES_BASE_DIR)

# %% tags=[]
OUTPUT_DIR = DELIVERABLES_BASE_DIR / "multiplier"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
display(OUTPUT_DIR)

# %% [markdown]
# # Load data

# %% [markdown] tags=[]
# ## MultiPLIER summary

# %% tags=[]
multiplier_model_summary = pd.read_pickle(conf.MULTIPLIER["MODEL_SUMMARY_FILE"])

# %% tags=[]
multiplier_model_summary.shape

# %% tags=[]
multiplier_model_summary.head()

# %% [markdown]
# ## Get pathway-aligned LVs

# %% tags=[]
well_aligned_lvs = multiplier_model_summary[multiplier_model_summary["FDR"] < 0.05]

display(well_aligned_lvs.shape)
display(well_aligned_lvs.head())

# %% tags=[]
well_aligned_lv_codes = set([f"LV{lvi}" for lvi in well_aligned_lvs["LV index"]])

# %% tags=[]
len(well_aligned_lv_codes)

# %% tags=[]
list(well_aligned_lv_codes)[:5]

# %% [markdown] tags=[]
# ## MultiPLIER Z (gene loadings)

# %%
multiplier_z = pd.read_pickle(conf.MULTIPLIER["MODEL_Z_MATRIX_FILE"])

# %%
multiplier_z.shape

# %%
multiplier_z.head()

# %% [markdown] tags=[]
# # Create LV-pathway dataframe

# %%
lv_pathway_df = multiplier_model_summary[
    ["LV index", "pathway", "AUC", "p-value", "FDR"]
]

# %%
lv_pathway_df["LV index"] = lv_pathway_df["LV index"].astype(int)

# %%
lv_pathway_df = lv_pathway_df.sort_values(["LV index", "FDR"])

# %%
lv_pathway_df["LV index"] = lv_pathway_df["LV index"].apply(lambda x: f"LV{x}")

# %%
lv_pathway_df.shape

# %%
lv_pathway_df = lv_pathway_df.rename(
    columns={"LV index": "LV identifier", "pathway": "Pathway"}
)

# %%
lv_pathway_df.head()

# %%
lv_pathway_df.tail()

# %% [markdown]
# ## Save

# %%
output_file = OUTPUT_DIR / "lv-pathways.xlsx"
display(output_file)

# %%
lv_pathway_df.to_excel(output_file, index=False)

# %%
# adjust column widths
wb = openpyxl.load_workbook(filename=output_file)
worksheet = wb.active

for col in worksheet.columns:
    max_length = 0
    column = get_column_letter(col[0].column)  # Get the column name
    for cell in col:
        if cell.coordinate in worksheet.merged_cells:  # not check merge_cells
            continue

        try:  # Necessary to avoid error on empty cells
            if len(str(cell.value)) > max_length:
                max_length = len(str(cell.value))
        except:
            pass
    adjusted_width = (max_length + 2) * 1.05
    worksheet.column_dimensions[column].width = adjusted_width

wb.save(output_file)

# %% [markdown] tags=[]
# # Create LV-gene dataframe

# %%
df = (
    multiplier_z.unstack()
    .to_frame()
    .reset_index()
    .rename(columns={0: "Weight", "level_0": "LV identifier", "level_1": "Gene symbol"})
)

# %%
df = df.assign(lv_index=df["LV identifier"].apply(lambda x: int(x[2:])))

# %%
df = df.sort_values(["lv_index", "Weight"], ascending=[True, False]).drop(
    columns=["lv_index"]
)

# %%
df.shape

# %%
df.head()

# %%
df.tail()

# %% [markdown]
# ## Save as TSV

# %%
# output_file = OUTPUT_DIR / "lv-gene_weights.tsv.gz"
# display(output_file)

# %%
# df.to_csv(output_file, sep="\t", index=False)

# %% [markdown]
# ## Save as Excel

# %%
output_file = OUTPUT_DIR / "lv-gene_weights.xlsx"
display(output_file)

# %%
with pd.ExcelWriter(output_file) as writer:
    for lv_id in df["LV identifier"].unique():
        print(lv_id, end=", ", flush=True)

        lv_data = df[df["LV identifier"] == lv_id].drop(columns=["LV identifier"])
        lv_data = lv_data.sort_values("Weight", ascending=False)

        lv_data.to_excel(writer, index=False, sheet_name=lv_id)

# %%
# adjust column widths
wb = openpyxl.load_workbook(filename=output_file)

for worksheet in wb.worksheets:
    for col in worksheet.columns:
        max_length = 0
        column = get_column_letter(col[0].column)  # Get the column name
        for cell in col:
            if cell.coordinate in worksheet.merged_cells:  # not check merge_cells
                continue

            try:  # Necessary to avoid error on empty cells
                if len(str(cell.value)) > max_length:
                    max_length = len(str(cell.value))
            except:
                pass
        adjusted_width = (max_length + 2) * 1.05
        worksheet.column_dimensions[column].width = adjusted_width

wb.save(output_file)

# %%
