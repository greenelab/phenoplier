# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all,-execution,-papermill,-trusted
#     formats: ipynb,py//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] tags=[]
# # Description

# %% [markdown] tags=[]
# Generates manubot tables for PhenomeXcan and eMERGE associations given an LV name (which is the only parameter that needs to be specified in the Settings section below).

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
LV_NAME = "LV30"

# %%
assert (
    conf.MANUSCRIPT["BASE_DIR"] is not None
), "The manuscript directory was not configured"

OUTPUT_FILE_PATH = conf.MANUSCRIPT["CONTENT_DIR"] / "50.00.supplementary_material.md"
display(OUTPUT_FILE_PATH)
assert OUTPUT_FILE_PATH.exists()

# %%
# result_set is either phenomexcan or emerge
LV_FILE_MARK_TEMPLATE = "<!-- {lv}:{result_set}_traits_assocs:{position} -->"

# %%
TABLE_CAPTION = "Table: Significant trait associations of {lv_name} in {result_set_name}. {table_id}"

# %%
TABLE_CAPTION_ID = "#tbl:sup:{result_set}_assocs:{lv_name_lower_case}"

# %%
RESULT_SET_NAMES = {
    "phenomexcan": "PhenomeXcan",
    "emerge": "eMERGE",
}

# %% [markdown] tags=[]
# # Load data

# %% [markdown]
# ## PhenomeXcan LV-trait associations

# %%
input_filepath = Path(conf.RESULTS["GLS"] / "gls-summary-phenomexcan.pkl.gz")
display(input_filepath)

# %%
phenomexcan_lv_trait_assocs = pd.read_pickle(input_filepath)

# %%
phenomexcan_lv_trait_assocs.shape

# %%
phenomexcan_lv_trait_assocs.head()

# %% [markdown]
# ## eMERGE LV-trait associations

# %%
input_filepath = Path(conf.RESULTS["GLS"] / "gls-summary-emerge.pkl.gz")
display(input_filepath)

# %%
emerge_lv_trait_assocs = pd.read_pickle(input_filepath)

# %%
emerge_lv_trait_assocs.shape

# %%
emerge_lv_trait_assocs.head()

# %% [markdown]
# ## eMERGE traits info

# %%
input_filepath = conf.EMERGE["DESC_FILE_WITH_SAMPLE_SIZE"]
display(input_filepath)

# %%
emerge_traits_info = pd.read_csv(
    input_filepath,
    sep="\t",
    dtype={"phecode": str},
    usecols=[
        "phecode",
        "phenotype",
        "category",
        "eMERGE_III_EUR_case",
        "eMERGE_III_EUR_control",
    ],
)

# %%
emerge_traits_info["phecode"] = emerge_traits_info["phecode"].apply(
    lambda x: f"EUR_{x}"
)

# %%
emerge_traits_info = emerge_traits_info.set_index("phecode").sort_index()

# %%
emerge_traits_info = emerge_traits_info.rename(
    columns={
        "eMERGE_III_EUR_case": "eur_n_cases",
        "eMERGE_III_EUR_control": "eur_n_controls",
    }
)

# %%
emerge_traits_info.shape

# %%
emerge_traits_info.head()

# %%
assert emerge_traits_info.index.is_unique

# %% [markdown]
# # Trait associations


# %% [markdown]
# ## PhenomeXcan

# %%
from traits import SHORT_TRAIT_NAMES

# %%
result_set = "phenomexcan"

# %%
def get_trait_objs(phenotype_full_code):
    if Trait.is_efo_label(phenotype_full_code):
        traits = Trait.get_traits_from_efo(phenotype_full_code)
    else:
        traits = [Trait.get_trait(code=phenotype_full_code)]

    # sort by sample size
    return sorted(traits, key=lambda x: x.n_cases / x.n, reverse=True)


def get_trait_description(phenotype_full_code):
    traits = get_trait_objs(phenotype_full_code)

    desc = traits[0].description
    if desc in SHORT_TRAIT_NAMES:
        return SHORT_TRAIT_NAMES[desc]

    return desc


def get_trait_n(phenotype_full_code):
    traits = get_trait_objs(phenotype_full_code)

    return traits[0].n


def get_trait_n_cases(phenotype_full_code):
    traits = get_trait_objs(phenotype_full_code)

    return traits[0].n_cases


def num_to_int_str(num):
    if pd.isnull(num):
        return ""

    return f"{num:,.0f}"


def get_part_clust(row):
    return f"{row.part_k} / {row.cluster_id}"


# %%
lv_assocs = phenomexcan_lv_trait_assocs[
    (phenomexcan_lv_trait_assocs["lv"] == LV_NAME)
    & (phenomexcan_lv_trait_assocs["fdr"] < 0.05)
].sort_values("fdr")

# %%
with pd.option_context(
    "display.max_rows", None, "display.max_columns", None, "display.max_colwidth", None
):
    display(lv_assocs)

# %%
lv_assocs = lv_assocs.assign(
    phenotype_desc=lv_assocs["phenotype"].apply(get_trait_description)
)

# %%
lv_assocs = lv_assocs.assign(n=lv_assocs["phenotype"].apply(get_trait_n))

# %%
lv_assocs = lv_assocs.assign(n_cases=lv_assocs["phenotype"].apply(get_trait_n_cases))

# %%
# lv_assocs = lv_assocs.assign(coef=lv_assocs["coef"].apply(lambda x: f"{x:.3f}"))

# %%
lv_assocs = lv_assocs.assign(
    fdr=lv_assocs["fdr"].apply(lambda x: f"{x:.2e}".replace("-", "&#8209;"))
)

# %%
lv_assocs = lv_assocs.assign(n=lv_assocs["n"].apply(num_to_int_str))

# %%
lv_assocs = lv_assocs.assign(n_cases=lv_assocs["n_cases"].apply(num_to_int_str))

# %%
# lv_assocs = lv_assocs.assign(part_clust="")  # lv_assocs.apply(get_part_clust, axis=1))

# %%
lv_assocs = lv_assocs.drop(columns=["phenotype"])

# %%
lv_assocs.shape

# %%
lv_assocs = lv_assocs[["phenotype_desc", "n", "n_cases", "fdr"]]

# %%
lv_assocs = lv_assocs.rename(
    columns={
        "part_clust": "Partition / cluster",
        "lv": "Latent variable (LV)",
        #         "coef": r"$\beta$",
        "fdr": "FDR",
        "phenotype_desc": "Trait description",
        "n": "Sample size",
        "n_cases": "Cases",
    }
)

# %%
with pd.option_context(
    "display.max_rows", None, "display.max_columns", None, "display.max_colwidth", None
):
    display(lv_assocs)

# %% [markdown]
# ### Fill empty

# %%
if lv_assocs.shape[0] == 0:
    lv_assocs.loc[0, "Trait description"] = "No significant associations"
    lv_assocs = lv_assocs.fillna("")

# %% [markdown]
# ### Save

# %%
# start
lv_file_mark_start = LV_FILE_MARK_TEMPLATE.format(
    result_set=result_set, lv=LV_NAME, position="start"
)
display(lv_file_mark_start)

# end
lv_file_mark_end = LV_FILE_MARK_TEMPLATE.format(
    result_set=result_set, lv=LV_NAME, position="end"
)
display(lv_file_mark_end)

# %%
new_content = lv_assocs.to_markdown(index=False, disable_numparse=True)

# %%
# add table caption
table_caption = TABLE_CAPTION.format(
    lv_name=LV_NAME,
    result_set_name=RESULT_SET_NAMES[result_set],
    table_id="{"
    + TABLE_CAPTION_ID.format(result_set=result_set, lv_name_lower_case=LV_NAME.lower())
    + "}",
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
# ## eMERGE

# %%
result_set = "emerge"

# %%
lv_assocs = emerge_lv_trait_assocs[
    (emerge_lv_trait_assocs["lv"] == LV_NAME) & (emerge_lv_trait_assocs["fdr"] < 0.05)
].sort_values("fdr")

# %%
with pd.option_context(
    "display.max_rows", None, "display.max_columns", None, "display.max_colwidth", None
):
    display(lv_assocs)

# %%
lv_assocs = lv_assocs.assign(
    phenotype_desc=lv_assocs["phenotype"].apply(
        lambda x: emerge_traits_info.loc[x, "phenotype"]
    )
)

# %%
lv_assocs = lv_assocs.assign(
    n=lv_assocs["phenotype"].apply(
        lambda x: emerge_traits_info.loc[x, ["eur_n_cases", "eur_n_controls"]].sum()
    )
)

# %%
lv_assocs = lv_assocs.assign(
    n_cases=lv_assocs["phenotype"].apply(
        lambda x: emerge_traits_info.loc[x, "eur_n_cases"]
    )
)

# %%
lv_assocs["phenotype"] = lv_assocs["phenotype"].apply(lambda x: x.split("EUR_")[1])

# %%
# lv_assocs = lv_assocs.assign(coef=lv_assocs["coef"].apply(lambda x: f"{x:.3f}"))

# %%
lv_assocs = lv_assocs.assign(
    fdr=lv_assocs["fdr"].apply(lambda x: f"{x:.2e}".replace("-", "&#8209;"))
)

# %%
lv_assocs = lv_assocs.assign(n=lv_assocs["n"].apply(num_to_int_str))

# %%
lv_assocs = lv_assocs.assign(n_cases=lv_assocs["n_cases"].apply(num_to_int_str))

# %%
lv_assocs = lv_assocs.rename(columns={"phenotype": "phecode"})

# %%
lv_assocs.shape

# %%
lv_assocs = lv_assocs[["phecode", "phenotype_desc", "n", "n_cases", "fdr"]]

# %%
lv_assocs = lv_assocs.rename(
    columns={
        "lv": "Latent variable (LV)",
        #         "coef": r"$\beta$",
        "fdr": "FDR",
        "phecode": "Phecode",
        "phenotype_desc": "Trait description",
        "n": "Sample size",
        "n_cases": "Cases",
    }
)

# %%
with pd.option_context(
    "display.max_rows", None, "display.max_columns", None, "display.max_colwidth", None
):
    display(lv_assocs)

# %% [markdown]
# ### Fill empty

# %%
if lv_assocs.shape[0] == 0:
    lv_assocs = pd.DataFrame(columns=lv_assocs.columns.copy()).astype(str)
    lv_assocs.loc[0, "Phecode"] = "No significant associations"
    lv_assocs = lv_assocs.fillna("")

    display(lv_assocs)

# %% [markdown]
# ### Save

# %%
# start
lv_file_mark_start = LV_FILE_MARK_TEMPLATE.format(
    result_set=result_set, lv=LV_NAME, position="start"
)
display(lv_file_mark_start)

# end
lv_file_mark_end = LV_FILE_MARK_TEMPLATE.format(
    result_set=result_set, lv=LV_NAME, position="end"
)
display(lv_file_mark_end)

# %%
new_content = lv_assocs.to_markdown(index=False, disable_numparse=True)

# %%
# add table caption
table_caption = TABLE_CAPTION.format(
    lv_name=LV_NAME,
    result_set_name=RESULT_SET_NAMES[result_set],
    table_id="{"
    + TABLE_CAPTION_ID.format(result_set=result_set, lv_name_lower_case=LV_NAME.lower())
    + "}",
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
