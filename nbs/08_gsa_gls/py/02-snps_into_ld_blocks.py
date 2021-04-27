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
# This notebook separates all SNPs in prediction models into LD blocks (LINK TO PAPER).

# %% [markdown] tags=[]
# # Modules

# %% tags=[]
# %load_ext autoreload
# %autoreload 2

# %% tags=[]
import numpy as np
import pandas as pd
import sqlite3

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

import conf

# %% tags=[]
load_rda = ro.r["load"]

# %% [markdown] tags=[]
# # Load data

# %% [markdown] tags=[]
# ## LD blocks

# %% tags=[]
input_file = str(conf.PHENOMEXCAN["LD_BLOCKS"]["LD_BLOCKS_FILE"])
display(input_file)

# %% tags=[]
load_rda(input_file)

# %% tags=[]
ro.r["ld_block_pickrell_eur_b38"]

# %% [markdown] tags=[]
# ### Show metadata

# %% tags=[]
ro.r["ld_block_pickrell_eur_b38"][1]

# %% tags=[]
np.array(ro.r["ld_block_pickrell_eur_b38"][1][0])

# %% tags=[]
np.array(ro.r["ld_block_pickrell_eur_b38"][1][1])

# %% [markdown] tags=[]
# ### Load LD blocks

# %% tags=[]
ld_block_r = ro.r["ld_block_pickrell_eur_b38"][0]

# %% tags=[]
ld_block_r.rownames

# %% tags=[]
ld_block_r.colnames

# %% tags=[]
with localconverter(ro.default_converter + pandas2ri.converter):
    ld_block_df = ro.conversion.rpy2py(ld_block_r)

# %% tags=[]
ld_block_df

# %% tags=[]
ld_block_df.dtypes

# %% [markdown] tags=[]
# ### Save in tsv

# %% tags=[]
output_file = conf.PHENOMEXCAN["LD_BLOCKS"]["LD_BLOCKS_FILE"].parent / (
    conf.PHENOMEXCAN["LD_BLOCKS"]["LD_BLOCKS_FILE"].stem + ".tsv"
)
display(output_file)

# %% tags=[]
ld_block_df.to_csv(output_file, sep="\t", index=False)

# %% [markdown] tags=[]
# ## SNPs in predictions models

# %% tags=[]
mashr_models_db_files = list(
    conf.PHENOMEXCAN["PREDICTION_MODELS"]["MASHR"].glob("*.db")
)

# %% tags=[]
assert len(mashr_models_db_files) == 49

# %% tags=[]
all_variants_ids = set()

# %% tags=[]
for m in mashr_models_db_files:
    print(f"Processing {m.name}")

    with sqlite3.connect(m) as conn:
        df = pd.read_sql("select varID from weights", conn)["varID"]
        all_variants_ids.update(set(df.values))

# %% tags=[]
len(all_variants_ids)

# %% tags=[]
list(all_variants_ids)[:10]

# %% [markdown] tags=[]
# # Assign each variant to an LD block

# %% [markdown] tags=[]
# ## Preprocess SNPs data

# %% tags=[]
variants_ld_block_df = pd.DataFrame({"varID": list(all_variants_ids)})

# %% tags=[]
variants_ld_block_df.shape

# %% tags=[]
variants_info = variants_ld_block_df["varID"].str.split("_", expand=True)

# %% tags=[]
variants_info.shape

# %% tags=[]
assert variants_ld_block_df.shape[0] == variants_info.shape[0]

# %% tags=[]
variants_ld_block_df = variants_ld_block_df.join(variants_info)[["varID", 0, 1, 2, 3]]

# %% tags=[]
assert variants_ld_block_df.shape[0] == variants_info.shape[0]

# %% tags=[]
variants_ld_block_df.head()

# %% tags=[]
variants_ld_block_df = variants_ld_block_df.rename(
    columns={
        0: "chr",
        1: "position",
        2: "ref_allele",
        3: "eff_allele",
    }
)

# %% tags=[]
variants_ld_block_df["chr"] = variants_ld_block_df["chr"].apply(lambda x: int(x[3:]))

# %% tags=[]
variants_ld_block_df["position"] = variants_ld_block_df["position"].astype(int)

# %% tags=[]
variants_ld_block_df.shape

# %% tags=[]
variants_ld_block_df.head()

# %% tags=[]
variants_ld_block_df.dtypes

# %% [markdown] tags=[]
# ## Assign LD blocks

# %% tags=[]
ld_block_df.shape

# %% tags=[]
ld_block_df.head()

# %% tags=[]
snps_ld_blocks = []

for index, ldb in ld_block_df.iterrows():
    snps_in = variants_ld_block_df[
        (variants_ld_block_df["chr"] == int(ldb["chromosome"][3:]))
        & (variants_ld_block_df["position"] >= ldb["start"])
        & (variants_ld_block_df["position"] < ldb["end"])
    ]
    snps_in = snps_in[["varID"]]
    snps_in = snps_in.assign(ld_block=ldb["region_name"])

    snps_ld_blocks.append(snps_in)

# %% tags=[]
display(len(snps_ld_blocks))
assert len(snps_ld_blocks) == ld_block_df.shape[0]

# %% tags=[]
all_snps_ld_blocks = pd.concat(snps_ld_blocks, ignore_index=True)

# %% tags=[]
all_snps_ld_blocks.shape

# %% tags=[]
all_snps_ld_blocks.head()

# %% tags=[]
variants_ld_block_df.shape

# %% tags=[]
_tmp0 = variants_ld_block_df.set_index("varID")
assert _tmp0.index.is_unique

_tmp1 = all_snps_ld_blocks.set_index("varID")
assert _tmp1.index.is_unique

_tmp_df = pd.merge(_tmp0, _tmp1, left_index=True, right_index=True, how="inner")

# %% tags=[]
display(_tmp_df.shape)
assert _tmp_df.shape[0] == all_snps_ld_blocks.shape[0]

# %% tags=[]
_tmp_df.head()

# %% tags=[]
variants_ld_block_df = _tmp_df

# %% [markdown] tags=[]
# # Testing

# %% tags=[]
_unique_chr_per_ld_block = variants_ld_block_df.groupby("ld_block").apply(
    lambda x: x["chr"].unique().shape[0]
)
display(_unique_chr_per_ld_block)

# %% tags=[]
display(_unique_chr_per_ld_block.unique())
assert _unique_chr_per_ld_block.unique().shape[0] == 1
assert _unique_chr_per_ld_block.unique()[0] == 1

# %% [markdown] tags=[]
# # Save

# %% tags=[]
variants_ld_block_df.head()

# %% tags=[]
output_file = conf.PHENOMEXCAN["LD_BLOCKS"]["BASE_DIR"] / "mashr_snps_ld_blocks.pkl"
display(output_file)

# %% tags=[]
variants_ld_block_df.to_pickle(output_file)

# %% tags=[]
