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
# (Please, take a look at the README.md file in this directory for instructions on how to run this notebook)
#
# This notebook computes predicted expression correlations between all genes in the MultiPLIER models.
#
# It also has a parameter set for papermill to run on a single chromosome to run in parallel (see under `Settings` below).
#
# This notebook is not directly run. See README.md.

# %% [markdown] tags=[]
# # Modules

# %% tags=[]
# %load_ext autoreload
# %autoreload 2

# %% tags=[]
import numpy as np
from scipy.spatial.distance import squareform
import pandas as pd
from tqdm import tqdm

import conf
from entity import Gene

# %% [markdown] tags=[]
# # Settings

# %% tags=["parameters"]
# reference panel
REFERENCE_PANEL = "GTEX_V8"
# REFERENCE_PANEL = "1000G"

# prediction models
## mashr
EQTL_MODEL = "MASHR"
EQTL_MODEL_FILES_PREFIX = "mashr_"

# ## elastic net
# EQTL_MODEL = "ELASTIC_NET"
# EQTL_MODEL_FILES_PREFIX = "en_"

# make it read the prefix from conf.py
EQTL_MODEL_FILES_PREFIX = None

# specifies a single chromosome value
# by default, run on all chromosomes
chromosome = "all"

# %%
if EQTL_MODEL_FILES_PREFIX is None:
    EQTL_MODEL_FILES_PREFIX = conf.PHENOMEXCAN["PREDICTION_MODELS"][
        f"{EQTL_MODEL}_PREFIX"
    ]

# %%
display(f"Using eQTL model: {EQTL_MODEL} / {EQTL_MODEL_FILES_PREFIX}")

# %%
REFERENCE_PANEL_DIR = conf.PHENOMEXCAN["LD_BLOCKS"][f"{REFERENCE_PANEL}_GENOTYPE_DIR"]

# %%
display(f"Using reference panel folder: {str(REFERENCE_PANEL_DIR)}")

# %%
OUTPUT_DIR_BASE = (
    conf.PHENOMEXCAN["LD_BLOCKS"][f"GENE_CORRS_DIR"]
    / REFERENCE_PANEL.lower()
    / EQTL_MODEL.lower()
)
OUTPUT_DIR_BASE.mkdir(parents=True, exist_ok=True)

# %%
display(f"Using output dir base: {OUTPUT_DIR_BASE}")

# %%
if chromosome == "all":
    from time import sleep

    message = """
    WARNING: you are going to compute correlations of gene predicted expression across all chromosomes without parallelism.
    It is recommended that you look at the README.md file in this subfolder (nbs/08_gsa_gls/README.md) to know how to do that.
    
    It will continue in 20 seconds.
    """
    print(message)
    sleep(20)

# %% [markdown] tags=[]
# # Load data

# %% [markdown] tags=[]
# ## MultiPLIER Z

# %% tags=[]
multiplier_z_genes = pd.read_pickle(
    conf.MULTIPLIER["MODEL_Z_MATRIX_FILE"]
).index.tolist()

# %% tags=[]
len(multiplier_z_genes)

# %% tags=[]
multiplier_z_genes[:10]

# %% [markdown] tags=[]
# ## Get gene objects

# %% tags=[]
multiplier_gene_obj = {
    gene_name: Gene(name=gene_name)
    for gene_name in multiplier_z_genes
    if gene_name in Gene.GENE_NAME_TO_ID_MAP
}

# %% tags=[]
len(multiplier_gene_obj)

# %% tags=[]
multiplier_gene_obj["GAS6"].ensembl_id

# %% tags=[]
_gene_obj = list(multiplier_gene_obj.values())

genes_info = pd.DataFrame(
    {
        "name": [g.name for g in _gene_obj],
        "id": [g.ensembl_id for g in _gene_obj],
        "chr": [g.chromosome for g in _gene_obj],
        "band": [g.band for g in _gene_obj],
    }
)

# %% tags=[]
genes_info.shape

# %% tags=[]
genes_info.head()

# %% [markdown] tags=[]
# ## Get tissues names

# %% tags=[]
tissues = conf.PHENOMEXCAN["PREDICTION_MODELS"][f"{EQTL_MODEL}_TISSUES"].split(" ")

# %% tags=[]
tissues[:5]

# %% tags=[]
assert len(tissues) == 49

# %% [markdown] tags=[]
# # Test

# %%
genes_info[genes_info["chr"] == "13"]

# %%
_gene_list = [
    Gene("ENSG00000134871"),
    Gene("ENSG00000187498"),
    Gene("ENSG00000183087"),
    Gene("ENSG00000073910"),
    Gene("ENSG00000133101"),
    Gene("ENSG00000122025"),
    Gene("ENSG00000120659"),
    Gene("ENSG00000133116"),
]

tissue = "Whole_Blood"

# %%
# %%timeit
for gene_idx1 in range(0, len(_gene_list) - 1):
    gene_obj1 = _gene_list[gene_idx1]

    for gene_idx2 in range(gene_idx1 + 1, len(_gene_list)):
        gene_obj2 = _gene_list[gene_idx2]

        c = gene_obj1.get_ssm_correlation(
            gene_obj2,
        )

        print(f"{gene_obj1.name} / {gene_obj2.name}: {c}")

# %% [markdown] tags=[]
# # Compute correlation per chromosome

# %%
import warnings

warnings.filterwarnings("error")

# %%
all_chrs = genes_info["chr"].dropna().unique()
assert all_chrs.shape[0] == 22

if chromosome != "all":
    chromosome = str(chromosome)
    assert chromosome in all_chrs

    # run only on the chromosome specified
    all_chrs = [chromosome]

# # For testing purposes
# all_chrs = ["13"]
# # tissues = ["Whole_Blood"]
# genes_info = genes_info[genes_info["id"].isin(["ENSG00000134871", "ENSG00000187498", "ENSG00000183087", "ENSG00000073910"])]


for chr_num in all_chrs:
    print(f"Chromosome {chr_num}", flush=True)

    # check if results exist
    output_dir = OUTPUT_DIR_BASE / "by_chr"
    output_file = output_dir / f"gene_corrs-chr{chr_num}.pkl"

    if output_file.exists():
        _tmp_data = pd.read_pickle(output_file)

        if _tmp_data.shape[0] > 0:
            print("Already run, stopping.")
            continue

    genes_chr = genes_info[genes_info["chr"] == chr_num]
    print(f"Genes in chromosome{genes_chr.shape}", flush=True)

    gene_chr_objs = [Gene(ensembl_id=gene_id) for gene_id in genes_chr["id"]]
    gene_chr_ids = [g.ensembl_id for g in gene_chr_objs]

    n = len(gene_chr_objs)
    n_comb = int(n * (n - 1) / 2.0)
    print(f"Number of gene combinations: {n_comb}", flush=True)

    gene_corrs = []

    pbar = tqdm(ncols=100, total=n_comb)
    i = 0
    for gene_idx1 in range(0, len(gene_chr_objs) - 1):
        gene_obj1 = gene_chr_objs[gene_idx1]

        # FIXME: get tissues for which we have results for gene_obj1 only from S-PrediXcan

        for gene_idx2 in range(gene_idx1 + 1, len(gene_chr_objs)):
            gene_obj2 = gene_chr_objs[gene_idx2]

            pbar.set_description(f"{gene_obj1.ensembl_id} / {gene_obj2.ensembl_id}")

            # FIXME: get tissues for which we have results for gene_obj2 only from S-PrediXcan

            try:
                gene_corrs.append(
                    gene_obj1.get_ssm_correlation(
                        other_gene=gene_obj2,
                        # tissues=tissues, FIXME
                        reference_panel=REFERENCE_PANEL,
                        model_type=EQTL_MODEL,
                    )
                )
            except Warning:
                print(
                    f"RuntimeWarning for genes {gene_obj1.ensembl_id} and {gene_obj2.ensembl_id}",
                    flush=True,
                )
                import traceback

                print(traceback.format_exc(), flush=True)
            except Exception as e:
                print(
                    f"Exception for genes {gene_obj1.ensembl_id} and {gene_obj2.ensembl_id}",
                    flush=True,
                )
                import traceback

                print(traceback.format_exc(), flush=True)
                gene_corrs.append(np.nan)

            pbar.update(1)

    pbar.close()

    # testing
    gene_corrs_flat = pd.Series(gene_corrs)
    print(f"Min/max values: {gene_corrs_flat.min()} / {gene_corrs_flat.max()}")
    assert gene_corrs_flat.min() >= -1.001
    assert gene_corrs_flat.max() <= 1.001

    # save
    # FIXME: consider saving only the condenced matrix here. See here for
    # more details: https://github.com/greenelab/phenoplier/pull/38#discussion_r634600813
    gene_corrs_data = squareform(np.array(gene_corrs, dtype=np.float64))
    np.fill_diagonal(gene_corrs_data, 1.0)

    gene_corrs_df = pd.DataFrame(
        data=gene_corrs_data,
        index=gene_chr_ids,
        columns=gene_chr_ids,
    )

    # FIXME: all values should be between 1.0 and -1.0 (change then if not)

    output_dir.mkdir(exist_ok=True, parents=True)
    display(output_file)

    gene_corrs_df.to_pickle(output_file)

# %%
gene_corrs_df.shape

# %%
gene_corrs_df

# %% [markdown]
# # Testing

# %% tags=[]
# data = pd.read_pickle(
#     conf.PHENOMEXCAN["LD_BLOCKS"]["BASE_DIR"] / "gene_corrs" / "Whole_Blood" / "gene_corrs-Whole_Blood-chr13.pkl"
# )

# %%
# assert data.loc["ENSG00000134871", "ENSG00000187498"] > 0.97

# %%
