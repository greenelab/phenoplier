"""
This file is used to compute the real correlation between predicted gene
expression for a pair of genes in a given tissue.

Modify parameters section below to change the genes or tissue.
"""

import sqlite3
import pandas as pd

import conf

# Parameter (change this as needed)
#   Genes should be in the same chromosome.
gene0_id = "ENSG00000169750"
gene1_id = "ENSG00000121101"
tissue_name = "Brain_Cortex"

# get gene prediction weights
base_prediction_model_dir = conf.PHENOMEXCAN["PREDICTION_MODELS"]["MASHR"]
input_db_file = base_prediction_model_dir / f"mashr_{tissue_name}.db"
with sqlite3.connect(input_db_file) as cnx:
    gene0 = pd.read_sql_query(
        f'select * from weights where gene like "{gene0_id}.%"', cnx
    )
    gene1 = pd.read_sql_query(
        f'select * from weights where gene like "{gene1_id}.%"', cnx
    )

# get genes' chromosomes
gene0_chr = gene0["varID"].str.split("_", n=1, expand=True)[0].unique()
assert len(gene0_chr) == 1
gene0_chr = gene0_chr[0]

gene1_chr = gene1["varID"].str.split("_", n=1, expand=True)[0].unique()
assert len(gene1_chr) == 1
gene1_chr = gene1_chr[0]

assert gene0_chr == gene1_chr

# get gene variants
gene_variants = gene0["varID"].tolist() + gene1["varID"].tolist()
assert len(gene_variants) == len(set(gene_variants))

# get individual level data
base_reference_panel_dir = conf.PHENOMEXCAN["LD_BLOCKS"]["1000G_GENOTYPE_DIR"]
ind_data = pd.read_parquet(
    base_reference_panel_dir / f"{gene0_chr}.variants.parquet",
    columns=["individual"] + gene_variants,
)

# predict expression for the two genes
def _predict_expression(gene_data):
    gene_expr = (
        ind_data[["individual"] + gene_data["varID"].tolist()].set_index("individual")
        @ gene_data[["varID", "weight"]].set_index("varID")
    ).squeeze()

    assert gene_expr.min() < gene_expr.max()

    return gene_expr


gene0_pred_expr = _predict_expression(gene0)
gene1_pred_expr = _predict_expression(gene1)

# compute the real correlation
gene_corr = gene0_pred_expr.corr(gene1_pred_expr)
print(f"{gene0_id} / {gene1_id} in {tissue_name}: {gene_corr}")
