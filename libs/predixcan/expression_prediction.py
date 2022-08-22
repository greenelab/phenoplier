"""
This file contains function to predict gene expression from genotype.
"""
import sqlite3
from functools import lru_cache

from fastparquet import ParquetFile
import pandas as pd

import conf
from entity import Gene


def get_gene_prediction_weights(gene_obj, gene_tissues, debug_messages: bool = True):
    base_prediction_model_dir = conf.PHENOMEXCAN["PREDICTION_MODELS"]["MASHR"]

    all_genes_data = []

    for tissue_name in gene_tissues:
        if debug_messages:
            print(tissue_name)

        input_db_file = base_prediction_model_dir / f"mashr_{tissue_name}.db"
        with sqlite3.connect(input_db_file) as cnx:
            gene0 = pd.read_sql_query(
                f'select * from weights where gene like "{gene_obj.ensembl_id}.%"', cnx
            )
            if gene0.shape[0] > 0:
                gene0["tissue"] = tissue_name
                all_genes_data.append(gene0)

    if len(all_genes_data) == 0:
        raise ValueError(f"No predictor SNPs for gene {gene_obj.ensembl_id}")

    return all_genes_data


@lru_cache(maxsize=1)
def load_genotypes_from_chr(
    chromosome: int, reference_panel: str, snps_subset: frozenset = None
):
    base_reference_panel_dir = conf.PHENOMEXCAN["LD_BLOCKS"][
        f"{reference_panel}_GENOTYPE_DIR"
    ]

    if reference_panel == "1000G":
        chr_file_template = "chr{chromosome}.variants.parquet"
    elif reference_panel == "GTEX_V8":
        chr_file_template = "gtex_v8_eur_filtered_maf0.01_monoallelic_variants.chr{chromosome}.variants.parquet"
    else:
        raise ValueError(f"Invalid reference panel: {reference_panel}")

    chr_parquet_file = base_reference_panel_dir / chr_file_template.format(
        chromosome=chromosome
    )
    pf = ParquetFile(str(chr_parquet_file))
    pf_variants = set(pf.columns)
    if snps_subset is not None:
        pf_variants = snps_subset.intersection(pf_variants)

    # get individual level data
    ind_data = pd.read_parquet(
        chr_parquet_file,
        columns=["individual"] + list(pf_variants),
    )

    return ind_data, pf_variants


def predict_expression(
    gene0_id,
    gene1_id,
    gene0_tissues=None,
    gene1_tissues=None,
    snps_subset: frozenset = None,
    reference_panel: str = "1000G",
    center_gene_expr=False,
    genotypes_filepath: str = None,
    debug_messages: bool = True,
):
    if gene0_tissues is None:
        gene0_tissues = conf.PHENOMEXCAN["PREDICTION_MODELS"]["MASHR_TISSUES"].split(
            " "
        )
        gene0_tissues = sorted(gene0_tissues)

    if gene0_id != gene1_id:
        if gene1_tissues is None:
            gene1_tissues = conf.PHENOMEXCAN["PREDICTION_MODELS"][
                "MASHR_TISSUES"
            ].split(" ")
            gene1_tissues = sorted(gene1_tissues)
    else:
        gene1_tissues = gene0_tissues

    # get genes' chromosomes
    gene0_obj = Gene(ensembl_id=gene0_id)
    if gene0_id != gene1_id:
        gene1_obj = Gene(ensembl_id=gene1_id)
        assert gene0_obj.chromosome == gene1_obj.chromosome
    else:
        gene1_obj = gene0_obj

    if debug_messages:
        print(f"Genes chromosome: {gene0_obj.chromosome}")

    # get gene prediction weights
    gene0_data = get_gene_prediction_weights(gene0_obj, gene0_tissues, debug_messages)
    if gene0_id != gene1_id:
        gene1_data = get_gene_prediction_weights(
            gene1_obj, gene1_tissues, debug_messages
        )
    else:
        gene1_data = gene0_data

    if gene0_id != gene1_id:
        all_genes_data = pd.concat(gene0_data + gene1_data, axis=0)
    else:
        all_genes_data = pd.concat(gene0_data, axis=0)
    assert not all_genes_data.isna().any().any()

    # get gene variants
    gene_variants = list(set(all_genes_data["varID"].tolist()))
    if debug_messages:
        print(f"Number of unique variants: {len(gene_variants)}")
    # keep only variants in snps_subset
    if snps_subset is not None:
        gene_variants = list(snps_subset.intersection(gene_variants))
        if debug_messages:
            print(f"Number of variants after filtering: {len(gene_variants)}")

    if genotypes_filepath is not None:
        ind_data = pd.read_pickle(genotypes_filepath)
        pf_variants = set([c for c in ind_data.columns if c.startswith("chr")])
    else:
        ind_data, pf_variants = load_genotypes_from_chr(
            chromosome=int(gene0_obj.chromosome),
            reference_panel=reference_panel,
            snps_subset=frozenset(gene_variants),
        )

    gene_variants = [gv for gv in gene_variants if gv in pf_variants]

    all_genes_data = all_genes_data[all_genes_data["varID"].isin(gene_variants)]
    if debug_messages:
        print(all_genes_data)

    ind_data = ind_data[["individual"] + gene_variants]

    # predict expression for the two genes
    def _predict_expression(gene_id, tissue_name):
        gene_data = all_genes_data[
            all_genes_data["gene"].str.startswith(gene_id + ".")
            & (all_genes_data["tissue"] == tissue_name)
        ].drop_duplicates(
            subset=["gene", "varID", "tissue"]
        )  # needed when the same gene/tissues are given

        gene_expr = (
            ind_data[["individual"] + gene_data["varID"].tolist()].set_index(
                "individual"
            )
            @ gene_data[["varID", "weight"]].set_index("varID")
        ).squeeze()

        if gene_expr.sum() == 0.0:
            return None

        if center_gene_expr:
            gene_expr = gene_expr - gene_expr.mean()

        return gene_expr

    gene0_pred_expr = pd.DataFrame(
        {t: _predict_expression(gene0_id, t) for t in gene0_tissues}
    ).dropna(how="all", axis=1)
    assert not gene0_pred_expr.isna().any().any()

    if gene0_id != gene1_id:
        gene1_pred_expr = pd.DataFrame(
            {t: _predict_expression(gene1_id, t) for t in gene1_tissues}
        ).dropna(how="all", axis=1)
        assert not gene1_pred_expr.isna().any().any()
    else:
        gene1_pred_expr = gene0_pred_expr

    return gene0_pred_expr, gene1_pred_expr
