import pandas as pd
import pytest

from entity import Gene


@pytest.mark.parametrize(
    "gene_property,gene_value",
    [("ensembl_id", "id_does_not_exist"), ("name", "name_does_not_exist")],
)
def test_gene_does_not_exist(gene_property, gene_value):
    with pytest.raises(ValueError):
        Gene(**{gene_property: gene_value})


@pytest.mark.parametrize(
    "gene_id,gene_name,gene_band",
    [
        ("ENSG00000003249", "DBNDD1", "16q24.3"),
        ("ENSG00000101440", "ASIP", "20q11.22"),
    ],
)
def test_gene_obj_from_gene_id(gene_id, gene_name, gene_band):
    gene = Gene(ensembl_id=gene_id)
    assert gene is not None
    assert gene.ensembl_id == gene_id
    assert gene.name == gene_name
    assert gene.band == gene_band

    gene = Gene(name=gene_name)
    assert gene is not None
    assert gene.ensembl_id == gene_id
    assert gene.name == gene_name
    assert gene.band == gene_band


@pytest.mark.parametrize(
    "gene_id,gene_name,gene_chr",
    [
        ("ENSG00000003249", "DBNDD1", "16"),
        ("ENSG00000101440", "ASIP", "20"),
    ],
)
def test_gene_obj_chromosome(gene_id, gene_name, gene_chr):
    gene = Gene(ensembl_id=gene_id)
    assert gene is not None
    assert gene.ensembl_id == gene_id
    assert gene.name == gene_name
    assert gene.chromosome == gene_chr

    gene = Gene(name=gene_name)
    assert gene is not None
    assert gene.ensembl_id == gene_id
    assert gene.name == gene_name
    assert gene.chromosome == gene_chr


@pytest.mark.parametrize(
    "tissue",
    [
        "Whole_Blood",
        "Testis",
    ],
)
def test_gene_get_tissue_connection(tissue):
    con = Gene._get_tissue_connection(tissue, model_type="MASHR")
    df = pd.read_sql(sql="select * from weights", con=con)
    assert df is not None
    assert df.shape[0] > 100
    assert df.shape[1] > 3


def test_gene_get_tissue_connection_tissue_does_not_exist():
    with pytest.raises(ValueError) as e:
        Gene._get_tissue_connection("NonExistent", model_type="MASHR")


@pytest.mark.parametrize(
    "gene_id,tissue,expected_snps_weights",
    [
        (
            "ENSG00000003249",
            "Whole_Blood",
            {
                "chr16_89979578_A_G_b38": 0.341560385316961,
                "chr16_90013161_G_C_b38": 0.493600048074392,
            },
        ),
        (
            "ENSG00000003249",
            "Testis",
            {
                "chr16_90015252_A_G_b38": 0.24632078046807,
            },
        ),
    ],
)
def test_gene_get_prediction_weights(gene_id, tissue, expected_snps_weights):
    w = Gene(ensembl_id=gene_id).get_prediction_weights(tissue, model_type="MASHR")
    assert w is not None
    assert w.shape[0] == len(expected_snps_weights)

    # assert "gene" in w.columns
    assert "varID" in w.columns
    assert "weight" in w.columns

    w = w.set_index("varID")
    for snp_id, snp_weight in expected_snps_weights.items():
        assert w.loc[snp_id, "weight"].round(5) == round(snp_weight, 5)


def test_gene_get_prediction_weights_empty():
    w = Gene(ensembl_id="ENSG00000183087").get_prediction_weights(
        "Whole_Blood", model_type="MASHR"
    )
    assert w is None


@pytest.mark.parametrize(
    "gene_pair,expected_snps1,expected_snps2,tissue",
    [
        (
            [
                "ENSG00000123200",
                "ENSG00000122026",
            ],
            [
                "chr13_46050849_T_C_b38",
                "chr13_46052725_T_C_b38",
            ],
            [
                "chr13_27242863_T_C_b38",
            ],
            "Whole_Blood",
        ),
        (
            [
                "ENSG00000123200",
                "ENSG00000122026",
            ],
            [
                "chr13_46029707_C_T_b38",
            ],
            [
                "chr13_27251616_G_C_b38",
                "chr13_27252069_G_A_b38",
            ],
            "Adipose_Subcutaneous",
        ),
    ],
)
def test_gene_get_snps_cov_genes_same_chromosome(
    gene_pair, expected_snps1, expected_snps2, tissue
):
    g1 = Gene(ensembl_id=gene_pair[0])
    g2 = Gene(ensembl_id=gene_pair[1])

    g1_snps = g1.get_prediction_weights(tissue, model_type="MASHR")["varID"]
    g2_snps = g2.get_prediction_weights(tissue, model_type="MASHR")["varID"]

    df = Gene._get_snps_cov(g1_snps, g2_snps)
    assert df is not None
    assert df.shape[0] == g1_snps.shape[0] == len(expected_snps1)
    assert df.shape[1] == g2_snps.shape[0] == len(expected_snps2)
    assert not df.isna().any().any()
    assert df.index.tolist() == expected_snps1
    assert df.columns.tolist() == expected_snps2


@pytest.mark.parametrize(
    "gene_pair,expected_snps1,expected_snps2,tissue",
    [
        (
            [
                "ENSG00000166821",
                "ENSG00000140545",
            ],
            [
                "chr15_89690738_G_C_b38",
            ],
            [
                "chr15_88913717_G_A_b38",
                "chr15_88917324_A_G_b38",
                # "chr15_88945286_AAAGTGCTGAGATCCTCCTACCTCT_A_b38", this one is not present in 1000G
            ],
            "Whole_Blood",
        ),
    ],
)
def test_gene_get_snps_cov_genes_same_chromosome_some_snps_missing_cov(
    gene_pair, expected_snps1, expected_snps2, tissue
):
    g1 = Gene(ensembl_id=gene_pair[0])
    g2 = Gene(ensembl_id=gene_pair[1])

    g1_snps = g1.get_prediction_weights(tissue, model_type="MASHR")["varID"]
    g2_snps = g2.get_prediction_weights(tissue, model_type="MASHR")["varID"]

    df = Gene._get_snps_cov(g1_snps, g2_snps)
    assert df is not None
    assert df.shape[0] == len(expected_snps1)
    assert df.shape[1] == len(expected_snps2)
    assert not df.isna().any().any()
    assert df.index.tolist() == expected_snps1
    assert df.columns.tolist() == expected_snps2


@pytest.mark.parametrize(
    "gene_pair,expected_snps1,tissue",
    [
        (
            [
                "ENSG00000123200",
            ],
            [
                "chr13_46050849_T_C_b38",
                "chr13_46052725_T_C_b38",
            ],
            "Whole_Blood",
        ),
        (
            [
                "ENSG00000122026",
            ],
            [
                "chr13_27251616_G_C_b38",
                "chr13_27252069_G_A_b38",
            ],
            "Adipose_Subcutaneous",
        ),
    ],
)
def test_gene_get_snps_cov_one_gene(gene_pair, expected_snps1, tissue):
    g1 = Gene(ensembl_id=gene_pair[0])

    g1_snps = g1.get_prediction_weights(tissue, model_type="MASHR")["varID"]

    df = Gene._get_snps_cov(g1_snps)
    assert df is not None
    assert df.shape[0] == df.shape[1] == g1_snps.shape[0] == len(expected_snps1)
    assert not df.isna().any().any()
    assert df.index.tolist() == expected_snps1
    assert df.columns.tolist() == expected_snps1


def test_gene_get_snps_cov_snp_list_different_chromosomes():
    g1_snps = [
        "chr13_45120003_C_T_b38",
        "chr13_49493306_T_C_b38",
        "chr12_25064415_T_C_b38",
    ]

    with pytest.raises(ValueError) as e:
        Gene._get_snps_cov(g1_snps)


def test_gene_get_snps_cov_genes_different_chromosomes():
    tissue = "Whole_Blood"

    g1 = Gene(ensembl_id="ENSG00000123200")
    g2 = Gene(ensembl_id="ENSG00000133065")

    g1_snps = g1.get_prediction_weights(tissue, model_type="MASHR")["varID"]
    g2_snps = g2.get_prediction_weights(tissue, model_type="MASHR")["varID"]

    with pytest.raises(Exception) as e:
        Gene._get_snps_cov(g1_snps, g2_snps)


@pytest.mark.parametrize(
    "gene_id,tissue,expected_var",
    [
        # FIXME add real expected values
        (
            "ENSG00000157227",
            "Whole_Blood",
            0.001,
        ),  # FIXME THIS IS NOT THE REAL VARIANCE VALUE!
        (
            "ENSG00000157227",
            "Testis",
            0.0001,
        ),  # FIXME THIS IS NOT THE REAL VARIANCE VALUE!
        (
            "ENSG00000096696",
            "Adipose_Subcutaneous",
            0.0712,
        ),  # FIXME THIS IS NOT THE REAL VARIANCE VALUE!
        (
            "ENSG00000096696",
            "Colon_Transverse",
            0.0002,
        ),  # FIXME THIS IS NOT THE REAL VARIANCE VALUE!
    ],
)
def test_gene_get_pred_expression_variance(gene_id, tissue, expected_var):
    g = Gene(ensembl_id=gene_id)
    g_var = g.get_pred_expression_variance(tissue, model_type="MASHR")
    assert g_var is not None
    assert isinstance(g_var, float)

    assert round(g_var, 4) == round(expected_var, 4)


def test_gene_get_pred_expression_variance_gene_not_in_tissue():
    g = Gene(ensembl_id="ENSG00000183087")
    g_var = g.get_pred_expression_variance(
        "Brain_Cerebellar_Hemisphere", model_type="MASHR"
    )
    assert g_var is None


@pytest.mark.parametrize(
    "gene_id1,gene_id2,tissue,expected_corr",
    [
        # FIXME add real expected values
        # case where some snps in the second gene are not in covariance snp matrix
        (
            "ENSG00000166821",
            "ENSG00000140545",
            "Whole_Blood",
            0.0477,
        ),  # FIXME THIS IS NOT THE REAL CORRELATION VALUE!
        # case where all snps are in cov matrix
        (
            "ENSG00000121101",
            "ENSG00000169750",
            "Brain_Cortex",
            0.05,
        ),  # FIXME THIS IS NOT THE REAL CORRELATION VALUE!
        # case of highly correlated genes
        (
            "ENSG00000134871",
            "ENSG00000187498",
            "Whole_Blood",
            0.9702,
        ),  # FIXME check if this value is right, not sure how to do that
        # case with more than 1 snps in each gene
        # FIXME ADD
        # corr of genes from different chromosomes is always zero
        # (same gene pair across different tissues)
        ("ENSG00000166821", "ENSG00000121101", "Whole_Blood", 0.00),
        ("ENSG00000166821", "ENSG00000121101", "Brain_Cortex", 0.00),
        ("ENSG00000166821", "ENSG00000121101", "Brain_Spinal_cord_cervical_c-1", 0.00),
        # corr of genes from different chromosomes is always zero
        ("ENSG00000134871", "ENSG00000169750", "Whole_Blood", 0.00),
        ("ENSG00000134871", "ENSG00000169750", "Brain_Cortex", 0.00),
        ("ENSG00000134871", "ENSG00000169750", "Testis", 0.00),
    ],
)
def test_gene_get_expression_correlation(gene_id1, gene_id2, tissue, expected_corr):
    gene1 = Gene(ensembl_id=gene_id1)
    gene2 = Gene(ensembl_id=gene_id2)

    genes_corr = gene1.get_expression_correlation(gene2, tissue)
    assert genes_corr is not None
    assert isinstance(genes_corr, float)
    assert 0.0 <= genes_corr <= 1.0

    # correlation should be asymmetric
    genes_corr_2 = gene2.get_expression_correlation(gene1, tissue)
    assert genes_corr_2 == genes_corr

    # correlation with itself should be 1.0
    assert gene1.get_expression_correlation(gene1, tissue) == 1.0
    assert gene2.get_expression_correlation(gene2, tissue) == 1.0

    assert round(genes_corr, 4) == round(expected_corr, 4)


@pytest.mark.parametrize(
    "gene_id1,gene_id2,tissue,expected_corr",
    [
        # case where there is no model for a gene in that tissue
        # here ENSG00000166821 has no snps in Ovary
        ("ENSG00000166821", "ENSG00000121101", "Ovary", 0.00),
        # case where snps for one gene are not in snp cov matrix
        # here ENSG00000163221 has one predictor snp, but it is not in cov
        # matrix
        ("ENSG00000134686", "ENSG00000163221", "Brain_Cerebellar_Hemisphere", 0.00),
    ],
)
def test_gene_get_expression_correlation_no_prediction_models(
    gene_id1, gene_id2, tissue, expected_corr
):
    gene1 = Gene(ensembl_id=gene_id1)
    gene2 = Gene(ensembl_id=gene_id2)

    genes_corr = gene1.get_expression_correlation(gene2, tissue)
    assert genes_corr is not None
    assert isinstance(genes_corr, float)
    assert 0.0 <= genes_corr <= 1.0

    # correlation should be asymmetric
    genes_corr_2 = gene2.get_expression_correlation(gene1, tissue)
    assert genes_corr_2 == genes_corr

    assert genes_corr == expected_corr


def test_gene_get_expression_correlation_gene_weights_are_zero():
    gene1 = Gene(ensembl_id="ENSG00000134686")

    # weights for this gene in whole blood are all zero
    gene2 = Gene(ensembl_id="ENSG00000117215")

    tissue = "Whole_Blood"

    genes_corr = gene1.get_expression_correlation(gene2, tissue)
    assert genes_corr is not None
    assert isinstance(genes_corr, float)
    assert genes_corr == 0.0
