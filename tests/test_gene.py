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
    "tissue",
    [
        "Whole_Blood",
        "Testis",
    ],
)
def test_gene_get_tissue_connection(tissue):
    con = Gene._get_tissue_connection(tissue)
    df = pd.read_sql(sql="select * from weights", con=con)
    assert df is not None
    assert df.shape[0] > 100
    assert df.shape[1] > 3


@pytest.mark.xfail(raises=ValueError)
def test_gene_get_tissue_connection_tissue_does_not_exist():
    Gene._get_tissue_connection("NonExistent")


@pytest.mark.parametrize(
    "tissue",
    [
        "Whole_Blood",
        "Testis",
    ],
)
def test_gene_get_prediction_weights(tissue):
    w = Gene._get_prediction_weights(tissue)
    assert w is not None

    assert "gene" in w.columns
    assert "varID" in w.columns
    assert "weight" in w.columns

    # check gene column has right format
    gene_id_length = w["gene"].apply(len)
    g_id_unique = gene_id_length.unique()
    assert g_id_unique.shape[0] == 1
    assert gene_id_length[0] == 15


@pytest.mark.parametrize(
    "tissue",
    [
        "Whole_Blood",
        "Testis",
    ],
)
def test_gene_get_snps_covariances(tissue):
    df = Gene._get_snps_covariances(tissue)
    assert df is not None

    assert "GENE" in df.columns
    assert "RSID1" in df.columns
    assert "RSID2" in df.columns
    assert "VALUE" in df.columns

    # check gene column has right format
    gene_id_length = df["GENE"].apply(len)
    g_id_unique = gene_id_length.unique()
    assert g_id_unique.shape[0] == 1
    assert gene_id_length[0] == 15


@pytest.mark.parametrize(
    "gene_id,tissue,expected_var",
    [
        # FIXME add real expected values
        ("ENSG00000003249", "Whole_Blood", -0.10),
        ("ENSG00000003249", "Testis", -0.10),
        ("ENSG00000238142", "Adipose_Subcutaneous", -0.10),
        ("ENSG00000238142", "Colon_Transverse", -0.10),
    ],
)
def test_gene_get_pred_expression_variance(gene_id, tissue, expected_var):
    g = Gene(ensembl_id=gene_id)
    g_var = g.get_pred_expression_variance(tissue)
    assert g_var is not None
    assert isinstance(g_var, float)
    assert round(g_var, 5) == expected_var


def test_gene_get_pred_expression_variance_gene_not_in_tissue():
    g = Gene(ensembl_id="ENSG00000183087")
    g_var = g.get_pred_expression_variance("Brain_Cerebellar_Hemisphere")
    assert g_var is None
