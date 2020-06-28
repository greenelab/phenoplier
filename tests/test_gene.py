import pytest

from entity import Gene


class TestGene:
    @pytest.mark.parametrize(
        "gene_property,gene_value",
        [("ensembl_id", "id_does_not_exist"), ("name", "name_does_not_exist")],
    )
    def test_gene_does_not_exist(self, gene_property, gene_value):
        try:
            Gene(**{gene_property: gene_value})
            pytest.fail("Should have failed")
        except ValueError:
            pass

    @pytest.mark.parametrize(
        "gene_id,gene_name,gene_band",
        [
            ("ENSG00000003249", "DBNDD1", "16q24.3"),
            ("ENSG00000101440", "ASIP", "20q11.22"),
        ],
    )
    def test_gene_obj_from_gene_id(self, gene_id, gene_name, gene_band):
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
