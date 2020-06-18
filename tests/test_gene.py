import pandas as pd
import unittest

from entity import Gene


class GeneTest(unittest.TestCase):
    def test_gene_id_does_not_exist(self):
        try:
            gene = Gene(ensembl_id='does_not_exist')
            self.fail('Should have failed')
        except ValueError:
            pass

    def test_gene_name_does_not_exist(self):
        try:
            gene = Gene(name='does_not_exist')
            self.fail('Should have failed')
        except ValueError:
            pass

    def test_gene_obj_from_gene_id(self):
        gene = Gene(ensembl_id='ENSG00000003249')
        assert gene is not None
        assert gene.ensembl_id == 'ENSG00000003249'
        assert gene.name == 'DBNDD1'
        assert gene.band == '16q24.3'

    def test_gene_obj_from_gene_id_from_another_band(self):
        gene = Gene(ensembl_id='ENSG00000101440')
        assert gene is not None
        assert gene.ensembl_id == 'ENSG00000101440'
        assert gene.name == 'ASIP'
        assert gene.band == '20q11.22'

    def test_gene_obj_from_gene_name(self):
        gene = Gene(name='DBNDD1')
        assert gene is not None
        assert gene.ensembl_id == 'ENSG00000003249'
        assert gene.name == 'DBNDD1'
        assert gene.band == '16q24.3'

    def test_gene_obj_from_gene_name_from_another_band(self):
        gene = Gene(name='ASIP')
        assert gene is not None
        assert gene.ensembl_id == 'ENSG00000101440'
        assert gene.name == 'ASIP'
        assert gene.band == '20q11.22'
