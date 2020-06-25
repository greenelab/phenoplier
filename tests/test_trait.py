import pandas as pd
import unittest

from entity import Trait, UKBiobankTrait, Study, GTEXGWASTrait


class TraitTest(unittest.TestCase):
    def test_ukb_trait_category_height(self):
        trait = UKBiobankTrait(code="50_raw")
        trait_cat = trait.category
        assert trait.category is not None
        assert trait_cat == "Body size measures"

    def test_ukb_trait_category_disease_first_occurence(self):
        trait = UKBiobankTrait(code="22617_1222")
        trait_cat = trait.category
        assert trait_cat is not None
        assert trait_cat == "Employment history"

    def test_ukb_trait_category_icd10(self):
        trait = UKBiobankTrait(code="G54")
        trait_cat = trait.category
        assert trait_cat is not None
        assert trait_cat == "Diseases (ICD10 main)"

    def test_ukb_trait_category_finngen(self):
        trait = UKBiobankTrait(code="M13_RHEUMATISM")
        trait_cat = trait.category
        assert trait_cat is not None
        assert trait_cat == "Diseases (FinnGen)"

    def test_ukb_trait_category_selfreported_20002_cardiovascular(self):
        trait = UKBiobankTrait(code="20002_1473")
        trait_cat = trait.category
        assert trait_cat is not None
        assert trait_cat == "Diseases (cardiovascular)", trait_cat

    def test_ukb_trait_category_selfreported_20002_neurology(self):
        trait = UKBiobankTrait(code="20002_1265")
        trait_cat = trait.category
        assert trait_cat is not None
        assert trait_cat == "Diseases (neurology/eye/psychiatry)", trait_cat

    def test_ukb_trait_category_selfreported_6150_vascular(self):
        trait = UKBiobankTrait(code="6150_1")
        trait_cat = trait.category
        assert trait_cat is not None
        assert trait_cat == "Diseases (cardiovascular)", trait_cat

    def test_ukb_trait_category_selfreported_6152_cardiovascular(self):
        trait = UKBiobankTrait(code="6152_5")
        trait_cat = trait.category
        assert trait_cat is not None
        assert trait_cat == "Diseases (cardiovascular)", trait_cat

        trait = UKBiobankTrait(code="6152_7")
        trait_cat = trait.category
        assert trait_cat is not None
        assert trait_cat == "Diseases (cardiovascular)", trait_cat

        trait = UKBiobankTrait(code="3627_raw")
        trait_cat = trait.category
        assert trait_cat is not None
        assert trait_cat == "Diseases (cardiovascular)", trait_cat

    def test_ukb_trait_category_selfreported_6152_respiratory(self):
        trait = UKBiobankTrait(code="6152_6")
        trait_cat = trait.category
        assert trait_cat is not None
        assert trait_cat == "Diseases (respiratory/ent)", trait_cat

        trait = UKBiobankTrait(code="6152_8")
        trait_cat = trait.category
        assert trait_cat is not None
        assert trait_cat == "Diseases (respiratory/ent)", trait_cat

    def test_ukb_trait_category_selfreported_6151_musculoskeletal_trauma(self):
        trait = UKBiobankTrait(code="6151_4")
        trait_cat = trait.category
        assert trait_cat is not None
        assert trait_cat == "Diseases (musculoskeletal/trauma)", trait_cat

        trait = UKBiobankTrait(code="6151_6")
        trait_cat = trait.category
        assert trait_cat is not None
        assert trait_cat == "Diseases (musculoskeletal/trauma)", trait_cat

    def test_ukb_trait_category_selfreported_6152_allergy(self):
        trait = UKBiobankTrait(code="6152_9")
        trait_cat = trait.category
        assert trait_cat is not None
        assert trait_cat == "Diseases (allergies)", trait_cat

    def test_ukb_trait_category_cancer_selfreported_20001_other(self):
        trait = UKBiobankTrait(code="20001_1068")
        trait_cat = trait.category
        assert trait_cat is not None
        assert trait_cat == "Cancer (other)", trait_cat

    def test_ukb_trait_category_cancer_selfreported_20001_gastroint(self):
        trait = UKBiobankTrait(code="20001_1020")
        trait_cat = trait.category
        assert trait_cat is not None
        assert trait_cat == "Cancer (gastrointestinal)", trait_cat

    def test_gtexgwas_trait_category_psychiatric(self):
        trait = GTEXGWASTrait(code="UKB_1160_Sleep_duration")
        trait_cat = trait.category
        assert trait_cat is not None
        assert trait_cat == "Psychiatric-neurologic"

    def test_gtexgwas_trait_category_digestive(self):
        trait = GTEXGWASTrait(
            code="UKB_20002_1154_self_reported_irritable_bowel_syndrome"
        )
        trait_cat = trait.category
        assert trait_cat is not None
        assert trait_cat == "Digestive system disease"

    def test_ukb_trait_no_cases_or_controls(self):
        pheno_from_code = UKBiobankTrait(code="50_raw")
        assert pheno_from_code is not None
        assert pheno_from_code.code == "50_raw"
        assert pheno_from_code.full_code == "50_raw-Standing_height"
        assert pheno_from_code.description == "Standing height"
        assert pheno_from_code.type == "continuous_raw"
        assert pheno_from_code.n == 360388
        assert pd.isnull(pheno_from_code.n_cases)
        assert pd.isnull(pheno_from_code.n_controls)
        assert pheno_from_code.source == "UK Biobank"
        assert pheno_from_code.study == Study.UK_BIOBANK
        assert pheno_from_code.get_plain_name() == "50_raw-Standing_height"

        pheno_from_full_code = UKBiobankTrait(
            full_code=pheno_from_code.get_plain_name()
        )
        assert pheno_from_code.code == pheno_from_full_code.code
        assert pheno_from_code.description == pheno_from_full_code.description
        assert pheno_from_code.type == pheno_from_full_code.type
        assert pheno_from_code.n == pheno_from_full_code.n
        assert pd.isnull(pheno_from_full_code.n_cases)
        assert pd.isnull(pheno_from_full_code.n_controls)
        assert pheno_from_code.source == pheno_from_full_code.source
        assert pheno_from_code.study == pheno_from_full_code.study
        assert pheno_from_code.get_plain_name() == pheno_from_full_code.get_plain_name()

    def test_ukb_trait_with_cases_and_controls(self):
        pheno_from_code = UKBiobankTrait(code="G54")
        assert pheno_from_code is not None
        assert pheno_from_code.code == "G54"
        assert (
            pheno_from_code.full_code
            == "G54-Diagnoses_main_ICD10_G54_Nerve_root_and_plexus_disorders"
        )
        assert (
            pheno_from_code.full_code
            == "G54-Diagnoses_main_ICD10_G54_Nerve_root_and_plexus_disorders"
        )
        assert (
            pheno_from_code.description
            == "Diagnoses - main ICD10: G54 Nerve root and plexus disorders"
        )
        assert pheno_from_code.type == "categorical"
        assert pheno_from_code.n == 361194
        assert pheno_from_code.n_cases == 143
        assert pheno_from_code.n_controls == 361051
        assert pheno_from_code.source == "UK Biobank"
        assert pheno_from_code.study == Study.UK_BIOBANK
        assert (
            pheno_from_code.get_plain_name()
            == "G54-Diagnoses_main_ICD10_G54_Nerve_root_and_plexus_disorders"
        )

        pheno_from_full_code = UKBiobankTrait(
            full_code=pheno_from_code.get_plain_name()
        )
        assert pheno_from_code.code == pheno_from_full_code.code
        assert pheno_from_code.description == pheno_from_full_code.description
        assert pheno_from_code.type == pheno_from_full_code.type
        assert pheno_from_code.n == pheno_from_full_code.n
        assert pheno_from_code.n_cases == pheno_from_full_code.n_cases
        assert pheno_from_code.n_controls == pheno_from_full_code.n_controls
        assert pheno_from_code.source == pheno_from_full_code.source
        assert pheno_from_code.study == pheno_from_full_code.study
        assert pheno_from_code.get_plain_name() == pheno_from_full_code.get_plain_name()

    def test_gtex_gwas_trait_no_cases_or_controls(self):
        pheno_from_code = GTEXGWASTrait(code="UKB_1160_Sleep_duration")
        assert pheno_from_code is not None
        assert pheno_from_code.code == "UKB_1160_Sleep_duration"
        assert pheno_from_code.description == "Sleep Duration UKB"
        assert pheno_from_code.type == "continuous_raw"
        assert pheno_from_code.n == 337119
        assert pd.isnull(pheno_from_code.n_cases)
        assert pd.isnull(pheno_from_code.n_controls)
        assert pheno_from_code.source == "UK Biobank"
        assert pheno_from_code.study == Study.GTEX_GWAS
        assert pheno_from_code.get_plain_name() == "UKB_1160_Sleep_duration"

    def test_gtex_gwas_trait_with_cases_and_controls(self):
        pheno_from_code = GTEXGWASTrait(
            code="UKB_20002_1094_self_reported_deep_venous_thrombosis_dvt"
        )
        assert pheno_from_code is not None
        assert (
            pheno_from_code.code
            == "UKB_20002_1094_self_reported_deep_venous_thrombosis_dvt"
        )
        assert pheno_from_code.description == "Deep Venous Thrombosis UKBS"
        assert pheno_from_code.type == "binary"
        assert pheno_from_code.n == 337119
        assert pheno_from_code.n_cases == 6767
        assert pheno_from_code.n_controls == 337119 - 6767
        assert pheno_from_code.source == "UK Biobank"
        assert pheno_from_code.study == Study.GTEX_GWAS
        assert (
            pheno_from_code.get_plain_name()
            == "UKB_20002_1094_self_reported_deep_venous_thrombosis_dvt"
        )

    def test_trait_get_trait_using_code_is_ukb(self):
        trait_code = "50_raw"

        trait = Trait.get_trait(trait_code)
        assert trait is not None
        assert isinstance(trait, UKBiobankTrait)
        assert trait.code == "50_raw"
        assert trait.get_plain_name() == "50_raw-Standing_height"

    def test_trait_get_trait_using_full_code_is_ukb(self):
        trait_code = "50_raw-Standing_height"

        trait = Trait.get_trait(full_code=trait_code)
        assert trait is not None
        assert isinstance(trait, UKBiobankTrait)
        assert trait.code == "50_raw"
        assert trait.get_plain_name() == "50_raw-Standing_height"

    def test_trait_get_trait_is_gtex_gwas(self):
        trait_code = "UKB_1160_Sleep_duration"

        trait = Trait.get_trait(trait_code)
        assert trait is not None
        assert isinstance(trait, GTEXGWASTrait)
        assert trait.code == "UKB_1160_Sleep_duration"
        assert trait.get_plain_name() == "UKB_1160_Sleep_duration"

    def test_trait_get_trait_using_full_code_is_gtex_gwas(self):
        trait_code = "UKB_1160_Sleep_duration"

        trait = Trait.get_trait(full_code=trait_code)
        assert trait is not None
        assert isinstance(trait, GTEXGWASTrait)
        assert trait.code == "UKB_1160_Sleep_duration"
        assert trait.get_plain_name() == "UKB_1160_Sleep_duration"
