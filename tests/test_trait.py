from pathlib import Path

import pytest
import numpy as np
import pandas as pd

from entity import Trait, UKBiobankTrait, Study, GTEXGWASTrait


@pytest.mark.parametrize(
    "trait_code,trait_category",
    [
        ("50_raw", "Body size measures"),
        ("22617_1222", "Employment history"),
        ("G54", "Diseases (ICD10 main)"),
        ("M13_RHEUMATISM", "Diseases (FinnGen)"),
        ("20002_1473", "Diseases (cardiovascular)"),
        ("20002_1265", "Diseases (neurology/eye/psychiatry)"),
        ("6150_1", "Diseases (cardiovascular)"),
        ("6152_5", "Diseases (cardiovascular)"),
        ("6152_7", "Diseases (cardiovascular)"),
        ("3627_raw", "Diseases (cardiovascular)"),
        ("6152_6", "Diseases (respiratory/ent)"),
        ("6152_8", "Diseases (respiratory/ent)"),
        ("6151_4", "Diseases (musculoskeletal/trauma)"),
        ("6151_6", "Diseases (musculoskeletal/trauma)"),
        ("6152_9", "Diseases (allergies)"),
        ("20001_1068", "Cancer (other)"),
        ("20001_1020", "Cancer (gastrointestinal)"),
    ],
)
def test_ukb_trait_category(trait_code, trait_category):
    trait = UKBiobankTrait(code=trait_code)
    assert trait.code == trait_code
    assert trait.category == trait_category


@pytest.mark.parametrize(
    "trait_code,trait_category",
    [
        ("UKB_1160_Sleep_duration", "Psychiatric-neurologic"),
        (
            "UKB_20002_1154_self_reported_irritable_bowel_syndrome",
            "Digestive system disease",
        ),
    ],
)
def test_gtexgwas_trait_category(trait_code, trait_category):
    trait = GTEXGWASTrait(code=trait_code)
    assert trait.code == trait_code
    assert trait.category == trait_category


def test_ukb_trait_no_cases_or_controls():
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

    pheno_from_full_code = UKBiobankTrait(full_code=pheno_from_code.get_plain_name())
    assert pheno_from_code.code == pheno_from_full_code.code
    assert pheno_from_code.description == pheno_from_full_code.description
    assert pheno_from_code.type == pheno_from_full_code.type
    assert pheno_from_code.n == pheno_from_full_code.n
    assert pd.isnull(pheno_from_full_code.n_cases)
    assert pd.isnull(pheno_from_full_code.n_controls)
    assert pheno_from_code.source == pheno_from_full_code.source
    assert pheno_from_code.study == pheno_from_full_code.study
    assert pheno_from_code.get_plain_name() == pheno_from_full_code.get_plain_name()


def test_ukb_trait_with_cases_and_controls():
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

    pheno_from_full_code = UKBiobankTrait(full_code=pheno_from_code.get_plain_name())
    assert pheno_from_code.code == pheno_from_full_code.code
    assert pheno_from_code.description == pheno_from_full_code.description
    assert pheno_from_code.type == pheno_from_full_code.type
    assert pheno_from_code.n == pheno_from_full_code.n
    assert pheno_from_code.n_cases == pheno_from_full_code.n_cases
    assert pheno_from_code.n_controls == pheno_from_full_code.n_controls
    assert pheno_from_code.source == pheno_from_full_code.source
    assert pheno_from_code.study == pheno_from_full_code.study
    assert pheno_from_code.get_plain_name() == pheno_from_full_code.get_plain_name()


def test_gtex_gwas_trait_no_cases_or_controls():
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


def test_gtex_gwas_trait_with_cases_and_controls():
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


def test_trait_get_trait_using_code_is_ukb():
    trait_code = "50_raw"

    trait = Trait.get_trait(trait_code)
    assert trait is not None
    assert isinstance(trait, UKBiobankTrait)
    assert trait.code == "50_raw"
    assert trait.get_plain_name() == "50_raw-Standing_height"


def test_trait_get_trait_using_full_code_is_ukb():
    trait_code = "50_raw-Standing_height"

    trait = Trait.get_trait(full_code=trait_code)
    assert trait is not None
    assert isinstance(trait, UKBiobankTrait)
    assert trait.code == "50_raw"
    assert trait.get_plain_name() == "50_raw-Standing_height"


def test_trait_get_trait_is_gtex_gwas():
    trait_code = "UKB_1160_Sleep_duration"

    trait = Trait.get_trait(trait_code)
    assert trait is not None
    assert isinstance(trait, GTEXGWASTrait)
    assert trait.code == "UKB_1160_Sleep_duration"
    assert trait.get_plain_name() == "UKB_1160_Sleep_duration"


def test_trait_get_trait_using_full_code_is_gtex_gwas():
    trait_code = "UKB_1160_Sleep_duration"

    trait = Trait.get_trait(full_code=trait_code)
    assert trait is not None
    assert isinstance(trait, GTEXGWASTrait)
    assert trait.code == "UKB_1160_Sleep_duration"
    assert trait.get_plain_name() == "UKB_1160_Sleep_duration"


def test_trait_get_trait_from_efo():
    trait_efo = "asthma"

    traits = Trait.get_traits_from_efo(trait_efo)
    assert traits is not None
    assert isinstance(traits, list)
    assert len(traits) == 3

    traits_codes = set([t.code for t in traits])
    assert len(traits_codes) == len(traits)
    assert "20002_1111" in traits_codes
    assert "22127" in traits_codes
    assert "J45" in traits_codes


@pytest.mark.parametrize(
    "trait_code,efo_code,efo_name",
    [
        ("20002_1111", "EFO:0000270", "asthma"),
        ("22127", "EFO:0000270", "asthma"),
        ("J45", "EFO:0000270", "asthma"),
        (
            "BCAC_Overall_BreastCancer_EUR",
            "EFO:0005606",
            "family history of breast cancer",
        ),
    ],
)
def test_ukb_trait_efo(trait_code, efo_code, efo_name):
    trait = Trait.get_trait(code=trait_code)
    trait_efo_info = trait.get_efo_info()
    assert trait_efo_info.id == efo_code
    assert trait_efo_info.label == efo_name


@pytest.mark.parametrize(
    "trait_code",
    [
        "50_raw",
        "49_raw",
    ],
)
def test_ukb_trait_efo_not_mapped(trait_code):
    trait = Trait.get_trait(code=trait_code)
    trait_efo_info = trait.get_efo_info()
    assert trait_efo_info is None


@pytest.mark.parametrize(
    "trait_code,do_id,do_name",
    [
        ("20002_1111", "DOID:2841", "asthma"),
        ("22127", "DOID:2841", "asthma"),
        ("J45", "DOID:2841", "asthma"),
        ("20002_1065", "DOID:10763", "hypertension"),
        ("20002_1440", "DOID:399", "tuberculosis"),
    ],
)
def test_ukb_trait_do(trait_code, do_id, do_name):
    trait = Trait.get_trait(code=trait_code)
    trait_do_info = trait.get_do_info()
    assert isinstance(trait_do_info.id, list)
    assert len(trait_do_info.id) == 1
    assert trait_do_info.id[0] == do_id
    # FIXME: not adding labels for DO for now, but should be included later
    # assert trait_do_info.label == do_name


@pytest.mark.parametrize(
    "trait_code",
    [
        "50_raw",
        "49_raw",
        "K10"
    ],
)
def test_ukb_trait_do_not_mapped(trait_code):
    trait = Trait.get_trait(code=trait_code)
    trait_do_info = trait.get_do_info()
    assert trait_do_info is None
    # FIXME: not adding labels for DO for now, but should be included later
    # assert trait_do_info.label == do_name


def test_map_to_doid_simple():
    data = pd.read_pickle(Path(
        Path(__file__).parent,
        "test_cases",
        "smultixcan_zscores",
        "smultixcan-slice01.pkl"
    ))

    # select traits
    data = data[[
        # asthma  EFO_0000270
        "20002_1111-Noncancer_illness_code_selfreported_asthma",
        "22127-Doctor_diagnosed_asthma",
        "J45-Diagnoses_main_ICD10_J45_Asthma",
        # hypertension    EFO_0000537
        "20002_1065-Noncancer_illness_code_selfreported_hypertension",
    ]]

    data_mapped = Trait.map_to_doid(data)

    assert data_mapped.shape == data.shape
    assert data_mapped.index.equals(data.index)

    assert not data_mapped.columns.equals(data.columns)
    assert data_mapped.columns.tolist() == [
        "DOID:2841", "DOID:2841", "DOID:2841", "DOID:10763"
    ]

    np.testing.assert_array_equal(data.values, data_mapped.values)


def test_map_to_doid_using_preferred_doid_list():
    data = pd.read_pickle(Path(
        Path(__file__).parent,
        "test_cases",
        "smultixcan_zscores",
        "smultixcan-slice01.pkl"
    ))

    # select traits
    data = data[[
        # asthma  EFO_0000270
        "20002_1111-Noncancer_illness_code_selfreported_asthma",
        "22127-Doctor_diagnosed_asthma",
        "J45-Diagnoses_main_ICD10_J45_Asthma",
        # labyrinthitis   EFO_0009604 -> this one maps to several DOID: DOID:3930, DOID:1468
        "20002_1499-Noncancer_illness_code_selfreported_labyrinthitis",
    ]]

    # using first DOID
    data_mapped = Trait.map_to_doid(data, preferred_doid_list=["DOID:3930"])

    assert data_mapped.shape == data.shape
    assert data_mapped.index.equals(data.index)

    assert not data_mapped.columns.equals(data.columns)
    assert data_mapped.columns.tolist() == [
        "DOID:2841", "DOID:2841", "DOID:2841", "DOID:3930"
    ]

    np.testing.assert_array_equal(data.values, data_mapped.values)

    # using second DOID
    data_mapped = Trait.map_to_doid(data, preferred_doid_list=["DOID:1468"])

    assert data_mapped.shape == data.shape
    assert data_mapped.index.equals(data.index)

    assert not data_mapped.columns.equals(data.columns)
    assert data_mapped.columns.tolist() == [
        "DOID:2841", "DOID:2841", "DOID:2841", "DOID:1468"
    ]

    np.testing.assert_array_equal(data.values, data_mapped.values)


def test_map_to_doid_simple_combine_max():
    data = pd.read_pickle(Path(
        Path(__file__).parent,
        "test_cases",
        "smultixcan_zscores",
        "smultixcan-slice01.pkl"
    ))

    # select traits
    data = data[[
        # asthma  EFO_0000270
        "20002_1111-Noncancer_illness_code_selfreported_asthma",
        "22127-Doctor_diagnosed_asthma",
        "J45-Diagnoses_main_ICD10_J45_Asthma",
        # hypertension    EFO_0000537
        "20002_1065-Noncancer_illness_code_selfreported_hypertension",
    ]]

    data_mapped = Trait.map_to_doid(data, combine="max")

    assert data_mapped.shape == (data.shape[0], 2)
    assert data_mapped.index.equals(data.index)

    assert not data_mapped.columns.equals(data.columns)
    assert set(data_mapped.columns) == set([
        "DOID:2841", "DOID:10763"
    ])

    assert data_mapped.loc["ENSG00000242715", "DOID:2841"].round(6) == 0.659602
    assert data_mapped.loc["ENSG00000237567", "DOID:2841"].round(6) == 1.292931
    assert data_mapped.loc["ENSG00000205643", "DOID:10763"].round(6) == 0.283507
    assert data_mapped.loc["ENSG00000229722", "DOID:10763"].round(6) == 0.694461


def test_map_to_doid_some_non_existent():
    data = pd.read_pickle(Path(
        Path(__file__).parent,
        "test_cases",
        "smultixcan_zscores",
        "smultixcan-slice01.pkl"
    ))

    # select traits
    data = data[[
        # preeclampsia    EFO_0000668
        "O16-Diagnoses_main_ICD10_O16_Unspecified_maternal_hypertension",
        "O14-Diagnoses_main_ICD10_O14_Gestational_pregnancyinduced_hypertension_with_significant_proteinuria",
        # Traits not mapped to DOID
        "50_raw-Standing_height",
        "20096_1-Size_of_red_wine_glass_drunk_small_125ml",
    ]]

    data_mapped = Trait.map_to_doid(data)

    assert data_mapped.shape == (data.shape[0], 2)
    assert data_mapped.index.equals(data.index)

    assert not data_mapped.columns.equals(data.columns)
    assert data_mapped.columns.tolist() == [
        "DOID:10591", "DOID:10591"
    ]

    np.testing.assert_array_equal(
        data.iloc[:, 0:2].values,
        data_mapped.iloc[:, 0:2].values
    )
