"""
Classes to ease access to attributes of common entities like Trait and Gene.
"""
from abc import ABCMeta, abstractmethod
from enum import Enum, auto

import pandas as pd

from utils import simplify_string


class Study(Enum):
    UK_BIOBANK = auto()
    GTEX_GWAS = auto()


class Trait(object, metaclass=ABCMeta):
    """Abstract class to represent a generic trait from different GWAS.

    This class represents traits from different studies (for instance,
    from the UK Biobank or from the GTEX GWAS project). Different studies
    provide different attributes for traits. To use this class, you either
    instanciate any of the subclasses or call the get_trait method providing
    the short or long code of the trait, and it will return an object with
    the appropriate Trait subclass.

        Typical usage example:

        # using the full code of trait
        ukb_trait = Trait.get_trait(full_code='50_raw-Standing_height')

        # using the short code of trait
        ukb_trait = Trait.get_trait(code='50_raw')

    Args:
        code (str): short code of trait.
        full_code (str): long code of trait.

    Attributes:
        code (str): short code.
        full_code (str): long code.
        description (str): description.
        type (str): whether the trait is binary (case/control) or continuous.
        n (int): sample size for the trait in the GWAS.
        n_cases (int): if binary, the number of cases.
        n_controls (int): if binary, the number of controls.
        source (str): source, typically the consortium name or dataset.
        study (Study): Study object representing the study.
        pheno_data (pandas.Series): contains all the information available for
        the trait.
    """

    def __init__(self, code=None, full_code=None):
        if code is None and full_code is None:
            raise ValueError("Either code or full_code must be specified.")

        if code is not None:
            self.code = code
        elif full_code is not None:
            self.full_code = full_code
            self.code = Trait.get_code_from_full_code(full_code)

        self.description = None
        self.type = None
        self.n = None
        self.n_cases = None
        self.n_controls = None
        self.source = None
        self.study = None
        self.pheno_data = None

        self.init_metadata()

    @staticmethod
    def get_code_from_full_code(full_code):
        """Given the full code of a trait, returns the short code.

        Args:
            full_code: full code of the trait.

        Returns:
            str: the short code of the trait.
        """

        pheno_split = full_code.split("-")

        if len(pheno_split) in (1, 2):
            code = pheno_split[0]
        else:
            code = "-".join(pheno_split[:2])

        return code

    @staticmethod
    def get_trait(code=None, full_code=None):
        """Returns the appropriate Trait subclass given the short or full code
        of the trait.

        Either code or full_code must be given.

        Args:
            code: short code of the trait.
            full_code: full code of the trait.

        Returns:
            A subclass of Trait corresponding to the study where the trait is
            present. None if the trait is not found in any study.
        """

        # FIXME this needs refatoring when more studies are included.
        if UKBiobankTrait.is_phenotype_from_study(code, full_code):
            return UKBiobankTrait(code, full_code)
        elif GTEXGWASTrait.is_phenotype_from_study(code, full_code):
            return GTEXGWASTrait(code, full_code)

    @abstractmethod
    def init_metadata(self):
        """Initializes all the trait attributes."""
        pass

    @staticmethod
    @abstractmethod
    def is_phenotype_from_study(code=None, full_code=None):
        """Given the code or full code of a trait, return if it belongs to the
        study."""
        pass

    @property
    @abstractmethod
    def category(self):
        """Returns the category of the trait."""
        pass

    def get_plain_name(self):
        """Returns the plain name of the trait, which coincides with the full
        code."""
        if self.study in (Study.GTEX_GWAS,):
            return self.code

        if not pd.isnull(self.description):
            return f"{self.code}-{simplify_string(self.description)}"
        else:
            return self.code

    def __repr__(self):
        return self.get_plain_name()


class UKBiobankTrait(Trait):
    """Trait subclass representing traits from the UK Biobank"""

    from metadata import RAPID_GWAS_PHENO_INFO, RAPID_GWAS_DATA_DICT, UK_BIOBANK_CODINGS

    CODE_STARTSWITH_CATEGORIES_MAP = {
        ("6150_", "3627_"): "Diseases (cardiovascular)",
        ("6151_",): "Diseases (musculoskeletal/trauma)",
    }

    CODE_IN_CATEGORIES_MAP = {
        ("6152_5", "6152_7"): "Diseases (cardiovascular)",
        ("6152_6", "6152_8"): "Diseases (respiratory/ent)",
        ("6152_9",): "Diseases (allergies)",
    }

    def get_selfreported_parent_category(self):
        """Return the top parent category for a given trait and coding number.

        Returns:
            pandas.Series: an object with hierarchical information about the
            top parent category of the trait.
        """

        primary_code = self.get_fieldid()
        subcode = float(self.code.split("_")[1])

        coding_number = int(self.RAPID_GWAS_DATA_DICT.loc[primary_code, "Coding"])

        ukb_coding = self.UK_BIOBANK_CODINGS[coding_number]

        parent = ukb_coding[ukb_coding["coding"] == subcode].iloc[0]

        while parent["parent_id"] > 0:
            parent = ukb_coding[ukb_coding["node_id"] == parent["parent_id"]].iloc[0]

        return parent

    @property
    def category(self):
        # FIXME the next two ifs need refactoring
        if self.code.startswith("20001_"):
            parent_cat = self.get_selfreported_parent_category()
            parent_name = parent_cat["meaning"].replace(" cancer", "")
            return f"Cancer ({parent_name})"

        if self.code.startswith("20002_"):
            parent_cat = self.get_selfreported_parent_category()
            parent_name = parent_cat["meaning"]
            return f"Diseases ({parent_name})"

        for k, v in self.CODE_STARTSWITH_CATEGORIES_MAP.items():
            if self.code.startswith(k):
                return v

        for k, v in self.CODE_IN_CATEGORIES_MAP.items():
            if self.code in k:
                return v

        if self.pheno_data["source"] == "icd10":
            return "Diseases (ICD10 main)"
        elif self.pheno_data["source"] == "finngen":
            return "Diseases (FinnGen)"

        field_id = self.get_fieldid()
        assert field_id is not None, self.code
        field_path = self.RAPID_GWAS_DATA_DICT.loc[field_id]["Path"]
        return field_path.split(" > ")[-1]

    @staticmethod
    def is_phenotype_from_study(code=None, full_code=None):
        if code is None:
            code = Trait.get_code_from_full_code(full_code)

        return code in UKBiobankTrait.RAPID_GWAS_PHENO_INFO.index

    def get_fieldid(self):
        """Returns the field id of the Trait."""
        if "_" in self.code:
            code = self.code.split("_")[0]
            if code.isdigit() and int(code) in self.RAPID_GWAS_DATA_DICT.index:
                return int(code)
        elif self.code.isdigit():
            return int(self.code)
        elif self.description.startswith("Diagnoses - main ICD10:"):
            return int(41202)

    def init_metadata(self):
        if not self.is_phenotype_from_study(self.code):
            raise ValueError(f"Invalid UK Biobank phenotype code: {self.code}")

        self.pheno_data = UKBiobankTrait.RAPID_GWAS_PHENO_INFO.loc[self.code]

        self.description = self.pheno_data["description"]
        self.type = self.pheno_data["variable_type"]
        self.n = self.pheno_data["n_non_missing"]
        self.n_cases = self.pheno_data["n_cases"]
        self.n_controls = self.pheno_data["n_controls"]
        self.source = "UK Biobank"
        self.study = Study.UK_BIOBANK


class GTEXGWASTrait(Trait):
    """Trait subclass representing traits from the GTEX GWAS"""

    from metadata import GTEX_GWAS_PHENO_INFO

    @property
    def category(self):
        return self.pheno_data["Category"]

    @staticmethod
    def is_phenotype_from_study(code=None, full_code=None):
        if code is None:
            code = Trait.get_code_from_full_code(full_code)

        return code in GTEXGWASTrait.GTEX_GWAS_PHENO_INFO.index

    def init_metadata(self):
        if not self.is_phenotype_from_study(self.code):
            raise ValueError(f"Invalid GWAS GWAS phenotype code: {self.code}")

        self.pheno_data = GTEXGWASTrait.GTEX_GWAS_PHENO_INFO.loc[self.code]

        self.description = self.pheno_data["new_Phenotype"].replace("_", " ")
        if self.pheno_data["Binary"] == 1:
            self.type = "binary"
        else:
            self.type = (
                # we don't know if it was transformed, actually
                "continuous_raw"
            )

        self.n = self.pheno_data["Sample_Size"]
        self.n_cases = self.pheno_data["Cases"]
        self.n_controls = self.n - self.n_cases
        self.source = self.pheno_data["Consortium"]
        self.study = Study.GTEX_GWAS


class Gene(object):
    from metadata import GENE_ID_TO_NAME_MAP, GENE_NAME_TO_ID_MAP, BIOMART_GENES

    def __init__(self, ensembl_id=None, name=None):
        if ensembl_id is not None:
            if ensembl_id not in self.GENE_ID_TO_NAME_MAP:
                raise ValueError("Ensembl ID not found.")

            self.ensembl_id = ensembl_id
            self.name = self.GENE_ID_TO_NAME_MAP[self.ensembl_id]
        elif name is not None:
            if name not in self.GENE_NAME_TO_ID_MAP:
                raise ValueError("Gene name not found.")

            self.name = name
            self.ensembl_id = self.GENE_NAME_TO_ID_MAP[self.name]

        self._band = None

    @property
    def band(self):
        """Returns the cytoband of a gene."""
        if self._band is not None:
            return self.band

        if self.ensembl_id not in Gene.BIOMART_GENES.index:
            return ""

        gene_data = Gene.BIOMART_GENES.loc[self.ensembl_id]
        chrom = gene_data["chromosome_name"]
        band = gene_data["band"]

        return f"{chrom}{band}"
