"""
Classes to ease access to attributes of common entities like Trait and Gene.
"""
from abc import ABCMeta, abstractmethod
from enum import Enum, auto
import re
from collections import namedtuple
from pathlib import Path
from functools import lru_cache

import pandas as pd

import conf
from data.cache import read_data


class Study(Enum):
    UK_BIOBANK = auto()
    GTEX_GWAS = auto()


class Trait(object, metaclass=ABCMeta):
    """Abstract class to represent a generic trait from different GWAS.

    This class represents traits from different studies (for instance, from the UK
    Biobank or from the GTEX GWAS project). Different studies provide different
    attributes for traits. To use this class, you either instanciate any of the
    subclasses or call the get_trait method providing the short or long code of the
    trait, and it will return an object with the appropriate Trait subclass.

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

    MAP_INFO = namedtuple("EfoInfo", ["id", "label"])

    # This file was downloaded from https://github.com/EBISPOT/EFO-UKB-mappings
    UKB_TO_EFO_MAP_FILE = Path(
        Path(__file__).parent,
        "data",
        conf.PHENOMEXCAN["TRAITS_FULLCODE_TO_EFO_MAP_FILE"].name,
    ).resolve()

    # This file was generated from the EFO ontology, which has a map to several
    # other ontology IDs
    EFO_XREFS_FILE = Path(
        Path(__file__).parent,
        "data",
        conf.GENERAL["TERM_ID_XREFS_FILE"].name,
    ).resolve()

    # This file was obtained from https://github.com/dhimmel/disease-ontology
    DO_XREFS_FILE = Path(Path(__file__).parent, "data", "xrefs-prop-slim.tsv").resolve()

    def __init__(self, code=None, full_code=None):
        if code is None and full_code is None:
            raise ValueError("Either code or full_code must be specified.")

        if code is not None:
            self.code = code
            self.full_code = None
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

        # now that all the data was initialized, I can fill full_code if was not given
        if self.full_code is None:
            self.full_code = self.get_plain_name()

    def get_efo_info(self, mapping_type=None):
        """
        It returns a EFO_INFO (namedtuple) with EFO code and label for this
        trait.

        Args:
            mapping_type (str): mapping type to be used for the UK Biobank
            mappings. It could be Exact, Broad or Narrow.

        Returns:
            An EFO_INFO (namedtupled) with the EFO code and label for this
            trait.
        """
        efo_map_data = self.get_traits_to_efo_map_data()

        if self.full_code not in efo_map_data.index:
            return None

        map_info = efo_map_data.loc[self.full_code]

        if mapping_type is not None:
            map_info = map_info[map_info["mapping_type"] == mapping_type]

        efo_codes = map_info["term_codes"]
        if not isinstance(efo_codes, str):
            efo_code = ", ".join(efo_codes.unique())
        else:
            efo_code = efo_codes

        label = map_info["current_term_label"]
        if not isinstance(label, str):
            label = label.unique()
            assert label.shape[0] == 1
            label = label[0]

        return self.MAP_INFO(id=efo_code, label=label)

    def get_do_info(self, mapping_type=None):
        """
        Get the Disease Ontology ID (DOID) for this Trait instance. It uses
        several sources to map the PhenomeXcan trait to a DOID.

        Args:
            mapping_type:
                A string that is passed to the function get_efo_info (see
                documentation there).

        Returns:
            A MAP_INFO namedtuple with two fields: id (which could be a list of
            IDs) and label (which here is always None).
        """
        # obtain the EFO data for this PhenomeXcan/UK Biobank trait
        efo_info = self.get_efo_info(mapping_type=mapping_type)
        if efo_info is None:
            return None

        efo_id_part = efo_info.id[4:]

        # now, look for a mapping from EFO to DOID using the EFO ontology
        # references
        efo_xrefs_data = self.get_efo_xrefs_data()
        efo_xrefs_data = efo_xrefs_data[
            (efo_xrefs_data["term_id"] == efo_info.id)
            & (efo_xrefs_data["target_id_type"] == "DOID")
        ]["target_id"].values

        # now do the same but using another resource
        do_xrefs_data = self.get_do_xrefs_data()
        do_xrefs_data = do_xrefs_data[
            (do_xrefs_data["resource"] == "EFO")
            & (do_xrefs_data["resource_id"] == efo_id_part)
        ]["doid_code"].values

        # merge different mappings
        doid_maps = set(efo_xrefs_data).union(set(do_xrefs_data))

        if len(doid_maps) == 0:
            return None

        doid_maps = list(doid_maps)
        doid_maps.sort()

        return self.MAP_INFO(id=doid_maps, label=None)

    def get_plain_name(self):
        """Returns the plain name of the trait, which coincides with the full
        code."""
        if self.study in (Study.GTEX_GWAS,):
            return self.code

        if not pd.isnull(self.description):
            return f"{self.code}-{self._simplify_trait_name(self.description)}"
        else:
            return self.code

    def __repr__(self):
        return self.get_plain_name()

    @staticmethod
    def get_efo_xrefs_data():
        return read_data(
            Trait.EFO_XREFS_FILE,
            sep="\t",
        )

    @staticmethod
    def get_do_xrefs_data():
        return read_data(Trait.DO_XREFS_FILE, sep="\t")

    @staticmethod
    @lru_cache(maxsize=None)
    def get_traits_to_efo_map_data():
        return read_data(Trait.UKB_TO_EFO_MAP_FILE, sep="\t", index_col="ukb_fullcode")

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
    def is_efo_label(trait_label: str):
        """
        Given a string, it returns true if it is an EFO label

        Args:
            trait_label: any string representing a trait code.

        Returns:
            True if trait_label is an EFO label. False otherwise.
        """
        efo_labels = set(
            Trait.get_traits_to_efo_map_data()["current_term_label"].values
        )
        return trait_label in efo_labels

    @staticmethod
    def get_traits_from_efo(efo_label: str):
        """
        It returns a map from an EFO label to a list of PhenomeXcan traits.

        Args:
            efo_label: an EFO label.

        Returns:
            A list of Trait instances that map to efo_label.
        """
        if efo_label is None:
            return None

        efo_map_data = (
            Trait.get_traits_to_efo_map_data()
            .reset_index()
            .set_index("current_term_label")
        )

        if efo_label not in efo_map_data.index:
            return None

        map_info = efo_map_data.loc[[efo_label]]

        label = map_info["ukb_fullcode"].values

        return [Trait.get_trait(full_code=fc) for fc in label]

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

    @staticmethod
    def _simplify_trait_name(s):
        """
        Given any string representing a trait name, it returns a simplified
        version of it. It removes any character that is not a word, a number
        or a separator. Then it replaces all separators by an underscore.

        Args:
            s (str): string to be simplified.

        Returns:
            str: string simplified.
        """
        # Remove all non-word characters (everything except numbers and letters)
        s = re.sub(r"[^\w\s]", "", s)

        # Replace all runs of whitespace with a single dash
        s = re.sub(r"\s+", "_", s)

        return s

    @staticmethod
    def _select_doid(doids: list, preferred_doid_list: set):
        """
        It takes a list of Disease Ontology IDs, and returns the first one that
        appears in a preferred DOID list (such as those present in the gold standard).
        If none in the list is there, then return the first one.
        """
        if doids is None or len(doids) == 0:
            raise ValueError("List of Disease Ontology IDs is empty or None")
        
        for doid in doids:
            if doid in preferred_doid_list:
                return doid

        return doids[0]

    @staticmethod
    def map_to_doid(
        data: pd.DataFrame,
        preferred_doid_list: set = None,
        combine: str = None,
    ) -> pd.DataFrame:
        """
        It maps traits in the columns of a pandas DataFrame to Disese Ontology
        IDs (DOID). If several traits map to the same DOID, then it can merge
        them using a strategy defined in `combine`.

        This function is mainly used for drug-disease prediction data, where
        drugs are in rows in traits in columns. If several traits point to the
        same DOID, then the strategy supported to merge them is to take the
        maximum prediction value for a particular drug.

        Args:
            data:
                A pandas dataframe where columns are traits.
            preferred_doid_list:
                A preferred list of DOIDs.
            combine:
                A strategy to combine traits that map to the same DOID. Actually,
                only "max" is supported (it takes the maximum value).

        Returns:
            A pandas dataframe where traits in columns are mapped to DOID.
        """
        if preferred_doid_list is None:
            preferred_doid_list = set()

        traits_full_code_to_do_map = {
            fc: Trait._select_doid(t.get_do_info().id, preferred_doid_list)
            for fc in data.columns.sort_values()
            if (t := Trait.get_trait(full_code=fc)).get_do_info() is not None
        }

        data_mapped = data.loc[
            :, data.columns.isin(traits_full_code_to_do_map.keys())
        ].rename(columns=traits_full_code_to_do_map)

        if combine == "max":
            data_mapped = data_mapped.groupby(data_mapped.columns, axis=1).max()
            assert data_mapped.columns.is_unique, "Columns not unique"

        return data_mapped


class UKBiobankTrait(Trait):
    """Trait subclass representing traits from the UK Biobank"""

    RAPID_GWAS_PHENO_INFO = read_data(conf.PHENOMEXCAN["RAPID_GWAS_PHENO_INFO_FILE"])
    RAPID_GWAS_DATA_DICT = read_data(conf.PHENOMEXCAN["RAPID_GWAS_DATA_DICT_FILE"])

    UK_BIOBANK_CODINGS = {
        3: read_data(conf.UK_BIOBANK["CODING_3_FILE"]),
        6: read_data(conf.UK_BIOBANK["CODING_6_FILE"]),
    }

    MAIN_ICD10_CODE = 41202

    CODE_STARTSWITH_CATEGORIES_MAP = {
        ("6150_", "3627_"): "Diseases (cardiovascular)",
        ("6151_",): "Diseases (musculoskeletal/trauma)",
    }

    CODE_IN_CATEGORIES_MAP = {
        ("6152_5", "6152_7"): "Diseases (cardiovascular)",
        ("6152_6", "6152_8"): "Diseases (respiratory/ent)",
        ("6152_9",): "Diseases (allergies)",
    }

    def _get_selfreported_parent_category(self):
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
            parent_cat = self._get_selfreported_parent_category()
            parent_name = parent_cat["meaning"].replace(" cancer", "")
            return f"Cancer ({parent_name})"

        if self.code.startswith("20002_"):
            parent_cat = self._get_selfreported_parent_category()
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
            return int(self.MAIN_ICD10_CODE)

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

    GTEX_GWAS_PHENO_INFO = read_data(conf.PHENOMEXCAN["GTEX_GWAS_PHENO_INFO_FILE"])

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

        # WARNING: this attribute retrieves the original EFO code in the GTEx
        # GWAS metadata. However, some EFO codes there are wrong. Use this
        # carefully, the get_efo_info is prefered.
        self.orig_efo_id = self.pheno_data["EFO"]


class Gene(object):
    """TODO complete docstring"""

    GENE_ID_TO_NAME_MAP = read_data(conf.PHENOMEXCAN["GENE_MAP_ID_TO_NAME"])
    GENE_NAME_TO_ID_MAP = read_data(conf.PHENOMEXCAN["GENE_MAP_NAME_TO_ID"])
    BIOMART_GENES = read_data(conf.GENERAL["BIOMART_GENES_INFO_FILE"])

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
