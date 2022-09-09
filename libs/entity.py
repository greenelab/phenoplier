"""
Classes to ease access to attributes of common entities like Trait and Gene.
"""
from abc import ABCMeta, abstractmethod
from enum import Enum, auto
import re
from collections import namedtuple
from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform

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
    """
    It represents a Gene with certain attributes (symbol, id, etc) and functions
    (correlations of predicted expression, etc).
    """

    GENE_ID_TO_NAME_MAP = read_data(conf.PHENOMEXCAN["GENE_MAP_ID_TO_NAME"])
    GENE_NAME_TO_ID_MAP = read_data(conf.PHENOMEXCAN["GENE_MAP_NAME_TO_ID"])
    BIOMART_GENES = read_data(conf.GENERAL["BIOMART_GENES_INFO_FILE"])

    DEFAULT_WITHIN_DISTANCE = 2.5e6

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
    @lru_cache(maxsize=None)
    def chromosome(self):
        """Returns the chromosome of the gene."""
        if self.ensembl_id not in Gene.BIOMART_GENES.index:
            return None

        gene_data = Gene.BIOMART_GENES.loc[self.ensembl_id]
        return gene_data["chromosome_name"]

    @property
    @lru_cache(maxsize=None)
    def band(self):
        """Returns the cytoband of the gene."""
        if self.ensembl_id not in Gene.BIOMART_GENES.index:
            return None

        gene_data = Gene.BIOMART_GENES.loc[self.ensembl_id]
        chrom = gene_data["chromosome_name"]
        band = gene_data["band"]

        return f"{chrom}{band}"

    @lru_cache(maxsize=None)
    def get_attribute(self, attribute_name):
        """Returns any attribute of the gene in BioMart."""
        if self.ensembl_id not in Gene.BIOMART_GENES.index:
            return None

        gene_data = Gene.BIOMART_GENES.loc[self.ensembl_id]
        attr = gene_data[attribute_name]

        return attr

    def within_distance(self, other_gene, distance_bp=None):
        """
        This function returns True if this genes is within a certain distance
        from other gene. It assumes that the two genes are in the same
        chromosome (it will give misleading results if this is not true).
        """
        if distance_bp is None:
            distance_bp = Gene.DEFAULT_WITHIN_DISTANCE

        this_start = self.get_attribute("start_position")
        this_end = self.get_attribute("end_position")
        if this_start is None or this_end is None:
            return False
        this_start = int(this_start) - distance_bp
        this_end = int(this_end) + distance_bp

        other_start = other_gene.get_attribute("start_position")
        other_end = other_gene.get_attribute("end_position")
        if other_start is None or other_end is None:
            return False
        other_start = int(other_start) - distance_bp
        other_end = int(other_end) + distance_bp

        return (other_start <= this_start <= other_end) or (
            this_start <= other_start <= this_end
        )

    @staticmethod
    def _get_tissue_connection(tissue: str, model_type: str):
        """
        Returns an SQLite connection to the prediction models of PrediXcan for
        a specified tissue. The caller is responsible of closing this connection
        object by calling the `close` method after using it.

        Args:
            tissue:
                The tissue name to get the connection from.
            model_type:
                The prediction model type, such as "MASHR" or "ELASTIC_NET" (see conf.py).

        Returns:
            An read-only SQLite connection object.
        """
        # check that the file for the tissue exists
        model_prefix = conf.PHENOMEXCAN["PREDICTION_MODELS"][f"{model_type}_PREFIX"]

        tissue_weights_file = (
            conf.PHENOMEXCAN["PREDICTION_MODELS"][model_type]
            / f"{model_prefix}{tissue}.db"
        )

        if not tissue_weights_file.exists():
            raise ValueError(
                f"Model file for tissue does not exist: {str(tissue_weights_file)}"
            )

        import sqlite3

        db_uri = f"file:{str(tissue_weights_file)}?mode=ro"
        return sqlite3.connect(db_uri, uri=True)

    @lru_cache(maxsize=None)
    def get_prediction_weights(
        self,
        tissue: str,
        model_type: str,
        snps_subset: frozenset = None,
    ):
        """
        Given a tissue and model type, it returns the prediction weights for
        this gene. The prediction weights are a list of SNPs with the weights
        learned from a regression model to predict its expression.

        Args:
            tissue:
                The tissue name.
            model_type:
                The prediction model type, such as "MASHR" or "ELASTIC_NET" (see
                conf.py).
            snps_subset:
                This is a subset of SNPs IDs (such as "chr1_33071920_A_C_b38")
                that will be considered to compute the correlation. All snps
                that are not present in this subset will be ignored. This is
                used to make more accurate predictions given the SNPs in a GWAS,
                for example.

        Returns:
            A pandas DataFrame with the variants' weight for the prediction of
             this gene expression. It has these columns: variant ID (in GTEx v8
             format) and its weight. It returns None if no predictors (SNPs) are
             available for the gene in this tissue/model.
        """
        sqlite_conn = Gene._get_tissue_connection(tissue, model_type)

        try:
            query = f"""
                select varID, weight
                from weights
                where gene like '{self.ensembl_id}.%'
            """

            df = pd.read_sql(query, sqlite_conn)

            if df.shape[0] == 0 or df["weight"].abs().sum() == 0.0:
                return None

            df = df.set_index("varID")["weight"]

            if snps_subset is not None and len(snps_subset) > 0:
                snps = snps_subset.intersection(set(df.index))
                if len(snps) == 0:
                    return None
                df = df.loc[snps]

            return df.sort_index()
        finally:
            sqlite_conn.close()

    @staticmethod
    @lru_cache(maxsize=1)
    def _read_snps_cov(snps_chr, reference_panel: str, model_type: str):
        """
        Returns the covariance matrix for all SNPs (in the predictions models)
        in a chromosome. The structure returned (a tuple with three elements,
        see below) is thought to speed up processing in other parts of the code.

        For this function, lru_cache/maxsize is 1 because the idea is to call
        always on the same chromosome in sequence to speed up but avoid high
        memory consumption.

        Args:
            snps_chr:
                A string specifying the chromosome in format "chr{num}".
            reference_panel:
                Reference panel used to compute SNP covariance matrix. Either
                GTEX_V8 or 1000G.
            model_type:
                The prediction model type, such as "MASHR" or "ELASTIC_NET" (see
                conf.py).
        Returns:
            A tuple with three elements:
                1. A square numpy array with SNPs covariances.
                2. A set with SNPs ids
                3. A dictionary with SNPs ids in keys and positions as values.
        """
        input_dir = (
            conf.RESULTS["GLS"]
            / "gene_corrs"
            / "reference_panels"
            / reference_panel.lower()
            / model_type.lower()
        )
        snps_cov_file = input_dir / "snps_chr_blocks_cov.h5"
        assert snps_cov_file.exists(), f"Input file does not exist: {snps_cov_file}"

        # go to disk and read the data
        with pd.HDFStore(snps_cov_file, mode="r") as store:
            snp_cov = store[snps_chr].sort_index(axis=0).sort_index(axis=1)
            snp_cov_variants = set(snp_cov.index)
            snp_index_dict = {
                snp_id: snp_idx for snp_idx, snp_id in enumerate(snp_cov.index)
            }

            return snp_cov.to_numpy(), snp_cov_variants, snp_index_dict

    @staticmethod
    @lru_cache(maxsize=None)
    def get_snps_variance(tissue: str, snps_list: tuple, model_type: str):
        """
        Returns the variance of a set of SNPs (snps_list) precomputed from one
        prediction model of PrediXcan (model_type) in a given tissue.
        """
        # check that the file for the tissue exists
        model_prefix = conf.PHENOMEXCAN["PREDICTION_MODELS"][f"{model_type}_PREFIX"]

        tissue_snps_var_file = (
            conf.PHENOMEXCAN["PREDICTION_MODELS"][model_type]
            / f"{model_prefix}{tissue}.txt.gz"
        )

        if not tissue_snps_var_file.exists():
            raise ValueError(
                f"SNPs variance file for tissue does not exist: {str(tissue_snps_var_file)}"
            )

        snps_var_data = pd.read_csv(tissue_snps_var_file, sep=" ")

        snp_vars = {}
        for snp_id in snps_list:
            data = snps_var_data[
                (snps_var_data["RSID1"] == snp_id) & (snps_var_data["RSID2"] == snp_id)
            ]
            data = data.iloc[0]
            snp_vars[snp_id] = data["VALUE"]

        return pd.Series(snp_vars)

    @staticmethod
    def _get_snps_cov(
        snps_ids_list1,
        snps_ids_list2=None,
        check=False,
        reference_panel="GTEX_V8",
        model_type="MASHR",
    ):
        """
        Given one or (optionally) two lists of SNPs IDs, it returns their
        covariance matrix. The structure returned is a uptle with three elements
        thought for convenience and performance.

        Args:
            snps_ids_list1:
                A list of SNPs IDs. When only this parameter is used, generally
                one wants to compute a gene's predicted expression variance.
            snps_ids_list2:
                (Optional) A second list of SNPs IDs. When this is used, it is
                generally the SNPs from a second gene, and to compute the
                correlations between genes.
            check:
                If should be checked that all SNPs are from the same chromosome.
            reference_panel:
                Reference panel used to compute SNP covariance matrix. Either
                GTEX_V8 or 1000G.
            model_type:
                The prediction model type, such as "MASHR" or "ELASTIC_NET" (see
                conf.py).

        Returns:
            Return a pandas dataframe with the SNPs specified in the arguments
            for which we have genotype data (otherwise we don't have its
            covariance).

            A tuple with three elements:
                1. A numpy array with SNP covariance matrix.
                2. A tuple with two elements: the SNPs IDs and positions (in the
                  matrix) for the first list of SNPs given.
                3. Same as before, for the second list of SNPs given.
        """
        snps_ids_list1 = list(snps_ids_list1)

        if len(snps_ids_list1) == 0:
            return None

        if snps_ids_list2 is None:
            snps_ids_list2 = snps_ids_list1
        else:
            snps_ids_list2 = list(snps_ids_list2)
            if len(snps_ids_list2) == 0:
                return None

        first_snp_id = snps_ids_list1[0]
        snps_chr = first_snp_id.split("_")[0]

        if check:
            # all snps must be from the same chromosome
            all_snps = pd.Series(list(set(snps_ids_list1 + snps_ids_list2)))
            all_snps_chr = all_snps.str.split("_", expand=True)[0]
            if all_snps_chr.unique().shape[0] != 1:
                raise ValueError("Only snps from the same chromosome are supported")

        # read the entire covariance matrix for this chromosome
        snps_cov, snps_cov_variants, snp_index_dict = Gene._read_snps_cov(
            snps_chr, reference_panel, model_type
        )

        # from the specified SNP lists, only keep those for which we have
        # genotypes
        def _get_snps_with_genotypes(snps_list):
            snps_ids_with_genotype = []
            snps_pos_with_genotype = []

            for v_idx, v in enumerate(snps_list):
                if v in snps_cov_variants:
                    snps_ids_with_genotype.append(v)
                    snps_pos_with_genotype.append(v_idx)

            return snps_ids_with_genotype, snps_pos_with_genotype

        snps_ids_list1, snps_pos_list1 = _get_snps_with_genotypes(snps_ids_list1)
        snps_ids_list2, snps_pos_list2 = _get_snps_with_genotypes(snps_ids_list2)

        snps_cov = snps_cov[
            np.ix_(
                [snp_index_dict[v] for v in snps_ids_list1],
                [snp_index_dict[v] for v in snps_ids_list2],
            )
        ]

        if snps_cov.shape[0] == 0 or snps_cov.shape[1] == 0:
            return None

        return (
            snps_cov,
            (snps_ids_list1, snps_pos_list1),
            (snps_ids_list2, snps_pos_list2),
        )

    @lru_cache(maxsize=None)
    def get_pred_expression_variance(
        self,
        tissue: str,
        reference_panel: str,
        model_type: str,
        snps_subset: frozenset = None,
    ):
        """
        Given a tissue, it computes the variance of the predicted gene
        expression.

        Args:
            tissue:
                The tissue name.
            reference_panel:
                Reference panel used to compute SNP covariance matrix. Either
                GTEX_V8 or 1000G.
            model_type:
                The prediction model type, such as "MASHR" or "ELASTIC_NET" (see
                conf.py).
            snps_subset:
                This is a subset of SNPs IDs (such as "chr1_33071920_A_C_b38")
                that will be considered to compute the correlation. All snps
                that are not present in this subset will be ignored. This is
                used to make more accurate predictions given the SNPs in a GWAS,
                for example.

        Returns:
            A float with the covariance of the gene predicted expression.
        """
        w = self.get_prediction_weights(tissue, model_type, snps_subset=snps_subset)
        if w is None:
            return None

        # LD of snps in gene model
        gene_snps_cov_data = Gene._get_snps_cov(
            w.index, reference_panel=reference_panel, model_type=model_type
        )
        if gene_snps_cov_data is None:
            return None

        gene_snps_cov, (_, snps_pos) = gene_snps_cov_data[:2]

        # gene model weights
        w = w.to_numpy()[np.ix_(snps_pos)]

        # return variance of gene's predicted expression using formula from:
        #   - MetaXcan paper: https://doi.org/10.1038/s41467-018-03621-1
        #   - MultiXcan paper: https://doi.org/10.1371/journal.pgen.1007889
        return w.T @ gene_snps_cov @ w

    def get_expression_correlation(
        self,
        other_gene,
        tissue: str,
        other_tissue: str = None,
        reference_panel: str = "GTEX_V8",
        model_type: str = "MASHR",
        use_within_distance: bool = True,
        snps_subset: frozenset = None,
    ):
        """
        It computes the correlation of predicted gene expression between two
        genes and their respective tissues.

        Args:
            other_gene:
                Another Gene object.
            tissue:
                The tissue name that will be used for both genes, or only this
                gene (self) if 'other_tissue' is provided.
            other_tissue:
                The tissue name that will be used for 'other_gene'. In that
                case, 'tissue' is for this gene (self).
            reference_panel:
                Reference panel used to compute SNP covariance matrix. Either
                GTEX_V8 or 1000G.
            model_type:
                The prediction model type, such as "MASHR" or "ELASTIC_NET" (see
                conf.py).
            use_within_distance:
                If True, it will use the function within_distance to determine
                if the correlation between a pair of genes will be computed or
                not.
            snps_subset:
                This is a subset of SNPs IDs (such as "chr1_33071920_A_C_b38")
                that will be considered to compute the correlation. All snps
                that are not present in this subset will be ignored. This is
                used to make more accurate predictions given the SNPs in a GWAS,
                for example.

        Returns:
            A float with the correlation between the two genes' predicted
            expression in the given tissues. None if any of the genes have no
            predictors (SNPs) in the tissue.
        """
        if self.chromosome != other_gene.chromosome:
            return 0.0

        if use_within_distance and not self.within_distance(other_gene):
            return 0.0

        other_gene_tissue = tissue
        if other_tissue is not None:
            other_gene_tissue = other_tissue

        gene_w = self.get_prediction_weights(
            tissue, model_type, snps_subset=snps_subset
        )
        if gene_w is None:
            return None

        other_gene_w = other_gene.get_prediction_weights(
            other_gene_tissue, model_type, snps_subset=snps_subset
        )
        if other_gene_w is None:
            return None

        # get genes' variances
        gene_var = self.get_pred_expression_variance(
            tissue, reference_panel, model_type, snps_subset=snps_subset
        )
        if gene_var is None or gene_var == 0.0:
            return None

        other_gene_var = other_gene.get_pred_expression_variance(
            other_gene_tissue, reference_panel, model_type, snps_subset=snps_subset
        )
        if other_gene_var is None or other_gene_var == 0.0:
            return None

        (snps_cov, (_, snps_pos_list1), (_, snps_pos_list2),) = self._get_snps_cov(
            gene_w.index,
            other_gene_w.index,
            reference_panel=reference_panel,
            model_type=model_type,
        )

        # align weights with snps cov
        gene_w = gene_w.to_numpy()[np.ix_(snps_pos_list1)]
        other_gene_w = other_gene_w.to_numpy()[np.ix_(snps_pos_list2)]

        # formula from the MultiXcan paper:
        #   https://doi.org/10.1371/journal.pgen.1007889
        return (gene_w.T @ snps_cov @ other_gene_w) / np.sqrt(gene_var * other_gene_var)

    @staticmethod
    def _get_tissues(tissues: tuple, model_type: str):
        """
        A convenience function that returns the tissues argument given or, if
        this is None and not empty, all the tissues for a given prediction
        model.

        Returns:
            A tuple with tissue names sorted.
        """
        if tissues is not None and len(tissues) > 0:
            return tissues

        all_tissues = conf.PHENOMEXCAN["PREDICTION_MODELS"][
            f"{model_type}_TISSUES"
        ].split(" ")

        return tuple(sorted(all_tissues))

    @lru_cache(maxsize=None)
    def get_tissues_correlations(
        self,
        other_gene,
        tissues: tuple = None,
        other_tissues: tuple = None,
        snps_subset: frozenset = None,
        reference_panel: str = "GTEX_V8",
        model_type: str = "MASHR",
        use_within_distance: bool = True,
        return_covariance: bool = False,
    ):
        """
        It computes the correlation matrix for two genes across all tissues.

        Args:
            other_gene:
                Another Gene object.
            tissues:
                A tuple with the tissue names that will be used for this gene
                (self). If None, all tissues in the given prediction model will
                be used.
            other_tissues:
                Same as 'tissues' but for 'other_gene.
            snps_subset:
                This is a subset of SNPs IDs (such as "chr1_33071920_A_C_b38")
                that will be considered to compute the correlation. All snps
                that are not present in this subset will be ignored. This is
                used to make more accurate predictions given the SNPs in a GWAS,
                for example.
            reference_panel:
                Reference panel used to compute SNP covariance matrix. Either
                GTEX_V8 or 1000G.
            model_type:
                The prediction model type, such as "MASHR" or "ELASTIC_NET" (see
                conf.py).
            use_within_distance:
                If True, it will use the function within_distance to determine
                if the correlation between a pair of genes will be computed or
                not.
            return_covariance:
                Return covariances instead of correlations.

        Returns:
            A dataframe with len(tissues) in rows and len(other_tissues) in
            columns, sorted. Tissues that have no SNPs predictors are removed.
            None if final dataframe is empty.
        """
        tissues = Gene._get_tissues(tissues, model_type)
        other_tissues = Gene._get_tissues(other_tissues, model_type)

        res = np.full((len(tissues), len(other_tissues)), fill_value=np.nan)

        for t1_idx, t1 in enumerate(tissues):
            for t2_idx, t2 in enumerate(other_tissues):
                ec = self.get_expression_correlation(
                    other_gene=other_gene,
                    tissue=t1,
                    other_tissue=t2,
                    snps_subset=snps_subset,
                    reference_panel=reference_panel,
                    model_type=model_type,
                    use_within_distance=use_within_distance,
                )

                if ec is not None and ec != 0.0 and return_covariance:
                    this_gene_variance = self.get_pred_expression_variance(
                        tissue=t1,
                        reference_panel=reference_panel,
                        model_type=model_type,
                        snps_subset=snps_subset,
                    )

                    other_gene_variance = other_gene.get_pred_expression_variance(
                        tissue=t2,
                        reference_panel=reference_panel,
                        model_type=model_type,
                        snps_subset=snps_subset,
                    )

                    ec = ec * np.sqrt(this_gene_variance * other_gene_variance)

                # ec could be None; that means that there are no SNP preditors for one
                # of the genes in the tissue
                res[t1_idx, t2_idx] = ec

        # Return a dataframe with tissues in rows and columns
        df = pd.DataFrame(res, index=tissues, columns=other_tissues)

        # remove tissues for which we don't have snps predictors for any of the genes
        df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
        if df.shape == (0, 0):
            return None

        return df

    @lru_cache(maxsize=None)
    def get_tissues_correlations_svd(
        self,
        tissues: tuple = None,
        snps_subset: frozenset = None,
        reference_panel: str = "GTEX_V8",
        model_type: str = "MASHR",
        condition_number: float = 30,
        use_covariance_matrix: bool = False,
    ):
        """
        Returns the SVD of the tissues correlations for a single gene.
        Top eigenvalues and eigenvectors are selected according to the
        condition_number given (default is S-MultiXcan default).

        Args:
            tissues:
                A tuple with the tissue names that will be used for this gene
                (self).
            snps_subset:
                This is a subset of SNPs IDs (such as "chr1_33071920_A_C_b38")
                that will be considered to compute the correlation. All snps
                that are not present in this subset will be ignored. This is
                used to make more accurate predictions given the SNPs in a GWAS,
                for example.
            reference_panel:
                Reference panel used to compute SNP covariance matrix. Either
                GTEX_V8 or 1000G.
            model_type:
                The prediction model type, such as "MASHR" or "ELASTIC_NET" (see
                conf.py).
            condition_number:
                The condition number used to select the top eigenvalues from the
                SVD decomposition, such that (max(s) / s[i]) < condition_number,
                where s is the array of eigenvalues. This is the procedure used
                in MultiXcan.
            use_covariance_matrix:
                Use covariance matrix instead of correlation matrix.

        Returns:
            A tuple with the three numpy arrays returned by numpy.linalg.svd,
            where only the top eigenvalues/eigenvectors are selected.
        """

        def _filter_eigen_values_from_max(s, ratio):
            s_max = np.max(s)
            return [i for i, x in enumerate(s) if x >= s_max * ratio]

        gene_corrs = self.get_tissues_correlations(
            other_gene=self,
            tissues=tissues,
            other_tissues=tissues,
            snps_subset=snps_subset,
            reference_panel=reference_panel,
            model_type=model_type,
            use_within_distance=True,
            return_covariance=use_covariance_matrix,
        )

        if gene_corrs is None:
            return None
        u_i, s_i, V_i = np.linalg.svd(gene_corrs)
        selected = _filter_eigen_values_from_max(s_i, 1.0 / condition_number)
        u_i = u_i[:, selected]
        s_i = s_i[selected]
        V_i = V_i[selected]

        return u_i, s_i, V_i

    def get_ssm_correlation(
        self,
        other_gene,
        tissues: tuple = None,
        other_tissues: tuple = None,
        snps_subset: frozenset = None,
        reference_panel: str = "GTEX_V8",
        model_type: str = "MASHR",
        condition_number: float = 30,
        use_within_distance: bool = True,
    ) -> float:
        """
        Computes the correlation of the model sum of squares (SSM) for a pair of
        genes assuming a null hypothesis of no association. For that, is uses
        the same predictors used by the MultiXcan approach, that is, the
        principal components of the expression matrix for one gene across
        tissues.

        The method is an adaptation of MAGMA's gene-set analysis
        (https://doi.org/10.1371/journal.pcbi.1004219) for TWAS.

        Args:
            other_gene:
                Another Gene object.
            tissues:
                A tuple with the tissue names that will be used for this gene
                (self). If None, all tissues in the given prediction model will
                be used.
            other_tissues:
                Same as 'tissues' but for 'other_gene.
            snps_subset:
                This is a subset of SNPs IDs (such as "chr1_33071920_A_C_b38")
                that will be considered to compute the correlation. All snps
                that are not present in this subset will be ignored. This is
                used to make more accurate predictions given the SNPs in a GWAS,
                for example.
            reference_panel:
                Reference panel used to compute SNP covariance matrix. Either
                GTEX_V8 or 1000G.
            model_type:
                The prediction model type, such as "MASHR" or "ELASTIC_NET" (see
                conf.py).
            condition_number:
                The condition number used to select the top eigenvalues from the
                SVD decomposition, such that (max(s) / s[i]) < condition_number,
                where s is the array of eigenvalues. This is the procedure used
                in MultiXcan.
            use_within_distance:
                If True, it will use the function within_distance to determine
                if the correlation between a pair of genes will be computed or
                not.

        Returns:
            It returns the correlation of the SSMs between two genes.
        """

        # Correlation between genes from different chromosomes is set to zero
        if self.chromosome != other_gene.chromosome:
            return 0.0

        # do not compute correlation if not within distance
        if use_within_distance and not self.within_distance(other_gene):
            return 0.0

        # SVD for this gene
        gene0_svd = self.get_tissues_correlations_svd(
            tissues=tissues,
            snps_subset=snps_subset,
            reference_panel=reference_panel,
            model_type=model_type,
            condition_number=condition_number,
        )
        if gene0_svd is None:
            return None
        u_i, s_i, V_i = gene0_svd

        # SVD for other gene
        gene1_svd = other_gene.get_tissues_correlations_svd(
            tissues=other_tissues,
            snps_subset=snps_subset,
            reference_panel=reference_panel,
            model_type=model_type,
            condition_number=condition_number,
        )
        if gene1_svd is None:
            return None
        u_j, s_j, V_j = gene1_svd

        # Compute correlations between genes
        gene0_gene1_corrs = self.get_tissues_correlations(
            other_gene=other_gene,
            tissues=tissues,
            other_tissues=other_tissues,
            snps_subset=snps_subset,
            reference_panel=reference_panel,
            model_type=model_type,
            use_within_distance=use_within_distance,
        )

        # compute covariance between SSMs of the two genes
        t0_t1_cov = (
            np.diag(s_i ** (-1 / 2))
            @ V_i
            @ gene0_gene1_corrs
            @ V_j.T
            @ np.diag(s_j ** (-1 / 2))
        )

        cov_ssm = 2 * np.trace(t0_t1_cov @ t0_t1_cov.T)

        # compute standard deviation for each gene
        t0_ssm_sd = np.sqrt(2 * V_i.shape[0])
        t1_ssm_sd = np.sqrt(2 * V_j.shape[0])

        # compute correlation between genes
        r = cov_ssm / (t0_ssm_sd * t1_ssm_sd)

        return r
