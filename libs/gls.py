from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.preprocessing import scale

import conf
from entity import Gene


class GLSPhenoplier(object):
    """
    Runs a generalized least squares (GLS) model with a latent variable (gene
    module) weights as predictor and a trait's gene associations (z-scores) as
    dependent variable. It account for gene predicted expression correlations
    by modeling the error term. The idea is very similar to the gene-property
    analysis from MAGMA:

      * https://doi.org/10.1371/journal.pcbi.1004219
      * https://doi.org/10.1038/s41467-018-06022-6

    After successfully trained, the object/self contains these attributes:
      * `lv_code` and `phenotype_code`: passed to the fit_named method.
      * `model`: the statsmodels's GLS object.
      * `results`: the object returned by self.model.fit()
      * `results_summary`: the object retuned by self.results.summary()

    Args:
        TODO: finish docs
        smultixcan_result_set_filepath:
            Filepath where gene-trait associations will be loaded from. It has
            to be a python pickle file containing a pandas dataframe with gene
            Ensembl IDs in rows and traits in columns. Values are expected to
            be z-scores or -log10(p-values) (although this last one was not
            tested).
        gene_corrs_file_path: path to file with gene corrs matrix; if not given, it loads the default gene correlation
            matrix trained from GTEX_V8 and MASHR models
        sigma:
            (optional) This parameter should never be modified, it is here just
            for debugging purposes. It is the sigma parameter of statsmodels's
            GLS class (the weighting matrix of the covariance). Internally, the
            gene predicted expression correlation matrix is provided as this
            argument, and this parameter allows to override that.
        logger:
            A Logger instance, the string "warnings_only" or None. If None, all logging and warning is disabled.
            If "warnings_only", then warnings will be raised as python warnings using the warnings module.
            If a Logger instance is provided, then the logger is used for everything (warnings and log messages).
            Default: "warnings_only"
    """

    def __init__(
        self,
        smultixcan_result_set_filepath: str = None,
        gene_corrs_file_path: Path = None,
        debug_use_ols: bool = False,
        debug_use_sub_gene_corr: bool = False,
        logger="warnings_only",
    ):
        self.smultixcan_result_set_filepath = conf.PHENOMEXCAN[
            "SMULTIXCAN_EFO_PARTIAL_MASHR_ZSCORES_FILE"
        ]
        if smultixcan_result_set_filepath is not None:
            self.smultixcan_result_set_filepath = smultixcan_result_set_filepath

        if gene_corrs_file_path is None:
            # by default, it loads gene correlations from GTEX_V8 and MASHR models
            input_dir_base = (
                conf.PHENOMEXCAN["LD_BLOCKS"]["GENE_CORRS_DIR"]
                / "GTEX_V8".lower()
                / "MASHR".lower()
            )

            input_filename = conf.PHENOMEXCAN["LD_BLOCKS"][
                "GENE_CORRS_FILE_NAME_TEMPLATES"
            ]["GENE_CORR_AVG"].format(
                prefix="",
                suffix=f"-mean-gene_symbols",
            )

            self.gene_corrs_file_path = input_dir_base / input_filename
        else:
            self.gene_corrs_file_path = gene_corrs_file_path
        # sigma is disabled, but left here for future reference (debugging)
        # self.sigma = sigma
        self.debug_use_ols = debug_use_ols
        self.debug_use_sub_gene_corr = debug_use_sub_gene_corr

        self.log_warning = None
        self.log_info = None
        self.set_logger(logger)

        self.lv_code = None
        self.phenotype_code = None
        self.model = None
        self.results = None
        self.results_summary = None

    def set_logger(self, logger=None):
        if logger == "warnings_only":
            import warnings

            self.log_warning = lambda x: warnings.warn(x)
            self.log_info = lambda x: None
        elif logger is None:
            self.log_warning = lambda x: None
            self.log_info = lambda x: None
        else:
            self.log_warning = logger.warning
            self.log_info = logger.info

    @staticmethod
    @lru_cache(maxsize=None)
    def _get_lv_weights(gene_loadings_file: str = None) -> pd.DataFrame:
        """
        It returns the gene loadings matrix from MultiPLIER. It contains genes in rows and LVs in columns.
        It accepts an optional file path, in that case it will load it from there. Otherwise, it returns the
        default MultiPLIER Z matrix.
        """
        # load gene loadings
        if gene_loadings_file is None:
            gene_loadings_file = conf.MULTIPLIER["MODEL_Z_MATRIX_FILE"]

        return pd.read_pickle(gene_loadings_file)

    @staticmethod
    @lru_cache(maxsize=None)
    def _get_gene_corrs(gene_corrs_file_path: str):
        """
        Returns a matrix with correlations between predicted gene expression loaded from the
        specified file.
        """
        return pd.read_pickle(gene_corrs_file_path)

    @staticmethod
    @lru_cache(maxsize=None)
    def _get_phenotype_assoc(smultixcan_result_set_filepath: str) -> pd.DataFrame:
        """
        Given a filepath pointing to the gene-trait associations file, it
        loads that one, rename gene IDs to symbols, remove duplicated gene symbols
        and returns it.

        Args:
            smultixcan_result_set_filepath:
                Filepath where gene-trait associations will be loaded from.
                We expect to have either gene symbols or Ensembl IDs in the rows
                to represent the genes. If Ensembl IDs are given, they will be
                converted to gene symbols.
        Returns:
            A pandas dataframe with gene-trait associations
        """

        # load gene-trait associations
        input_filepath = smultixcan_result_set_filepath
        phenotype_assocs = pd.read_pickle(input_filepath)
        phenotype_assocs = phenotype_assocs.rename(index=Gene.GENE_ID_TO_NAME_MAP)
        phenotype_assocs = phenotype_assocs[
            ~phenotype_assocs.index.duplicated()
        ].dropna()
        assert phenotype_assocs.index.is_unique
        assert not phenotype_assocs.isna().any().any()

        return phenotype_assocs

    @staticmethod
    def match_and_align_genes(
        gene_phenotype_assoc: pd.Series,
        gene_lv_weights: pd.Series,
        gene_correlations: pd.DataFrame = None,
    ):
        """
        Given the gene-trait associations, gene-lv weights and gene correlation matrices, it returns a version of all
        them with the same genes (present in all of them) and aligned (same order). This is used to prefer data for
        fitting the GLS model.

        Args:
            gene_phenotype_assoc: gene IDs in index
            gene_lv_weights: gene IDs in index
            gene_correlations: (optional) gene IDs in index and column

        Returns:
            A tuple with three elements: gene-trait associations, gene-lv weights and gene correlations, all
            aligned with the same genes.
        """
        common_genes = gene_phenotype_assoc.index.intersection(gene_lv_weights.index)

        if gene_correlations is not None:
            common_genes = common_genes.intersection(gene_correlations.index)

        return (
            gene_phenotype_assoc.loc[common_genes],
            gene_lv_weights.loc[common_genes],
            gene_correlations.loc[common_genes, common_genes]
            if gene_correlations is not None
            else None,
        )

    def _fit_named_internal(self, lv_code: str, phenotype: str):
        """
        Fits the GLS model with the given LV code/name and trait/phenotype
        name/code. Intended to be used internally, where phenotype associations are read from
        a results file (S-MultiXcan z-scores).

        Args:
            lv_code:
                An LV code. For example: LV136
            phenotype:
                A phenotype code that has to be present in the columns of the gene-trait association matrix.

        Returns:
            self
        """
        # obtain the needed matrices
        lv_weights = GLSPhenoplier._get_lv_weights()
        gene_corrs = GLSPhenoplier._get_gene_corrs(self.gene_corrs_file_path)
        phenotype_assocs = GLSPhenoplier._get_phenotype_assoc(
            self.smultixcan_result_set_filepath
        )

        # I leave this code here for future reference (debugging)
        #
        # if self.sigma is not None:
        #     import warnings
        #
        #     warnings.warn(
        #         "Using user-provided sigma matrix. It's the user's "
        #         "responsibility to make sure it is aligned with gene "
        #         "associations and module loadings."
        #     )
        #     assert self.sigma.shape[0] == self.sigma.shape[1]
        #     assert (
        #         self.sigma.shape[0] == phenotype_assocs.shape[0] == lv_weights.shape[0]
        #     )
        #
        #     gene_corrs = self.sigma

        # predictor
        x = lv_weights[lv_code]

        # dependent variable
        y = phenotype_assocs[phenotype]

        return self._fit_general(x, y, gene_corrs)

    def _fit_named_cli(
        self, lv_code: str, phenotype: pd.Series, lv_weights_file: str = None
    ):
        """
        TODO
        """
        lv_weights = GLSPhenoplier._get_lv_weights(lv_weights_file)
        gene_corrs = None
        if not self.debug_use_ols:
            gene_corrs = GLSPhenoplier._get_gene_corrs(self.gene_corrs_file_path)

        x = lv_weights[lv_code]

        if self.debug_use_sub_gene_corr:
            self.log_info(
                f"Using submatrix of gene correlations with nonzero genes in {lv_code}"
            )

            corr_mat_sub = pd.DataFrame(
                np.identity(gene_corrs.shape[0]),
                index=gene_corrs.index.copy(),
                columns=gene_corrs.columns.copy(),
            )

            lv_nonzero_genes = x[x > 0].index
            lv_nonzero_genes = lv_nonzero_genes.intersection(gene_corrs.index)
            corr_mat_sub.loc[lv_nonzero_genes, lv_nonzero_genes] = gene_corrs.loc[
                lv_nonzero_genes, lv_nonzero_genes
            ]

            gene_corrs = corr_mat_sub

        return self._fit_general(x, phenotype, gene_corrs)

    def _fit_general(self, x: pd.Series, y: pd.Series, gene_corrs: pd.DataFrame):
        """
        General function to fit a GLS model given the gene-trait associations, gene-LV weights and gene correlations
        matrices. It performs a series of standard steps, like removing missing data from input parameters, aligning
        genes in all three matrices and logging warnings and other checks. Then it add the results in the GLSPhenoplier
        object.

        Args:
            x: MUST HAVE a valid LV name
            y: MUST HAVE a valid name
            gene_corrs: if None, then use a standard OLS model

        Returns:
            self
        """
        lv_code = x.name
        assert lv_code is not None
        assert isinstance(
            lv_code, str
        ), "The name property of x has to have a valid LV identifier (str starting with 'LV')"
        assert lv_code.startswith(
            "LV"
        ), "The name property of x has to have a valid LV identifier (str starting with 'LV')"

        # remove missing values from gene-trait associations
        n_genes_orig = y.shape[0]
        y = y.dropna()
        n_genes_without_nan = y.shape[0]
        if n_genes_orig != n_genes_without_nan:
            self.log_warning(
                f"{n_genes_orig- n_genes_without_nan} genes with missing values have been removed"
            )

        # make sure data is aligned
        n_genes_orig_phenotype = y.shape[0]
        y, x, gene_corrs = GLSPhenoplier.match_and_align_genes(y, x, gene_corrs)

        if n_genes_orig_phenotype > y.shape[0]:
            self.log_warning(
                f"{n_genes_orig_phenotype} genes in phenotype associations, but only {y.shape[0]} were found in LV models"
            )

        # create training data
        data = pd.DataFrame({"i": 1.0, "lv": x, "phenotype": y})
        data = data.apply(lambda d: scale(d) if d.name != "i" else d)
        assert not data.isna().any().any(), "Data contains NaN"

        self.log_info(f"Final number of genes in training data: {data.shape[0]}")

        # create GLS model and fit
        if gene_corrs is not None:
            self.log_info("Using a Generalized Least Squares (GLS) model")
            gls_model = sm.GLS(data["phenotype"], data[["i", "lv"]], sigma=gene_corrs)
            gls_results = gls_model.fit()
        else:
            self.log_info("Using a Ordinary Least Squares (OLS) model")
            gls_model = sm.OLS(data["phenotype"], data[["i", "lv"]])
            gls_results = gls_model.fit()

        # add one-sided pvalue
        # in this case we are only interested in testing whether the coeficient
        # is positive (positive correlation between gene weights in an LV and
        # gene associations for a trait). We are not interested if it is
        # negative, since it does not seem to make much sense.
        gls_results.pvalues_onesided = gls_results.pvalues.copy()
        idx = gls_results.pvalues_onesided.index.tolist()
        gls_results.pvalues_onesided.loc[idx] = stats.t.sf(
            gls_results.tvalues.loc[idx], gls_results.df_resid
        )

        # save results
        self.lv_code = lv_code
        self.phenotype_code = y.name if isinstance(y, pd.Series) else None
        self.model = gls_model

        self.results = gls_results
        self.results_summary = gls_results.summary()

        return self

    def fit_named(self, lv_code: str, phenotype):
        """
        Fits the GLS model with the given LV code/name and trait/phenotype
        name/code or data. According to the type of the 'phenotype' parameter, it calls either
        '_fit_named_internal' (str) or '_fit_named_cli' (pd.Series).

        Args:
            lv_code:
                An LV code. For example: LV136
            phenotype:
                Either a phenotype code (str) that has to be present in the
                columns of the gene-trait association matrix; or the phenotype
                data itself as a pandas series (with gene symbols in index, and
                trait associations as values).

        Returns:
            self
        """
        if isinstance(phenotype, str):
            return self._fit_named_internal(lv_code, phenotype)
        elif isinstance(phenotype, pd.Series):
            return self._fit_named_cli(lv_code, phenotype)
        else:
            raise ValueError(
                "Wrong phenotype data type. Should be str or pandas.Series (with gene symbols as index)"
            )
