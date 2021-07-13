from functools import lru_cache

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
        smultixcan_result_set_filepath:
            Filepath where gene-trait associations will be loaded from. It has
            to be a python pickle file containing a pandas dataframe with gene
            Ensembl IDs in rows and traits in columns. Values are expected to
            be z-scores or -log10(p-values) (although this last one was not
            tested).
        sigma:
            (optional) This parameter should never be modified, it is here just
            for debugging purposes. It is the sigma parameter of statsmodels's
            GLS class (the weighting matrix of the covariance). Internally, the
            gene predicted expression correlation matrix is provided as this
            argument, and this parameter allows to override that.
    """

    def __init__(
        self,
        smultixcan_result_set_filepath: str = None,
        sigma=None,
    ):
        self.smultixcan_result_set_filepath = conf.PHENOMEXCAN[
            "SMULTIXCAN_EFO_PARTIAL_MASHR_ZSCORES_FILE"
        ]
        if smultixcan_result_set_filepath is not None:
            self.smultixcan_result_set_filepath = smultixcan_result_set_filepath

        self.sigma = sigma

        self.lv_code = None
        self.phenotype_code = None
        self.model = None
        self.results = None
        self.results_summary = None

    @staticmethod
    @lru_cache(maxsize=None)
    def _get_data(smultixcan_result_set_filepath):
        """
        Given a filepath pointing to the gene-trait associations file, it
        loads that one and also the gene correlations and MultiPLIER Z matrix
        (gene loadings). Then it aligns genes (rows) in all three matrices, so
        it is ready to run GLS.

        Args:
            smultixcan_result_set_filepath:
                Filepath where gene-trait associations will be loaded from.
                We expect to have either gene symbols or Ensembl IDs in the rows
                to represent the genes. If Ensembl IDs are given, they will be
                converted to gene symbols.
        Returns:
            A tuple with three matrices in this order: gene correlations,
            gene-trait associations and gene loadings (Z).
        """
        # load gene correlations (with gene symbols)
        input_filepath = (
            conf.PHENOMEXCAN["LD_BLOCKS"]["BASE_DIR"]
            / "multiplier_genes-pred_expression_corr_avg-gene_names.pkl"
        )
        gene_corrs = pd.read_pickle(input_filepath)

        # load gene-trait associations
        input_filepath = smultixcan_result_set_filepath
        phenotype_assocs = pd.read_pickle(input_filepath)
        phenotype_assocs = phenotype_assocs.rename(index=Gene.GENE_ID_TO_NAME_MAP)
        phenotype_assocs = phenotype_assocs[
            ~phenotype_assocs.index.duplicated()
        ].dropna()
        assert phenotype_assocs.index.is_unique
        assert not phenotype_assocs.isna().any().any()

        # load gene loadings
        input_filepath = conf.MULTIPLIER["MODEL_Z_MATRIX_FILE"]
        lv_weights = pd.read_pickle(input_filepath)

        # get common genes and align all three matrices
        # all common genes IDs will be gene symbols at this point
        common_genes = gene_corrs.index.intersection(
            phenotype_assocs.index
        ).intersection(lv_weights.index)

        return (
            gene_corrs.loc[common_genes, common_genes],
            phenotype_assocs.loc[common_genes],
            lv_weights.loc[common_genes],
        )

    def fit_named(self, lv_code: str, phenotype_code: str):
        """
        Fits the GLS model with the given LV code/name and trait/phenotype
        code/name.

        Args:
            lv_code:
                An LV code. For example: LV136
            phenotype_code:
                A phenotype code that has to be present in the columns of the
                gene-trait association matrix.

        Returns:
            self
        """
        # obtain the needed matrices
        gene_corrs, phenotype_assocs, lv_weights = GLSPhenoplier._get_data(
            self.smultixcan_result_set_filepath
        )

        if self.sigma is not None:
            import warnings

            warnings.warn(
                "Using user-provided sigma matrix. It's the user's "
                "responsibility to make sure it is aligned with gene "
                "associations and module loadings."
            )
            assert self.sigma.shape[0] == self.sigma.shape[1]
            assert (
                self.sigma.shape[0] == phenotype_assocs.shape[0] == lv_weights.shape[0]
            )

            gene_corrs = self.sigma

        # predictor
        x = lv_weights[lv_code]
        # dependent variable
        y = phenotype_assocs[phenotype_code]

        # merge both variables plus contant (intercept) into one dataframe,
        # and scale them
        data = pd.DataFrame({"i": 1.0, "lv": x, "phenotype": y})
        data = data.apply(lambda d: scale(d) if d.name != "i" else d)

        # create GLS model and fit
        gls_model = sm.GLS(data["phenotype"], data[["i", "lv"]], sigma=gene_corrs)
        gls_results = gls_model.fit()

        # add one-sided pvalue
        gls_results.pvalues_onesided = gls_results.pvalues.copy()
        idx = gls_results.pvalues_onesided.index.tolist()
        gls_results.pvalues_onesided.loc[idx] = stats.t.sf(
            gls_results.tvalues.loc[idx],
            gls_results.df_resid
        )

        # save results
        self.lv_code = lv_code
        self.phenotype_code = phenotype_code
        self.model = gls_model

        self.results = gls_results
        self.results_summary = gls_results.summary()

        return self
