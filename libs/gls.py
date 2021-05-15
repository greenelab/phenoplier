from functools import lru_cache

import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import scale

import conf
from entity import Gene


class GLSPhenoplier(object):
    def __init__(
        self,
        smultixcan_result_set_filepath=None,
        sigma=None,
    ):
        self.smultixcan_result_set_filepath = conf.PHENOMEXCAN[
            "SMULTIXCAN_EFO_PARTIAL_MASHR_ZSCORES_FILE"
        ]
        if smultixcan_result_set_filepath is not None:
            self.smultixcan_result_set_filepath = smultixcan_result_set_filepath

        # FIXME: add in documentation that sigma should not be changed, this is just for
        #  debugging purposes.
        self.sigma = sigma

        self.lv_code = None
        self.phenotype_code = None
        self.model = None
        self.results = None
        self.results_summary = None

    @staticmethod
    @lru_cache(maxsize=None)
    def _get_data(smultixcan_result_set_filepath):
        input_filepath = (
            conf.PHENOMEXCAN["LD_BLOCKS"]["BASE_DIR"]
            / "multiplier_genes-pred_expression_corr_avg-gene_names.pkl"
        )
        gene_corrs = pd.read_pickle(input_filepath)

        input_filepath = smultixcan_result_set_filepath
        phenotype_assocs = pd.read_pickle(input_filepath)
        phenotype_assocs = phenotype_assocs.rename(index=Gene.GENE_ID_TO_NAME_MAP)
        phenotype_assocs = phenotype_assocs[
            ~phenotype_assocs.index.duplicated()
        ].dropna()
        assert phenotype_assocs.index.is_unique
        assert not phenotype_assocs.isna().any().any()

        input_filepath = conf.MULTIPLIER["MODEL_Z_MATRIX_FILE"]
        lv_weights = pd.read_pickle(input_filepath)

        common_genes = gene_corrs.index.intersection(
            phenotype_assocs.index
        ).intersection(lv_weights.index)

        return (
            gene_corrs.loc[common_genes, common_genes],
            phenotype_assocs.loc[common_genes],
            lv_weights.loc[common_genes],
        )

    def fit_named(self, lv_code, phenotype_code):
        gene_corrs, phenotype_assocs, lv_weights = GLSPhenoplier._get_data(
            self.smultixcan_result_set_filepath
        )

        if self.sigma is not None:
            import warnings

            warnings.warn(
                "Using user-provided sigma matrix. It's the user's "
                "responsability to make sure it is aligned with gene "
                "associations and module loadings"
            )
            assert self.sigma.shape[0] == self.sigma.shape[1]
            assert (
                self.sigma.shape[0] == phenotype_assocs.shape[0] == lv_weights.shape[0]
            )

            gene_corrs = self.sigma

        x = lv_weights[lv_code]
        y = phenotype_assocs[phenotype_code]
        data = pd.DataFrame({"i": 1.0, "lv": x, "phenotype": y})
        data = data.apply(lambda d: scale(d) if d.name != "i" else d)

        gls_model = sm.GLS(data["phenotype"], data[["i", "lv"]], sigma=gene_corrs)
        gls_results = gls_model.fit()

        self.lv_code = lv_code
        self.phenotype_code = phenotype_code
        self.model = gls_model

        self.results = gls_results
        self.results_summary = gls_results.summary()

        self.results.phenotype = phenotype_code
        self.results.lv = lv_code
