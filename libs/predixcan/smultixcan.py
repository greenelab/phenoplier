# from functools import lru_cache

import numpy as np
from numpy.core import asarray, product, dot, transpose, multiply, newaxis
import pandas as pd

from entity import Gene
from predixcan.expression_prediction import load_genotypes_from_chr
from predixcan.multixcan import _pca_data, _design_matrices


class RegressionResults(object):
    pass


class CutoffEigenRatio(object):
    def __init__(self, cutoff_ratio):
        self.cutoff_ratio = float(cutoff_ratio)

    def __call__(self, matrix):
        # conceptual shotcut
        if self.cutoff_ratio == 0:
            return 0.0
        w, v = np.linalg.eigh(matrix)
        w = -np.sort(-w)
        cutoff = self.cutoff_ratio * w[0]
        return cutoff


def capinv(a, condition_number=30, epsilon=None):
    """Pseudo inverse, absolute cutoff"""
    rcond = CutoffEigenRatio(1.0 / condition_number)
    return _inv(a, rcond(a), epsilon)


def _inv(a, rcond, epsilon=None):
    """
    modified pseudo inverse
    """

    def _assertNoEmpty2d(*arrays):
        for a in arrays:
            if a.size == 0 and product(a.shape[-2:]) == 0:
                raise RuntimeError("Arrays cannot be empty")

    def _makearray(a):
        new = asarray(a)
        wrap = getattr(a, "__array_prepare__", new.__array_wrap__)
        return new, wrap

    a, wrap = _makearray(a)
    _assertNoEmpty2d(a)

    if epsilon is not None:
        epsilon = np.repeat(epsilon, a.shape[0])
        epsilon = np.diag(epsilon)
        a = a + epsilon
    a = a.conjugate()
    # WARNING! the "s" eigenvalues might not equal the eigenvalues of eigh
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    m = u.shape[0]
    n = vt.shape[1]
    eigen = np.copy(s)

    # cutoff = rcond*maximum.reduce(s)
    cutoff = rcond  # cf(s, rcond)
    for i in range(min(n, m)):
        # The first Singular Value will always be selected because we want at
        # least one, and the first is the highest
        if s[i] >= cutoff or i == 0:
            s[i] = 1.0 / s[i]
        else:
            s[i] = 0.0

    n_indep = np.count_nonzero(s)
    res = dot(transpose(vt), multiply(s[:, newaxis], transpose(u)))
    return wrap(res), n_indep, eigen


# @lru_cache(maxsize=None)
def get_gwas_betas(y, snp_ids, reference_panel):
    # FIXME: a variant of the approach here is that in S-PrediXcan they use the
    #  variance of a SNP from a subset of GTEx samples

    snp_chrs = [int(s.split("chr")[1].split("_")[0]) for s in snp_ids]
    assert len(set(snp_chrs)) == 1
    chromosome = snp_chrs[0]

    ind_data = load_genotypes_from_chr(
        chromosome=chromosome,
        reference_panel=reference_panel,
        snps_subset=frozenset(snp_ids),
    )[0]
    ind_data = ind_data.set_index("individual")

    snp_vars = ind_data.var()

    return pd.Series({s: (ind_data[s].cov(y) / snp_vars[s]) for s in ind_data.columns})


def get_spredixcan_betas(
    y, gene_obj, gene_tissues, reference_panel, model_type, snps_subset
):
    betas = []
    tissues = []

    # first, get all gene's predictor SNPs across all tissues
    gene_snps_weights = []
    for t in gene_tissues:
        gene_weights = gene_obj.get_prediction_weights(
            tissue=t,
            model_type=model_type,
            snps_subset=snps_subset,
        ).to_frame()

        gene_weights = gene_weights.assign(tissue=t)
        gene_snps_weights.append(gene_weights)

    gene_snps_weights_df = pd.concat(gene_snps_weights, axis=0)

    # now get the GWAS effect sizes for all SNPs
    unique_snps = sorted(list(set(gene_snps_weights_df.index)))
    all_gwas_betas = get_gwas_betas(
        y, snp_ids=unique_snps, reference_panel=reference_panel
    )

    # organize by tissue and varID
    gene_snps_weights_df = (
        gene_snps_weights_df.reset_index().set_index(["tissue", "varID"]).sort_index()
    )
    gene_snps_weights_df = gene_snps_weights_df["weight"]

    for t in gene_tissues:
        if t not in gene_snps_weights_df.index:
            continue

        gene_weights = gene_snps_weights_df.loc[t]

        gene_snps_var = Gene.get_snps_variance(
            tissue=t,
            snps_list=tuple(gene_weights.index),
            model_type=model_type,
        ).rename_axis(gene_weights.index.name)

        gwas_betas = all_gwas_betas.loc[gene_weights.index]

        gene_expr_var = gene_obj.get_pred_expression_variance(
            tissue=t,
            reference_panel=reference_panel,
            model_type=model_type,
            snps_subset=snps_subset,
        )
        gene_snps_var = gene_snps_var.loc[gene_weights.index]

        spredixcan_beta = ((gene_weights * gwas_betas) @ gene_snps_var) / gene_expr_var
        betas.append(spredixcan_beta)

        tissues.append(t)

    return pd.Series(betas, index=tissues)


# def get_spredixcan_betas(
#     random_phenotype_number, gene_id
# ):
#     pass


def run_smultixcan(
    y,
    gene_pred_expr,
    gene_obj,
    # gene_tissues=None,
    snps_subset=None,
    reference_panel="GTEX_V8",
    model_type="MASHR",
    # unit_var=True,
):
    # assert y.name is not None and len(y.name) > 0, "Phenotype array needs a name"

    # n = y.shape[0]

    gene_tissues = gene_pred_expr.columns.tolist()

    # gene_obj = Gene(ensembl_id=gene_id)
    gene_corrs = gene_obj.get_tissues_correlations(
        other_gene=gene_obj,
        tissues=frozenset(gene_tissues),
        other_tissues=frozenset(gene_tissues),
        snps_subset=snps_subset,
        reference_panel=reference_panel,
        model_type=model_type,
    )
    # n_tissues = gene_corrs.shape[0]
    # gene_corrs = gene_pred_expr.corr()

    # d = np.diag(np.ones(n_tissues) * (n - 1))
    gene_corrs_inv, n_indep, eigen = capinv(gene_corrs)

    spredixcan_betas = get_spredixcan_betas(
        y=y,
        gene_obj=gene_obj,
        gene_tissues=gene_tissues,
        reference_panel=reference_panel,
        model_type=model_type,
        snps_subset=snps_subset,
    )

    # FIXME: I'm not sure if this is equivalent, because what we need are the
    #  effect sizes of the PCs, not the effect sizes of each gene-tissue expresion.
    #  if we use this formula, is it the same/equivalent?
    # smultixcan_betas = (1.0 / (n - 1)) * gene_corrs_inv @ d @ spredixcan_betas
    smultixcan_betas = gene_corrs_inv @ spredixcan_betas

    # compute fitted values using the S-MultiXcan betas (derived from S-PrediXcan
    #  betas).
    # model_keys = gene_pred_expr.columns.tolist()
    # e_ = gene_pred_expr.assign(pheno=y)
    # e_, model_keys, *_tmp_rest = _pca_data(e_, model_keys, unit_var=unit_var)
    # assert (e_.shape[1] - 1) == n_indep

    # y, X = _design_matrices(e_, model_keys)
    # X = X.drop(columns="Intercept")

    result = RegressionResults()
    result.fittedvalues = gene_pred_expr @ smultixcan_betas

    # model = sm.OLS(y, X)
    # result = model.fit()
    return result  # , X, y
