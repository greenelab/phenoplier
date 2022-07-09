import sqlite3

from fastparquet import ParquetFile
from tqdm import tqdm

import conf
from entity import Gene


def predict_expression(gene0_id, gene1_id, center_gene_expr=False):
    all_tissues = conf.PHENOMEXCAN["PREDICTION_MODELS"]["MASHR_TISSUES"].split(" ")
    all_tissues = sorted(all_tissues)
    print(f"Number of tissues {len(all_tissues)}")
    print(all_tissues[:5])

    # get genes' chromosomes
    gene0_obj = Gene(ensembl_id=gene0_id)
    gene1_obj = Gene(ensembl_id=gene1_id)
    assert gene0_obj.chromosome == gene1_obj.chromosome
    print(f"Genes chromosome: {gene0_obj.chromosome}")

    # get gene prediction weights
    base_prediction_model_dir = conf.PHENOMEXCAN["PREDICTION_MODELS"]["MASHR"]
    all_genes_data = []
    for tissue_name in all_tissues:
        print(tissue_name)

        input_db_file = base_prediction_model_dir / f"mashr_{tissue_name}.db"
        with sqlite3.connect(input_db_file) as cnx:
            gene0 = pd.read_sql_query(
                f'select * from weights where gene like "{gene0_id}.%"', cnx
            )
            if gene0.shape[0] > 0:
                gene0["tissue"] = tissue_name
                all_genes_data.append(gene0)

            gene1 = pd.read_sql_query(
                f'select * from weights where gene like "{gene1_id}.%"', cnx
            )
            if gene1.shape[0] > 0:
                gene1["tissue"] = tissue_name
                all_genes_data.append(gene1)

    all_genes_data = pd.concat(all_genes_data, axis=0)
    assert not all_genes_data.isna().any().any()

    # get gene variants
    gene_variants = list(set(all_genes_data["varID"].tolist()))
    print(f"Number of variants: {len(gene_variants)}")

    # get intersection of gene variants with variants in parquet file
    base_reference_panel_dir = conf.PHENOMEXCAN["LD_BLOCKS"]["1000G_GENOTYPE_DIR"]

    chr_parquet_file = (
        base_reference_panel_dir / f"chr{gene0_obj.chromosome}.variants.parquet"
    )
    pf = ParquetFile(str(chr_parquet_file))
    pf_variants = set(pf.columns)
    gene_variants = [gv for gv in gene_variants if gv in pf_variants]

    all_genes_data = all_genes_data[all_genes_data["varID"].isin(gene_variants)]
    print(all_genes_data)

    # get individual level data
    ind_data = pd.read_parquet(
        chr_parquet_file,
        columns=["individual"] + gene_variants,
    )

    # predict expression for the two genes
    def _predict_expression(gene_id, tissue_name):
        gene_data = all_genes_data[
            all_genes_data["gene"].str.startswith(gene_id + ".")
            & (all_genes_data["tissue"] == tissue_name)
        ]

        gene_expr = (
            ind_data[["individual"] + gene_data["varID"].tolist()].set_index(
                "individual"
            )
            @ gene_data[["varID", "weight"]].set_index("varID")
        ).squeeze()

        if gene_expr.sum() == 0.0:
            return None

        if center_gene_expr:
            gene_expr = gene_expr - gene_expr.mean()

        return gene_expr

    gene0_pred_expr = pd.DataFrame(
        {t: _predict_expression(gene0_id, t) for t in all_tissues}
    ).dropna(how="all", axis=1)
    assert not gene0_pred_expr.isna().any().any()
    gene1_pred_expr = pd.DataFrame(
        {t: _predict_expression(gene1_id, t) for t in all_tissues}
    ).dropna(how="all", axis=1)
    assert not gene1_pred_expr.isna().any().any()

    return gene0_pred_expr, gene1_pred_expr


# MultiXcan


# modules
from patsy import dmatrices
import numpy as np
from numpy import dot as _dot
import statsmodels.api as sm
import pandas as pd

from sklearn.preprocessing import scale

# functions
def _design_matrices(e_, keys):
    formula = "pheno ~ {}".format(" + ".join(keys))
    y, X = dmatrices(formula, data=e_, return_type="dataframe")
    return y, X


def _filter_eigen_values_from_max(s, ratio):
    s_max = np.max(s)
    return [i for i, x in enumerate(s) if x >= s_max * ratio]


def pc_filter(x, cond_num=30):
    return _filter_eigen_values_from_max(x, 1.0 / cond_num)


class Math:
    def standardize(x, unit_var=True):
        mean = np.mean(x)
        # follow R's convention, ddof=1
        scale = np.std(x, ddof=1)
        if scale == 0:
            return None
        x = x - mean
        if unit_var:
            x = x / scale
        return x


def _get_pc_input(e_, model_keys, unit_var=True):
    Xc = []
    _mk = []
    for key in model_keys:
        x = Math.standardize(e_[key], unit_var)
        if x is not None:
            Xc.append(x)
            _mk.append(key)
    return Xc, _mk


def _pca_data(e_, model_keys, unit_var=True):
    if e_.shape[1] == 2:
        return e_, model_keys, model_keys, 1, 1, 1, 1, 1
    # numpy.svd can't handle typical data size in UK Biobank. So we do PCA through the covariance matrix
    # That is: we compute ths SVD of a covariance matrix, and use those coefficients to get the SVD of input data
    # Shamelessly designed from https://stats.stackexchange.com/questions/134282/relationship-between-svd-and-pca-how-to-use-svd-to-perform-pca
    # In numpy.cov, each row is a variable and each column an observation. Exactly opposite to standard PCA notation: it is transposed, then.
    Xc_t, original_keys = _get_pc_input(e_, model_keys, unit_var)
    k = np.cov(Xc_t)
    u, s, vt = np.linalg.svd(k)
    # we want to keep only those components with significant variance, to reduce dimensionality
    selected = pc_filter(s)

    variance = s[selected]
    vt_projection = vt[selected]
    Xc_t_ = _dot(vt_projection, Xc_t)
    pca_keys = ["pc{}".format(i) for i in range(0, len(selected))]
    _data = {pca_keys[i]: x for i, x in enumerate(Xc_t_)}
    _data["pheno"] = e_.pheno
    pca_data = pd.DataFrame(_data)
    return (
        pca_data,
        pca_keys,
        original_keys,
        np.max(s),
        np.min(s),
        np.min(s[selected]),
        vt_projection,
        variance,
    )


def run_multixcan(y, gene_pred_expr):
    model_keys = gene_pred_expr.columns.tolist()

    e_ = gene_pred_expr.assign(pheno=y)

    e_, model_keys, *_tmp_rest = _pca_data(e_, model_keys)

    y, X = _design_matrices(e_, model_keys)

    model = sm.OLS(y, X)
    result = model.fit()
    return result, X


def get_y_hat(multixcan_model_result, X_data):
    return multixcan_model_result.predict(X_data)


def get_ssm(multixcan_model_result, X_data, y_data):
    y_hat = get_y_hat(multixcan_model_result, X_data)
    return np.power(y_hat - y_data.mean(), 2).sum()


# Main
# # FGR (1p35.3) and AK2 (1p35.1)
# gene0_id = "ENSG00000000938"
# gene1_id = "ENSG00000004455"

# # COL4A2 (13q34) and COL4A1 (13q34)
# gene0_id = "ENSG00000134871"
# gene1_id = "ENSG00000187498"

# # NOC2L (1p36.33) and HES4 (1p36.33)
# gene0_id = "ENSG00000188976"
# gene1_id = "ENSG00000188290"

# # ARSA (22q13.33) and SHANK3 (22q13.33)
# gene0_id = "ENSG00000100299"
# gene1_id = "ENSG00000251322"

# # IRF4 (6p25.3) and TBP (6q27)
# gene0_id = "ENSG00000137265"
# gene1_id = "ENSG00000112592"

# # IKZF3 (17q21.1) and PNMT (17q12)
# gene0_id = "ENSG00000161405"
# gene1_id = "ENSG00000141744"

# # CCL2 (17q12) and CCL7 (17q12)
# gene0_id = "ENSG00000108691"
# gene1_id = "ENSG00000108688"

# # CCL2 (17q12) and CCL8 (17q12)
# gene0_id = "ENSG00000108691"
# gene1_id = "ENSG00000108700"

# # HIST2H2BF (1q21.2) and HIST3H2A (1q42.13)
# gene0_id = "ENSG00000203814"
# gene1_id = "ENSG00000181218"

# # HIST2H2BF (1q21.2) and HIST3H2BB (1q42.13)
# gene0_id = "ENSG00000203814"
# gene1_id = "ENSG00000196890"

# # HIST3H2A (1q42.13) and HIST3H2BB (1q42.13)
# gene0_id = "ENSG00000181218"
# gene1_id = "ENSG00000196890"

# # HIST1H2BC (6p22.2) and HIST1H2AC (6p22.2)
# gene0_id = "ENSG00000180596"
# gene1_id = "ENSG00000180573"

# # HIST1H2BO (6p22.1) and HIST1H2BK (6p22.1)
# gene0_id = "ENSG00000274641"
# gene1_id = "ENSG00000197903"

# HIST1H2BO (6p22.1) and HIST1H2BF (6p22.2)
gene0_id = "ENSG00000274641"
gene1_id = "ENSG00000277224"


N_PHENOTYPES = 10000

gene0_obj = Gene(ensembl_id=gene0_id)
gene1_obj = Gene(ensembl_id=gene1_id)

gene0_pred_expr, gene1_pred_expr = predict_expression(gene0_id, gene1_id)

rs = np.random.RandomState(0)

random_phenotypes = []
for pheno_i in range(N_PHENOTYPES):
    y = pd.Series(
        rs.normal(size=gene0_pred_expr.shape[0]), index=gene0_pred_expr.index.tolist()
    )
    # y = y - y.mean()
    random_phenotypes.append(y)

gene0_ssms = []
gene1_ssms = []
for y_idx, y in tqdm(
    enumerate(random_phenotypes), total=len(random_phenotypes), ncols=100
):
    gene0_model_result, gene0_data = run_multixcan(y, gene0_pred_expr)
    gene1_model_result, gene1_data = run_multixcan(y, gene1_pred_expr)

    gene0_ssms.append(get_ssm(gene0_model_result, gene0_data, y))
    gene1_ssms.append(get_ssm(gene1_model_result, gene1_data, y))

gene0_ssms = pd.Series(gene0_ssms)
gene1_ssms = pd.Series(gene1_ssms)

print(f"Correlation from null: {gene0_ssms.corr(gene1_ssms)}")


n = gene0_pred_expr.shape[0]

# code to compute ssms correlation using MAGMA method
# in the original formula, the numerator is really the predictors but scaled
gene0_pcs = gene0_data.drop(columns=["Intercept"]).apply(scale)
gene1_pcs = gene1_data.drop(columns=["Intercept"]).apply(scale)

cov_ssm = 2 * np.trace(gene0_pcs.T @ gene1_pcs @ gene1_pcs.T @ gene0_pcs)
t0_ssm_sd = np.sqrt(2 * gene0_pcs.shape[1]) * (gene0_pcs.shape[0] - 1)
t1_ssm_sd = np.sqrt(2 * gene1_pcs.shape[1]) * (gene1_pcs.shape[0] - 1)
print(f"Correlation from genotype: {cov_ssm / (t0_ssm_sd * t1_ssm_sd)}")


# modified version to use covariance in the numerator
gene0_pcs = gene0_data.drop(columns=["Intercept"]).apply(scale)
gene1_pcs = gene1_data.drop(columns=["Intercept"]).apply(scale)

gene0_pcs_gene1_pcs_cov = (gene0_pcs.T @ gene1_pcs) / (n - 1)

cov_ssm = 2 * np.trace(gene0_pcs_gene1_pcs_cov @ gene0_pcs_gene1_pcs_cov.T)
t0_ssm_sd = np.sqrt(2 * gene0_pcs.shape[1])
t1_ssm_sd = np.sqrt(2 * gene1_pcs.shape[1])
print(f"Correlation from genotype: {cov_ssm / (t0_ssm_sd * t1_ssm_sd)}")


# FINAL VERSION ??? without using predicted gene expression
def _filter_eigen_values_from_max(s, ratio):
    s_max = np.max(s)
    return [i for i, x in enumerate(s) if x >= s_max * ratio]


def pc_filter(x, cond_num):
    return _filter_eigen_values_from_max(x, 1.0 / cond_num)


reference_panel = "1000G"
model_type = "MASHR"
condition_number = 30
use_within_distance = False

# these are for debugging only
T_i = gene0_pred_expr.apply(scale)
T_j = gene1_pred_expr.apply(scale)

gene0_corrs = gene0_obj.get_tissues_correlations(
    gene0_obj,
    reference_panel=reference_panel,
    model_type=model_type,
    use_within_distance=use_within_distance,
)
u_i, s_i, V_i = np.linalg.svd(gene0_corrs)
selected = pc_filter(s_i, condition_number)
s_i = s_i[selected]
V_i = V_i[selected]

gene1_corrs = gene1_obj.get_tissues_correlations(
    gene1_obj,
    reference_panel=reference_panel,
    model_type=model_type,
    use_within_distance=use_within_distance,
)
u_j, s_j, V_j = np.linalg.svd(gene1_corrs)
selected = pc_filter(s_j, condition_number)
s_j = s_j[selected]
V_j = V_j[selected]

gene0_gene1_corrs = gene0_obj.get_tissues_correlations(
    gene1_obj,
    reference_panel=reference_panel,
    model_type=model_type,
    use_within_distance=use_within_distance,
)

t0_t1_cov = (
    np.diag(s_i ** (-1 / 2))
    @ V_i
    @ gene0_gene1_corrs
    @ V_j.T
    @ np.diag(s_j ** (-1 / 2))
)

cov_ssm = 2 * np.trace(t0_t1_cov @ t0_t1_cov.T)
t0_ssm_sd = np.sqrt(2 * V_i.shape[0])
t1_ssm_sd = np.sqrt(2 * V_j.shape[0])

print(f"Correlation final: {cov_ssm / (t0_ssm_sd * t1_ssm_sd)}")
