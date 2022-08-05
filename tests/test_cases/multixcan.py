"""
It has code to run MultiXcan on randomly generated phenotypes and compute
the correlation between two genes' sum of squares for model (SSM).
"""
import sqlite3
from concurrent.futures import ProcessPoolExecutor, as_completed

from fastparquet import ParquetFile
from tqdm import tqdm

import conf
from entity import Gene


def get_gene_prediction_weights(gene_obj, gene_tissues):
    base_prediction_model_dir = conf.PHENOMEXCAN["PREDICTION_MODELS"]["MASHR"]

    all_genes_data = []

    for tissue_name in gene_tissues:
        print(tissue_name)

        input_db_file = base_prediction_model_dir / f"mashr_{tissue_name}.db"
        with sqlite3.connect(input_db_file) as cnx:
            gene0 = pd.read_sql_query(
                f'select * from weights where gene like "{gene_obj.ensembl_id}.%"', cnx
            )
            if gene0.shape[0] > 0:
                gene0["tissue"] = tissue_name
                all_genes_data.append(gene0)

    if len(all_genes_data) == 0:
        raise ValueError(f"No predictor SNPs for gene {gene_obj.ensembl_id}")

    return all_genes_data


def predict_expression(
    gene0_id,
    gene1_id,
    gene0_tissues=None,
    gene1_tissues=None,
    snps_subset: set = None,
    reference_panel: str = "1000G",
    center_gene_expr=False,
):
    if gene0_tissues is None:
        gene0_tissues = conf.PHENOMEXCAN["PREDICTION_MODELS"]["MASHR_TISSUES"].split(
            " "
        )
        gene0_tissues = sorted(gene0_tissues)

    if gene1_tissues is None:
        gene1_tissues = conf.PHENOMEXCAN["PREDICTION_MODELS"]["MASHR_TISSUES"].split(
            " "
        )
        gene1_tissues = sorted(gene1_tissues)

    # get genes' chromosomes
    gene0_obj = Gene(ensembl_id=gene0_id)
    gene1_obj = Gene(ensembl_id=gene1_id)
    assert gene0_obj.chromosome == gene1_obj.chromosome
    print(f"Genes chromosome: {gene0_obj.chromosome}")

    # get gene prediction weights
    gene0_data = get_gene_prediction_weights(gene0_obj, gene0_tissues)
    gene1_data = get_gene_prediction_weights(gene1_obj, gene1_tissues)

    all_genes_data = pd.concat(gene0_data + gene1_data, axis=0)
    assert not all_genes_data.isna().any().any()

    # get gene variants
    gene_variants = list(set(all_genes_data["varID"].tolist()))
    print(f"Number of unique variants: {len(gene_variants)}")
    # keep only variants in snps_subset
    if snps_subset is not None:
        gene_variants = list(snps_subset.intersection(gene_variants))
        print(f"Number of variants after filtering: {len(gene_variants)}")

    # get intersection of gene variants with variants in parquet file
    base_reference_panel_dir = conf.PHENOMEXCAN["LD_BLOCKS"][
        f"{reference_panel}_GENOTYPE_DIR"
    ]

    if reference_panel == "1000G":
        chr_file_template = "chr{chromosome}.variants.parquet"
    elif reference_panel == "GTEX_V8":
        chr_file_template = "gtex_v8_eur_filtered_maf0.01_monoallelic_variants.chr{chromosome}.variants.parquet"
    else:
        raise ValueError(f"Invalid reference panel: {reference_panel}")

    chr_parquet_file = base_reference_panel_dir / chr_file_template.format(
        chromosome=gene0_obj.chromosome
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
        ].drop_duplicates(
            subset=["gene", "varID", "tissue"]
        )  # needed when the same gene/tissues are given

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
        {t: _predict_expression(gene0_id, t) for t in gene0_tissues}
    ).dropna(how="all", axis=1)
    assert not gene0_pred_expr.isna().any().any()

    gene1_pred_expr = pd.DataFrame(
        {t: _predict_expression(gene1_id, t) for t in gene1_tissues}
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

    return (pca_data, pca_keys, selected, u, s, vt)

    # original return:
    # return (
    #     pca_data,
    #     pca_keys,
    #     original_keys,
    #     np.max(s),
    #     np.min(s),
    #     np.min(s[selected]),
    #     vt_projection,
    #     variance,
    # )


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

N_PHENOTYPES = 10000
REFERENCE_PANEL = "GTEX_V8"
# ALL_TISSUES = conf.PHENOMEXCAN["PREDICTION_MODELS"]["MASHR_TISSUES"].split(" ")

gene0_id = "ENSG00000180596"
gene0_tissues = ("Small_Intestine_Terminal_Ileum", "Uterus")
gene1_id = "ENSG00000180573"
gene1_tissues = (
    "Brain_Cerebellum",
    "Esophagus_Gastroesophageal_Junction",
    "Artery_Coronary",
)
# when interested in one gene only, the following code can be useful to
# "ignore" gene1
# gene1_id = gene0_id
# gene1_tissues = gene0_tissues

# code to select all tissues
# gene0_tissues = ALL_TISSUES

# code to disable snps_subset:
# snps_subset = None

snps_subset = {
    # first gene:
    #  Small_Intestine_Terminal_Ileum
    # "chr6_26124075_T_C_b38",  # remove, this removes this tissue
    "chr6_26124202_C_T_b38",  # this SNP is not in the genotype
    #  Uterus
    # "chr6_26124075_T_C_b38",  # (same as other one)
    "chr6_26124406_C_T_b38",
    # second gene:
    #  Brain_Cerebellum
    # "chr6_26124075_T_C_b38",  # (same as other one)
    "chr6_26124015_G_A_b38",
    "chr6_26124406_C_T_b38",
    #  Esophagus_Gastroesophageal_Junction
    # "chr6_26124075_T_C_b38",  # (same as other one)
    "chr6_26124015_G_A_b38",
    # Artery_Coronary
    # "chr6_26124075_T_C_b38",  # (same as other one)
    "chr6_26124015_G_A_b38",
}

gene0_obj = Gene(ensembl_id=gene0_id)
gene1_obj = Gene(ensembl_id=gene1_id)

# to quickly test whether genes have predictors in tissues:
#     get_gene_prediction_weights(gene0_obj, gene0_tissues)
# or to explore how many snps are in each model across all tissues
#     all_tissues = conf.PHENOMEXCAN["PREDICTION_MODELS"][
#                   f"MASHR_TISSUES"
#     ].split(" ")
#
#     for t in all_tissues:
#         try:
#             _tmp = get_gene_prediction_weights(gene1_obj, (t,))[0]
#             print(f"  {_tmp.shape[0]}", flush=True)
#         except:
#             print(f"[FAILED] {t}")

# predict expression
gene0_pred_expr, gene1_pred_expr = predict_expression(
    gene0_id,
    gene1_id,
    gene0_tissues=gene0_tissues,
    gene1_tissues=gene1_tissues,
    snps_subset=snps_subset,
    reference_panel=REFERENCE_PANEL,
)
# assert gene0_pred_expr.shape[1] == len(gene0_tissues)
# assert gene1_pred_expr.shape[1] == len(gene1_tissues)

# generate random phenotypes
rs = np.random.RandomState(0)

random_phenotypes = []
for pheno_i in range(N_PHENOTYPES):
    y = pd.Series(
        rs.normal(size=gene0_pred_expr.shape[0]), index=gene0_pred_expr.index.tolist()
    )
    # y = y - y.mean()
    random_phenotypes.append(y)

# run multixcan, get SSMs
def _run_job(y):
    gene0_model_result, gene0_data = run_multixcan(y, gene0_pred_expr)
    gene1_model_result, gene1_data = run_multixcan(y, gene1_pred_expr)

    return (
        get_ssm(gene0_model_result, gene0_data, y),
        gene0_data,
        get_ssm(gene1_model_result, gene1_data, y),
        gene1_data,
    )


gene0_ssms = []
gene1_ssms = []
with ProcessPoolExecutor(max_workers=conf.GENERAL["N_JOBS"]) as executor:
    futures = {executor.submit(_run_job, y) for y in random_phenotypes}

    for fut in tqdm(as_completed(futures), total=len(random_phenotypes), ncols=100):
        gene0_result, gene0_data, gene1_result, gene1_data = fut.result()

        gene0_ssms.append(gene0_result)
        gene1_ssms.append(gene1_result)

gene0_ssms = pd.Series(gene0_ssms)
gene1_ssms = pd.Series(gene1_ssms)

# compute empirical correlation between SSMs
print(f"Correlation from null: {gene0_ssms.corr(gene1_ssms)}")


# code to compute ssms correlation using MAGMA method
# in the original formula, the numerator is really the predictors but scaled

gene0_pcs = gene0_data.drop(columns=["Intercept"]).apply(scale)
gene1_pcs = gene1_data.drop(columns=["Intercept"]).apply(scale)

cov_ssm = 2 * np.trace(gene0_pcs.T @ gene1_pcs @ gene1_pcs.T @ gene0_pcs)
t0_ssm_sd = np.sqrt(2 * gene0_pcs.shape[1]) * (gene0_pcs.shape[0] - 1)
t1_ssm_sd = np.sqrt(2 * gene1_pcs.shape[1]) * (gene1_pcs.shape[0] - 1)
print(f"Correlation from genotype: {cov_ssm / (t0_ssm_sd * t1_ssm_sd)}")
