"""
It has code to run MultiXcan on randomly generated phenotypes and compute
the correlation between two genes' sum of squares for model (SSM).
"""
import sqlite3
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

from fastparquet import ParquetFile
from tqdm import tqdm
import pandas as pd

import conf
from entity import Gene


def get_gene_prediction_weights(gene_obj, gene_tissues, debug_messages: bool = True):
    base_prediction_model_dir = conf.PHENOMEXCAN["PREDICTION_MODELS"]["MASHR"]

    all_genes_data = []

    for tissue_name in gene_tissues:
        if debug_messages:
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


@lru_cache(maxsize=1)
def load_genotypes_from_chr(
    chromosome: int, reference_panel: str, snps_subset: set = None
):
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
        chromosome=chromosome
    )
    pf = ParquetFile(str(chr_parquet_file))
    pf_variants = set(pf.columns)
    if snps_subset is not None:
        pf_variants = snps_subset.intersection(pf_variants)

    # get individual level data
    ind_data = pd.read_parquet(
        chr_parquet_file,
        columns=["individual"] + list(pf_variants),
    )

    return ind_data, pf_variants


def predict_expression(
    gene0_id,
    gene1_id,
    gene0_tissues=None,
    gene1_tissues=None,
    snps_subset: frozenset = None,
    reference_panel: str = "1000G",
    center_gene_expr=False,
    genotypes_filepath: str = None,
    debug_messages: bool = True,
):
    if gene0_tissues is None:
        gene0_tissues = conf.PHENOMEXCAN["PREDICTION_MODELS"]["MASHR_TISSUES"].split(
            " "
        )
        gene0_tissues = sorted(gene0_tissues)

    if gene0_id != gene1_id:
        if gene1_tissues is None:
            gene1_tissues = conf.PHENOMEXCAN["PREDICTION_MODELS"][
                "MASHR_TISSUES"
            ].split(" ")
            gene1_tissues = sorted(gene1_tissues)
    else:
        gene1_tissues = gene0_tissues

    # get genes' chromosomes
    gene0_obj = Gene(ensembl_id=gene0_id)
    if gene0_id != gene1_id:
        gene1_obj = Gene(ensembl_id=gene1_id)
        assert gene0_obj.chromosome == gene1_obj.chromosome
    else:
        gene1_obj = gene0_obj

    if debug_messages:
        print(f"Genes chromosome: {gene0_obj.chromosome}")

    # get gene prediction weights
    gene0_data = get_gene_prediction_weights(gene0_obj, gene0_tissues, debug_messages)
    if gene0_id != gene1_id:
        gene1_data = get_gene_prediction_weights(
            gene1_obj, gene1_tissues, debug_messages
        )
    else:
        gene1_data = gene0_data

    if gene0_id != gene1_id:
        all_genes_data = pd.concat(gene0_data + gene1_data, axis=0)
    else:
        all_genes_data = pd.concat(gene0_data, axis=0)
    assert not all_genes_data.isna().any().any()

    # get gene variants
    gene_variants = list(set(all_genes_data["varID"].tolist()))
    if debug_messages:
        print(f"Number of unique variants: {len(gene_variants)}")
    # keep only variants in snps_subset
    if snps_subset is not None:
        gene_variants = list(snps_subset.intersection(gene_variants))
        if debug_messages:
            print(f"Number of variants after filtering: {len(gene_variants)}")

    if genotypes_filepath is not None:
        ind_data = pd.read_pickle(genotypes_filepath)
        pf_variants = set([c for c in ind_data.columns if c.startswith("chr")])
    else:
        ind_data, pf_variants = load_genotypes_from_chr(
            chromosome=int(gene0_obj.chromosome),
            reference_panel=reference_panel,
            snps_subset=snps_subset,
        )

    gene_variants = [gv for gv in gene_variants if gv in pf_variants]

    all_genes_data = all_genes_data[all_genes_data["varID"].isin(gene_variants)]
    if debug_messages:
        print(all_genes_data)

    ind_data = ind_data[["individual"] + gene_variants]

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

    if gene0_id != gene1_id:
        gene1_pred_expr = pd.DataFrame(
            {t: _predict_expression(gene1_id, t) for t in gene1_tissues}
        ).dropna(how="all", axis=1)
        assert not gene1_pred_expr.isna().any().any()
    else:
        gene1_pred_expr = gene0_pred_expr

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
ALL_TISSUES = conf.PHENOMEXCAN["PREDICTION_MODELS"]["MASHR_TISSUES"].split(" ")

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
assert gene0_pred_expr.shape[1] == len(gene0_tissues)
assert gene1_pred_expr.shape[1] == len(gene1_tissues)

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


#
# attempt to run multixcan on null results from 1000G
#

REFERENCE_PANEL = "1000G"
EQTL_MODEL = "MASHR"
PHENOTYPE_CODE = "pheno0"
SPREDIXCAN_FOLDER = Path("/opt/data/results/gls/null_sims/twas/spredixcan/").resolve()
SPREDIXCAN_FILE_PATTERN = f"random.{PHENOTYPE_CODE}-gtex_v8-mashr-" + "{tissue}.csv"
SMULTIXCAN_FILE = Path(
    f"/opt/data/results/gls/null_sims/twas/smultixcan/random.{PHENOTYPE_CODE}-gtex_v8-mashr-smultixcan.txt"
).resolve()

EQTL_MODEL_FILES_PREFIX = conf.PHENOMEXCAN["PREDICTION_MODELS"][f"{EQTL_MODEL}_PREFIX"]

prediction_model_tissues = conf.PHENOMEXCAN["PREDICTION_MODELS"][
    f"{EQTL_MODEL}_TISSUES"
].split(" ")

spredixcan_result_files = {
    t: SPREDIXCAN_FOLDER / SPREDIXCAN_FILE_PATTERN.format(tissue=t)
    for t in prediction_model_tissues
}

spredixcan_dfs = [
    pd.read_csv(
        f,
        usecols=[
            "gene",
            "zscore",
            "pvalue",
            "n_snps_used",
            "n_snps_in_model",
        ],
    )
    .dropna(subset=["gene", "zscore", "pvalue"])
    .assign(tissue=t)
    for t, f in spredixcan_result_files.items()
]

assert len(spredixcan_dfs) == len(prediction_model_tissues)

spredixcan_dfs = pd.concat(spredixcan_dfs)

spredixcan_dfs = spredixcan_dfs.assign(
    gene_id=spredixcan_dfs["gene"].apply(lambda g: g.split(".")[0])
)
spredixcan_dfs = spredixcan_dfs.set_index("gene_id").sort_index()


def get_gene_tissues(gene_id):
    gene_tissues = spredixcan_dfs.loc[[gene_id]]["tissue"]
    assert gene_tissues.is_unique
    return set(gene_tissues)


smultixcan_results = pd.read_csv(
    SMULTIXCAN_FILE, sep="\t", usecols=["gene", "gene_name", "pvalue", "n", "n_indep"]
).dropna()

smultixcan_results = smultixcan_results.assign(
    gene_id=smultixcan_results["gene"].apply(lambda g: g.split(".")[0])
)

gene_id_to_full_id_map = (
    smultixcan_results[["gene", "gene_id"]]
    .dropna()
    .set_index("gene_id")["gene"]
    .to_dict()
)

smultixcan_genes_tissues = {
    gene_id: get_gene_tissues(gene_id) for gene_id in smultixcan_results["gene_id"]
}
assert len(smultixcan_genes_tissues) == smultixcan_results.shape[0]

gwas_phenotypes = pd.read_csv(
    conf.A1000G["GENOTYPES_DIR"] / "subsets" / "all_phase3.8.random_pheno.txt",
    sep=" ",
    index_col="IID",
)
random_phenotypes = [
    gwas_phenotypes[c] for c in gwas_phenotypes.columns if c.startswith("pheno")
]

# get all variants in prediction models
mashr_models_db_files = list(
    conf.PHENOMEXCAN["PREDICTION_MODELS"][EQTL_MODEL].glob("*.db")
)
assert len(mashr_models_db_files) == 49

all_variants_ids = []

for m in mashr_models_db_files:
    print(f"Processing {m.name}")
    tissue = m.name.split(EQTL_MODEL_FILES_PREFIX)[1].split(".db")[0]

    with sqlite3.connect(m) as conn:
        df = pd.read_sql("select gene, varID from weights", conn)
        df["gene"] = df["gene"].apply(lambda x: x.split(".")[0])
        df = df.assign(tissue=tissue)

        all_variants_ids.append(df)

all_gene_snps = pd.concat(all_variants_ids, ignore_index=True)
all_gene_snps = all_gene_snps.assign(
    chr=all_gene_snps["varID"].apply(lambda x: int(x.split("chr")[1].split("_")[0]))
)
all_gene_snps = all_gene_snps.set_index("chr")
all_gene_snps = (
    all_gene_snps.sort_index()
)  # to improve performance in prediction expression
all_snps_in_models_per_chr = {
    chromosome: frozenset(all_gene_snps.loc[chromosome, "varID"])
    for chromosome in all_gene_snps.index.unique()
}

multiplier_genes = pd.read_pickle(conf.MULTIPLIER["MODEL_Z_MATRIX_FILE"]).index
multiplier_genes = set(
    [Gene(name=g).ensembl_id for g in multiplier_genes if g in Gene.GENE_NAME_TO_ID_MAP]
)
common_genes = multiplier_genes.intersection(
    set(smultixcan_results["gene_id"].tolist())
)


# predict expression for genes in all tissues
def _run_predict_expression(gene_id, genotypes_filepath):
    gene_tissues = smultixcan_genes_tissues[gene_id]

    # predict gene expression
    gene_pred_expr, _ = predict_expression(
        gene_id,
        gene_id,
        gene0_tissues=gene_tissues,
        gene1_tissues=gene_tissues,
        reference_panel=REFERENCE_PANEL,
        genotypes_filepath=genotypes_filepath,
        debug_messages=False,
    )

    return gene_pred_expr


_genes = all_gene_snps.drop_duplicates(subset=["gene"]).sort_index()
selected_genes = {
    c: [g for g in _genes.loc[c]["gene"] if g in common_genes]
    for c in _genes.index.unique().sort_values()
}

genes_predicted_expression = {}
genotypes_file = Path("/tmp/genotypes.pkl").resolve()
with ProcessPoolExecutor(max_workers=conf.GENERAL["N_JOBS"]) as executor:
    for genes_chr, genes in selected_genes.items():
        # get genotype from chromosome
        print(f"Reading genotypes for chromosome {genes_chr}")

        ind_data = load_genotypes_from_chr(
            chromosome=genes_chr,
            reference_panel=REFERENCE_PANEL,
            snps_subset=all_snps_in_models_per_chr[genes_chr],
        )[0]

        print("  Saving genotypes in temporary file")
        ind_data.to_pickle(genotypes_file)

        # run
        futures = {
            executor.submit(_run_predict_expression, gene_id, genotypes_file): gene_id
            for gene_id in genes
        }

        for fut in tqdm(as_completed(futures), total=len(genes), ncols=100):
            gene_id = futures[fut]
            gene_pred_expr = fut.result()
            genes_predicted_expression[gene_id] = gene_pred_expr


# run MultiXcan
def _run_multixcan(y, gene_id):
    gene_pred_expression = genes_predicted_expression[gene_id]

    # align y and gene_pred_expression genes
    index_inter = gene_pred_expression.index.intersection(y.index)

    y = y.loc[index_inter]
    gene_pred_expression = gene_pred_expression.loc[index_inter]

    gene_model_result, gene_data = run_multixcan(
        y,
        gene_pred_expression,
    )

    return {
        "phenotype": y.name,
        "gene": gene_id_to_full_id_map[gene_id],
        "gene_name": Gene(ensembl_id=gene_id).name,
        "pvalue": gene_model_result.f_pvalue,
        "n": gene_pred_expression.shape[1],
        "n_indep": gene_data.shape[1] - 1,  # minus intercept
    }


gene_pheno_assoc = []
current_batch_number = 0
with ProcessPoolExecutor(max_workers=conf.GENERAL["N_JOBS"]) as executor:
    futures = {
        executor.submit(_run_multixcan, y, gene_id)
        for y in random_phenotypes
        for gene_id in genes_predicted_expression.keys()
    }

    for fut_idx, fut in tqdm(
        enumerate(as_completed(futures)),
        total=int(len(random_phenotypes) * len(genes_predicted_expression)),
        ncols=100,
    ):
        pheno_gene_results = fut.result()

        gene_pheno_assoc.append(pheno_gene_results)

        # save
        if (fut_idx > 0) and (fut_idx % int(10 * len(genes_predicted_expression)) == 0):
            pd.DataFrame(gene_pheno_assoc).to_pickle(
                f"base/data/tmp/res_{current_batch_number}.pkl"
            )
            current_batch_number += 1
            gene_pheno_assoc = []

# save
pd.DataFrame(gene_pheno_assoc).to_pickle(
    f"base/data/tmp/res_{current_batch_number}.pkl"
)

# save results separately
multixcan_nulls_dir = Path("base/results/gls/null_sims/twas/multixcan/").resolve()
multixcan_nulls_dir.mkdir(exist_ok=True, parents=True)

results_files = list(Path("base/data/tmp/").glob("res_*.pkl"))
all_results = []

for res_f in results_files:
    res = pd.read_pickle(res_f)
    all_results.append(res)

all_results = pd.concat(all_results, axis=0)
phenotypes_genes_count = all_results["phenotype"].value_counts()
phenotypes_ready = phenotypes_genes_count[phenotypes_genes_count > 10].index
# assert phenotypes_genes_count.unique().shape[0] == 1

for pheno_number in phenotypes_ready:
    pheno_res = (
        all_results[all_results["phenotype"] == pheno_number]
        .drop(columns=["phenotype"])
        .sort_values("pvalue")
    )

    pheno_res.to_csv(
        multixcan_nulls_dir / f"random.{pheno_number}-gtex_v8-mashr-multixcan.txt",
        index=False,
        sep="\t",
    )
