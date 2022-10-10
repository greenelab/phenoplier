"""
It has code to run MultiXcan on randomly generated phenotypes and compute
the correlation between two genes' sum of squares for model (SSM).
"""
import sqlite3
import pickle
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

from predixcan.expression_prediction import load_genotypes_from_chr, predict_expression
from predixcan.multixcan import run_multixcan, get_ssm
import conf
from entity import Gene


def scale(x):
    return (x - x.mean()) / x.std(ddof=1)


#
# Compute correlations between two genes
#

N_PHENOTYPES = 10000
REFERENCE_PANEL = "GTEX_V8"
ALL_TISSUES = conf.PHENOMEXCAN["PREDICTION_MODELS"]["MASHR_TISSUES"].split(" ")

# this is for testing purposes, but MultiXcan standardizes (x - mean / std) gene
# expression data before running PCA
USE_CORR_MATRIX_BEFORE_PCA = True

gene0_id = "ENSG00000000938"
gene0_tissues = tuple(ALL_TISSUES)
gene1_id = "ENSG00000004455"
gene1_tissues = tuple(ALL_TISSUES)
# when interested in one gene only, the following code can be useful to
# "ignore" gene1
# gene1_id = gene0_id
# gene1_tissues = gene0_tissues

# code to select all tissues
# gene0_tissues = ALL_TISSUES

# code to disable snps_subset:
snps_subset = None

# snps_subset = {
#     # first gene:
#     #  Small_Intestine_Terminal_Ileum
#     # "chr6_26124075_T_C_b38",  # remove, this removes this tissue
#     "chr6_26124202_C_T_b38",  # this SNP is not in the genotype
#     #  Uterus
#     # "chr6_26124075_T_C_b38",  # (same as other one)
#     "chr6_26124406_C_T_b38",
#     # second gene:
#     #  Brain_Cerebellum
#     # "chr6_26124075_T_C_b38",  # (same as other one)
#     "chr6_26124015_G_A_b38",
#     "chr6_26124406_C_T_b38",
#     #  Esophagus_Gastroesophageal_Junction
#     # "chr6_26124075_T_C_b38",  # (same as other one)
#     "chr6_26124015_G_A_b38",
#     # Artery_Coronary
#     # "chr6_26124075_T_C_b38",  # (same as other one)
#     "chr6_26124015_G_A_b38",
# }

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
# the checks below might not be necessary if you are considering all tissues
# assert gene0_pred_expr.shape[1] == len(gene0_tissues)
# assert gene1_pred_expr.shape[1] == len(gene1_tissues)

# generate random phenotypes
rs = np.random.RandomState(0)

random_phenotypes = []
for pheno_i in range(N_PHENOTYPES):
    y = pd.Series(
        rs.normal(size=gene0_pred_expr.shape[0]), index=gene0_pred_expr.index.tolist()
    ).rename(f"pheno{pheno_i}")
    random_phenotypes.append(y)

# run multixcan, get SSMs
def _run_job(y, y_idx):
    gene0_model_result, gene0_data, _ = run_multixcan(
        y, gene0_pred_expr, unit_var=USE_CORR_MATRIX_BEFORE_PCA
    )
    gene1_model_result, gene1_data, _ = run_multixcan(
        y, gene1_pred_expr, unit_var=USE_CORR_MATRIX_BEFORE_PCA
    )

    return (
        y_idx,
        get_ssm(gene0_model_result, y),
        gene0_data,
        get_ssm(gene1_model_result, y),
        gene1_data,
    )


y_indices = []
gene0_ssms = []
gene1_ssms = []
with ProcessPoolExecutor(max_workers=conf.GENERAL["N_JOBS"]) as executor:
    futures = {
        executor.submit(_run_job, y, y_idx) for y_idx, y in enumerate(random_phenotypes)
    }

    for fut in tqdm(as_completed(futures), total=len(random_phenotypes), ncols=100):
        y_idx, gene0_result, gene0_data, gene1_result, gene1_data = fut.result()

        y_indices.append(y_idx)
        gene0_ssms.append(gene0_result)
        gene1_ssms.append(gene1_result)

gene0_ssms = pd.Series(gene0_ssms, y_indices).sort_index()
gene1_ssms = pd.Series(gene1_ssms, y_indices).sort_index()

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
# Run multixcan on null results from 1000G or GTEX_V8
#

COHORT_NAME = "1000G_EUR"
REFERENCE_PANEL = "GTEX_V8"
EQTL_MODEL = "MASHR"
PHENOTYPE_CODE = "pheno0"
SPREDIXCAN_FOLDER = Path(
    conf.RESULTS["GLS_NULL_SIMS"] / "twas" / "spredixcan"
).resolve()
SPREDIXCAN_FILE_PATTERN = f"random.{PHENOTYPE_CODE}-gtex_v8-mashr-" + "{tissue}.csv"
SMULTIXCAN_FILE = Path(
    conf.RESULTS["GLS_NULL_SIMS"]
    / "twas"
    / "smultixcan"
    / f"random.{PHENOTYPE_CODE}-gtex_v8-mashr-smultixcan.txt"
).resolve()

EQTL_MODEL_FILES_PREFIX = conf.PHENOMEXCAN["PREDICTION_MODELS"][f"{EQTL_MODEL}_PREFIX"]

N_PHENOTYPES = 1000

INPUT_DIR_BASE = (
    conf.RESULTS["GLS"]
    / "gene_corrs"
    / "cohorts"
    / COHORT_NAME.lower()
    / REFERENCE_PANEL.lower()
    / EQTL_MODEL.lower()
)
assert INPUT_DIR_BASE.exists()

spredixcan_gene_tissues = pd.read_pickle(INPUT_DIR_BASE / "gene_tissues.pkl")

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
    gene_id: spredixcan_gene_tissues.loc[gene_id, "tissue"]
    for gene_id in smultixcan_results["gene_id"]
    if gene_id in spredixcan_gene_tissues.index
}

# code below is to use GWAS data and adjust phenotype for covariates before
# running MultiXcan
#
# gwas_phenotypes = pd.read_csv(
#     conf.A1000G["GENOTYPES_DIR"] / "subsets" / "all_phase3.8.random_pheno.txt",
#     sep=" ",
#     index_col="IID",
# )
#
# # get residuals of phenotype by adjusting to covariates
# pca_eigenvec = (
#     pd.read_csv(
#         conf.A1000G["GENOTYPES_DIR"] / "subsets" / "all_phase3.7.pca_covar.eigenvec",
#         sep=" ",
#         header=None,
#     )
#     .rename(columns={1: "IID"})
#     .drop(columns=[0])
#     .set_index("IID")
# )
# assert pca_eigenvec.index.is_unique
# pca_eigenvec = pca_eigenvec.rename(
#     columns={c: f"pc{c-1}" for c in pca_eigenvec.columns}
# )
#
# assert gwas_phenotypes.index.equals(pca_eigenvec.index)
#
# individuals_sex = (
#     pd.read_csv(
#         conf.A1000G["GENOTYPES_DIR"] / "subsets" / "all_phase3.8.fam",
#         sep=" ",
#         header=None,
#         usecols=[1, 4],
#     )
#     .rename(columns={1: "IID", 4: "sex"})
#     .set_index("IID")
# )
#
# assert individuals_sex.index.equals(pca_eigenvec.index)
#
# covariates = pd.concat([individuals_sex, pca_eigenvec], axis=1)
# assert gwas_phenotypes.index.equals(covariates.index)
#
#
# def _get_residual(pheno):
#     keys = covariates.columns.tolist()
#     e_ = covariates.assign(pheno=pheno)
#     assert not e_.isna().any(None)
#
#     y, X = dmatrices(
#         "pheno ~ {}".format(" + ".join(keys)), data=e_, return_type="dataframe"
#     )
#     model = sm.OLS(y, X)
#     result = model.fit()
#     e_["residual"] = result.resid
#     return e_["residual"].rename(pheno.name)
#
# random_phenotypes = [
#     _get_residual(gwas_phenotypes[c])
#     for c in gwas_phenotypes.columns
#     if c.startswith("pheno")
# ]

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

# get intersection with 1000G GWAS' variants also
# so it's compatible with gene expression correlations
with open(INPUT_DIR_BASE / "gwas_variant_ids.pkl", "rb") as handle:
    gwas_variants_ids_set = pickle.load(handle)

all_snps_in_models_per_chr = {
    chromosome: frozenset(
        set(all_gene_snps.loc[chromosome, "varID"]).intersection(gwas_variants_ids_set)
    )
    for chromosome in all_gene_snps.index.unique()
}

# get common genes
multiplier_genes = pd.read_pickle(conf.MULTIPLIER["MODEL_Z_MATRIX_FILE"]).index
multiplier_genes = set(
    [Gene(name=g).ensembl_id for g in multiplier_genes if g in Gene.GENE_NAME_TO_ID_MAP]
)
common_genes = multiplier_genes.intersection(set(spredixcan_gene_tissues.index))


# predict expression for genes in all tissues
output_dir = conf.DATA_DIR / "tmp" / "predict_expression" / REFERENCE_PANEL.lower()

# OPTIONAL: or load the results with:
# assert output_dir.exists()
# with open(
#     output_dir
#     / "genes_predicted_expression.pkl",
#     "rb",
# ) as h:
#     genes_predicted_expression = pickle.load(h)


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
genotypes_file = Path(f"/tmp/genotypes_{REFERENCE_PANEL}.pkl").resolve()
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

# save results
output_dir.mkdir(exist_ok=True, parents=True)
with open(
    output_dir / "genes_predicted_expression.pkl",
    "wb",
) as h:
    pickle.dump(genes_predicted_expression, h)

# generate random phenotypes
rs = np.random.RandomState(0)

random_phenotypes = []
gene_pred_expr = genes_predicted_expression["ENSG00000173614"]
for pheno_i in range(N_PHENOTYPES):
    y = pd.Series(
        rs.normal(size=gene_pred_expr.shape[0]), index=gene_pred_expr.index.tolist()
    ).rename(f"pheno{pheno_i}")
    random_phenotypes.append(y)


# Free some memory
del (
    spredixcan_gene_tissues,
    # spredixcan_result_files,
    smultixcan_results,
    smultixcan_genes_tissues,
    mashr_models_db_files,
    _genes,
    selected_genes,
    # prediction_model_tissues,
    # pca_eigenvec,
    # individuals_sex,
    # covariates,
    multiplier_genes,
    # gwas_phenotypes,
    all_gene_snps,
    all_snps_in_models_per_chr,
    all_variants_ids,
    common_genes,
)

import gc

n = gc.collect()
print("Number of unreachable objects collected by GC:", n)

# run MultiXcan
def _run_multixcan(y, gene_id):
    gene_pred_expression = genes_predicted_expression[gene_id]

    # align y and gene_pred_expression genes
    index_inter = gene_pred_expression.index.intersection(y.index)

    y = y.loc[index_inter]
    gene_pred_expression = gene_pred_expression.loc[index_inter]

    gene_model_result, gene_data, _ = run_multixcan(
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


_tmp = _run_multixcan(random_phenotypes[0], "ENSG00000142166")
assert _tmp is not None
print(_tmp)

gene_pheno_assoc = []
current_batch_number = 0
(output_dir / "multixcan").mkdir(exist_ok=True, parents=True)
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
                output_dir / "multixcan" / f"res_{current_batch_number}.pkl"
            )
            current_batch_number += 1
            gene_pheno_assoc = []
            gc.collect()

# save
pd.DataFrame(gene_pheno_assoc).to_pickle(
    output_dir / "multixcan" / f"res_{current_batch_number}.pkl"
)

# save results separately
multixcan_nulls_dir = Path(
    conf.RESULTS["GLS_NULL_SIMS"] / "twas" / f"multixcan_{REFERENCE_PANEL.lower()}"
).resolve()
multixcan_nulls_dir.mkdir(exist_ok=True, parents=True)

results_files = list(Path(output_dir / "multixcan").glob("res_*.pkl"))
assert len(results_files) > 0
all_results = []

for res_f in results_files:
    res = pd.read_pickle(res_f)
    all_results.append(res)

all_results = pd.concat(all_results, axis=0)
phenotypes_genes_count = all_results["phenotype"].value_counts()
phenotypes_ready = phenotypes_genes_count[phenotypes_genes_count > 10].index
assert len(phenotypes_ready) > 0

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
