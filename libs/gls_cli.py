"""
TODO
"""
from pathlib import Path
import sys
import argparse
import logging

import numpy as np
import pandas as pd
from scipy import stats

from gls import GLSPhenoplier

LOG_FORMAT = "[%(asctime)s] %(levelname)s: %(message)s"
logging.basicConfig(format=LOG_FORMAT, level=logging.INFO)
logger = logging.getLogger("root")


COVAR_OPTIONS = [
    "all",
    "gene_size",
    "gene_size_log",
    "gene_density",
    "gene_density_log",
]


def run():
    parser = argparse.ArgumentParser(
        description="""
    PhenoPLIER command line tool to compute gene module-trait associations.
    """.strip()
    )

    parser.add_argument(
        "-i",
        "--input-file",
        required=True,
        type=str,
        help="File path to S-MultiXcan result file (tab-separated and with at least columns 'gene' and 'pvalue'",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        required=True,
        type=str,
        help="File path where results will be written to.",
    )
    parser.add_argument(
        "-p",
        "--lv-model-file",
        required=False,
        type=str,
        help="A file containing the LV model. It has to be in pickle format, with gene symbols in rows and LVs in columns.",
    )
    parser.add_argument(
        "-g",
        "--gene-corr-file",
        required=False,
        help="TODO; if not specified, it lets GLSPhenoplier to specify a default gene corrs file",
    )
    parser.add_argument(
        # "-g",
        "--debug-use-ols",
        required=False,
        action="store_true",
        help="It uses a standard OLS model instead of GLS. For debugging purposes.",
    )
    parser.add_argument(
        # "-g",
        "--debug-use-sub-gene-corr",
        required=False,
        action="store_true",
        help="TODO.",
    )
    parser.add_argument(
        "-l",
        "--lv-list",
        required=False,
        nargs="+",
        default=[],
        help="List of LV (gene modules) identifiers on which an association will be computed.",
    )
    parser.add_argument(
        "--covars",
        required=False,
        nargs="+",
        choices=COVAR_OPTIONS,
        # default="keep-first",
        help="List of covariates to use.",
    )
    parser.add_argument(
        "--batch-id",
        required=False,
        type=int,
        help="TODO",
    )
    parser.add_argument(
        "--batch-n-splits",
        required=False,
        type=int,
        help="TODO",
    )
    parser.add_argument(
        "--duplicated-genes-action",
        required=False,
        choices=["keep-first", "keep-last", "remove-all"],
        # default="keep-first",
        help="TODO",
    )
    # FIXME: add debug level
    # FIXME: add z-score or -log10(p) transformations
    # FIXME: covariates

    # FIXME: when building the files related to a prediction model (mashr, etc), cnosider this:
    #  - a file with lv weights (independent of predixcan prediction model type)
    #  - gene corrs (dependent on prediction model type, because it uses weights)

    args = parser.parse_args()

    # check compatibility of parameteres
    if len(args.lv_list) > 0 and (
        args.batch_id is not None or args.batch_n_splits is not None
    ):
        logger.error(
            "Incompatible parameters: LV list and batches cannot be used together"
        )
        sys.exit(2)

    if (args.batch_id is not None and args.batch_n_splits is None) or (
        args.batch_id is None and args.batch_n_splits is not None
    ):
        logger.error(
            "Both --batch-id and --batch-n-splits have to be provided (not only one of them)"
        )
        sys.exit(2)

    if args.batch_id is not None and args.batch_id < 1:
        logger.error("--batch-id must be >= 1")
        sys.exit(2)

    if (
        args.batch_id is not None
        and args.batch_n_splits is not None
        and args.batch_id > args.batch_n_splits
    ):
        logger.error("--batch-id must be <= --batch-n-splits")
        sys.exit(2)

    # check output file (should not exist)
    output_file = Path(args.output_file)
    if output_file.exists():
        logger.warning(f"Skipping, output file exists: {str(output_file)}")
        sys.exit(0)

    if not output_file.parent.exists():
        logger.error(
            f"Parent directory of output file does not exist: {str(output_file.parent)}"
        )
        sys.exit(1)

    # check input file
    logger.info(f"Reading input file {args.input_file}")
    input_file = Path(args.input_file).resolve()
    if not input_file.exists():
        logger.error("Input file does not exist")
        sys.exit(2)

    # read
    data = pd.read_csv(input_file, sep="\t")
    logger.info(f"Input file has {data.shape[0]} genes")

    if "gene_name" not in data.columns:
        logger.error("Mandatory columns not present in data 'gene_name'")
        sys.exit(1)

    if "pvalue" not in data.columns:
        logger.error("Mandatory columns not present in data 'pvalue'")
        sys.exit(1)

    data = data.set_index("gene_name")  # [["gene_name", "pvalue"]]

    # remove duplicated gene entries
    if args.duplicated_genes_action is not None:
        if args.duplicated_genes_action.startswith("keep"):
            keep_action = args.duplicated_genes_action.split("-")[1]
        elif args.duplicated_genes_action == "remove-all":
            keep_action = False
        else:
            raise ValueError("Wrong --duplicated-genes-action value")

        data = data.loc[~data.index.duplicated(keep=keep_action)]

        logger.info(
            f"Removed duplicated genes symbols using '{args.duplicated_genes_action}'. Data now has {data.shape[0]} genes"
        )

    # unique index (gene names)
    if not data.index.is_unique:
        logger.error(
            "Duplicated genes in input data. Use option --duplicated-genes-action if you want to skip them."
        )
        sys.exit(1)

    # pvalues stats
    _data_pvalues = data["pvalue"]
    n_missing = _data_pvalues.isna().sum()
    n = _data_pvalues.shape[0]
    min_pval = _data_pvalues.min()
    mean_pval = _data_pvalues.mean()
    max_pval = _data_pvalues.max()

    logger.info(
        f"p-values statistics: min={min_pval:.1e} | mean={mean_pval:.1e} | max={max_pval:.1e} | # missing={n_missing} ({(n_missing / n) * 100:.1f}%)"
    )

    if min_pval < 0.0:
        logger.warning("Some p-values are smaller than 0.0")

    if max_pval > 1.0:
        logger.warning("Some p-values are greater than 1.0")

    final_data = data.loc[:, ["pvalue"]].rename(
        columns={
            "pvalue": "y",
        }
    )

    if args.covars is not None and len(args.covars) > 0:
        logger.info(f"Using covariates: {args.covars}")
        covars_selected = args.covars

        if "all" in covars_selected:
            covars_selected = [c for c in COVAR_OPTIONS if c != "all"]

        covars_selected = sorted(covars_selected)

        # get necessary columns from results
        covars = data[["pvalue", "n", "n_indep"]]
        covars = covars.rename(
            columns={
                "n_indep": "gene_size",
            }
        )

        if "gene_size_log" in covars_selected:
            covars["gene_size_log"] = np.log(covars["gene_size"])

        if "gene_density" in covars_selected:
            covars = covars.assign(
                gene_density=covars.apply(lambda x: x["gene_size"] / x["n"], axis=1)
            )

        if "gene_density_log" in covars_selected:
            covars["gene_density_log"] = -np.log(covars["gene_density"])

        final_data = pd.concat([final_data, covars[covars_selected]], axis=1)

    # convert p-values
    # TODO: add optional parameter to convert using either -log10 or z-score
    final_data["y"] = np.abs(stats.norm.ppf(final_data["y"] / 2))
    # final_data["y"] = -np.log10(final_data)

    if final_data.shape[1] == 1:
        final_data = final_data.squeeze().rename(input_file.stem)

    if args.debug_use_ols and args.gene_corr_file is not None:
        logger.error(
            "Incompatible arguments: you cannot specify both --gene-corr-file and --debug-use-ols"
        )
        sys.exit(1)

    if args.gene_corr_file is not None:
        logger.info(f"Using gene correlation file: {args.gene_corr_file}")
    else:
        if not args.debug_use_ols:
            logger.warning(
                "No gene correlations file specified. The default will be used"
            )

    if args.lv_model_file is not None:
        lv_model_file = Path(args.lv_model_file)
        # FIXME: check that file exists
        logger.info(f"Reading LV model file: {str(lv_model_file)}")
        full_lvs_list = GLSPhenoplier._get_lv_weights(lv_model_file).columns.tolist()
    else:
        full_lvs_list = GLSPhenoplier._get_lv_weights().columns.tolist()

    if args.batch_n_splits is not None and args.batch_n_splits > len(full_lvs_list):
        logger.error(
            f"--batch-n-splits cannot be greater than LVs in the model ({len(full_lvs_list)} LVs)"
        )
        sys.exit(2)

    full_lvs_set = set(full_lvs_list)
    logger.info(f"{len(full_lvs_set)} LVs (gene modules) were found in LV model")

    if len(args.lv_list) > 0:
        selected_lvs = [lv for lv in args.lv_list if lv in full_lvs_set]
        logger.info(
            f"A list of {len(args.lv_list)} LVs was provided, and {len(selected_lvs)} are present in LV models"
        )
    else:
        selected_lvs = full_lvs_list
        logger.info("All LVs in models will be used")

    if args.batch_id is not None and args.batch_n_splits is not None:
        selected_lvs_chunks = [
            ar.tolist() for ar in np.array_split(selected_lvs, args.batch_n_splits)
        ]
        selected_lvs = selected_lvs_chunks[args.batch_id - 1]
        logger.info(
            f"Using batch {args.batch_id} out of {args.batch_n_splits} ({len(selected_lvs)} LVs selected)"
        )

    if len(selected_lvs) == 0:
        logger.error("No LVs were selected")
        sys.exit(1)

    model = GLSPhenoplier(
        gene_corrs_file_path=args.gene_corr_file,
        debug_use_ols=args.debug_use_ols,
        debug_use_sub_gene_corr=args.debug_use_sub_gene_corr,
        use_own_implementation=True,
        logger=logger,
    )

    results = []

    # TODO: add tqdm?

    for lv_idx, lv_code in enumerate(selected_lvs):
        logger.info(f"Computing for {lv_code}")

        # show warnings or logs only in the first run
        if lv_idx == 0:
            model.set_logger(logger)
        else:
            model.set_logger(None)

        model.fit_named(lv_code, final_data)

        res = model.results

        results.append(
            {
                "lv": lv_code,
                # FIXME: lv_with_pathways is very cool!
                # "lv_with_pathway": lv_code in well_aligned_lv_codes,
                "beta": res.params.loc["lv"],
                "beta_se": res.bse.loc["lv"],
                "t": res.tvalues.loc["lv"],
                "pvalue_twosided": res.pvalues.loc["lv"],
                "pvalue_onesided": res.pvalues_onesided.loc["lv"],
                # "pvalue_twosided": res.pvalues.loc["lv"],
                # "summary": gls_model.results_summary,
            }
        )

    results = pd.DataFrame(results).set_index("lv")
    logger.info(f"Writing results to {str(output_file)}")
    results.to_csv(output_file, sep="\t", na_rep="NA")


if __name__ == "__main__":
    run()
