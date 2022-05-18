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
        "-m",
        "--predixcan-model-type",
        required=False,
        choices=["MASHR", "ELASTIC_NET"],
        default="MASHR",
        help="TODO",
    )
    parser.add_argument(
        "-l",
        "--lv-list",
        required=False,
        nargs="+",
        default=[],
        help="List of LV identifiers on which an association will be computed.",
    )
    # FIXME: add debug level
    # FIXME: add z-score or -log10(p) transformations
    # FIXME: covariates

    # FIXME: check output file does not exist
    # FIXME: check output file parent DOES exist

    args = parser.parse_args()

    # check input file
    logger.info(f"Reading input file {args.input_file}")
    input_file = Path(args.input_file).resolve()
    if not input_file.exists():
        logger.error("Input file does not exist")
        sys.exit(1)

    # read
    data = pd.read_csv(input_file, sep="\t")
    logger.info(f"Input file has {data.shape[0]} genes")

    if "gene_name" not in data.columns:
        logger.error("Mandatory columns not present in data 'gene_name'")
        sys.exit(1)

    if "pvalue" not in data.columns:
        logger.error("Mandatory columns not present in data 'pvalue'")
        sys.exit(1)

    input_phenotype_name = input_file.stem
    data = (
        data[["gene_name", "pvalue"]]
        .set_index("gene_name")
        .squeeze()
        .rename(input_phenotype_name)
    )

    # unique index (gene names)
    if not data.index.is_unique:
        logger.error("Duplicated gene names in input data")
        sys.exit(1)

    # pvalues stats
    n_missing = data.isna().sum()
    n = data.shape[0]
    min_pval = data.min()
    mean_pval = data.mean()
    max_pval = data.max()

    logger.info(
        f"p-values statistics: min={min_pval:.1e} | mean={mean_pval:.1e} | max={max_pval:.1e} | # missing={n_missing} ({(n_missing / n) * 100:.1f}%)"
    )

    if min_pval < 0.0:
        logger.warning("Some p-values are smaller than 0.0")

    if max_pval > 1.0:
        logger.warning("Some p-values are greater than 1.0")

    # TODO: convert to -log10 or z-score
    data = pd.Series(data=np.abs(stats.norm.ppf(data / 2)), index=data.index.copy())
    # data = -np.log10(data)

    logger.info(f"Prediction models used: {args.predixcan_model_type}")

    if args.lv_model_file is not None:
        lv_model_file = Path(args.lv_model_file)
        # FIXME: check that file exists
        logger.info(f"Reading LV model file: {str(lv_model_file)}")
        full_lvs_set = GLSPhenoplier._get_lv_weights(lv_model_file).columns
    else:
        full_lvs_set = GLSPhenoplier._get_lv_weights().columns

    full_lvs_set = set(full_lvs_set)
    logger.info(f"{len(full_lvs_set)} gene modules in LV models")

    if len(args.lv_list) > 0:
        selected_lvs = [lv for lv in args.lv_list if lv in full_lvs_set]
        logger.info(
            f"A list of {len(args.lv_list)} LVs was provided, and {len(selected_lvs)} are present in LV models"
        )
    else:
        selected_lvs = list(full_lvs_set)
        logger.info("All LVs in models will be used")

    if len(selected_lvs) == 0:
        logger.error("No LVs were selected")
        sys.exit(1)

    model = GLSPhenoplier(
        model_type=args.predixcan_model_type,
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

        model.fit_named(lv_code, data)

        res = model.results

        results.append(
            {
                "lv": lv_code,
                # FIXME: lv_with_pathways is very cool!
                # "lv_with_pathway": lv_code in well_aligned_lv_codes,
                "coef": res.params.loc["lv"],
                "pvalue": res.pvalues_onesided.loc["lv"],
                # "pvalue_twosided": res.pvalues.loc["lv"],
                # "summary": gls_model.results_summary,
            }
        )

    results = pd.DataFrame(results).set_index("lv")
    logger.info(f"Writing results to {args.output_file}")
    results.to_csv(args.output_file, sep="\t", na_rep="NA")


if __name__ == "__main__":
    run()
