"""
TODO
"""
from pathlib import Path
import sys
import argparse
import logging

import numpy as np
import pandas as pd

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
        "-m",
        "--predixcan-model-type",
        required=False,
        choices=["MASHR", "ELASTIC_NET"],
        default="MASHR",
        help="TODO",
    )
    # FIXME: add debug level
    # FIXME: add z-score or -log10(p) transformations
    # FIXME: covariates

    # FIXME: check output file does not exist
    # FIXME: check output file parent DOES exist

    args = parser.parse_args()

    print(args.predixcan_model_type)

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
    data = -np.log10(data)

    logger.info(f"Prediction models used: {args.predixcan_model_type}")

    # FIXME: if lv weights is a filepath given in arguments, use that filepath
    lvs_subset = GLSPhenoplier._get_lv_weights().columns.tolist()

    model = GLSPhenoplier(
        model_type=args.predixcan_model_type,
        warnings_logger=logger,
    )

    results = []

    # FIXME: add tqdm?

    # FIXME: remove top 5 lvs here, this is just for debugging
    for lv_code in lvs_subset[:5]:
        logger.info(f"Computing for {lv_code}")
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

    results = pd.DataFrame(results)
    logger.info(f"Writing to {args.output_file}")
    results.to_csv(args.output_file, sep="\t", na_rep="NA")

    # FIXME: when saving to a file, use input_phenotype_name


if __name__ == "__main__":
    run()
