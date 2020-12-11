"""
It sets up the file/folder structure by downloading the necessary files.
"""
import os
from pathlib import Path

import conf
from utils import curl
from log import get_logger

logger = get_logger("setup")


# Methods names (that download files) which should not be included in testing mode (see
# below).
AVOID_IN_TESTING_MODE = {
    "download_phenomexcan_smultixcan_mashr_zscores",
    "download_phenomexcan_smultixcan_mashr_pvalues",
    "download_multiplier_recount2_model",
}


def _create_directories(node=conf.__dict__):
    """Creates directories for all setting entries pointing to a folder.

    Args:
        node (dict): a dictionary with key names pointing to different settings. By
        default it uses all attributes from the conf module.
    """
    for k, v in node.items():
        if isinstance(v, str) and not k.endswith("_DIR"):
            continue

        if isinstance(v, dict):
            # if the key itself is a dictionary, then walk through its values
            _create_directories(v)
        elif k.endswith("_DIR"):
            if v is None:
                continue

            os.makedirs(v, exist_ok=True)


def download_phenomexcan_rapid_gwas_pheno_info(**kwargs):
    output_file = conf.PHENOMEXCAN["RAPID_GWAS_PHENO_INFO_FILE"]
    curl(
        "https://upenn.box.com/shared/static/163mkzgd4uosk4pnzx0xsj7n0reu8yjv.gz",
        output_file,
        "cba910ee6f93eaed9d318edcd3f1ce18",
    )


def download_phenomexcan_rapid_gwas_data_dict_file(**kwargs):
    output_file = conf.PHENOMEXCAN["RAPID_GWAS_DATA_DICT_FILE"]
    curl(
        "https://upenn.box.com/shared/static/u3po287ku1cj0jubbnsi7c4xawsaked5.tsv",
        output_file,
        "c4b5938a7fdb0b1525f984cfb815bda5",
    )


def download_phenomexcan_gtex_gwas_pheno_info(**kwargs):
    output_file = conf.PHENOMEXCAN["GTEX_GWAS_PHENO_INFO_FILE"]
    curl(
        "https://upenn.box.com/shared/static/gur0ug0qg7hs88ybrsgrwx7eeymmxay1.tsv",
        output_file,
        "982434335f07acb1abfb83e57532f2c0",
    )


def download_gene_map_name_to_id(**kwargs):
    output_file = conf.PHENOMEXCAN["GENE_MAP_NAME_TO_ID"]
    curl(
        "https://upenn.box.com/shared/static/t33a6iv4jtwc2pv2c1nllpnq0nlrfxkt.pkl",
        output_file,
        "582d93c30c18027eefd465516733170f",
    )


def download_gene_map_id_to_name(**kwargs):
    output_file = conf.PHENOMEXCAN["GENE_MAP_ID_TO_NAME"]
    curl(
        "https://upenn.box.com/shared/static/p20w0ikxhvo04xf1b2zai53cpoqb4ljz.pkl",
        output_file,
        "63ac3ad54930d1b1490c6d02a68feb61",
    )


def download_biomart_genes_hg38(**kwargs):
    output_file = conf.GENERAL["BIOMART_GENES_INFO_FILE"]
    curl(
        "https://upenn.box.com/shared/static/ks998wwlwble7rcb5cdthwjg1l0j1alb.gz",
        output_file,
        "c4d74e156e968267278587d3ce30e5eb",
    )


def download_uk_biobank_coding_3(**kwargs):
    output_file = conf.UK_BIOBANK["CODING_3_FILE"]
    curl(
        "https://upenn.box.com/shared/static/1f5yjg31qxemvf5hqkoz559cau14xr68.tsv",
        output_file,
        "c02c65888793d4190fc190182128cc02",
    )


def download_uk_biobank_coding_6(**kwargs):
    output_file = conf.UK_BIOBANK["CODING_6_FILE"]
    curl(
        "https://upenn.box.com/shared/static/libgz7998c2lsytjon8we1ouhabvh1z1.tsv",
        output_file,
        "23a2bca99ea0bf25d141fc8573f67fce",
    )


def download_phenomexcan_smultixcan_mashr_zscores(**kwargs):
    output_file = conf.PHENOMEXCAN["SMULTIXCAN_MASHR_ZSCORES_FILE"]
    curl(
        "https://upenn.box.com/shared/static/taj1ex9ircek0ymi909of9anmjnj90k4.pkl",
        output_file,
        "83ded01d34c906092d64c1f5cc382fb0",
    )


def download_phenomexcan_smultixcan_mashr_pvalues(**kwargs):
    output_file = conf.PHENOMEXCAN["SMULTIXCAN_MASHR_PVALUES_FILE"]
    curl(
        "https://upenn.box.com/shared/static/wvrbt0v2ddrtb25g7dgw1be09yt9l14l.pkl",
        output_file,
        "3436a41e9a70fc2a206e9b13153ebd12",
    )


def download_multiplier_model_z_pkl(**kwargs):
    output_file = conf.MULTIPLIER["MODEL_Z_MATRIX_FILE"]
    curl(
        "https://upenn.box.com/shared/static/pz07jiy99f8yx0fx2grle8i5cstpn7fz.pkl",
        output_file,
        "c3c84d70250ab34d06625eedc3d5ff29",
    )


def download_multiplier_model_metadata_pkl(**kwargs):
    output_file = conf.MULTIPLIER["MODEL_METADATA_FILE"]
    curl(
        "https://upenn.box.com/shared/static/efeulwvivjtucunvrx2nwq06pyzs3pkq.pkl",
        output_file,
        "21cfd84270d04ad30ac2bca7049c7dab",
    )


def download_multiplier_recount2_model(**kwargs):
    """
    This method downloads the MultiPLIER model on recount2. Since this file is inside
    a public zip file, it first downloads the zip file and extracts only the requested
    file.
    """
    # TODO: refactor this method into a generic one to download files within zip files.
    from utils import md5_matches

    output_file = conf.MULTIPLIER["RECOUNT2_MODEL_FILE"]

    # do not download file again if it exists and MD5 matches the expected one
    if output_file.exists() and md5_matches(
        "fc7446ff989d0bd0f1aae1851d192dc6", output_file
    ):
        logger.info(f"File already downloaded: {output_file}")
        return

    # download zip file
    parent_dir = conf.MULTIPLIER["RECOUNT2_MODEL_FILE"].parent
    zip_file_path = Path(parent_dir, "recount2_PLIER_data.zip").resolve()

    curl(
        "https://ndownloader.figshare.com/files/10881866",
        zip_file_path,
    )

    # extract model from zip file
    zip_internal_filename = Path("recount2_PLIER_data", "recount_PLIER_model.RDS")
    logger.info(f"Extracting {zip_internal_filename}")
    import zipfile

    with zipfile.ZipFile(zip_file_path, "r") as z:
        z.extract(str(zip_internal_filename), path=parent_dir)

    # rename file
    Path(parent_dir, zip_internal_filename).rename(output_file)
    Path(parent_dir, zip_internal_filename.parent).rmdir()

    # delete zip file
    zip_file_path.unlink()


if __name__ == "__main__":
    import argparse
    from collections import defaultdict

    # create a list of available options:
    #   --mode=full:    it downloads all the data.
    #   --mode=testing: it downloads minimal data needed for running unit tests.
    #                   This is useful for Github Action workflows.
    AVAILABLE_ACTIONS = defaultdict(list)

    # Obtain all local attributes of this module and run functions to download files
    local_items = list(locals().items())
    for key, value in local_items:
        # iterate only on download_* methods
        if not (
            callable(value)
            and value.__module__ == __name__
            and key.startswith("download_")
        ):
            continue

        if key not in AVOID_IN_TESTING_MODE:
            AVAILABLE_ACTIONS["testing"].append((key, value))

        AVAILABLE_ACTIONS["full"].append((key, value))

    parser = argparse.ArgumentParser(description="PhenoPLIER data setup.")
    parser.add_argument(
        "--mode",
        choices=list(AVAILABLE_ACTIONS.keys()),
        default="full",
        help="Specifies which kind of data should be downloaded. It "
        "could be all the data or just different subsets.",
    )
    args = parser.parse_args()

    # create all directories specified in the configuration
    _create_directories()

    method_args = vars(args)

    for method_name, method in AVAILABLE_ACTIONS[args.mode]:
        method(**method_args)
