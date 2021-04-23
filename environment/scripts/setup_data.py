"""
It sets up the file/folder structure by downloading the necessary files.
"""
import os
from pathlib import Path

import conf
from utils import curl, md5_matches
from log import get_logger

logger = get_logger("setup")


#
# These are methods names (which download files) which should be included
# in the full mode only (see __main__ below).
#
DATA_IN_FULL_MODE_ONLY = {
    "download_phenomexcan_smultixcan_mashr_zscores",
    "download_phenomexcan_smultixcan_mashr_pvalues",
    "download_multiplier_recount2_model",
    "download_multiplier_recount2_data",
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
        logger=logger,
    )


def download_phenomexcan_rapid_gwas_data_dict_file(**kwargs):
    output_file = conf.PHENOMEXCAN["RAPID_GWAS_DATA_DICT_FILE"]
    curl(
        "https://upenn.box.com/shared/static/u3po287ku1cj0jubbnsi7c4xawsaked5.tsv",
        output_file,
        "c4b5938a7fdb0b1525f984cfb815bda5",
        logger=logger,
    )


def download_phenomexcan_gtex_gwas_pheno_info(**kwargs):
    output_file = conf.PHENOMEXCAN["GTEX_GWAS_PHENO_INFO_FILE"]
    curl(
        "https://upenn.box.com/shared/static/gur0ug0qg7hs88ybrsgrwx7eeymmxay1.tsv",
        output_file,
        "982434335f07acb1abfb83e57532f2c0",
        logger=logger,
    )


def download_gene_map_name_to_id(**kwargs):
    output_file = conf.PHENOMEXCAN["GENE_MAP_NAME_TO_ID"]
    curl(
        "https://upenn.box.com/shared/static/t33a6iv4jtwc2pv2c1nllpnq0nlrfxkt.pkl",
        output_file,
        "582d93c30c18027eefd465516733170f",
        logger=logger,
    )


def download_gene_map_id_to_name(**kwargs):
    output_file = conf.PHENOMEXCAN["GENE_MAP_ID_TO_NAME"]
    curl(
        "https://upenn.box.com/shared/static/p20w0ikxhvo04xf1b2zai53cpoqb4ljz.pkl",
        output_file,
        "63ac3ad54930d1b1490c6d02a68feb61",
        logger=logger,
    )


def download_biomart_genes_hg38(**kwargs):
    output_file = conf.GENERAL["BIOMART_GENES_INFO_FILE"]
    curl(
        "https://upenn.box.com/shared/static/ks998wwlwble7rcb5cdthwjg1l0j1alb.gz",
        output_file,
        "c4d74e156e968267278587d3ce30e5eb",
        logger=logger,
    )


def download_uk_biobank_coding_3(**kwargs):
    output_file = conf.UK_BIOBANK["CODING_3_FILE"]
    curl(
        "https://upenn.box.com/shared/static/1f5yjg31qxemvf5hqkoz559cau14xr68.tsv",
        output_file,
        "c02c65888793d4190fc190182128cc02",
        logger=logger,
    )


def download_uk_biobank_coding_6(**kwargs):
    output_file = conf.UK_BIOBANK["CODING_6_FILE"]
    curl(
        "https://upenn.box.com/shared/static/libgz7998c2lsytjon8we1ouhabvh1z1.tsv",
        output_file,
        "23a2bca99ea0bf25d141fc8573f67fce",
        logger=logger,
    )


def download_phenomexcan_smultixcan_mashr_zscores(**kwargs):
    output_file = conf.PHENOMEXCAN["SMULTIXCAN_MASHR_ZSCORES_FILE"]
    curl(
        "https://upenn.box.com/shared/static/taj1ex9ircek0ymi909of9anmjnj90k4.pkl",
        output_file,
        "83ded01d34c906092d64c1f5cc382fb0",
        logger=logger,
    )


def download_phenomexcan_smultixcan_mashr_pvalues(**kwargs):
    output_file = conf.PHENOMEXCAN["SMULTIXCAN_MASHR_PVALUES_FILE"]
    curl(
        "https://upenn.box.com/shared/static/wvrbt0v2ddrtb25g7dgw1be09yt9l14l.pkl",
        output_file,
        "3436a41e9a70fc2a206e9b13153ebd12",
        logger=logger,
    )


def download_phenomexcan_fastenloc_rcp(**kwargs):
    output_file = conf.PHENOMEXCAN["FASTENLOC_TORUS_RCP_FILE"]
    curl(
        "https://upenn.box.com/shared/static/qgghpf4nyuj45su5a184e8geg4egjd20.pkl",
        output_file,
        "a1b12c552c0b41db3f3b0131910aa974",
        logger=logger,
    )


def download_multiplier_model_summary_pkl(**kwargs):
    output_file = conf.MULTIPLIER["MODEL_SUMMARY_FILE"]
    curl(
        "https://upenn.box.com/shared/static/xfaez2u5wr258qb58lpexllyrvc7jolr.pkl",
        output_file,
        "1fdcd5dbee984b617dddb44937910710",
        logger=logger,
    )


def download_multiplier_model_z_pkl(**kwargs):
    output_file = conf.MULTIPLIER["MODEL_Z_MATRIX_FILE"]
    curl(
        "https://upenn.box.com/shared/static/pz07jiy99f8yx0fx2grle8i5cstpn7fz.pkl",
        output_file,
        "c3c84d70250ab34d06625eedc3d5ff29",
        logger=logger,
    )


def download_multiplier_model_b_pkl(**kwargs):
    output_file = conf.MULTIPLIER["MODEL_B_MATRIX_FILE"]
    curl(
        "https://upenn.box.com/shared/static/26n3l20t3755fjaihx9os783tk5hh2sa.pkl",
        output_file,
        "ef67e80b282781ec08beeb39f1bce07f",
        logger=logger,
    )


def download_multiplier_model_metadata_pkl(**kwargs):
    output_file = conf.MULTIPLIER["MODEL_METADATA_FILE"]
    curl(
        "https://upenn.box.com/shared/static/efeulwvivjtucunvrx2nwq06pyzs3pkq.pkl",
        output_file,
        "21cfd84270d04ad30ac2bca7049c7dab",
        logger=logger,
    )


def download_ukb_to_efo_map_tsv(**kwargs):
    # The original file was downloaded from:
    # https://github.com/EBISPOT/EFO-UKB-mappings/blob/master/UK_Biobank_master_file.tsv
    # on Nov 19, 2020
    output_file = conf.UK_BIOBANK["UKBCODE_TO_EFO_MAP_FILE"]
    curl(
        "https://upenn.box.com/shared/static/hwlpdlp3pq9buv955q5grlkxwwfxt6ul.tsv",
        output_file,
        "bfa56310d40e28f89c1f1b5d4ade0bf0",
        logger=logger,
    )


def download_efo_ontology(**kwargs):
    # The original file was download from:
    # http://www.ebi.ac.uk/efo/efo.obo
    # on Nov 16, 2020
    output_file = conf.GENERAL["EFO_ONTOLOGY_OBO_FILE"]
    curl(
        "https://upenn.box.com/shared/static/nsrxx3szg4s69j84dg2oakx6mwjxoarb.obo",
        output_file,
        "2bf23581ff6365514a0b3b1b5ae4651a",
        logger=logger,
    )


def download_emerge_phenotypes_description(**kwargs):
    output_file = conf.EMERGE["DESC_FILE_WITH_SAMPLE_SIZE"]
    curl(
        "https://upenn.box.com/shared/static/jvjaclxyckv4qd9gqh89qdc53147uctg.txt",
        output_file,
        "e8ed06025fc393e3216c1af9d6e16615",
    )


def download_multiplier_banchereau_mcp_neutrophils(**kwargs):
    output_file = conf.MULTIPLIER["BANCHEREAU_MCPCOUNTER_NEUTROPHIL_FILE"]
    curl(
        "https://raw.githubusercontent.com/greenelab/multi-plier/master/results/40/Banchereau_MCPcounter_neutrophil_LV.tsv",
        output_file,
        "2ed8d71d9fdcf857a44b7fd1a42035f0",
    )


def download_crispr_lipids_gene_sets_file(**kwargs):
    output_file = conf.CRISPR["LIPIDS_GENE_SETS_FILE"]
    curl(
        "https://upenn.box.com/shared/static/amiu6epztbuqjoad7eq9e50fpfdzdrvt.csv",
        output_file,
        "987eeef1987421b596988eba92e6305f",
        logger=logger,
    )


def download_pharmacotherapydb_indications(**kwargs):
    output_file = conf.PHARMACOTHERAPYDB["INDICATIONS_FILE"]
    output_file.parent.mkdir(exist_ok=True, parents=True)
    curl(
        "https://ndownloader.figshare.com/files/4823950",
        output_file,
        "33585132777601dedd3bed35caf718e2",
        logger=logger,
    )


def download_lincs_consensus_signatures(**kwargs):
    output_file = conf.LINCS["CONSENSUS_SIGNATURES_FILE"]
    output_file.parent.mkdir(exist_ok=True, parents=True)
    curl(
        "https://ndownloader.figshare.com/files/4797607",
        output_file,
        "891e257037adc15212405af461ffbfd6",
        logger=logger,
    )


def download_spredixcan_hdf5_results(**kwargs):
    output_folder = conf.PHENOMEXCAN["SPREDIXCAN_MASHR_ZSCORES_FOLDER"] / "hdf5"
    output_folder.parent.mkdir(exist_ok=True, parents=True)

    output_tar_file = Path(
        conf.PHENOMEXCAN["GENE_ASSOC_DIR"], "spredixcan-mashr-zscores.tar"
    ).resolve()
    output_tar_file_md5 = "502cc184948c80c16ecea130a3523ebd"

    if not Path(output_tar_file).exists() or not md5_matches(
        output_tar_file_md5, output_tar_file
    ):
        # download the two parts
        part0_filename = output_tar_file.parent / (output_tar_file.name + ".part0")
        curl(
            "https://upenn.box.com/shared/static/hayzpeowlvgjy6ctmp1gpochagq2ob62.tar",
            part0_filename,
            "333ae4a9b9fc215a1aa1c0628e03d65e",
            logger=logger,
        )

        part1_filename = output_tar_file.parent / (output_tar_file.name + ".part1")
        curl(
            "https://upenn.box.com/shared/static/pl7hsqq7hqqf18kf4x0185xn1x45ov4i.tar",
            part1_filename,
            "3d41bc3e4d511081849a102f018af1a8",
            logger=logger,
        )

        # combine
        logger.info("Concatenating parts")
        with open(output_tar_file, "wb") as output_f, open(
            part0_filename, "rb"
        ) as part0_f, open(part1_filename, "rb") as part1_f:
            output_f.write(part0_f.read())
            output_f.write(part1_f.read())

        assert md5_matches(output_tar_file_md5, output_tar_file), "Concatenation failed"

        part0_filename.unlink()
        part1_filename.unlink()

    # uncompress file
    import tarfile

    logger.info(f"Extracting {output_tar_file}")
    with tarfile.open(output_tar_file, "r") as f:
        tar_members = f.getmembers()
        members_dict = {t.name: t for t in tar_members}

        assert (
            output_folder.name in members_dict
        ), "Output folder name not inside tar file"

        f.extractall(output_folder.parent)


def _get_file_from_zip(
    zip_file_url,
    zip_file_path,
    zip_file_md5,
    zip_internal_filename,
    output_file,
    output_file_md5,
):
    """
    This method downloads a zip file and extracts a particular file inside
    it.

    TODO: finish documentation of arguments

    Args:
        zip_file_url:
        zip_file_path:
        zip_file_md5:
        zip_internal_filename:
        output_file:
        output_file_md5:
    """
    from utils import md5_matches

    # output_file = conf.MULTIPLIER["RECOUNT2_MODEL_FILE"]

    # do not download file again if it exists and MD5 matches the expected one
    if output_file.exists() and md5_matches(output_file_md5, output_file):
        logger.info(f"File already downloaded: {output_file}")
        return

    # download zip file
    parent_dir = Path(zip_file_path).parent
    # parent_dir = conf.MULTIPLIER["RECOUNT2_MODEL_FILE"].parent
    # zip_file_path = Path(parent_dir, "recount2_PLIER_data.zip").resolve()

    curl(
        zip_file_url,
        zip_file_path,
        zip_file_md5,
        logger=logger,
    )

    # extract model from zip file
    # zip_internal_filename = Path("recount2_PLIER_data", "recount_PLIER_model.RDS")
    logger.info(f"Extracting {zip_internal_filename}")
    import zipfile

    with zipfile.ZipFile(zip_file_path, "r") as z:
        z.extract(str(zip_internal_filename), path=parent_dir)

    # rename file
    Path(parent_dir, zip_internal_filename).rename(output_file)
    Path(parent_dir, zip_internal_filename.parent).rmdir()

    # delete zip file
    # zip_file_path.unlink()


def download_multiplier_recount2_model(**kwargs):
    """
    This method downloads the MultiPLIER model on recount2.
    """
    _get_file_from_zip(
        zip_file_url="https://ndownloader.figshare.com/files/10881866",
        zip_file_path=Path(
            conf.MULTIPLIER["RECOUNT2_MODEL_FILE"].parent, "recount2_PLIER_data.zip"
        ).resolve(),
        zip_file_md5="f084992c5d91817820a2782c9441b9f6",
        zip_internal_filename=Path("recount2_PLIER_data", "recount_PLIER_model.RDS"),
        output_file=conf.MULTIPLIER["RECOUNT2_MODEL_FILE"],
        output_file_md5="fc7446ff989d0bd0f1aae1851d192dc6",
    )


# def download_multiplier_recount2_data(**kwargs):
#     """
#     This method downloads the recount2 data used in MultiPLIER.
#     """
#     _get_file_from_zip(
#         zip_file_url="https://ndownloader.figshare.com/files/10881866",
#         zip_file_path=Path(
#             conf.MULTIPLIER["RECOUNT2_MODEL_FILE"].parent, "recount2_PLIER_data.zip"
#         ).resolve(),
#         zip_file_md5="f084992c5d91817820a2782c9441b9f6",
#         zip_internal_filename=Path(
#             "recount2_PLIER_data", "recount_data_prep_PLIER.RDS"
#         ),
#         output_file=conf.RECOUNT2["PREPROCESSED_GENE_EXPRESSION_FILE"],
#         output_file_md5="4f806e06069fd339f8fcff7c98cecff0",
#     )


if __name__ == "__main__":
    import argparse
    from collections import defaultdict

    # create a list of available options:
    #   --mode=full:  it downloads all the data.
    #   --mode=light: it downloads a smaller set of the data. This is useful for
    #                 Github Action workflows.
    AVAILABLE_ACTIONS = defaultdict(dict)

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

        if key not in DATA_IN_FULL_MODE_ONLY:
            AVAILABLE_ACTIONS["light"][key] = value

        AVAILABLE_ACTIONS["full"][key] = value

    parser = argparse.ArgumentParser(description="PhenoPLIER data setup.")
    parser.add_argument(
        "--mode",
        choices=list(AVAILABLE_ACTIONS.keys()),
        default="light",
        help="Specifies which kind of data should be downloaded. It "
        "could be all the data (full) or a small subset (light, which is "
        "used by default).",
    )
    parser.add_argument(
        "--action",
        help="Specifies a single action to be executed. It could be any of "
        "the following: " + " ".join(AVAILABLE_ACTIONS["full"].keys()),
    )
    args = parser.parse_args()

    # create all directories specified in the configuration
    _create_directories()

    method_args = vars(args)

    methods_to_run = {}

    if args.action is not None:
        if args.action not in AVAILABLE_ACTIONS["full"]:
            import sys

            logger.error(f"The action does not exist: {args.action}")
            sys.exit(1)

        methods_to_run[args.action] = AVAILABLE_ACTIONS["full"][args.action]
    else:
        methods_to_run = AVAILABLE_ACTIONS[args.mode]

    for method_name, method in methods_to_run.items():
        method(**method_args)
