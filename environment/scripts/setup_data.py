"""
It sets up the file/folder structure by downloading the necessary files.
"""
import os
import sys
import subprocess
from pathlib import Path

import conf
from utils import curl, md5_matches
from log import get_logger

logger = get_logger("setup")


#
# These variables specify methods names (that download files) which should only
# be executed in specific modes. For example, "testing" for unit testing, or
# "demo" for data needed only for the demo.
#

MODES_ACTIONS = {
    "testing": {
        "download_phenomexcan_rapid_gwas_pheno_info",
        "download_phenomexcan_rapid_gwas_data_dict_file",
        "download_uk_biobank_coding_3",
        "download_uk_biobank_coding_6",
        "download_phenomexcan_gtex_gwas_pheno_info",
        "download_gene_map_name_to_id",
        "download_gene_map_id_to_name",
        "download_biomart_genes_hg38",
        "download_multiplier_model_z_pkl",
        "download_multiplier_model_metadata_pkl",
        "download_predixcan_mashr_prediction_models",
        "download_precomputed_gene_correlations",
        "download_phenomexcan_smultixcan_mashr_zscores",
        "download_snps_covariance",
    },
    "demo": {
        "download_phenomexcan_rapid_gwas_pheno_info",
        "download_phenomexcan_gtex_gwas_pheno_info",
        "download_phenomexcan_rapid_gwas_data_dict_file",
        "download_uk_biobank_coding_3",
        "download_uk_biobank_coding_6",
        "download_biomart_genes_hg38",
        "download_gene_map_id_to_name",
        "download_gene_map_name_to_id",
        "download_multiplier_model_z_pkl",
        "download_multiplier_model_b_pkl",
        "download_multiplier_model_summary_pkl",
        "download_phenomexcan_smultixcan_mashr_pvalues",
        "download_precomputed_gene_correlations",
    },
    "full": {},  # empty means all actions/methods
}


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
    curl(
        "https://ndownloader.figshare.com/files/4823950",
        output_file,
        "33585132777601dedd3bed35caf718e2",
        logger=logger,
    )


def download_lincs_consensus_signatures(**kwargs):
    output_file = conf.LINCS["CONSENSUS_SIGNATURES_FILE"]
    curl(
        "https://ndownloader.figshare.com/files/4797607",
        output_file,
        "891e257037adc15212405af461ffbfd6",
        logger=logger,
    )


def download_precomputed_gene_correlations(**kwargs):
    output_file = conf.PHENOMEXCAN["LD_BLOCKS"]["GENE_NAMES_CORR_AVG"]
    curl(
        "https://upenn.box.com/shared/static/xopvxtlj44t4tswz3zzz8uxr6z7dzk0w.pkl",
        output_file,
        "cb2f7cabc1aa399a80699076b162a261",
        logger=logger,
    )


def download_snps_covariance(**kwargs):
    output_file = conf.PHENOMEXCAN["LD_BLOCKS"]["SNPS_COVARIANCE_FILE"]
    curl(
        "https://upenn.box.com/shared/static/cm9my5bo8jw6nawsxy5y6z0ceh9wnecv.h5",
        output_file,
        "bac23b32561f49f1683c421965a4d237",
        logger=logger,
    )


def download_predixcan_mashr_prediction_models(**kwargs):
    output_folder = conf.PHENOMEXCAN["PREDICTION_MODELS"]["MASHR"]
    if output_folder.exists():
        logger.warning(f"Output directory already exists ({output_folder}). Skipping.")
        return

    output_folder.parent.mkdir(exist_ok=True, parents=True)

    output_tar_file = Path(
        conf.PHENOMEXCAN["PREDICTION_MODELS"]["BASE_DIR"], "mashr_eqtl.tar"
    ).resolve()
    output_tar_file_md5 = "87f3470bf2676043c748b684fb35fa7d"

    if not Path(output_tar_file).exists() or not md5_matches(
        output_tar_file_md5, output_tar_file
    ):
        # download
        curl(
            "https://zenodo.org/record/3518299/files/mashr_eqtl.tar?download=1",
            output_tar_file,
            output_tar_file_md5,
            logger=logger,
        )

    # uncompress file
    import tarfile

    logger.info(f"Extracting {output_tar_file}")
    with tarfile.open(output_tar_file, "r") as f:
        f.extractall(output_folder.parent)

        # rename folder
        (output_folder.parent / "eqtl" / "mashr").rename(output_folder)
        (output_folder.parent / "eqtl").rmdir()


def download_spredixcan_hdf5_results(**kwargs):
    output_folder = conf.PHENOMEXCAN["SPREDIXCAN_MASHR_ZSCORES_FOLDER"] / "hdf5"
    if output_folder.exists():
        logger.warning(f"Output directory already exists ({output_folder}). Skipping.")
        return

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


def download_1000g_genotype_data(**kwargs):
    output_folder = conf.PHENOMEXCAN["LD_BLOCKS"]["1000G_GENOTYPE_DIR"]
    if output_folder.exists():
        logger.warning(f"Output directory already exists ({output_folder}). Skipping.")
        return

    output_folder.parent.mkdir(exist_ok=True, parents=True)

    output_tar_file = Path(
        conf.PHENOMEXCAN["LD_BLOCKS"]["BASE_DIR"], "sample_data.tar"
    ).resolve()
    output_tar_file_md5 = "8b42c388953d016e1112051d3b6140ed"

    if not Path(output_tar_file).exists() or not md5_matches(
        output_tar_file_md5, output_tar_file
    ):
        # download
        curl(
            "https://zenodo.org/record/3657902/files/sample_data.tar?download=1",
            output_tar_file,
            output_tar_file_md5,
            logger=logger,
        )

    # uncompress file
    import tarfile

    logger.info(f"Extracting {output_tar_file}")
    with tarfile.open(output_tar_file, "r") as f:
        selected_folder = [
            tarinfo
            for tarinfo in f.getmembers()
            if tarinfo.name.startswith("data/reference_panel_1000G/")
        ]

        f.extractall(output_folder.parent, members=selected_folder)

        # rename folder
        (output_folder.parent / "data" / "reference_panel_1000G").rename(output_folder)
        (output_folder.parent / "data").rmdir()


def _get_file_from_zip(
    zip_file_url,
    zip_file_path,
    zip_file_md5,
    zip_internal_filename,
    output_file,
    output_file_md5=None,
):
    """
    This method downloads a zip file and extracts a particular file inside
    it.

    TODO: finish documentation of arguments

    Args:
        zip_file_url:
        zip_file_path:
        zip_file_md5:
        zip_internal_filename: if it ends with "/", then it is treated as a folder and all members will be extracted
        output_file:
        output_file_md5:
    """
    from utils import md5_matches

    # output_file = conf.MULTIPLIER["RECOUNT2_MODEL_FILE"]
    _internal_file = str(zip_internal_filename)

    # do not download file again if it exists and MD5 matches the expected one
    if (
        not _internal_file.endswith("/")
        and output_file.exists()
        and (output_file_md5 is not None or md5_matches(output_file_md5, output_file))
    ):
        logger.info(f"File already downloaded: {output_file}")
        return

    # download zip file
    parent_dir = output_file.parent

    curl(
        zip_file_url,
        zip_file_path,
        zip_file_md5,
        logger=logger,
    )

    # extract model from zip file
    logger.info(f"Extracting {_internal_file}")
    import zipfile

    with zipfile.ZipFile(zip_file_path, "r") as z:

        if _internal_file.endswith("/"):
            # it's a folder
            # in this case, output_file points to the output folder
            output_folder = output_file
            if output_folder.exists():
                logger.warning(
                    f"Output folder exists, skipping: '{str(output_folder)}'"
                )
                return

            for i in z.namelist():
                if i.startswith(_internal_file):
                    z.extract(i, parent_dir)
        else:
            # it's a file
            z.extract(_internal_file, path=parent_dir)

            # TODO: check output_file_md5 ?

        # rename file
        Path(parent_dir, zip_internal_filename).rename(output_file)
        if Path(zip_internal_filename).parent != Path("."):
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


def download_1000g_genotype_data_from_plink(**kwargs):
    output_file = conf.A1000G["GENOTYPES_DIR"] / "all_phase3.pgen.zst"
    curl(
        "https://www.dropbox.com/s/y6ytfoybz48dc0u/all_phase3.pgen.zst?dl=1",
        output_file,
        "a52dcc2ad7ee09b29895df8fa044012e",
        logger=logger,
    )

    output_file = conf.A1000G["GENOTYPES_DIR"] / "all_phase3.pvar.zst"
    curl(
        "https://www.dropbox.com/s/odlexvo8fummcvt/all_phase3.pvar.zst?dl=1",
        output_file,
        "aa99c03e3fd9c5fe702239d07161a153",
        logger=logger,
    )

    output_file = conf.A1000G["GENOTYPES_DIR"] / "all_phase3.psam"
    curl(
        "https://www.dropbox.com/s/6ppo144ikdzery5/phase3_corrected.psam?dl=1",
        output_file,
        "b9a6d22dbf794f335ed122e465faef1d",
        logger=logger,
    )

    output_file = conf.A1000G["GENOTYPES_DIR"] / "deg2_phase3.king.cutoff.out.id"
    curl(
        "https://www.dropbox.com/s/zj8d14vv9mp6x3c/deg2_phase3.king.cutoff.out.id?dl=1",
        output_file,
        "9b047ac7ffb14a5e2be2ee7a68a95f8a",
        logger=logger,
    )


def _download_plink_generic(
    plink_zip_file,
    plink_executable_filename,
    output_file,
    platform_parameters,
):
    import platform

    current_system = platform.system()
    if current_system not in platform_parameters:
        raise ValueError("plink download for your platform was not added")

    platform_parameters = platform_parameters[current_system]
    zip_file_url = platform_parameters["zip_file_url"]
    zip_file_md5 = platform_parameters["zip_file_md5"]
    output_file_md5 = platform_parameters["output_file_md5"]

    # generic parameters
    zip_file_path = plink_zip_file
    zip_internal_filename = plink_executable_filename
    output_file = output_file

    _get_file_from_zip(
        zip_file_url=zip_file_url,
        zip_file_path=zip_file_path,
        zip_file_md5=zip_file_md5,
        zip_internal_filename=zip_internal_filename,
        output_file=output_file,
        output_file_md5=output_file_md5,
    )

    # make plink executable
    import os
    import stat

    st = os.stat(output_file)
    os.chmod(output_file, st.st_mode | stat.S_IEXEC)


def download_plink19(**kwargs):
    _download_plink_generic(
        plink_zip_file=Path(conf.PLINK["BASE_DIR"], "plink.zip").resolve(),
        plink_executable_filename=Path("plink"),
        output_file=conf.PLINK["EXECUTABLE_VERSION_1_9"],
        platform_parameters={
            "Linux": {
                "zip_file_url": "https://s3.amazonaws.com/plink1-assets/plink_linux_x86_64_20220402.zip",
                "zip_file_md5": "446600c3930997a031476b5961ed372f",
                "output_file_md5": "f285ab12811ab3063952a2e20adf9860",
            },
            "Darwin": {
                "zip_file_url": "https://s3.amazonaws.com/plink1-assets/plink_mac_20220402.zip",
                "zip_file_md5": "f5e78f0f4f8da2b60cfa77dc60d5847f",
                "output_file_md5": "626fb1c3452de35d2365715e16c03034",
            },
        },
    )


def download_plink2(**kwargs):
    _download_plink_generic(
        plink_zip_file=Path(conf.PLINK["BASE_DIR"], "plink2.zip").resolve(),
        plink_executable_filename=Path("plink2"),
        output_file=conf.PLINK["EXECUTABLE_VERSION_2"],
        platform_parameters={
            "Linux": {
                "zip_file_url": "https://s3.amazonaws.com/plink2-assets/plink2_linux_x86_64_20220426.zip",
                "zip_file_md5": "2e8e5d134a583f9f869a94fb11477208",
                "output_file_md5": "064529cc22083c44e4c6beeff33c206d",
            },
            "Darwin": {
                "zip_file_url": "https://s3.amazonaws.com/plink2-assets/plink2_mac_20220426.zip",
                "zip_file_md5": "51729ba53ccba1fb0de10158df289e45",
                "output_file_md5": "b62cbb4841d1bf062952f279f167fb2b",
            },
        },
    )


def _create_conda_environment(environment_folder: Path, environment_spec: Path):
    """
    TODO

    Args:
        environment_folder:
        environment_spec: yaml file with conda specification
    """
    # make sure parent folder exists
    environment_folder.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Creating conda environment in '{environment_folder}' using specification in '{environment_spec}'"
    )

    cmd = subprocess.check_call(
        [
            "conda",
            "env",
            "create",
            "-f",
            str(environment_spec),
            "-p",
            str(environment_folder),
        ],
        stdout=sys.stdout,
        stderr=subprocess.STDOUT,
    )


def download_setup_summary_gwas_imputation(**kwargs):
    _get_file_from_zip(
        zip_file_url="https://github.com/hakyimlab/summary-gwas-imputation/archive/206dac587824a6f207e137ce8c2d7b15d81d5869.zip",
        zip_file_path=Path(conf.SOFTWARE_DIR, "summary-gwas-imputation.zip").resolve(),
        zip_file_md5="b2e9ea5587c7cf35d42e7e16411efeb5",
        zip_internal_filename="summary-gwas-imputation-206dac587824a6f207e137ce8c2d7b15d81d5869/",
        output_file=conf.GWAS_IMPUTATION["BASE_DIR"],
        # output_file_md5="fc7446ff989d0bd0f1aae1851d192dc6",
    )

    if not conf.GWAS_IMPUTATION["CONDA_ENV"].exists():
        _create_conda_environment(
            environment_folder=conf.GWAS_IMPUTATION["CONDA_ENV"],
            environment_spec=conf.GWAS_IMPUTATION["BASE_DIR"] / "src/conda_env.yaml",
        )


def download_liftover_hg19tohg38_chain(**kwargs):
    output_file = conf.GENERAL["LIFTOVER"]["HG19_TO_HG38"]
    curl(
        "http://hgdownload.cse.ucsc.edu/goldenpath/hg19/liftOver/hg19ToHg38.over.chain.gz",
        output_file,
        "35887f73fe5e2231656504d1f6430900",
        logger=logger,
    )


def download_eur_ld_regions(**kwargs):
    output_file = conf.GENERAL["EUR_LD_REGIONS_FILE"]
    curl(
        "https://upenn.box.com/shared/static/yo25wpoct6vl8fk2s1rgxrs2wym7k1yb.gz",
        output_file,
        "900e4a7d3a14ae87de25ee48f7083dba",
        logger=logger,
    )


if __name__ == "__main__":
    import argparse
    from collections import defaultdict

    # create a list of available options. For example:
    #   --mode=full:    it downloads all the data.
    #   --mode=testing: it downloads a smaller set of the data. This is useful for
    #                   Github Action workflows.
    #   --mode=demo:    it downloads the data needed for the demo
    # (other modes might be specified in MODES_ACTION
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

        for mode, mode_actions in MODES_ACTIONS.items():
            if len(mode_actions) == 0:
                # if modes_actions is empty, it means all actions should be
                # added to that mode (e.g. "full" mode)
                AVAILABLE_ACTIONS[mode][key] = value
            elif key in mode_actions:
                AVAILABLE_ACTIONS[mode][key] = value

    parser = argparse.ArgumentParser(description="PhenoPLIER data setup.")
    parser.add_argument(
        "--mode",
        choices=list(AVAILABLE_ACTIONS.keys()),
        default="full",
        help="Specifies which kind of data should be downloaded. For example, "
        "it could be all the data (full) or a small subset (testing, which is "
        "used by unit tests).",
    )
    parser.add_argument(
        "--actions",
        nargs="+",
        help="Specifies a list of actions to be executed. It could be any of "
        "the following: " + " ".join(AVAILABLE_ACTIONS["full"].keys()),
    )
    args = parser.parse_args()

    method_args = vars(args)

    methods_to_run = {}

    if args.actions is not None:
        for a in args.actions:
            if a not in AVAILABLE_ACTIONS["full"]:
                import sys

                logger.error(f"The action does not exist: {a}")
                sys.exit(1)

            methods_to_run[a] = AVAILABLE_ACTIONS["full"][a]
    else:
        methods_to_run = AVAILABLE_ACTIONS[args.mode]

    for method_name, method in methods_to_run.items():
        method(**method_args)
