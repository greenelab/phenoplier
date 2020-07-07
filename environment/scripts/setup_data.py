"""
It sets up the file/folder structure by downloading the necessary files.
"""
import os

import conf
from utils import curl, check_md5


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
    )
    check_md5("cba910ee6f93eaed9d318edcd3f1ce18", output_file)


def download_phenomexcan_rapid_gwas_data_dict_file(**kwargs):
    output_file = conf.PHENOMEXCAN["RAPID_GWAS_DATA_DICT_FILE"]
    curl(
        "https://upenn.box.com/shared/static/u3po287ku1cj0jubbnsi7c4xawsaked5.tsv",
        output_file,
    )
    check_md5("c4b5938a7fdb0b1525f984cfb815bda5", output_file)


def download_phenomexcan_gtex_gwas_pheno_info(**kwargs):
    output_file = conf.PHENOMEXCAN["GTEX_GWAS_PHENO_INFO_FILE"]
    curl(
        "https://upenn.box.com/shared/static/gur0ug0qg7hs88ybrsgrwx7eeymmxay1.tsv",
        output_file,
    )
    check_md5("982434335f07acb1abfb83e57532f2c0", output_file)


def download_gene_map_name_to_id(**kwargs):
    output_file = conf.PHENOMEXCAN["GENE_MAP_NAME_TO_ID"]
    curl(
        "https://upenn.box.com/shared/static/t33a6iv4jtwc2pv2c1nllpnq0nlrfxkt.pkl",
        output_file,
    )
    check_md5("582d93c30c18027eefd465516733170f", output_file)


def download_gene_map_id_to_name(**kwargs):
    output_file = conf.PHENOMEXCAN["GENE_MAP_ID_TO_NAME"]
    curl(
        "https://upenn.box.com/shared/static/p20w0ikxhvo04xf1b2zai53cpoqb4ljz.pkl",
        output_file,
    )
    check_md5("63ac3ad54930d1b1490c6d02a68feb61", output_file)


def download_biomart_genes_hg38(**kwargs):
    output_file = conf.GENERAL["BIOMART_GENES_INFO_FILE"]
    curl(
        "https://upenn.box.com/shared/static/ks998wwlwble7rcb5cdthwjg1l0j1alb.gz",
        output_file,
    )
    check_md5("c4d74e156e968267278587d3ce30e5eb", output_file)


def download_uk_biobank_coding_3(**kwargs):
    output_file = conf.UK_BIOBANK["CODING_3_FILE"]
    curl(
        "https://upenn.box.com/shared/static/1f5yjg31qxemvf5hqkoz559cau14xr68.tsv",
        output_file,
    )
    check_md5("c02c65888793d4190fc190182128cc02", output_file)


def download_uk_biobank_coding_6(**kwargs):
    output_file = conf.UK_BIOBANK["CODING_6_FILE"]
    curl(
        "https://upenn.box.com/shared/static/libgz7998c2lsytjon8we1ouhabvh1z1.tsv",
        output_file,
    )
    check_md5("23a2bca99ea0bf25d141fc8573f67fce", output_file)


if __name__ == "__main__":
    _create_directories()

    # Obtain all local attributes of this module and run functions to download files
    local_items = list(locals().items())
    for key, value in local_items:
        if (
            callable(value)
            and value.__module__ == __name__
            and key.startswith("download_")
        ):
            value()
