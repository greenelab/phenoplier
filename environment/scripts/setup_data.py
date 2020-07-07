"""
It sets up the file/folder structure by downloading the necessary files.
"""
import os
import hashlib

import conf


def _check_md5(expected_md5, filepath):
    """Checks the MD5 hash for a given filename and compares with the expected value.

    Args:
        expected_md5 (str): expected MD5 hash.
        filepath (str): file for which MD5 will be computed.

    Raises:
        AssertionError: if the expected MD5 differs from the actual MD5 value.
    """
    with open(filepath, "rb") as f:
        current_md5 = hashlib.md5(f.read()).hexdigest()
        assert expected_md5 == current_md5, f'md5 mismatch for "{filepath}"'
    print(f"md5 file ok for {filepath}")


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
            os.makedirs(v, exist_ok=True)


def download_phenomexcan_rapid_gwas_pheno_info(**kwargs):
    output_file = conf.PHENOMEXCAN["RAPID_GWAS_PHENO_INFO_FILE"]
    os.system(
        f"wget https://upenn.box.com/shared/static/163mkzgd4uosk4pnzx0xsj7n0reu8yjv.gz -O {output_file}"
    )
    _check_md5("cba910ee6f93eaed9d318edcd3f1ce18", output_file)


def download_phenomexcan_rapid_gwas_data_dict_file(**kwargs):
    output_file = conf.PHENOMEXCAN["RAPID_GWAS_DATA_DICT_FILE"]
    os.system(
        f"wget https://upenn.box.com/shared/static/u3po287ku1cj0jubbnsi7c4xawsaked5.tsv -O {output_file}"
    )
    _check_md5("c4b5938a7fdb0b1525f984cfb815bda5", output_file)


def download_phenomexcan_gtex_gwas_pheno_info(**kwargs):
    output_file = conf.PHENOMEXCAN["GTEX_GWAS_PHENO_INFO_FILE"]
    os.system(
        f"wget https://upenn.box.com/shared/static/gur0ug0qg7hs88ybrsgrwx7eeymmxay1.tsv -O {output_file}"
    )
    _check_md5("982434335f07acb1abfb83e57532f2c0", output_file)


def download_gene_map_name_to_id(**kwargs):
    output_file = conf.PHENOMEXCAN["GENE_MAP_NAME_TO_ID"]
    os.system(
        f"wget https://upenn.box.com/shared/static/t33a6iv4jtwc2pv2c1nllpnq0nlrfxkt.pkl -O {output_file}"
    )
    _check_md5("582d93c30c18027eefd465516733170f", output_file)


def download_gene_map_id_to_name(**kwargs):
    output_file = conf.PHENOMEXCAN["GENE_MAP_ID_TO_NAME"]
    os.system(
        f"wget https://upenn.box.com/shared/static/p20w0ikxhvo04xf1b2zai53cpoqb4ljz.pkl -O {output_file}"
    )
    _check_md5("63ac3ad54930d1b1490c6d02a68feb61", output_file)


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
