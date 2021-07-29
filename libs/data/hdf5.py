"""
Contains functions and utilities to read data in HDF5 format.
"""
import re

HDF5_FILE_PATTERN = re.compile(r"spredixcan-(?P<tissue>.+)-zscore\.h5")
HDF5_KEY_NO_PATTERN = re.compile(r"[^0-9a-zA-Z_]")


def simplify_trait_fullcode(trait_full_code: str, prefix: str = "c") -> str:
    """
    Simplifies a phenotype's full code for HDF5.

    Function copied from: https://github.com/hakyimlab/phenomexcan/blob/master/src/utils.py

    Args:
        trait_full_code: The full code of a PhenomeXcan's trait
        prefix: prefix for trait's full code. By default it's "c" to avoid
            problems with HDF5 key names.

    Returns:
         A new version of the trait's full code only with allowed characters
         for a key in a HDF5 file.
    """
    clean_col = re.sub(HDF5_KEY_NO_PATTERN, "_", trait_full_code)
    return prefix + clean_col
