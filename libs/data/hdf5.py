import re
from pathlib import Path

import pandas as pd


HDF5_FILE_PATTERN = re.compile(r"spredixcan-(?P<tissue>.+)-zscore\.h5")
HDF5_KEY_NO_PATTERN = re.compile(r"[^0-9a-zA-Z_]")


def simplify_trait_fullcode(trait_full_code: str) -> str:
    """
    Simplifies a phenotype's full code for HDF5.

    Function copied from: https://github.com/hakyimlab/phenomexcan/blob/master/src/utils.py

    Args:
        trait_full_code: The full code of a PhenomeXcan's trait

    Returns:
         A new version of the trait's full code only with allowed characters
         for a key in a HDF5 file.
    """
    clean_col = re.sub(HDF5_KEY_NO_PATTERN, "_", trait_full_code)
    return "c" + clean_col


def read_spredixcan(folder, trait, tissue):
    """
    TODO: finish

    Args:
        folder:
        trait:
        tissue:

    Returns:

    """
    filename = f"spredixcan-{tissue}-zscore.h5"
    input_filepath = Path(folder, filename)

    trait_hdf5 = simplify_trait_fullcode(trait)

    with pd.HDFStore(input_filepath, mode="r") as store:
        return store[trait_hdf5].rename_axis("gene_id")
