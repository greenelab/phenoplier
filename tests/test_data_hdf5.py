from pathlib import Path
import re

import pytest
from data.hdf5 import HDF5_FILE_PATTERN


@pytest.mark.parametrize(
    "filename,expected_tissue",
    [
        ("spredixcan-Esophagus_Muscularis-zscore.h5", "Esophagus_Muscularis"),
        (
            "spredixcan-Brain_Anterior_cingulate_cortex_BA24-zscore.h5",
            "Brain_Anterior_cingulate_cortex_BA24",
        ),
    ],
)
def test_hdf5_file_pattern(filename, expected_tissue):
    m = re.search(HDF5_FILE_PATTERN, filename)
    assert m.group("tissue") == expected_tissue
