from pathlib import Path
import re

import pytest
from data.hdf5 import HDF5_FILE_PATTERN, read_spredixcan


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


@pytest.mark.parametrize(
    "trait, tissue, gene, expected_zscore",
    [
        ("50_raw-Standing_height", "Whole_Blood", "ENSG00000101019", -34.024),
        ("50_raw-Standing_height", "Whole_Blood", "ENSG00000109805", -22.855),
        ("50_raw-Standing_height", "Whole_Blood", "ENSG00000177311", 33.819),
        ("pgc.scz2", "Prostate", "ENSG00000233822", 10.752),
        ("pgc.scz2", "Prostate", "ENSG00000137312", -8.827),
        ("pgc.scz2", "Prostate", "ENSG00000204257", -7.965),
    ],
)
def test_spredixcan(trait, tissue, gene, expected_zscore):
    import conf

    SPREDIXCAN_H5_FOLDER = Path(
        conf.PHENOMEXCAN["SPREDIXCAN_MASHR_ZSCORES_FOLDER"],
        "hdf5",
    )

    assert (
        read_spredixcan(SPREDIXCAN_H5_FOLDER, trait, tissue).loc[gene].round(3)
        == expected_zscore
    )
