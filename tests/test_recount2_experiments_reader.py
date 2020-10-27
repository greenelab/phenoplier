from pathlib import Path

from data.recount2 import ExperimentDataReader

SRP_DIR = Path(__file__).resolve().parent / "data" / "recount2" / "srp"


def test_read_data():
    """
    Simply test whether it can load the SRP data into a Pandas DataFrame
    """
    srp_code = "SRP026537"
    edr = ExperimentDataReader(srp_code, srp_dir=SRP_DIR)

    assert hasattr(edr, "data")
    assert edr.data is not None
    assert hasattr(edr.data, "shape")
    assert edr.data.shape[0] == 64
    assert "project" in edr.data.columns
    assert edr.data["project"].unique() == srp_code

    assert "title" in edr.data.columns
    assert "characteristics" in edr.data.columns


def test_read_data_compact():
    """
    Read in compact mode (only most useful columns)
    """
    srp_code = "SRP026537"
    edr = ExperimentDataReader(srp_code, srp_dir=SRP_DIR, compact=True)

    # assert hasattr(edr, 'data')
    assert edr.data is not None
    assert hasattr(edr.data, "shape")
    assert edr.data.shape == (64, 3 + 3)

    assert "project" in edr.data.columns
    assert "run" in edr.data.columns
    assert "characteristics" in edr.data.columns
    assert "cell line" in edr.data.columns
    assert "cell type" in edr.data.columns
    assert "subtype" in edr.data.columns

    assert edr.data["project"].unique() == srp_code


def test_read_characteristics_standard():
    """
    Test whether the 'characteristics' column is correctly read and new columns are added.
    """
    srp_code = "SRP026537"
    edr = ExperimentDataReader(srp_code, srp_dir=SRP_DIR)

    assert "cell line" in edr.data.columns
    assert "cell type" in edr.data.columns
    assert "subtype" in edr.data.columns

    assert hasattr(edr, "characteristics_column_names")
    assert edr.characteristics_column_names is not None
    assert len(edr.characteristics_column_names) == 3
    assert "cell line" in edr.characteristics_column_names
    assert "cell type" in edr.characteristics_column_names
    assert "subtype" in edr.characteristics_column_names

    # make sure none of these three columns have null values
    assert (
        edr.data.shape
        == edr.data.dropna(subset=["cell line", "cell type", "subtype"]).shape
    )

    assert edr.data.iloc[0]["cell line"] == "184A1"
    assert edr.data.iloc[0]["cell type"] == "breast cancer"
    assert edr.data.iloc[0]["subtype"] == "Non-malignant"

    assert edr.data.iloc[10]["cell line"] == "BT483"
    assert edr.data.iloc[10]["cell type"] == "breast cancer"
    assert edr.data.iloc[10]["subtype"] == "Luminal"


def test_read_characteristics_non_r_array():
    """
    'characteristics' column has a non-standard format, such as: tissue: Breast Cancer Cell Line
    """
    srp_code = "SRP042620"
    edr = ExperimentDataReader(srp_code, srp_dir=SRP_DIR)

    assert "tissue" in edr.data.columns

    assert hasattr(edr, "characteristics_column_names")
    assert edr.characteristics_column_names is not None
    assert len(edr.characteristics_column_names) == 1
    assert "tissue" in edr.characteristics_column_names

    # make sure none of these three columns have null values
    assert edr.data.shape == edr.data.dropna(subset=["tissue"]).shape

    assert edr.data.iloc[0]["tissue"] == "Breast Cancer Cell Line"

    assert edr.data.iloc[28]["tissue"] == "ER+ Breast Cancer Primary Tumor"


def test_non_existent_srp_file():
    """
    Tests whether the class downloads the non-existent file and stores it in
    the designated directory.
    """
    srp_code = "SRP064259"

    # make sure the file does not exist in the SRP_DIR folder
    srp_file = Path(SRP_DIR, f"{srp_code}.tsv").resolve()
    srp_file.unlink(missing_ok=True)

    edr = ExperimentDataReader(srp_code, srp_dir=SRP_DIR)

    assert hasattr(edr, "data")
    assert edr.data is not None
    assert hasattr(edr.data, "shape")
    assert edr.data.shape[0] == 82
    assert "project" in edr.data.columns
    assert edr.data["project"].unique() == srp_code
