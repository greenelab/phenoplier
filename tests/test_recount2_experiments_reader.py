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
    'characteristics' column has a non-standard format, such as:
        tissue: Breast Cancer Cell Line
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


def test_read_characteristics_non_r_array_2():
    """
    'characteristics' column has another non-standard format, such as:
        cell type: MCF10a human breast cancer cells
    """
    srp_code = "SRP055569"
    edr = ExperimentDataReader(srp_code, srp_dir=SRP_DIR)

    assert "cell type" in edr.data.columns

    assert hasattr(edr, "characteristics_column_names")
    assert edr.characteristics_column_names is not None
    assert len(edr.characteristics_column_names) == 1
    assert "cell type" in edr.characteristics_column_names

    # make sure none of these three columns have null values
    assert edr.data.shape == edr.data.dropna(subset=["cell type"]).shape

    assert edr.data.iloc[0]["cell type"] == "MCF10a human breast cancer cells"

    assert edr.data.iloc[636]["cell type"] == "U87 human glioma cells"


def test_read_characteristics_non_r_array_3():
    """
    'characteristics' column has another non-standard format, such as:
        cervix cc cell line
    """
    srp_code = "SRP000599"
    edr = ExperimentDataReader(srp_code, srp_dir=SRP_DIR)

    unknown_col = ExperimentDataReader.UNKNOWN_ATTRIBUTE_COLUMN_NAME

    assert unknown_col in edr.data.columns

    assert hasattr(edr, "characteristics_column_names")
    assert edr.characteristics_column_names is not None
    assert len(edr.characteristics_column_names) == 1
    assert unknown_col in edr.characteristics_column_names

    # make sure none of these three columns have null values
    assert edr.data.shape == edr.data.dropna(subset=[unknown_col]).shape

    assert edr.data.iloc[0][unknown_col] == "cervix cc cell line"

    assert edr.data.iloc[23][unknown_col] == "liver carcinoma cell line"


def test_read_characteristics_non_r_array_4():
    """
    'characteristics' column has another non-standard format, such as:
        c(..., "who histotype: TC:SCC", ...)

    Note that the "value" has an ":" character
    """
    srp_code = "SRP042184"
    edr = ExperimentDataReader(srp_code, srp_dir=SRP_DIR)

    assert "who histotype" in edr.data.columns

    assert hasattr(edr, "characteristics_column_names")
    assert edr.characteristics_column_names is not None
    assert len(edr.characteristics_column_names) == 6
    assert "sample type" in edr.characteristics_column_names
    assert "who histotype" in edr.characteristics_column_names
    assert "final gtf2i mutation status" in edr.characteristics_column_names

    # make sure none of these three columns have null values
    assert (
        edr.data.shape
        == edr.data.dropna(
            subset=["sample type", "who histotype", "final gtf2i mutation status"]
        ).shape
    )

    assert edr.data.iloc[0]["sample type"] == "Frozen"
    assert edr.data.iloc[0]["who histotype"] == "B3"
    assert edr.data.iloc[0]["final gtf2i mutation status"] == "WT"

    assert edr.data.iloc[3]["sample type"] == "Frozen"
    assert edr.data.iloc[3]["who histotype"] == "TC:SCC"
    assert edr.data.iloc[3]["final gtf2i mutation status"] == "WT"


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
