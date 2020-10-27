from pathlib import Path

import pytest
import numpy as np
import pandas as pd

import rpy2.robjects as ro
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import pandas2ri

from multiplier import MultiplierProjection


def read_rds(test_case_number: int, kind: str):
    """Reads a test case data from an RDS file.

    Args:
        test_case_number (int): test case number to be read.
        kind (str): kind of data; it could be 'input_data' or 'output_data'.
    """
    readRDS = ro.r["readRDS"]
    rds_file = (
        Path(__file__).resolve().parent
        / "data"
        / "multiplier"
        / f"test_case{test_case_number}/{kind}.rds"
    )

    df = readRDS(str(rds_file))

    with localconverter(ro.default_converter + pandas2ri.converter):
        d = ro.conversion.rpy2py(df)
        return pd.DataFrame(data=d, index=df.rownames, columns=df.colnames)


def run_saved_test_case_simple_check(test_case_number, test_function=np.allclose):
    # prepare
    np.random.seed(0)
    input_data = read_rds(test_case_number, "input_data")

    # run
    mproj = MultiplierProjection()
    proj_data = mproj.transform(input_data)

    # evaluate
    assert proj_data is not None
    assert proj_data.shape == (987, input_data.shape[1])
    assert isinstance(proj_data, pd.DataFrame)

    expected_output_data = read_rds(test_case_number, "output_data")
    assert expected_output_data.shape == proj_data.shape
    assert test_function(expected_output_data.values, proj_data.values)


@pytest.mark.parametrize(
    "test_case_number",
    # these three cases include simple and small dataset with just a few genes and
    # traits (columns)
    [1, 2, 3],
)
def test_project_simple_data(test_case_number):
    run_saved_test_case_simple_check(test_case_number)


@pytest.mark.parametrize("test_case_number", [4])
def test_project_data_with_nan(test_case_number):
    run_saved_test_case_simple_check(
        test_case_number, lambda x, y: np.allclose(x, y, equal_nan=True)
    )


@pytest.mark.parametrize("test_case_number", [5])
def test_project_phenomexcan_subsample(test_case_number):
    run_saved_test_case_simple_check(test_case_number)
