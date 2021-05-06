import pytest


def test_gls_no_correlation_structure():
    # check that, if no correlation structure is given, results should match
    # R's nmle::gls function
    pass


def test_gls_artificial_data():
    # check that, with artificial data and correlation, results should match
    # R's nmle::gls function
    pass


def test_gls_real_data_original_correlation():
    # slice gene correlation data and test with LV136
    # should be the same as with R gls function
    pass


def test_gls_real_data_modified_positive_correlation():
    # artificially positively increase correlation between genes COL4A1 and COL4A2
    # results should be less significant
    pass


def test_gls_real_data_modified_negative_correlation():
    # artificially positively increase correlation between genes COL4A1 and COL4A2
    # results should be more significant
    pass
