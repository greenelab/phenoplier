import numpy as np

from clustering.utils import compare_arrays


def test_compare_arrays_simple():
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([6, 7, 8, 9, 0])

    def my_func(a, b):
        return np.sum(a) + np.sum(b)

    xy = compare_arrays(x, y, my_func)

    assert xy is not None
    assert xy == (1 + 2 + 3 + 4 + 5) + (6 + 7 + 8 + 9 + 0)


def test_compare_arrays_simple2():
    x = np.array([1, 2, 3, 4, 5]) * 2
    y = np.array([6, 7, 8, 9, 0]) * 2

    def my_func(a, b):
        return np.sum(a) + np.sum(b)

    xy = compare_arrays(x, y, my_func)

    assert xy is not None
    assert xy == (2 + 4 + 6 + 8 + 10) + (12 + 14 + 16 + 18 + 0)


def test_compare_arrays_with_nan():
    x = np.array([1, 2, np.nan, 4, 5]) * 2
    y = np.array([6, 7, 8, 9, 0]) * 2

    def my_func(a, b):
        return np.sum(a) + np.sum(b)

    xy = compare_arrays(x, y, my_func)

    assert xy is not None
    assert not np.isnan(xy)
    assert xy == (2 + 4 + 0 + 8 + 10) + (12 + 14 + 0 + 18 + 0)


def test_compare_arrays_with_more_nans():
    x = np.array([1, 2, np.nan, 4, 5]) * 2
    y = np.array([np.nan, 7, 8, 9, 0]) * 2

    def my_func(a, b):
        return np.sum(a) + np.sum(b)

    xy = compare_arrays(x, y, my_func)

    assert xy is not None
    assert not np.isnan(xy)
    assert xy == (0 + 4 + 0 + 8 + 10) + (0 + 14 + 0 + 18 + 0)


def test_compare_arrays_use_weighting():
    x = np.array([1, 2, np.nan, 4, 5])
    y = np.array([np.nan, 7, 8, 9, 0])

    def my_func(a, b):
        return np.sum(a) + np.sum(b)

    xy = compare_arrays(x, y, my_func, use_weighting=True)

    assert xy is not None
    assert not np.isnan(xy)
    expected_weight = 3.0 / 5.0
    assert xy == expected_weight * ((0 + 2 + 0 + 4 + 5) + (0 + 7 + 0 + 9 + 0))


def test_compare_arrays_use_weighting_more_data():
    x = np.array([1, 2, np.nan, 4, 5, 6, 7])
    y = np.array([10, 20, 30, 40, 50, 60, 70])

    def my_func(a, b):
        return np.sum(a) + np.sum(b)

    xy = compare_arrays(x, y, my_func, use_weighting=True)

    assert xy is not None
    assert not np.isnan(xy)
    expected_weight = 6.0 / 7.0
    assert xy == expected_weight * (
        (1 + 2 + 0 + 4 + 5 + 6 + 7) + (10 + 20 + 0 + 40 + 50 + 60 + 70)
    )
