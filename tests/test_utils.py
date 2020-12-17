import pytest

from utils import generate_result_set_name


@pytest.mark.parametrize(
    "method_options,expected_file_name",
    [
        # single parameter
        (
            {
                "opt1option": "opt1value",
            },
            "opt1option_opt1value",
        ),
        # single parameter upper case
        (
            {
                "OPT1option": "opt1value",
            },
            "OPT1option_opt1value",
        ),
        # single parameter with dash
        (
            {
                "opt1-option": "opt1value",
            },
            "opt1_option_opt1value",
        ),
        # two parameters
        (
            {
                "opt1option": "opt1value",
                "opt2-option": "opt2value",
            },
            "opt1option_opt1value-opt2_option_opt2value",
        ),
    ],
)
def test_generate_result_set_name_values_are_strings(
    method_options, expected_file_name
):
    file_name = generate_result_set_name(method_options)
    assert file_name == expected_file_name


@pytest.mark.parametrize(
    "method_options,expected_file_name",
    [
        # single parameter
        (
            {
                "opt1option": 1,
            },
            "opt1option_1",
        ),
        # two parameters
        (
            {
                "opt1option": "opt1value",
                "opt2-option": 2,
            },
            "opt1option_opt1value-opt2_option_2",
        ),
    ],
)
def test_generate_result_set_name_values_are_integers_also(
    method_options, expected_file_name
):
    file_name = generate_result_set_name(method_options)
    assert file_name == expected_file_name


@pytest.mark.parametrize(
    "method_options,options_sep,expected_file_name",
    [
        # single parameter
        (
            {
                "opt1option": "opt1value",
                "opt2-option": 10,
            },
            "-",
            "opt1option_opt1value-opt2_option_10",
        ),
        # two parameters
        (
            {
                "opt1option": "opt1value",
                "opt2-option": 2,
            },
            "#",
            "opt1option_opt1value#opt2_option_2",
        ),
    ],
)
def test_generate_result_set_name_options_separator(
    method_options, options_sep, expected_file_name
):
    file_name = generate_result_set_name(method_options, options_sep=options_sep)
    assert file_name == expected_file_name


@pytest.mark.parametrize(
    "method_options,prefix,expected_file_name",
    [
        # single parameter
        (
            {
                "opt1option": "opt1value",
            },
            "myfileprefix-",
            "myfileprefix-opt1option_opt1value",
        ),
    ],
)
def test_generate_result_set_name_with_prefix(
    method_options, prefix, expected_file_name
):
    file_name = generate_result_set_name(method_options, prefix=prefix)
    assert file_name == expected_file_name


@pytest.mark.parametrize(
    "method_options,suffix,expected_file_name",
    [
        # single parameter
        (
            {
                "opt1option": "opt1value",
            },
            ".pkl",
            "opt1option_opt1value.pkl",
        ),
    ],
)
def test_generate_result_set_name_with_suffix(
    method_options, suffix, expected_file_name
):
    file_name = generate_result_set_name(method_options, suffix=suffix)
    assert file_name == expected_file_name


@pytest.mark.parametrize(
    "method_options,prefix,suffix,expected_file_name",
    [
        # single parameter
        (
            {
                "opt1option": "opt1value",
                "OPT2option": "opt2value",
            },
            "another_prefix-",
            ".pkl",
            "another_prefix-OPT2option_opt2value-opt1option_opt1value.pkl",
        ),
    ],
)
def test_generate_result_set_name_with_prefix_and_suffix(
    method_options, prefix, suffix, expected_file_name
):
    file_name = generate_result_set_name(method_options, prefix=prefix, suffix=suffix)
    assert file_name == expected_file_name
