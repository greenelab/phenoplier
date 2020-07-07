"""
General utility functions.
"""
import hashlib
import os
import re
import pickle


def load_pickle(filepath):
    """Shortcut function to load a pickle file given by filepath.

    Args:
        filepath (str): file path of the pickle file to load.

    Returns:
        The object stored in the pickle file.
    """
    with open(filepath, "rb") as f:
        return pickle.load(f)


def simplify_string(s):
    """Given any string, it returns a simplified version of it.

    It removes any character that is not a word, a number or a separator. Then
    it replaces all separators by an underscore.

    Args:
        s (str): string to be simplified.

    Returns:
        str: string simplified.
    """
    # Remove all non-word characters (everything except numbers and letters)
    s = re.sub(r"[^\w\s]", "", s)

    # Replace all runs of whitespace with a single dash
    s = re.sub(r"\s+", "_", s)

    return s


def is_number(s):
    """
    Checks whether s is a number or not.

    Args:
        s (object): the object to check whether is a number or not.

    Returns:
        bool: Either True (s is a number) or False (s is not a number).
    """
    try:
        float(s)
        return True
    except ValueError:
        return False


def curl(url, output_file):
    """Downloads a file from an URL.

    Args:
        url (str): URL to download.
        output_file (str): path of file to store content.
    """
    print(f"Downloading {output_file}")
    os.system(f"curl -s -L {url} -o {output_file}")


def check_md5(expected_md5, filepath):
    """Checks the MD5 hash for a given filename and compares with the expected value.

    Args:
        expected_md5 (str): expected MD5 hash.
        filepath (str): file for which MD5 will be computed.

    Raises:
        AssertionError: if the expected MD5 differs from the actual MD5 value.
    """
    with open(filepath, "rb") as f:
        current_md5 = hashlib.md5(f.read()).hexdigest()
        assert expected_md5 == current_md5, f'md5 mismatch for "{filepath}"'
    print(f"md5 file ok for {filepath}")
