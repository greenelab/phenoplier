"""
General utility functions.
"""
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
