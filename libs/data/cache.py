"""
Provides functions to read frequently used files (which are cached) and returns
pandas.DataFrame objects.
"""
from data.readers import DATA_READERS

DATA_CACHE = {}


def read_data(filepath):
    """Reads any data file given and returns a pandas.DataFrame object.

    Args:
        filepath (str): any file path present in the conf module and which has a
        data reader (data.readers.DATA_READER).

    Returns:
        A pandas.DataFrame instance.

    Raises:
        ValueError: if the file path has no data reader specified in
        data.readers.DATA_READER.
    """
    if filepath not in DATA_READERS:
        raise ValueError(f"{filepath} does not exist in the configuration.")

    if filepath not in DATA_CACHE:
        DATA_CACHE[filepath] = DATA_READERS[filepath]()

    return DATA_CACHE[filepath]
