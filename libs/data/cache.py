"""
Provides functions to read frequently used files (which are cached) and returns
pandas.DataFrame objects.
"""
from pathlib import Path

from data.readers import DATA_READERS, DATA_FORMAT_READERS

DATA_CACHE = {}


def read_data(filepath: Path, **kwargs):
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
    file_extensions = ''.join(filepath.suffixes)

    if filepath in DATA_READERS:
        reading_function = DATA_READERS[filepath]
    elif file_extensions in DATA_FORMAT_READERS:
        reading_function = DATA_FORMAT_READERS[file_extensions](filepath, **kwargs)
    else:
        raise ValueError(
            f"{filepath}: there is no function that can read this file."
        )

    if filepath not in DATA_CACHE:
        DATA_CACHE[filepath] = reading_function()

    return DATA_CACHE[filepath]
