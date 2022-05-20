"""
General utility functions.
"""
import re
import hashlib
import subprocess
from pathlib import Path
from subprocess import run
from typing import Dict

from log import get_logger


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


def chunker(seq, size):
    """
    Divides a sequence in chunks according to the given size. For example, if
    given a list

        [0,1,2,3,4,5,6,7]

    and size is 3, it will return

        [[0, 1, 2], [3, 4, 5], [6, 7]]
    """
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))


def curl(url: str, output_file: str, md5hash: str = None, logger=None):
    """Downloads a file from an URL. If the md5hash option is specified, it checks
    if the file was successfully downloaded (whether MD5 matches).

    Before starting the download, it checks if output_file exists. If so, and md5hash
    is None, it quits without downloading again. If md5hash is not None, it checks if
    it matches the file.

    Args:
        url: URL of file to download.
        output_file: path of file to store content.
        md5hash: expected MD5 hash of file to download.
        logger: Logger instance.
    """
    logger = logger or get_logger("none")

    Path(output_file).resolve().parent.mkdir(parents=True, exist_ok=True)

    if Path(output_file).exists() and (
        md5hash is None or md5_matches(md5hash, output_file)
    ):
        logger.info(f"File already downloaded: {output_file}")
        return

    logger.info(f"Downloading {output_file}")
    run(["curl", "-s", "-L", url, "-o", output_file])

    if md5hash is not None and not md5_matches(md5hash, output_file):
        msg = "MD5 does not match"
        logger.error(msg)
        raise AssertionError(msg)


def md5_matches(expected_md5: str, filepath: str) -> bool:
    """Checks the MD5 hash for a given filename and compares with the expected value.

    Args:
        expected_md5: expected MD5 hash.
        filepath: file for which MD5 will be computed.

    Returns:
        True if MD5 matches, False otherwise.
    """
    with open(filepath, "rb") as f:
        current_md5 = hashlib.md5(f.read()).hexdigest()
        return expected_md5 == current_md5


def generate_result_set_name(
    method_options: Dict, options_sep: str = "-", prefix: str = None, suffix: str = None
) -> str:
    """Generates a filename for a result set with the method's options given.

    When a method is run with several options (like a clustering/classification
    algorithm) and its results need to be saved to a file, this method generates a
    descriptive filename using the given options.

    Args:
        method_options: dictionary with parameter names and their values.
        options_sep: options separator.
        prefix: optional prefix for the filename.
        suffix: optional suffix (like a filename extension).
    Returns:
        A filename as a str object.
    """

    def simplify_option_name(s: str) -> str:
        # s = s.lower()

        # remove any non-allowed character
        s = re.sub(r"[^\w\s\-_]", "", s)

        s = re.sub(r"-", "_", s)

        return s

    def simplify_option_value(s) -> str:
        if isinstance(s, str):
            return simplify_option_name(s)
        elif isinstance(s, (list, tuple, set)):
            return "_".join(simplify_option_name(str(x)) for x in s)
        else:
            return simplify_option_name(str(s))

    output_file_suffix = options_sep.join(
        [
            f"{simplify_option_name(k)}_{simplify_option_value(v)}"
            for k, v in sorted(method_options.items(), reverse=False)
        ]
    )

    filename = output_file_suffix

    if prefix is not None:
        filename = f"{prefix}{filename}"

    if suffix is not None:
        filename = f"{filename}{suffix}"

    return filename


def get_git_repository_path():
    """
    Returns the Git repository path. If for any reason running git fails, it
    returns the operating system temporary folder.
    """
    try:
        results = run(["git", "rev-parse", "--show-toplevel"], stdout=subprocess.PIPE)

        return Path(results.stdout.decode("utf-8").strip())
    except Exception:
        import tempfile

        return Path(tempfile.gettempdir())
