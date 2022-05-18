from pathlib import Path
import subprocess
import tempfile

import gls_cli

GLS_CLI_PATH = Path(gls_cli.__file__).resolve()
assert GLS_CLI_PATH is not None
assert GLS_CLI_PATH.exists()

DATA_DIR = (Path(__file__).parent / "data" / "gls").resolve()
assert DATA_DIR.exists()

TEMP_DIR = tempfile.mkdtemp()


def test_gls_cli_without_parameters():
    r = subprocess.run(
        [
            "python",
            GLS_CLI_PATH,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert r is not None
    assert r.returncode == 2
    r_output = r.stdout.decode("utf-8")
    assert r_output is not None
    assert len(r_output) > 1, r_output
    assert "error:" in r_output


def test_gls_cli_help():
    r = subprocess.run(
        ["python", GLS_CLI_PATH, "-h"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert r is not None
    assert r.returncode == 0
    r_output = r.stdout.decode("utf-8")
    assert r_output is not None
    assert len(r_output) > 1, r_output
    assert "PhenoPLIER command line tool" in r_output


def test_gls_cli_input_file_does_not_exist():
    r = subprocess.run(
        [
            "python",
            GLS_CLI_PATH,
            "-i",
            str(DATA_DIR / "does_not_exist.txt"),
            "-o",
            Path(TEMP_DIR) / "out.tsv",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert r is not None
    assert r.returncode == 1
    r_output = r.stdout.decode("utf-8")
    assert r_output is not None
    assert len(r_output) > 1, r_output
    assert "ERROR:" in r_output


def test_gls_cli_single_smultixcan_no_gene_name_column():
    r = subprocess.run(
        [
            "python",
            GLS_CLI_PATH,
            "-i",
            str(DATA_DIR / "random.pheno0-smultixcan-no_gene_name_column.txt"),
            "-o",
            Path(TEMP_DIR) / "out.tsv",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert r is not None
    assert r.returncode == 1
    r_output = r.stdout.decode("utf-8")
    assert r_output is not None
    assert len(r_output) > 1, r_output
    assert "Reading input file" in r_output
    assert "ERROR:" in r_output
    assert "'gene_name'" in r_output


def test_gls_cli_single_smultixcan_no_pvalue_column():
    r = subprocess.run(
        [
            "python",
            GLS_CLI_PATH,
            "-i",
            str(DATA_DIR / "random.pheno0-smultixcan-no_pvalue_column.txt"),
            "-o",
            Path(TEMP_DIR) / "out.tsv",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert r is not None
    assert r.returncode == 1
    r_output = r.stdout.decode("utf-8")
    assert r_output is not None
    assert len(r_output) > 1, r_output
    assert "Reading input file" in r_output
    assert "ERROR:" in r_output
    assert "'pvalue'" in r_output


def test_gls_cli_single_smultixcan_repeated_gene_names():
    r = subprocess.run(
        [
            "python",
            GLS_CLI_PATH,
            "-i",
            str(DATA_DIR / "random.pheno0-smultixcan-repeated_gene_names.txt"),
            "-o",
            Path(TEMP_DIR) / "out.tsv",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert r is not None
    assert r.returncode == 1
    r_output = r.stdout.decode("utf-8")
    assert r_output is not None
    assert len(r_output) > 1, r_output
    assert "Reading input file" in r_output
    assert "ERROR:" in r_output
    assert "Duplicated gene names" in r_output


def test_gls_cli_single_smultixcan_input_full():
    r = subprocess.run(
        [
            "python",
            GLS_CLI_PATH,
            "-i",
            str(DATA_DIR / "random.pheno0-smultixcan-full.txt"),
            "-o",
            Path(TEMP_DIR) / "out.tsv",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert r is not None
    r_output = r.stdout.decode("utf-8")
    # print(r_output)

    assert r.returncode == 0
    assert r_output is not None
    assert len(r_output) > 1, r_output
    assert "Reading input file" in r_output
    assert "Input file has 13 genes" in r_output
    assert (
        "p-values statistics: min=3.2e-05 | mean=1.2e-03 | max=2.4e-03 | # missing=3 (6.8%)"
        in r_output
    )

    # TODO: check output file
    # TODO: check number of genes being used (present in models)
