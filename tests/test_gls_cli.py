from pathlib import Path
import subprocess
import tempfile

import numpy as np
import pandas as pd

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
    assert r.returncode == 2
    r_output = r.stdout.decode("utf-8")
    assert r_output is not None
    assert len(r_output) > 1, r_output
    assert "Input file does not exist" in r_output


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


def test_gls_cli_single_smultixcan_input_full_subset_of_lvs():
    output_file = Path(TEMP_DIR) / "out.tsv"

    r = subprocess.run(
        [
            "python",
            GLS_CLI_PATH,
            "-i",
            str(DATA_DIR / "random.pheno0-smultixcan-full.txt"),
            "-o",
            output_file,
            "-l",
            "LV1",
            "LV2",
            "LV3",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert r is not None
    r_output = r.stdout.decode("utf-8")
    print("\n" + r_output)

    assert r.returncode == 0
    assert r_output is not None
    assert len(r_output) > 1, r_output
    assert "Reading input file" in r_output
    assert "Input file has 54 genes" in r_output
    assert (
        "p-values statistics: min=3.2e-05 | mean=2.2e-03 | max=6.3e-03 | # missing=3 (5.6%)"
        in r_output
    )

    assert output_file.exists()
    output_data = pd.read_csv(output_file, sep="\t")
    assert output_data.shape[0] == 3  # 3 lvs tested
    assert "lv" in output_data.columns
    assert "coef" in output_data.columns
    assert "pvalue" in output_data.columns
    _lvs = set(output_data["lv"].tolist())
    assert "LV1" in _lvs
    assert "LV2" in _lvs
    assert "LV3" in _lvs
    assert not output_data.isna().any().any()


def test_gls_cli_single_smultixcan_input_full_subset_of_lvs_none_exist_in_models():
    output_file = Path(TEMP_DIR) / "out.tsv"

    r = subprocess.run(
        [
            "python",
            GLS_CLI_PATH,
            "-i",
            str(DATA_DIR / "random.pheno0-smultixcan-full.txt"),
            "-o",
            output_file,
            "-l",
            "LV1a",
            "LV2b",
            "LV3c",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert r is not None
    r_output = r.stdout.decode("utf-8")
    print("\n" + r_output)

    assert r.returncode == 1
    assert r_output is not None
    assert len(r_output) > 1, r_output
    assert "A list of 3 LVs was provided, and 0 are present in LV models" in r_output
    assert "No LVs were selected" in r_output


def test_gls_cli_single_smultixcan_input_full_all_lvs_in_model_file():
    output_file = Path(TEMP_DIR) / "out.tsv"

    r = subprocess.run(
        [
            "python",
            GLS_CLI_PATH,
            "-i",
            str(DATA_DIR / "random.pheno0-smultixcan-full.txt"),
            "-o",
            output_file,
            "-p",
            str(DATA_DIR / "sample-lv-model.pkl"),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert r is not None
    r_output = r.stdout.decode("utf-8")
    print("\n" + r_output)

    assert r.returncode == 0
    assert r_output is not None
    assert len(r_output) > 1, r_output
    assert "Reading input file" in r_output
    assert "Input file has 54 genes" in r_output
    assert (
        "p-values statistics: min=3.2e-05 | mean=2.2e-03 | max=6.3e-03 | # missing=3 (5.6%)"
        in r_output
    )

    assert output_file.exists()
    output_data = pd.read_csv(output_file, sep="\t")
    assert output_data.shape[0] == 5  # 5 lvs tested (all in the model file)
    assert "lv" in output_data.columns
    assert "coef" in output_data.columns
    assert "pvalue" in output_data.columns
    _lvs = set(output_data["lv"].tolist())
    assert "LV1" in _lvs
    assert "LV2" in _lvs
    assert "LV3" in _lvs
    assert "LV4" in _lvs
    assert "LV5" in _lvs
    assert not output_data.isna().any().any()


def test_gls_cli_use_incompatible_parameters_batch_and_lv_list():
    output_file = Path(TEMP_DIR) / "out.tsv"

    # batch 1
    r = subprocess.run(
        [
            "python",
            GLS_CLI_PATH,
            "-i",
            str(DATA_DIR / "random.pheno0-smultixcan-full.txt"),
            "-o",
            output_file,
            "-p",
            str(DATA_DIR / "sample-lv-model.pkl"),
            "-l",
            "LV1a",
            "LV2b",
            "LV3c",
            "--batch-id",
            "1",
            "--batch-n-splits",
            "3",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert r is not None
    r_output = r.stdout.decode("utf-8")
    print("\n" + r_output)

    assert r.returncode == 2
    assert r_output is not None
    assert len(r_output) > 1, r_output
    assert "Incompatible parameters" in r_output


def test_gls_cli_batch_parameters_batch_n_splits_missing():
    output_file = Path(TEMP_DIR) / "out.tsv"

    # batch 1
    r = subprocess.run(
        [
            "python",
            GLS_CLI_PATH,
            "-i",
            str(DATA_DIR / "random.pheno0-smultixcan-full.txt"),
            "-o",
            output_file,
            "-p",
            str(DATA_DIR / "sample-lv-model.pkl"),
            "--batch-id",
            "1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert r is not None
    r_output = r.stdout.decode("utf-8")
    print("\n" + r_output)

    assert r.returncode == 2
    assert r_output is not None
    assert len(r_output) > 1, r_output
    assert "Both --batch-id and --batch-n-splits" in r_output


def test_gls_cli_batch_parameters_batch_id_missing():
    output_file = Path(TEMP_DIR) / "out.tsv"

    # batch 1
    r = subprocess.run(
        [
            "python",
            GLS_CLI_PATH,
            "-i",
            str(DATA_DIR / "random.pheno0-smultixcan-full.txt"),
            "-o",
            output_file,
            "-p",
            str(DATA_DIR / "sample-lv-model.pkl"),
            "--batch-n-splits",
            "3",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert r is not None
    r_output = r.stdout.decode("utf-8")
    print("\n" + r_output)

    assert r.returncode == 2
    assert r_output is not None
    assert len(r_output) > 1, r_output
    assert "Both --batch-id and --batch-n-splits" in r_output


def test_gls_cli_batch_parameters_batch_id_value_invalid():
    output_file = Path(TEMP_DIR) / "out.tsv"

    # batch id is not an integer
    r = subprocess.run(
        [
            "python",
            GLS_CLI_PATH,
            "-i",
            str(DATA_DIR / "random.pheno0-smultixcan-full.txt"),
            "-o",
            output_file,
            "-p",
            str(DATA_DIR / "sample-lv-model.pkl"),
            "--batch-id",
            "a",
            "--batch-n-splits",
            "3",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert r is not None
    r_output = r.stdout.decode("utf-8")
    print("\n" + r_output)

    assert r.returncode == 2
    assert r_output is not None
    assert len(r_output) > 1, r_output
    assert "error: argument --batch-id" in r_output

    # batch id is negative
    r = subprocess.run(
        [
            "python",
            GLS_CLI_PATH,
            "-i",
            str(DATA_DIR / "random.pheno0-smultixcan-full.txt"),
            "-o",
            output_file,
            "-p",
            str(DATA_DIR / "sample-lv-model.pkl"),
            "--batch-id",
            "-1",
            "--batch-n-splits",
            "3",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert r is not None
    r_output = r.stdout.decode("utf-8")
    print("\n" + r_output)

    assert r.returncode == 2
    assert r_output is not None
    assert len(r_output) > 1, r_output
    assert "ERROR: --batch-id must be" in r_output

    # batch id is zero
    r = subprocess.run(
        [
            "python",
            GLS_CLI_PATH,
            "-i",
            str(DATA_DIR / "random.pheno0-smultixcan-full.txt"),
            "-o",
            output_file,
            "-p",
            str(DATA_DIR / "sample-lv-model.pkl"),
            "--batch-id",
            "0",
            "--batch-n-splits",
            "3",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert r is not None
    r_output = r.stdout.decode("utf-8")
    print("\n" + r_output)

    assert r.returncode == 2
    assert r_output is not None
    assert len(r_output) > 1, r_output
    assert "ERROR: --batch-id must be" in r_output

    # batch id is larger than --batch-n-splits
    r = subprocess.run(
        [
            "python",
            GLS_CLI_PATH,
            "-i",
            str(DATA_DIR / "random.pheno0-smultixcan-full.txt"),
            "-o",
            output_file,
            "-p",
            str(DATA_DIR / "sample-lv-model.pkl"),
            "--batch-id",
            "4",
            "--batch-n-splits",
            "3",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert r is not None
    r_output = r.stdout.decode("utf-8")
    print("\n" + r_output)

    assert r.returncode == 2
    assert r_output is not None
    assert len(r_output) > 1, r_output
    assert "ERROR: --batch-id must be <= --batch-n-splits" in r_output


def test_gls_cli_batch_parameters_batch_n_splits_value_invalid():
    output_file = Path(TEMP_DIR) / "out.tsv"

    # batch n splits is not an integer
    r = subprocess.run(
        [
            "python",
            GLS_CLI_PATH,
            "-i",
            str(DATA_DIR / "random.pheno0-smultixcan-full.txt"),
            "-o",
            output_file,
            "-p",
            str(DATA_DIR / "sample-lv-model.pkl"),
            "--batch-id",
            "1",
            "--batch-n-splits",
            "a",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert r is not None
    r_output = r.stdout.decode("utf-8")
    print("\n" + r_output)

    assert r.returncode == 2
    assert r_output is not None
    assert len(r_output) > 1, r_output
    assert "error: argument --batch-n-splits" in r_output

    # batch n splits is smaller than batch id
    r = subprocess.run(
        [
            "python",
            GLS_CLI_PATH,
            "-i",
            str(DATA_DIR / "random.pheno0-smultixcan-full.txt"),
            "-o",
            output_file,
            "-p",
            str(DATA_DIR / "sample-lv-model.pkl"),
            "--batch-id",
            "3",
            "--batch-n-splits",
            "0",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert r is not None
    r_output = r.stdout.decode("utf-8")
    print("\n" + r_output)

    assert r.returncode == 2
    assert r_output is not None
    assert len(r_output) > 1, r_output
    assert "ERROR: --batch-id must be <= --batch-n-splits" in r_output

    # batch n splits is smaller than batch id
    r = subprocess.run(
        [
            "python",
            GLS_CLI_PATH,
            "-i",
            str(DATA_DIR / "random.pheno0-smultixcan-full.txt"),
            "-o",
            output_file,
            "-p",
            str(DATA_DIR / "sample-lv-model.pkl"),
            "--batch-id",
            "3",
            "--batch-n-splits",
            "-2",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert r is not None
    r_output = r.stdout.decode("utf-8")
    print("\n" + r_output)

    assert r.returncode == 2
    assert r_output is not None
    assert len(r_output) > 1, r_output
    assert "ERROR: --batch-id must be <= --batch-n-splits" in r_output

    # batch n splits larger than LVs in the model
    r = subprocess.run(
        [
            "python",
            GLS_CLI_PATH,
            "-i",
            str(DATA_DIR / "random.pheno0-smultixcan-full.txt"),
            "-o",
            output_file,
            "-p",
            str(DATA_DIR / "sample-lv-model.pkl"),
            "--batch-id",
            "3",
            "--batch-n-splits",
            "6",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert r is not None
    r_output = r.stdout.decode("utf-8")
    print("\n" + r_output)

    assert r.returncode == 2
    assert r_output is not None
    assert len(r_output) > 1, r_output
    assert "ERROR: --batch-n-splits cannot be greater than LVs in the model (5 LVs)" in r_output


def test_gls_cli_single_smultixcan_input_full_use_batches_with_n_splits():
    output_file = Path(TEMP_DIR) / "out.tsv"

    # batch 1
    r = subprocess.run(
        [
            "python",
            GLS_CLI_PATH,
            "-i",
            str(DATA_DIR / "random.pheno0-smultixcan-full.txt"),
            "-o",
            output_file,
            "-p",
            str(DATA_DIR / "sample-lv-model.pkl"),
            "--batch-id",
            "1",
            "--batch-n-splits",
            "3",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert r is not None
    r_output = r.stdout.decode("utf-8")
    print("\n" + r_output)

    assert r.returncode == 0
    assert r_output is not None
    assert len(r_output) > 1, r_output
    assert "Using batch 1 out of 3" in r_output

    assert output_file.exists()
    output_data = pd.read_csv(output_file, sep="\t")
    assert output_data.shape[0] == 2  # 5 lvs tested (all in the model file)
    assert "lv" in output_data.columns
    assert "coef" in output_data.columns
    assert "pvalue" in output_data.columns
    _lvs = set(output_data["lv"].tolist())
    assert "LV1" in _lvs
    assert "LV2" in _lvs
    assert not output_data.isna().any().any()
    batch1_values = output_data

    # batch 2
    r = subprocess.run(
        [
            "python",
            GLS_CLI_PATH,
            "-i",
            str(DATA_DIR / "random.pheno0-smultixcan-full.txt"),
            "-o",
            output_file,
            "-p",
            str(DATA_DIR / "sample-lv-model.pkl"),
            "--batch-id",
            "2",
            "--batch-n-splits",
            "3",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert r is not None
    r_output = r.stdout.decode("utf-8")
    print("\n" + r_output)

    assert r.returncode == 0
    assert r_output is not None
    assert len(r_output) > 1, r_output
    assert "Using batch 2 out of 3" in r_output

    assert output_file.exists()
    output_data = pd.read_csv(output_file, sep="\t")
    assert output_data.shape[0] == 2  # 5 lvs tested (all in the model file)
    assert "lv" in output_data.columns
    assert "coef" in output_data.columns
    assert "pvalue" in output_data.columns
    _lvs = set(output_data["lv"].tolist())
    assert "LV3" in _lvs
    assert "LV4" in _lvs
    assert not output_data.isna().any().any()
    batch2_values = output_data

    # batch 3
    r = subprocess.run(
        [
            "python",
            GLS_CLI_PATH,
            "-i",
            str(DATA_DIR / "random.pheno0-smultixcan-full.txt"),
            "-o",
            output_file,
            "-p",
            str(DATA_DIR / "sample-lv-model.pkl"),
            "--batch-id",
            "3",
            "--batch-n-splits",
            "3",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert r is not None
    r_output = r.stdout.decode("utf-8")
    print("\n" + r_output)

    assert r.returncode == 0
    assert r_output is not None
    assert len(r_output) > 1, r_output
    assert "Using batch 3 out of 3" in r_output

    assert output_file.exists()
    output_data = pd.read_csv(output_file, sep="\t")
    assert output_data.shape[0] == 1  # 5 lvs tested (all in the model file)
    assert "lv" in output_data.columns
    assert "coef" in output_data.columns
    assert "pvalue" in output_data.columns
    _lvs = set(output_data["lv"].tolist())
    assert "LV5" in _lvs
    assert not output_data.isna().any().any()
    batch3_values = output_data

    # results should be different across batches
    assert not np.allclose(
        batch1_values["coef"].to_numpy(),
        batch2_values["coef"].to_numpy(),
    )
    assert not np.allclose(
        batch1_values["pvalue"].to_numpy(),
        batch2_values["pvalue"].to_numpy(),
    )

    assert not np.allclose(
        batch1_values["coef"].to_numpy(),
        batch3_values["coef"].to_numpy(),
    )
    assert not np.allclose(
        batch1_values["pvalue"].to_numpy(),
        batch3_values["pvalue"].to_numpy(),
    )

    assert not np.allclose(
        batch2_values["coef"].to_numpy(),
        batch3_values["coef"].to_numpy(),
    )
    assert not np.allclose(
        batch2_values["pvalue"].to_numpy(),
        batch3_values["pvalue"].to_numpy(),
    )


def test_gls_cli_single_smultixcan_input_full_use_batches_with_n_splits_chunks_same_size_of_1():
    output_file = Path(TEMP_DIR) / "out.tsv"

    # batch 1
    r = subprocess.run(
        [
            "python",
            GLS_CLI_PATH,
            "-i",
            str(DATA_DIR / "random.pheno0-smultixcan-full.txt"),
            "-o",
            output_file,
            "-p",
            str(DATA_DIR / "sample-lv-model.pkl"),
            "--batch-id",
            "1",
            "--batch-n-splits",
            "5",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert r is not None
    r_output = r.stdout.decode("utf-8")
    print("\n" + r_output)

    assert r.returncode == 0
    assert r_output is not None
    assert len(r_output) > 1, r_output
    assert "Using batch 1 out of 5" in r_output

    assert output_file.exists()
    output_data = pd.read_csv(output_file, sep="\t")
    assert output_data.shape[0] == 1  # 1 lvs tested
    assert "lv" in output_data.columns
    assert "coef" in output_data.columns
    assert "pvalue" in output_data.columns
    _lvs = set(output_data["lv"].tolist())
    assert "LV1" in _lvs
    assert not output_data.isna().any().any()

    # batch 2
    r = subprocess.run(
        [
            "python",
            GLS_CLI_PATH,
            "-i",
            str(DATA_DIR / "random.pheno0-smultixcan-full.txt"),
            "-o",
            output_file,
            "-p",
            str(DATA_DIR / "sample-lv-model.pkl"),
            "--batch-id",
            "2",
            "--batch-n-splits",
            "5",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert r is not None
    r_output = r.stdout.decode("utf-8")
    print("\n" + r_output)

    assert r.returncode == 0
    assert r_output is not None
    assert len(r_output) > 1, r_output
    assert "Using batch 2 out of 5" in r_output

    assert output_file.exists()
    output_data = pd.read_csv(output_file, sep="\t")
    assert output_data.shape[0] == 1  # 1 lvs tested
    assert "lv" in output_data.columns
    assert "coef" in output_data.columns
    assert "pvalue" in output_data.columns
    _lvs = set(output_data["lv"].tolist())
    assert "LV2" in _lvs
    assert not output_data.isna().any().any()

    # batch 3
    r = subprocess.run(
        [
            "python",
            GLS_CLI_PATH,
            "-i",
            str(DATA_DIR / "random.pheno0-smultixcan-full.txt"),
            "-o",
            output_file,
            "-p",
            str(DATA_DIR / "sample-lv-model.pkl"),
            "--batch-id",
            "3",
            "--batch-n-splits",
            "5",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert r is not None
    r_output = r.stdout.decode("utf-8")
    print("\n" + r_output)

    assert r.returncode == 0
    assert r_output is not None
    assert len(r_output) > 1, r_output
    assert "Using batch 3 out of 5" in r_output

    assert output_file.exists()
    output_data = pd.read_csv(output_file, sep="\t")
    assert output_data.shape[0] == 1  # 1 lvs tested
    assert "lv" in output_data.columns
    assert "coef" in output_data.columns
    assert "pvalue" in output_data.columns
    _lvs = set(output_data["lv"].tolist())
    assert "LV3" in _lvs
    assert not output_data.isna().any().any()

    # batch 4
    r = subprocess.run(
        [
            "python",
            GLS_CLI_PATH,
            "-i",
            str(DATA_DIR / "random.pheno0-smultixcan-full.txt"),
            "-o",
            output_file,
            "-p",
            str(DATA_DIR / "sample-lv-model.pkl"),
            "--batch-id",
            "4",
            "--batch-n-splits",
            "5",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert r is not None
    r_output = r.stdout.decode("utf-8")
    print("\n" + r_output)

    assert r.returncode == 0
    assert r_output is not None
    assert len(r_output) > 1, r_output
    assert "Using batch 4 out of 5" in r_output

    assert output_file.exists()
    output_data = pd.read_csv(output_file, sep="\t")
    assert output_data.shape[0] == 1  # 1 lvs tested
    assert "lv" in output_data.columns
    assert "coef" in output_data.columns
    assert "pvalue" in output_data.columns
    _lvs = set(output_data["lv"].tolist())
    assert "LV4" in _lvs
    assert not output_data.isna().any().any()

    # batch 5
    r = subprocess.run(
        [
            "python",
            GLS_CLI_PATH,
            "-i",
            str(DATA_DIR / "random.pheno0-smultixcan-full.txt"),
            "-o",
            output_file,
            "-p",
            str(DATA_DIR / "sample-lv-model.pkl"),
            "--batch-id",
            "5",
            "--batch-n-splits",
            "5",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert r is not None
    r_output = r.stdout.decode("utf-8")
    print("\n" + r_output)

    assert r.returncode == 0
    assert r_output is not None
    assert len(r_output) > 1, r_output
    assert "Using batch 5 out of 5" in r_output

    assert output_file.exists()
    output_data = pd.read_csv(output_file, sep="\t")
    assert output_data.shape[0] == 1  # 1 lvs tested
    assert "lv" in output_data.columns
    assert "coef" in output_data.columns
    assert "pvalue" in output_data.columns
    _lvs = set(output_data["lv"].tolist())
    assert "LV5" in _lvs
    assert not output_data.isna().any().any()


def test_gls_cli_single_smultixcan_input_full_use_batches_with_n_splits_is_1():
    output_file = Path(TEMP_DIR) / "out.tsv"

    # batch 1
    r = subprocess.run(
        [
            "python",
            GLS_CLI_PATH,
            "-i",
            str(DATA_DIR / "random.pheno0-smultixcan-full.txt"),
            "-o",
            output_file,
            "-p",
            str(DATA_DIR / "sample-lv-model.pkl"),
            "--batch-id",
            "1",
            "--batch-n-splits",
            "1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert r is not None
    r_output = r.stdout.decode("utf-8")
    print("\n" + r_output)

    assert r.returncode == 0
    assert r_output is not None
    assert len(r_output) > 1, r_output
    assert "Using batch 1 out of 1" in r_output

    assert output_file.exists()
    output_data = pd.read_csv(output_file, sep="\t")
    assert output_data.shape[0] == 5  # 1 lvs tested
    assert "lv" in output_data.columns
    assert "coef" in output_data.columns
    assert "pvalue" in output_data.columns
    _lvs = set(output_data["lv"].tolist())
    assert "LV1" in _lvs
    assert "LV2" in _lvs
    assert "LV3" in _lvs
    assert "LV4" in _lvs
    assert "LV5" in _lvs
    assert not output_data.isna().any().any()