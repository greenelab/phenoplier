# Conda environment and data

If you want to run scripts/notebook from PhenoPLIER, you need to follow these
steps to create a conda environment and download the necessary data.

Keep in mind that the software only runs on Linux or macOS, **Windows is not
supported** now.

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
1. Open a terminal, run `cd environment` from the folder where you cloned this
   repo, and execute all following steps in it.
1. Run:
 
    ```bash
    conda env create --name phenoplier --file environment.yml
    conda activate phenoplier
    bash scripts/install_other_packages.sh
    ```

1. Adjust your environment variables:

    ```bash
    # Root directory where all data will be downloaded to.
    export PHENOPLIER_ROOT_DIR=/tmp/phenoplier

    # Adjust the number of cores available for general tasks
    export PHENOPLIER_N_JOBS=2

    # If you run the notebooks from the command line (using the
    # `nbs/run_nbs.sh` script), this variable tells papermill to automatically
    # override notebooks once finished (recommended). If not set, it will
    # ask what you want to do.
    export PHENOPLIER_RUN_NBS_OVERRIDE=1
   
    # Export this variable if you downloaded the manuscript sources and want to
    # generate the figures for it.
    export PHENOPLIER_MANUSCRIPT_DIR=/tmp/manuscript_dir
    ```

1. Adjust other settings (i.e. root directory, available computational
   resources, etc) by modifying the file `../libs/settings.py`

1. Adjust your `PYTHONPATH` variable to include the `libs` directory:

    ```bash
    export PYTHONPATH=`readlink -f ../libs/`:$PYTHONPATH
    ```

1. Download the data:

    ```bash
    python scripts/setup_data.py --mode light
    ```

    Note that if you want to download all the data you have to use ``--mode
    full``.  The full data set contains the original MultiPLIER RDS file with
    the models, which is only necessary if you need to process it again.


# Developer usage

These steps are only for PhenoPLIER developers.

1. Modify `scripts/environment_base.yml` accordingly (if needed).
1. Run:
 
    ```bash
    bash scripts/create_env_from_scratch.sh
    conda activate phenoplier
    bash scripts/install_other_packages.sh
    ```


1. Install JupyterLab extensions:
 
    ```bash
    jupyter labextension install @jupyterlab/toc
    ```

1. Export conda environment:

    ```
    conda env export --name phenoplier --file environment.yml
    ```

1. Modify `environment.yml` and leave only manually installed packages (not their dependencies).


## Optional packages

These optional packages might be useful for a developer, but are not necessary
for a user.

```bash
conda install -c conda-forge jupytext
```

