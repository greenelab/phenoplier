# Manual conda environment installation and data download

If you want to run scripts/notebook from PhenoPLIER, you need to follow these
steps to create a conda environment and download the necessary data.

Keep in mind that the software only runs on Linux or macOS, **Windows is not
supported** now. If you want to run on Windows, use the Docker image instead.

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda.

1. Open a terminal, run `cd environment` from the `phenoplier` folder repo.

1. (optional) Adjust your environment variables:

    ```bash
    # (optional, will default to subfolder 'phenoplier' under the system's temporary directory)
    # Root directory where all data will be downloaded to
    export PHENOPLIER_ROOT_DIR=/tmp/phenoplier

    # (optional, will default to half the number of cores)
    # Adjust the number of cores available for general tasks
    export PHENOPLIER_N_JOBS=2

    # (optional)
    # Export this variable if you downloaded the manuscript sources and want to
    # generate the figures for it
    export PHENOPLIER_MANUSCRIPT_DIR=/tmp/manuscript_dir
    ```

1. (optional) Adjust other settings (i.e. root directory, available computational
   resources, etc.) by modifying the file `../libs/settings.py`

1. Adjust your `PYTHONPATH` variable to include the `libs` directory:

    ```bash
    export PYTHONPATH=`readlink -f ../libs/`:$PYTHONPATH
    ```

    `readlink` might not work on macOS. In that case, simply replace it with
    the absolute path to the `../libs/` folder.

1. Run `bash scripts/setup_environment.sh`. This will create a conda environment and download the data needed to run the analyses.
This will download ~70 GB, so it will take a while to finish.


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
