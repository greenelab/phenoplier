# Conda environment and data

If you want to run scripts/notebook from PhenoPLIER, you need to follow these steps to create a
conda environment and download the necessary data.

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
1. Open a terminal and execute all following steps in it.
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
   
    # Export this variable if you downloaded the manuscript sources and want to
    # generate the figures for it (see additional # instructions for manuscript
    # figures below).
    export PHENOPLIER_MANUSCRIPT_DIR=/tmp/manuscript_dir
    ```

1. Adjust other settings (i.e. available computational resources) by modifying the
file `../libs/settings.py`

1. Adjust your `PYTHONPATH` variable to include the `libs` directory:

    ```bash
    export PYTHONPATH=`readlink -f ../libs/`:$PYTHONPATH
    ```

1. Download the data:

    ```bash
    python scripts/setup_data.py
    ```


# Manuscript files

The code in this repository also optionally generates figures and other files for the manuscript.
You need to manually install these dependencies for your operating system:

DO WE REALLY NEED THIS?

1. `pdf2svg` ([github](https://github.com/dawbarton/pdf2svg))


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

