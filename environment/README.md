## Conda environment

If you want to run scripts/notebook from PhenoPLIER, you need to follow these steps to create a
conda environment.

 1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
 1. Run:
 
    ```bash
    $ conda env create -n phenoplier -f environment.yml
    [...]
    $ conda activate phenoplier
    $ bash scripts/install_other_packages.sh
    [...]
    ```

## Developer usage

These steps are only for PhenoPLIER developers.

 1. Modify `scripts/create_env_from_scratch.sh` accordingly (if needed).
 1. Run:
 
    ```bash
    $ bash scripts/create_env_from_scratch.sh
    [...]
    $ bash scripts/install_other_packages.sh
    [...]
    ```

 1. Install JupyterLab extensions:
 
    ```bash
    $ jupyter labextension install @jupyterlab/toc
    ```

 1. Export conda environment:

    ```
    $ conda env export -n phenoplier -f environment.yml
    ```

 1. Modify `environment.yml` and leave only manually installed packages (not their dependencies).
