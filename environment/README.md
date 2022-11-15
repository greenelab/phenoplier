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

    # (optional, will default to 1 core)
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

1. Create a conda environment and install main packages:

    ```bash
   conda config --set channel_priority strict
   conda env create --name phenoplier --file environment.yml
   conda run -n phenoplier --no-capture-output bash scripts/install_other_packages.sh
    ```
1. Download the data:

   ```bash
   conda run -n phenoplier --no-capture-output python scripts/setup_data.py
   ```

This will download ~70 GB of data needed to run the analyses.


# Developer usage

These steps are only for PhenoPLIER developers.
All steps are run from the root directory (not within `environment/`).

It is a good idea to try to build the environment locally first and, when all issues have been solved, then create the Docker image.
A usual problem is to use a too recent Python version that produces several conflicts in conda.
In that case, the previous Python version should be used instead.

1. Modify `environment/scripts/environment_base.yml` accordingly (if needed).
Usually, this involves updating to the latest Python and R versions.

1. (if creating a local environment) Run:
 
    ```bash
    conda config --set channel_priority strict
    conda env create --name phenoplier --file environment/scripts/environment_base.yml
    conda run -n phenoplier --no-capture-output bash environment/scripts/install_other_packages.sh
    ```

1. (if creating a new Docker image) Run:
    ```bash
    cp environment/scripts/environment_base.yml environment/environment.yml
    ```

    Now open `scripts/create_docker_image.sh` and change settings according to instructions in the file.
    Then run:

    ```bash
    # IMPORTANT: the script below will build two images: base and final.
    #  The base image will only be rebuilt if the version in settings (see
    #  the script) is change. If for some reason you want to force building the
    #  the image (for example, you fix something in the Dockerfile), you have to
    #  pass the following argument: -f

    bash scripts/create_docker_image.sh
    ```

1. Export conda environment:

    ```bash
    # if creating a local environment:
    conda env export --name phenoplier --file environment/environment.yml

    # if creating a new Docker image:
    bash scripts/run_docker.sh conda env export --name phenoplier --file environment/environment.yml
    ```

1. Modify `environment/environment.yml` and leave only manually installed packages (not their dependencies).

1. (if creating a new Docker image) Push the new Docker images.
See at the end of `scripts/create_docker_image.sh` for examples.

