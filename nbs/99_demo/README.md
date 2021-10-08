# PhenoPLIER demo

The fastest way to quickly test PhenoPLIER through an example is to [install Docker](https://docs.docker.com/get-docker/) and run the PhenoPLIER container.
The instructions below should work on any Linux or macOS operating system.
Some minor adjustments should be done if you are using Windows.

Run the commands below (you can change folder in `DATA_FOLDER`) and ignore warnings related to PermissionError:

```bash
# pull the docker image
docker pull miltondp/phenoplier

# specify a directory in your computer where data will be stored
export DATA_FOLDER="/tmp/phenoplier_data"
mkdir -p ${DATA_FOLDER}

# download data to run the demo
docker run --rm \
  -v "${DATA_FOLDER}:/opt/phenoplier_data" \
  --user "$(id -u):$(id -g)" \
  miltondp/phenoplier \
  /bin/bash -c "python environment/scripts/setup_data.py --mode demo"
  
# run jupyter lab
docker run --rm \
  -p 8888:8892 \
  -v "${DATA_FOLDER}:/opt/phenoplier_data" \
  --user "$(id -u):$(id -g)" \
  miltondp/phenoplier
```

You can access the web interface by going to http://localhost:8888/lab/tree/nbs/99_demo.
From here, you should open the notebooks in the order presented: first `01-LV_trait_association-...`, then `02-LV_cell_types-...`, etc.
Once a notebook is opened in your browser, you can run each cell by clicking on the "Play" icon or pressing Shift+Enter.
Go through each cell and follow the instructions.

Then you can also try it with your own GWAS/TWAS results (you can copy your files to the `${DATA_FOLDER}` you specified below, and it will be available to the container).
Remember, however, that changes in the notebooks/code using this approach will not be preserved after you stop the Docker container.
If you are interested in keeping your code changes, you should also mount the Github repo (after cloning it) into the Docker container, like this:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone git@github.com:greenelab/phenoplier.git
cd phenoplier
export CODE_FOLDER=`pwd`
cd

docker run --rm \
  -p 8888:8892 \
  -v "${DATA_FOLDER}:/opt/phenoplier_data" \
  -v "${CODE_FOLDER}:/opt/phenoplier_code" \
  --user "$(id -u):$(id -g)" \
  miltondp/phenoplier
```
