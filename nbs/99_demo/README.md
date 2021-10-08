# PhenoPLIER demo

The fastest way to quickly test PhenoPLIER through an example is to [install Docker](https://docs.docker.com/get-docker/) and run the container:
```bash
# pull the docker image
docker pull miltondp/phenoplier

# download data to run the demo
docker run --rm \
  miltondp/phenoplier \
  /bin/bash -c "python environment/scripts/setup_data.py --mode demo"
  
# run jupyter lab
docker run --rm \
  -p 8888:8892 \
  miltondp/phenoplier
```

and access the web interface by going to `http://localhost:8888`.



AT THE END: 

After following the demo, you might want to use your own files or copy files from the docker container.
If that's the case, you can run jupyter lab in this way:
```bash
docker run --rm \
  -p 8888:8892 \
  -v "/tmp/phenoplier_data:/opt/phenoplier_data" \
  miltondp/phenoplier
```

The `-v` parameter specifies a local folder in your machine that will be mounted in `/opt/phenoplier_data`.
So everything you copy in `/tmp/phenoplier_data/` will be appear in `/opt/phenoplier_data/` and be accessible from the Jupyter notebooks.
