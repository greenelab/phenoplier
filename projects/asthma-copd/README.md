# PhenoPLIER analyses of Asthma-COPD Overlap Syndrome (ACOS)

## Contents

 * [Overview](#overview)
 * [Quick demo](#quick-demo)
 * [Code and data](#code-and-data)
 * [Setup](#setup)
 * [Running the code](#running-the-code)


## Overview

**TODO**


## Quick demo

**TODO**


## Code and data

**TODO**


## Setup

```bash
docker pull miltondp/phenoplier:asthma-copd
```


## Running the code

**TODO**


## Instructions for developers

**You very likely do not need to follow these steps**, unless you are a developer working on PhenoPLIER.

### Setup Docker image

**This only needs to be done once.**

Pull the right Docker image for this project and tag it accordingly:

```bash
bash projects/asthma-copd/scripts/create_docker_image.sh
```

### Load project-specific configuration

```bash
. projects/asthma-copd/scripts/env.sh
```

### Download data/software

```bash
bash scripts/run_docker_dev.sh python environment/scripts/setup_data.py --mode asthma-copd
```

### Start JupyterLab server

```bash
bash scripts/run_docker_dev.sh
```


### Run notebook from command-line

```bash
bash scripts/run_docker_dev.sh bash nbs/run_nbs.sh projects/asthma-copd/nbs/05_gwas/05-gwas-inflation_factor.ipynb
```
