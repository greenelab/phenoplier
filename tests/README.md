# Unit tests

## Run

These are the instructions to run the unit tests. It is assumed that you already 
followed the steps to setup the environment and download the needed data, and that
your `PYTHONPATH` and `PHENOPLIER_ROOT_DIR` variables are adjusted appropriately.

Execute these commands to run the unit tests:

```bash
pytest -rs --color=yes
```

## Generate data for test cases

Some test cases need data to be generated (which is already present in the repository), which represent small
samples of huge datasets. These are small `.rds` files that can be generated again by running:

```bash
python generate_test_cases.py
```

## Misc

The file `scripts/plier_util.R` was downloaded from
[here](https://github.com/greenelab/multi-plier/blob/v0.2.0/util/plier_util.R).
It is used to generate test data.
