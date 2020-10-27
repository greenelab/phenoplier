"""
This script creates data for some tests.
"""

from pathlib import Path
from subprocess import run

import conf

output_dir = Path("data", "multiplier")
output_dir.mkdir(exist_ok=True)

# take a small sample of the S-MultiXcan results from PhenomeXcan
run(["python", "scripts/generate_phenomexcan_data_sample.py", output_dir])

print("\n")

# generate some use cases using the GetNewDataB function from MultiPLIER. The results
# generated here will be used as the expected results for the Python implementation of
# the GetNewDataB function.
run(
    [
        "Rscript",
        "scripts/generate_test_data.r",
        conf.MULTIPLIER["RECOUNT2_MODEL_FILE"],
        output_dir,
    ]
)
