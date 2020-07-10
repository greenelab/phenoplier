"""
This script creates data for some tests.
"""

from pathlib import Path
from subprocess import run

import conf

output_dir = Path('test_cases')
output_dir.mkdir(exist_ok=True)

run([
    'python',
    'scripts/generate_phenomexcan_data_sample.py',
    output_dir
])

print('\n')

run([
    'Rscript',
    'scripts/generate_test_data.r',
    conf.MULTIPLIER["RECOUNT2_MODEL_FILE"],
    output_dir
])
