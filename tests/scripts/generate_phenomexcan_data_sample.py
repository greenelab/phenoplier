"""
This script takes a small sample from the PhenomeXcan results and saves it into an RDS
(R) format. This small file will be used later to project the data using MultiPLIER
code in unit tests.
"""
from pathlib import Path

import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

import conf
from data.cache import read_data


# Read gene mappings
gene_id_to_name = read_data(conf.PHENOMEXCAN["GENE_MAP_ID_TO_NAME"])

# Read S-MultiXcan results
smultixcan_results_filename = conf.PHENOMEXCAN["SMULTIXCAN_MASHR_ZSCORES_FILE"]
smultixcan_results = pd.read_pickle(smultixcan_results_filename)

smultixcan_results = smultixcan_results.rename(index=gene_id_to_name)
smultixcan_results = smultixcan_results.T.sample(n=3, random_state=0).T
smultixcan_results = smultixcan_results.loc[
    ~smultixcan_results.index.duplicated(keep="first")
]
smultixcan_results = smultixcan_results.dropna(how="all")
smultixcan_results = smultixcan_results.fillna(0)

print("This is the head of the sample:")
print(smultixcan_results.head())

output_dir = Path("test_cases", "test_case5")
output_dir.mkdir(exist_ok=True)

saveRDS = ro.r["saveRDS"]
with localconverter(ro.default_converter + pandas2ri.converter):
    r_from_pd_df = ro.conversion.py2rpy(smultixcan_results)
    saveRDS(r_from_pd_df, str(Path(output_dir, "phenomexcan_sample.rds")))
