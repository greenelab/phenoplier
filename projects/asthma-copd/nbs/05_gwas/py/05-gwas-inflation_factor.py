# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all,-execution,-papermill,-trusted
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] tags=[]
# # Description

# %% [markdown] tags=[]
# It runs PLINK2 on GWAS results to check that the genomic inflation factor is withint acceptable limits.

# %% [markdown] tags=[]
# # Modules

# %% tags=[]
import re
import subprocess
from pathlib import Path
import tempfile
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

import conf
from utils import chunker

# %% [markdown] tags=[]
# # Paths

# %% tags=[]
INPUT_GWAS_DIR = conf.PROJECTS["ASTHMA_COPD"]["DATA_DIR"] / "gwas"
display(INPUT_GWAS_DIR)
assert INPUT_GWAS_DIR.exists()

# %% tags=[]
PLINK2 = conf.PLINK["EXECUTABLE_VERSION_2"]
display(PLINK2)
assert PLINK2.exists()

# %% [markdown] tags=[]
# # GWAS results files

# %% tags=[]
gwas_files = sorted(list(INPUT_GWAS_DIR.glob("*_info0.7.txt")))
display(len(gwas_files))
display(gwas_files[:10])

# %% [markdown] tags=[]
# # Check inflation factor

# %% tags=[]
PAT = re.compile(
    r"Genomic inflation est\. lambda \(based on median chisq\) = (?P<inf_factor>[0-9\.]+)\."
)

# %% tags=[]
# testing
input_text = """
PLINK v2.00a3LM 64-bit Intel (26 Apr 2022)     www.cog-genomics.org/plink/2.0/
(C) 2005-2022 Shaun Purcell, Christopher Chang   GNU General Public License v3
Logging to plink2.log.
Options in effect:
  --adjust-file /opt/data/data/1000g/genotypes/gwas/random.pheno0.glm.linear test=ADD

Start time: Fri Apr 29 16:12:24 2022
64185 MiB RAM detected; reserving 32092 MiB for main workspace.
Using up to 4 compute threads.
--adjust: Genomic inflation est. lambda (based on median chisq) = 1.00316.
--adjust-file values (5923554 tests) written to plink2.adjusted .
End time: Fri Apr 29 16:12:33 2022
"""

m = PAT.search(input_text)
assert m.group("inf_factor") == "1.00316"

# %% tags=[]
# testing
input_text = """
PLINK v2.00a3LM 64-bit Intel (26 Apr 2022)     www.cog-genomics.org/plink/2.0/
(C) 2005-2022 Shaun Purcell, Christopher Chang   GNU General Public License v3
Logging to plink2.log.
Options in effect:
  --adjust-file base/data/1000g/genotypes/gwas/random.pheno1.glm.linear test=ADD

Start time: Fri Apr 29 12:19:51 2022
64185 MiB RAM detected; reserving 32092 MiB for main workspace.
Using up to 4 compute threads.
--adjust: Genomic inflation est. lambda (based on median chisq) = 1.
--adjust-file values (5923554 tests) written to plink2.adjusted .
End time: Fri Apr 29 12:19:59 2022
"""

m = PAT.search(input_text)
display(m.group("inf_factor"))
assert m.group("inf_factor") == "1"


# %% tags=[]
def _compute_inflation_factor(gwas_files_group):
    res = {}
    for gwas_file in gwas_files_group:
        output_dir = Path(tempfile.mkdtemp(prefix="plink-adjust-"))
        output_file = output_dir / "outfile"
        result = subprocess.run(
            [
                PLINK2,
                "--adjust-file",
                str(gwas_file),
                "test=ADD",
                "--threads",
                str(conf.GENERAL["N_JOBS"]),
                "--out",
                str(output_file),
            ],
            stdout=subprocess.PIPE,
        )

        assert result.returncode == 0

        result_output = result.stdout.decode("utf-8")
        inf_factor = float(PAT.search(result_output).group("inf_factor"))
        res[gwas_file.name] = inf_factor

        # delete temporary folder
        shutil.rmtree(output_dir)

    return res


# %% tags=[]
gwas_files_chunks = list(
    chunker(
        gwas_files,
        int(min(10, max(len(gwas_files) / conf.GENERAL["N_JOBS"], 1))),
    )
)

# %% tags=[]
len(gwas_files_chunks)

# %% tags=[]
all_results = {}

with ProcessPoolExecutor(max_workers=conf.GENERAL["N_JOBS"]) as executor:
    tasks = [
        executor.submit(_compute_inflation_factor, chunk) for chunk in gwas_files_chunks
    ]
    for future in as_completed(tasks):
        res = future.result()
        all_results.update(res)

# %% tags=[]
assert len(all_results) == len(gwas_files)

# %% [markdown] tags=[]
# # Create dataframe

# %% tags=[]
all_results_df = pd.Series(all_results, name="inflation_factor").rename_axis(
    "phenotype_code"
)

# %% tags=[]
all_results_df.shape

# %% tags=[]
all_results_df.head()

# %% [markdown] tags=[]
# # Checks

# %% tags=[]
all_results_df.describe()

# %% tags=[]
assert all_results_df.min() >= 1.10
assert all_results_df.max() <= 1.14

# %% tags=[]
all_results_df.sort_values(ascending=False).head(20)

# %% tags=[]
