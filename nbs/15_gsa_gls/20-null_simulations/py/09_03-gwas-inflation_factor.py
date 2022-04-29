# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all,-execution,-papermill,-trusted
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
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

import conf

# %% [markdown] tags=[]
# # Paths

# %% tags=[]
GWAS_DIR = conf.A1000G["GENOTYPES_DIR"] / "gwas"
display(GWAS_DIR)

# %% tags=[]
PLINK2 = conf.PLINK["EXECUTABLE_VERSION_2"]
display(PLINK2)

# %% [markdown] tags=[]
# # GWAS results files

# %% tags=[]
gwas_files = sorted(list(GWAS_DIR.glob("*.glm.linear")))
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
for gwas_file in gwas_files:
    result = subprocess.run(
        [PLINK2, "--adjust-file", str(gwas_file), "test=ADD"], stdout=subprocess.PIPE
    )

    assert result.returncode == 0

    result_output = result.stdout.decode("utf-8")
    inf_factor = float(PAT.search(result_output).group("inf_factor"))
    display(f"{gwas_file.name}: {inf_factor}")
    assert 1.015 >= inf_factor >= 1.0

# %% tags=[]
