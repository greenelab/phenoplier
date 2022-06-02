"""
General settings. This file is intended to be modified by the user. Each entry
also provides an alternative way to specify its value using an environment variable.
"""

# Instead of changing this file, you can also use the environment variable name
# specified for each entry (environment variables supersede these settings).

#
# Default paths
#

# Specifies the main directory where all data and results generated are stored.
# When setting up the environment for the first time, input data will be
# automatically downloaded into a subfolder of ROOT_DIR. If not specified, it
# defaults to the 'phenoplier' subfolder in the temporary directory of the
# operating system (i.e. '/tmp/phenoplier' in Unix systems).
#
# Environment variable: PHENOPLIER_ROOT_DIR
ROOT_DIR = None

# Specifies the directory where the manuscript git repository was
# cloned/downloaded to. If None, manuscript figures and other related files will
# not be generated.
#
# Environment variable: PHENOPLIER_MANUSCRIPT_DIR
MANUSCRIPT_DIR = None

# GTEx v8 data directory: it contains protected data files
#
# Environment variable: PHENOPLIER_GTEX_V8_DIR
GTEX_V8_DIR = None


#
# CPU usage
#

# Amount of cores to use for general usage.
# Default: half of available cores.
N_JOBS = None

# Number of cores to use for low-computational tasks (IO, etc). This number
# can be greater than N_JOBS.
# Default: same as N_JOBS.
N_JOBS_HIGH = None
