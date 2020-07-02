"""
General settings. This file is intended to be modified by the user.
"""

#
# Default paths
#

# Use these variables to change the paths where data will be downloaded to. You can
# also use the environment variable name specified in each line (commented out) to
# override settings (environment variables supersede these settings).

# Environment variable: PHENOPLIER_ROOT_DIR
ROOT_DIR = "/tmp/phenoplier"

# Environment variable: PHENOPLIER_MANUSCRIPT_DIR
MANUSCRIPT_DIR = None


#
# CPU usage
#

# Amount of cores to use for general usage.
# Default: half of available cores.
N_JOBS = None

# Amount of cores to use for for low-computational tasks (IO, etc).
# Default: all available cores.
N_JOBS_HIGH = None
