# This script installs R packages. When installing BiocManager, the script updates all R packages
# currently installed (options update=TRUE, ask=FALSE in BiocManager::install).


default_repo = 'http://cran.us.r-project.org'

if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager", repos=default_repo)
BiocManager::install(version = "3.10", update=TRUE, ask=FALSE)

# clusterProfiler
BiocManager::install("clusterProfiler", update=FALSE, ask=FALSE)

# org.Hs.eg.db
BiocManager::install("org.Hs.eg.db", update=FALSE, ask=FALSE)

# clustree
BiocManager::install("clustree", update=FALSE, ask=FALSE)

# PLIER
library(devtools)
install_github("wgmao/PLIER", ref="v0.1.4")

