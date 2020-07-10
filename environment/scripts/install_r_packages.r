# This script installs R packages. When installing BiocManager, the script updates all R packages
# currently installed (options update=TRUE, ask=FALSE in BiocManager::install).

library(devtools)

default_repo = 'http://cran.us.r-project.org'

if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager", repos=default_repo)
BiocManager::install(version = "3.10", update=TRUE, ask=FALSE)

# clusterProfiler
#BiocManager::install("clusterProfiler", update=TRUE, ask=FALSE)

# org.Hs.eg.db
#BiocManager::install("org.Hs.eg.db", update=TRUE, ask=FALSE)

# PLIER
install_github("wgmao/PLIER", ref="v0.1.4")

