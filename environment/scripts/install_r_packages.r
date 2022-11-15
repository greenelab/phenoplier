# This script installs R packages. When installing BiocManager, the script updates all R packages
# currently installed (options update=TRUE, ask=FALSE in BiocManager::install).

default_repo = 'http://cran.us.r-project.org'

if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager", repos=default_repo)
BiocManager::install(version = "3.16", update=TRUE, ask=FALSE)

# styler
BiocManager::install("styler", update = FALSE, ask = FALSE)

# clusterProfiler
BiocManager::install("clusterProfiler", update=FALSE, ask=FALSE)

# org.Hs.eg.db
BiocManager::install("org.Hs.eg.db", update=FALSE, ask=FALSE)

# clustree
BiocManager::install("clustree", update=FALSE, ask=FALSE)

# qqman
BiocManager::install("qqman", update=FALSE, ask=FALSE)

# fgsea
devtools::install_github("ctlab/fgsea", ref="v1.19.2")

# PLIER
#install_github("wgmao/PLIER", ref="v0.1.4")

