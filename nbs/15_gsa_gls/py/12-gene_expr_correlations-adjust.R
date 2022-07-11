# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all,-execution,-papermill,-trusted
#     text_representation:
#       extension: .R
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: R
#     language: R
#     name: ir
# ---

# %% [markdown]
# # Description

# %% [markdown]
# **TODO**

# %% [markdown]
# # Modules

# %%
library(reticulate)
pd <- import("pandas")

# %% [markdown]
# # Settings

# %% tags=["parameters"]
# reference panel
REFERENCE_PANEL <- "GTEX_V8"
# REFERENCE_PANEL = "1000G"

# prediction models
## mashr
EQTL_MODEL <- "MASHR"

chromosome <- NULL

# %%
paste0("Using reference panel: ", REFERENCE_PANEL)

# %%
paste0("Using eQTL model: ", EQTL_MODEL)

# %%
# chromosome must be provided as parameter
stopifnot(!is.null(chromosome))

# %% [markdown]
# # Paths

# %%
GENE_CORRS_DIR <- Sys.getenv("PHENOPLIER_PHENOMEXCAN_LD_BLOCKS_GENE_CORRS_DIR")
IRdisplay::display(GENE_CORRS_DIR)

# %%
INPUT_DIR <- file.path(GENE_CORRS_DIR, tolower(REFERENCE_PANEL), tolower(EQTL_MODEL), "by_chr")
IRdisplay::display(INPUT_DIR)

# %%
INPUT_FILE <- file.path(INPUT_DIR, paste0("gene_corrs-chr", chromosome, ".pkl"))
IRdisplay::display(INPUT_FILE)
stopifnot(file.exists(INPUT_FILE))

# %%
OUTPUT_DIR <- file.path(INPUT_DIR, "corrected_positive_definite")
IRdisplay::display(OUTPUT_DIR)
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

# %%
OUTPUT_FILE <- file.path(OUTPUT_DIR, paste0("gene_corrs-chr", chromosome, ".pkl"))
IRdisplay::display(OUTPUT_FILE)
if (file.exists(OUTPUT_FILE)) {
  IRdisplay::display("Output file exists, it will be overwritten")
}

# %% [markdown]
# # Functions

# %%
# taken from https://www.r-bloggers.com/2013/08/correcting-a-pseudo-correlation-matrix-to-be-positive-semidefinite/
# TODO: add documentation
CorrectCM <- function(CM, p = 0) {
  n <- dim(var(CM))[1L]
  E <- eigen(CM)
  CM1 <- E$vectors %*% tcrossprod(diag(pmax(E$values, p), n), E$vectors)
  Balance <- diag(1 / sqrt(diag(CM1)))
  CM2 <- Balance %*% CM1 %*% Balance
  return(CM2)
}

# %%
is_positive_definite <- function(data) {
  # eigenvalues <- eigen(data)$values
  # nonpositive_eigenvalues <- eigenvalues[eigenvalues <= 0]

  cholStatus <- try(u <- chol(data), silent = FALSE)
  cholError <- ifelse(class(cholStatus) == "try-error", TRUE, FALSE)

  if (cholError) {
    return(FALSE)
  } else {
    return(TRUE)
  }
}

# %% [markdown]
# # Load data

# %%
gene_corrs <- pd$read_pickle(INPUT_FILE)

# %%
dim(gene_corrs)

# %%
head(gene_corrs[1:10, 1:10])

# %% [markdown]
# # Check positive definiteness

# %%
orig_matrix_is_positive_definite <- is_positive_definite(gene_corrs)

# %%
if (orig_matrix_is_positive_definite) {
  IRdisplay::display("Matrix is already positive definite!")
  file.copy(INPUT_FILE, OUTPUT_FILE)
} else {
  IRdisplay::display("We need to correct the data and make the matrix positive definite")

  eigenvalues <- eigen(gene_corrs)$values
  nonpositive_eigenvalues <- eigenvalues[eigenvalues <= 0]
  IRdisplay::display(length(nonpositive_eigenvalues))
  IRdisplay::display(nonpositive_eigenvalues)
    
  # Make matrix positive definite
  gene_corrs_corrected <- CorrectCM(gene_corrs, 1e-10)
  dimnames(gene_corrs_corrected)[[1]] <- rownames(gene_corrs)
  dimnames(gene_corrs_corrected)[[2]] <- colnames(gene_corrs)
  gene_corrs_corrected <- as.data.frame(gene_corrs_corrected)
    
  # check if new matrix is really positive definite
    matrix_is_positive_definite <- is_positive_definite(gene_corrs_corrected)
    if (matrix_is_positive_definite) {
        
      IRdisplay::display("It worked!")
      IRdisplay::display(head(gene_corrs[1:10, 1:10]))
      IRdisplay::display(head(gene_corrs_corrected[1:10, 1:10]))
      py_save_object(gene_corrs_corrected, OUTPUT_FILE)
        
    } else {
        
      eigenvalues <- eigen(gene_corrs_corrected)$values
      nonpositive_eigenvalues <- eigenvalues[eigenvalues <= 0]
      IRdisplay::display(length(nonpositive_eigenvalues))
      IRdisplay::display(nonpositive_eigenvalues)

      stop("Method failed to adjust matrix")
        
    }
}

# %% [markdown]
# Both matrices should "look" similar. We are not interested in perfectly accurate correlation values (they are already inaccurate).

# %%
