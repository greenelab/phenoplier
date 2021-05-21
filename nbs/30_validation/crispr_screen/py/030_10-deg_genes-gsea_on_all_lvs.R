# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all,-execution,-papermill,-trusted
#     formats: ipynb,py//R:percent
#     text_representation:
#       extension: .R
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: R
#     language: R
#     name: ir
# ---

# %% [markdown] tags=[]
# # Module loading

# %% tags=[]
library(IRdisplay)
library(readr)
library(fgsea)
library(dplyr)
library(tidyverse)
library(reticulate)

# %% tags=[]
pd <- import("pandas")

# %% [markdown] tags=[]
# # Settings

# %% tags=[]
OUTPUT_DIR <- Sys.getenv("PHENOPLIER_RESULTS_CRISPR_ANALYSES_BASE_DIR")

# %% tags=[]
OUTPUT_DIR

# %% tags=[]
dir.create(OUTPUT_DIR, recursive=TRUE)

# %% [markdown] tags=[]
# # Data loading

# %% [markdown] tags=[]
# ## Lipids gene sets

# %% tags=[]
input_file <- Sys.getenv("PHENOPLIER_CRISPR_LIPIDS_GENE_SETS_FILE")
display(input_file)

# %% tags=[]
all_genes_ranked <- read_csv(input_file)

# %% tags=[]
orig_deg_gene_sets <- list()

for (r in unique(all_genes_ranked$rank)) {
    if (r == 0) {
        next
    }
    
    data <- all_genes_ranked[all_genes_ranked$rank == r,]
    #q <- quantile(data, 0.50, names=FALSE)
    
    orig_deg_gene_sets[[paste0("gene_set_", r)]] <- data$gene_name
}

# %% tags=[]
length(orig_deg_gene_sets)

# %% [markdown] tags=[]
# ### Combine gene sets into "increase lipids" and "decrease lipids"

# %% tags=[]
deg_gene_sets <- list()

# %% tags=[]
# genes that increase lipids
deg_gene_sets[["gene_set_increase"]] <- c(
    orig_deg_gene_sets[["gene_set_2"]],
    orig_deg_gene_sets[["gene_set_3"]]
)

# %% tags=[]
# genes that decrease lipids
deg_gene_sets[["gene_set_decrease"]] <- c(
    orig_deg_gene_sets[["gene_set_-2"]],
    orig_deg_gene_sets[["gene_set_-3"]]
)

# %% tags=[]
length(deg_gene_sets)

# %% tags=[]
length(deg_gene_sets[["gene_set_increase"]])

# %% tags=[]
stopifnot(length(deg_gene_sets[["gene_set_increase"]]) == 175)

# %% tags=[]
length(deg_gene_sets[["gene_set_decrease"]])

# %% tags=[]
stopifnot(length(deg_gene_sets[["gene_set_decrease"]]) == 96)

# %% tags=[]
# test new increase set
new_set <- deg_gene_sets[["gene_set_increase"]]
expected_set <- union(
    orig_deg_gene_sets[["gene_set_2"]],
    orig_deg_gene_sets[["gene_set_3"]]
)

stopifnot(length(new_set) == length(unique(new_set)))

stopifnot(
    length(new_set) == 
    length(
        intersect(
            new_set,
            expected_set
        )
    )
)

# %% tags=[]
# test new decrease set
new_set <- deg_gene_sets[["gene_set_decrease"]]
expected_set <- union(
    orig_deg_gene_sets[["gene_set_-2"]],
    orig_deg_gene_sets[["gene_set_-3"]]
)

stopifnot(length(new_set) == length(unique(new_set)))

stopifnot(
    length(new_set) == 
    length(
        intersect(
            new_set,
            expected_set
        )
    )
)

# %% [markdown] tags=[]
# ## MultiPLIER Z

# %% tags=[]
multiplier_z = pd$read_pickle(
    Sys.getenv("PHENOPLIER_MULTIPLIER_MODEL_Z_MATRIX_FILE")
)

# %% tags=[]
dim(multiplier_z)

# %% tags=[]
head(multiplier_z)

# %% [markdown] tags=[]
# # Prepare LVs list

# %% tags=[]
lvs = list()
z_gene_names <- rownames(multiplier_z)

for (cidx in 1:ncol(multiplier_z)) {
    data <- multiplier_z[, cidx]
    names(data) <- z_gene_names
    
    lvs[[paste0("LV", cidx)]] <- data[data > 0.0]
}

# %% tags=[]
display(length(lvs))
stopifnot(length(lvs) == 987)

# %% [markdown] tags=[]
# # Compute enrichment on all LVs

# %% tags=[]
n_reps = 10

# %% tags=[]
set.seed(0)

# %% tags=[]
results = list()

for (lv in names(lvs)) {
    repetitions = list()
    
    for (i in 1:n_reps) {
        rep_res <- fgsea(pathways = deg_gene_sets, stats = lvs[[lv]], scoreType = "pos", eps = 0.0)[order(pval), ]
        rep_res[, "lv"] <- lv
        rep_res[, "rep_idx"] <- i
        
        repetitions[[i]] <- rep_res
    }
    
    res <- do.call(rbind, repetitions)

    results[[lv]] <- res
}

# %% tags=[]
length(results)

# %% tags=[]
df <- do.call(rbind, results)

# %% tags=[]
df <- df %>% mutate(leadingEdge = map_chr(leadingEdge, toString))

# %% tags=[]
dim(df)

# %% tags=[]
head(df)

# %% [markdown] tags=[]
# ## Save

# %% tags=[]
output_file <- file.path(OUTPUT_DIR, "fgsea-all_lvs.tsv")
display(output_file)

# %% tags=[]
write_tsv(df, output_file)

# %% [markdown] tags=[]
# # Quick analyses/tests

# %% [markdown] tags=[]
# ## See how one LV looks like

# %% tags=[]
df %>% filter(lv == "LV100" & pathway == "gene_set_increase") %>% arrange(desc(padj))

# %% [markdown] tags=[]
# ## Show significant LVs

# %% tags=[]
df_signif <- df %>% group_by(lv, pathway) %>% summarize(max_padj = max(padj)) %>% filter(max_padj < 0.05)

# %% tags=[]
nrow(df_signif)

# %% tags=[]
stopifnot(nrow(df_signif) > 50)

# %% tags=[]
df_signif %>% arrange(max_padj)

# %% tags=[]
