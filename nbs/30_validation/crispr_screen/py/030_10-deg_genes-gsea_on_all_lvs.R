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

# %% [markdown]
# # Module loading

# %%
library(IRdisplay)
library(readr)
library(fgsea)
library(dplyr)
library(reticulate)

# %%
pd <- import("pandas")

# %% [markdown]
# # Settings

# %%
OUTPUT_DIR <- Sys.getenv("PHENOPLIER_RESULTS_CRISPR_ANALYSES_BASE_DIR")

# %%
OUTPUT_DIR

# %%
dir.create(OUTPUT_DIR, recursive=TRUE)

# %% [markdown]
# # Data loading

# %% [markdown]
# ## Lipids gene sets

# %%
input_file <- Sys.getenv("PHENOPLIER_CRISPR_LIPIDS_GENE_SETS_FILE")
display(input_file)

# %%
all_genes_ranked <- read_csv(input_file)

# %%
orig_deg_gene_sets <- list()

for (r in unique(all_genes_ranked$rank)) {
    if (r == 0) {
        next
    }
    
    data <- all_genes_ranked[all_genes_ranked$rank == r,]
    #q <- quantile(data, 0.50, names=FALSE)
    
    orig_deg_gene_sets[[paste0("gene_set_", r)]] <- data$gene_name
}

# %%
length(orig_deg_gene_sets)

# %% [markdown]
# ### Combine gene sets into "increase lipids" and "decrease lipids"

# %%
deg_gene_sets <- list()

# %%
# genes that increase lipids
deg_gene_sets[["gene_set_increase"]] <- c(
    orig_deg_gene_sets[["gene_set_2"]],
    orig_deg_gene_sets[["gene_set_3"]]
)

# %%
# genes that decrease lipids
deg_gene_sets[["gene_set_decrease"]] <- c(
    orig_deg_gene_sets[["gene_set_-2"]],
    orig_deg_gene_sets[["gene_set_-3"]]
)

# %%
length(deg_gene_sets)

# %%
length(deg_gene_sets[["gene_set_increase"]])

# %%
stopifnot(length(deg_gene_sets[["gene_set_increase"]]) == 175)

# %%
length(deg_gene_sets[["gene_set_decrease"]])

# %%
stopifnot(length(deg_gene_sets[["gene_set_decrease"]]) == 96)

# %%
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

# %%
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

# %% [markdown]
# ## MultiPLIER Z

# %%
multiplier_z = pd$read_pickle(
    Sys.getenv("PHENOPLIER_MULTIPLIER_MODEL_Z_MATRIX_FILE")
)

# %%
dim(multiplier_z)

# %%
head(multiplier_z)

# %% [markdown]
# # Prepare LVs list

# %%
lvs = list()
z_gene_names <- rownames(multiplier_z)

for (cidx in 1:ncol(multiplier_z)) {
    data <- multiplier_z[, cidx]
    names(data) <- z_gene_names
    # q <- quantile(data, 0.75, names=FALSE)
    q <- 0.0
    
    lvs[[paste0("LV", cidx)]] <- data[data > q]
}

# %% [markdown]
# # Compute enrichment on all LVs

# %%
n_reps = 10

# %%
set.seed(0)

# %%
results = list()

for (lv in names(lvs)) {
    repetitions = list()
    
    for (i in 1:n_reps) {
        rep_res <- fgsea(pathways = deg_gene_sets, stats = lvs[[lv]], scoreType = "pos", eps = 0.0)[order(pval), ]
        rep_res[, "leadingEdge"] <- sapply(rep_res$leadingEdge, paste, collapse=",")
        rep_res[, "lv"] <- lv
        rep_res[, "rep_idx"] <- i
        
        repetitions[[i]] <- rep_res
    }
    
    res <- do.call(rbind, repetitions)

    results[[lv]] <- res
}

# %%
length(results)

# %%
df <- do.call(rbind, results)

# %%
dim(df)

# %%
head(df)

# %% [markdown]
# ## Save

# %%
output_file <- file.path(OUTPUT_DIR, "fgsea-all_lvs.tsv")
display(output_file)

# %%
write_tsv(df, output_file)

# %% [markdown]
# # Quick analyses/tests

# %% [markdown]
# ## See how one LV looks like

# %%
df %>% filter(lv == "LV100" & pathway == "gene_set_increase") %>% arrange(desc(padj))

# %% [markdown]
# ## Show significant LVs

# %%
df_signif <- df %>% group_by(lv, pathway) %>% summarize(max_padj = max(padj)) %>% filter(max_padj < 0.05)

# %%
nrow(df_signif)

# %%
stopifnot(nrow(df_signif) > 50)

# %%
df_signif %>% arrange(max_padj)

# %%
