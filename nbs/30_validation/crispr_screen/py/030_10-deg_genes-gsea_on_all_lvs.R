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

# %%
library(readr)
library(fgsea)
library(readr)
library(dplyr)

# %%
all_genes_ranked <- read_csv("/home/miltondp/projects/labs/greenelab/phenoplier/base/data/crispr_screen/lipid_DEG.csv")

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
deg_gene_sets <- list()

# %%
# genes that increase lipids
deg_gene_sets[["gene_set_increase_2_and_3"]] <- c(
    orig_deg_gene_sets[["gene_set_2"]],
    orig_deg_gene_sets[["gene_set_3"]]
)

# %%
# genes that decrease lipids
deg_gene_sets[["gene_set_decrease_-2_and_-3"]] <- c(
    orig_deg_gene_sets[["gene_set_-2"]],
    orig_deg_gene_sets[["gene_set_-3"]]
)

# %%
# MultiPLIER LVs
multiplier_z = readRDS("/media/miltondp/Elements1/projects/multiplier/recount2_PLIER_data/recount_PLIER_model.RDS")$Z

lvs = list()
for (cidx in 1:ncol(multiplier_z)) {
    data <- multiplier_z[, cidx]
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
#     res <- fgsea(pathways = deg_gene_sets, stats = lvs[[lv]], scoreType = "pos", eps = 0.0)[order(pval), ]
    res <- do.call(rbind, repetitions)
#     res[, "leadingEdge"] <- sapply(res$leadingEdge, paste, collapse=",")
#     res[, "lv"] <- lv
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

# %%
write_tsv(df, "/home/miltondp/projects/labs/greenelab/phenoplier/base/data/crispr_screen/fsgea-all_lvs.tsv")

# %% [markdown]
# # Quick analyses

# %%
df %>% filter(lv == "LV100" & pathway == "gene_set_increase_2_and_3") %>% arrange(desc(padj))

# %%
df_signif <- df %>% group_by(lv, pathway) %>% summarize(max_padj = max(padj)) %>% filter(max_padj < 0.05)

# %%
df_signif %>% arrange(max_padj)

# %%
