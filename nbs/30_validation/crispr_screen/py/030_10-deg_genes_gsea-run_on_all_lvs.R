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

# %%
all_genes_ranked <- read_csv("/home/miltondp/projects/labs/greenelab/phenoplier/base/data/crispr_screen/lipid_DEG.csv")

# %%
deg_gene_sets = list()

for (r in unique(all_genes_ranked$rank)) {
    if (r == 0) {
        next
    }
    
    data <- all_genes_ranked[all_genes_ranked$rank == r,]
    #q <- quantile(data, 0.50, names=FALSE)
    
    deg_gene_sets[[paste0("gene_set_", r)]] <- data$gene_name
}

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
results = list()

# %%
set.seed(42)

# %%
for (lv in names(lvs)) {
    res <- fgsea(pathways = deg_gene_sets, stats = lvs[[lv]], scoreType = "pos", eps = 0.0)[order(pval), ]
    res[, "leadingEdge"] <- sapply(res$leadingEdge, paste, collapse=",")
    res[, "lv"] <- lv
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
df_signif <- df[df$padj < 0.05]

# %%
dim(df_signif)

# %%
df_signif[order(padj),]

# %%
