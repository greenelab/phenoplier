#!/usr/bin/env Rscript

# File adapted from: https://github.com/hakyimlab/gtex-miscellaneous-processing/blob/master/src/_samples_by_ethnicity.R

library(readr)
library(dplyr)
library(tidyr)

#Modify these paths to whatever you want.
GTEX_DIR <- Sys.getenv("PHENOPLIER_EXTERNAL_GTEX_V8_DIR")
ind_ <- file.path(GTEX_DIR, "generated", "GTEx_ethnicity.txt")
flagged_ <- file.path(GTEX_DIR, "analysis_supplement", "GTEx_Analysis_2017-06-05_v8_WholeGenomeSeq_flagged_donors.txt")
eur_ <-  file.path(GTEX_DIR, "generated", "eur_samples.txt")
afr_ <-  file.path(GTEX_DIR, "generated", "afr_samples.txt")

ethnicity <- read_tsv(ind_) %>% rename(id = SAMPLE_ID)
flagged <- read_tsv(flagged_) %>% rename(id = Donor_ID)

ethnicity <- ethnicity %>%
    separate(id, sep="-", into=c("id1", "id2", "id3", "id4", "id5")) %>%
    mutate(short_id=paste(id1, id2, sep="-", collapse=NULL), id=paste(id1, id2, id3, id4, id5, sep="-", collapse=NULL)) %>%
    select(-id1, -id2, -id3, -id3, -id4, -id5)

#must match the ids in the vcf Header! thatś why we use the short id
eur <- ethnicity %>% filter(!(short_id %in% flagged$id), InferredAncestry == "EUR" )
write.table(eur %>% select(short_id), eur_, row.names=FALSE, col.names=FALSE, quote=FALSE)

afr <- ethnicity %>% filter(!(short_id %in% flagged$id), InferredAncestry == "AFR" )
write.table(afr %>% select(short_id), afr_, row.names=FALSE, col.names=FALSE, quote=FALSE)

