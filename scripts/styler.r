#!/usr/bin/env Rscript

# It takes a filepath as argument (an R script) and apply a code style using
# the R package styler (it's similar to black in Python).

args <- commandArgs(trailingOnly = TRUE)
file_name <- args[1L]
styler::style_file(file_name)

