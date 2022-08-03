#!/bin/bash
# BSUB -J compile_gwas_twas
# BSUB -cwd _tmp/compile_gwas_twas
# BSUB -oo compile_gwas_twas.%J.out
# BSUB -eo compile_gwas_twas.%J.error
# -#BSUB -u miltondp@gmail.com
# -#BSUB -N
# BSUB -n 1
# BSUB -R "rusage[mem=7GB]"
# BSUB -M 10GB
# BSUB -W 10:00

# IMPORTANT: this is not a ready-for-submission script, it's a template.
#   see README.md to know how to generate the actual job scripts.

# make sure we use the number of CPUs specified
export n_jobs="${PHENOPLIER_N_JOBS}"
if [ -z "${n_jobs}" ]; then
  n_jobs=1
fi
export PHENOPLIER_N_JOBS=${n_jobs}
export NUMBA_NUM_THREADS=${n_jobs}
export MKL_NUM_THREADS=${n_jobs}
export OPEN_BLAS_NUM_THREADS=${n_jobs}
export NUMEXPR_NUM_THREADS=${n_jobs}
export OMP_NUM_THREADS=${n_jobs}


compile_gwas_twas () {
  cohort_name="$1"
  gwas_file="$2"
  spredixcan_folder="$3"
  spredixcan_file_pattern="$4"
  smultixcan_file="$5"
  ref_panel="$6"
  eqtl_models="$7"

  cd ${PHENOPLIER_CODE_DIR}

  notebook_output_folder="gene_corrs/cohorts/${cohort_name,,}/${ref_panel,,}/${eqtl_models,,}"
  full_notebook_output_folder="nbs/15_gsa_gls/${notebook_output_folder}"
  mkdir -p $full_notebook_output_folder

  bash nbs/run_nbs.sh \
    nbs/15_gsa_gls/07-compile_gwas_snps_and_twas_genes.ipynb \
    ${notebook_output_folder}/07-compile_gwas_snps_and_twas_genes.run.ipynb \
    -p COHORT_NAME $cohort_name \
    -p REFERENCE_PANEL $ref_panel \
    -p EQTL_MODEL $eqtl_models \
    -p GWAS_FILE $gwas_file \
    -p SPREDIXCAN_FOLDER $spredixcan_folder \
    -p SPREDIXCAN_FILE_PATTERN $spredixcan_file_pattern \
    -p SMULTIXCAN_FILE $smultixcan_file
}
export -f compile_gwas_twas


compile_gwas_twas \
    ${cohort_name_param} \
    ${gwas_file_param} \
    ${spredixcan_folder_param} \
    ${spredixcan_file_pattern_param} \
    ${smultixcan_file_param} \
    ${ref_panel_param} \
    ${eqtl_model_param}

