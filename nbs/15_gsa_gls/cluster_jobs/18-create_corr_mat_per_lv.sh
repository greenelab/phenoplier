#!/bin/bash
# BSUB -J corr_mat_per_lv${lv_code_param}
# BSUB -cwd _tmp/corr_mat_per_lv
# BSUB -oo corr_mat_per_lv.%J.out
# BSUB -eo corr_mat_per_lv.%J.error
# -#BSUB -u miltondp@gmail.com
# -#BSUB -N
# BSUB -n 1
# BSUB -R "rusage[mem=5GB]"
# BSUB -M 10GB
# BSUB -W 00:05
# BSUB -R 'select[hname!=lambda25]'

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

create_corr_mat_per_lv () {
  cohort_name="$1"
  ref_panel="$2"
  eqtl_models="$3"
  smultixcan_file="$4"
  lv_code="$5"

  cd ${PHENOPLIER_CODE_DIR}
  
  notebook_output_folder="gene_corrs/cohorts/${cohort_name,,}/${ref_panel,,}/${eqtl_models,,}/18-corr_mat_per_lv"
  full_notebook_output_folder="nbs/15_gsa_gls/${notebook_output_folder}"
  mkdir -p $full_notebook_output_folder
  
  bash nbs/run_nbs.sh \
    nbs/15_gsa_gls/18-create_corr_mat_per_lv.ipynb \
    ${notebook_output_folder}/18-create_corr_mat_per_lv-${lv_code}.run.ipynb \
    -p COHORT_NAME $cohort_name \
    -p REFERENCE_PANEL $ref_panel \
    -p EQTL_MODEL $eqtl_models \
    -p SMULTIXCAN_FILE $smultixcan_file \
    -p LV_CODE $lv_code
}
export -f create_corr_mat_per_lv

create_corr_mat_per_lv \
    ${cohort_name_param} \
    ${ref_panel_param} \
    ${eqtl_model_param} \
    ${smultixcan_file_param} \
    ${lv_code_param}

