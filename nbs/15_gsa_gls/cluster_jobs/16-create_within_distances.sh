#!/bin/bash
# BSUB -J create_within_dist
# BSUB -cwd _tmp/create_within_dist
# BSUB -oo post_gene_corrs.%J.out
# BSUB -eo post_gene_corrs.%J.error
# -#BSUB -u miltondp@gmail.com
# -#BSUB -N
# BSUB -n 1
# BSUB -R "rusage[mem=6GB]"
# BSUB -M 10GB
# BSUB -W 00:30
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


create_within_distance_matrices () {
  cohort_name="$1"
  ref_panel="$2"
  eqtl_models="$3"

  cd ${PHENOPLIER_CODE_DIR}
  
  notebook_output_folder="gene_corrs/cohorts/${cohort_name,,}/${ref_panel,,}/${eqtl_models,,}"
  full_notebook_output_folder="nbs/15_gsa_gls/${notebook_output_folder}"
  mkdir -p $full_notebook_output_folder
  
  bash nbs/run_nbs.sh \
    nbs/15_gsa_gls/16-create_within_distance_matrices.ipynb \
    ${notebook_output_folder}/16-create_within_distance_matrices.run.ipynb \
    -p COHORT_NAME $cohort_name \
    -p REFERENCE_PANEL $ref_panel \
    -p EQTL_MODEL $eqtl_models
}
export -f create_within_distance_matrices


create_within_distance_matrices ${cohort_name_param} ${ref_panel_param} ${eqtl_model_param}
