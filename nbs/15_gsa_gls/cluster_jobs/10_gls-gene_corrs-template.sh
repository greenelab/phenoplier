#!/bin/bash
# BSUB -J gene_corrs${chr_param}
# BSUB -cwd _tmp/gene_corrs
# BSUB -oo gene_corrs${chr_param}.%J.out
# BSUB -eo gene_corrs${chr_param}.%J.error
# -#BSUB -u miltondp@gmail.com
# -#BSUB -N
# BSUB -n 1
# BSUB -R "rusage[mem=6GB]"
# BSUB -M 10GB
# BSUB -W 20:00

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


compute_correlations () {
  cohort_name="$1"
  ref_panel="$2"
  eqtl_models="$3"
  chromosome=$4

  cd ${PHENOPLIER_CODE_DIR}
  
  notebook_output_folder="gene_corrs/cohorts/${cohort_name,,}/${ref_panel,,}/${eqtl_models,,}"
  full_notebook_output_folder="nbs/15_gsa_gls/${notebook_output_folder}"
  mkdir -p $full_notebook_output_folder
  
  bash nbs/run_nbs.sh \
    nbs/15_gsa_gls/10-gene_expr_correlations.ipynb \
    ${notebook_output_folder}/10-gene_expr_correlations-chr${chromosome}.run.ipynb \
    -p COHORT_NAME $cohort_name \
    -p REFERENCE_PANEL $ref_panel \
    -p EQTL_MODEL $eqtl_models \
    -p CHROMOSOME $chromosome
}
export -f compute_correlations


compute_correlations ${cohort_name_param} ${ref_panel_param} ${eqtl_model_param} ${chr_param}

