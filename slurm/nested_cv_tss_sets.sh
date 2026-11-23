#!/bin/sh
#SBATCH --time=05:59:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=24GB
#SBATCH --mail-type=END
#SBATCH --job-name=nested_cv_tss_sets
#SBATCH --output=logs/%j_nested_cv_tss_sets.out
#SBATCH --error=logs/%j_nested_cv_tss_sets.out

# ============================================================================
# Nested CV - TSS features aggregated over Hallmark pathways or chromosome arms
# (rebuttal R2.3 / R3.4). One predefined matrix (samples x sets, one fragmentomic
# feature, averaged or pooled; src/comp/build_tss_set_matrices.py) through the
# same nested CV as the ATAC comparator, including its per-sample row
# standardization across the set features (the ATAC matrix is likewise one
# value per sample and aggregated region set). Submit all arms with
# ./slurm/submit_tss_set_panels.sh; a single arm:
#   EXPERIMENT_NAME=brca_crc_tss_sets_arm_pooled_griffin_diff_nodeep \
#     PREDEFINED_STEM=extracted_features/tss_sets/arm_pooled_griffin_diff \
#     CANCER_TYPES="breast colorectal" EXCLUDE_SAMPLES=accessory_files/cris_exclude_deep_healthy.txt \
#     sbatch slurm/nested_cv_tss_sets.sh
# ============================================================================

MODE="predefined_tss_sets"
JOB_NAME="nested_cv_tss_sets"
DATASET=${1:-cris}
STRATEGY=${2:-mean_diff}
EXPERIMENT_NAME=${EXPERIMENT_NAME:?EXPERIMENT_NAME required}
PREDEFINED_STEM=${PREDEFINED_STEM:?PREDEFINED_STEM required}
MODE_ARGS="--run_nested_cv --predefined_feature_matrix ${PREDEFINED_STEM}.npy --predefined_metadata_csv ${PREDEFINED_STEM}.metadata.csv --store_fitted_models"
DEFAULT_METADATA="accessory_files/cris_manifest.csv"

. slurm/_model_hpc_exec.sh
print_job_header

set -- \
	-c ${SLURM_CPUS_PER_TASK:-16} -t ${CANCER_TYPES:-PANCAN} \
	${MODE_ARGS} \
	--metadata_file ${METADATA_FILE:-${DEFAULT_METADATA}} \
	-e "${EXPERIMENT_NAME}" \
	-o ${OUTPUT_DIR:-nested_cv_results_${DATASET}} \
	-s ${STRATEGY:-mean_diff}

run_model_hpc "$@"
