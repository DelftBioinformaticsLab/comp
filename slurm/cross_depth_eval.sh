#!/bin/sh
#SBATCH --time=03:59:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=48GB
#SBATCH --mail-type=END
#SBATCH --job-name=cross_depth
#SBATCH --output=logs/%j_cross_depth.out
#SBATCH --error=logs/%j_cross_depth.out
. slurm/_site.sh   # site settings: slurm/site.env (scheduler, storage paths, container, Apptainer cache)

# ============================================================================
# CROSS-DEPTH EVALUATION (reviewer arm: must FUTURE samples be sequenced deep?)
# ============================================================================
# Refits each outer fold's stored winning configuration from a completed fully
# nested run on its full-depth training partition (deterministic - verified
# against the stored predictions), then scores the SAME test folds on the
# depth-downsampled feature matrices. One cheap single-task job: ten refits of
# one configuration each; the runtime is dominated by feature-CSV loading.
#
# Usage:
#   sbatch slurm/cross_depth_eval.sh                       # brca_crc (defaults)
#   CANCER_TYPES=PANCAN EXPERIMENT_NAME=pancan_nested_full \
#     sbatch slurm/cross_depth_eval.sh                     # pancan
#
# Env overrides:
#   EXPERIMENT_NAME  source experiment (default brca_crc_nested_full); the
#                    source directory is nested_cv_results_cris/nested_full_
#                    <EXPERIMENT_NAME> and the output lands in
#                    nested_cv_results_cris/cross_depth_<EXPERIMENT_NAME>/
#   SOURCE_DIR       override the source result directory explicitly
#   EVAL_DIRS        name=dir pairs of downsampled feature dirs (default: the
#                    cov1x + cov0.1x seed-42 runs)
#   CANCER_TYPES     task selection, as in nested_full.sh (default brca+crc)
# ============================================================================

DATASET=${DATASET:-cris}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-brca_crc_nested_full}
CANCER_TYPES=${CANCER_TYPES:-"breast colorectal"}
OUTPUT_DIR=${OUTPUT_DIR:-nested_cv_results_${DATASET}}
SOURCE_DIR=${SOURCE_DIR:-${OUTPUT_DIR}/nested_full_${EXPERIMENT_NAME}}
FEATURES_DIR=${FEATURES_DIR:-extracted_features/biomart_10kb}
EVAL_DIRS=${EVAL_DIRS:-"cov1x=extracted_features/downsampled/cov1x_seed42/biomart_10kb cov0.1x=extracted_features/downsampled/cov0.1x_seed42/biomart_10kb"}

echo "============================================"
echo "Cross-depth evaluation"
echo "Source:   ${SOURCE_DIR}"
echo "Eval:     ${EVAL_DIRS}"
echo "Task:     ${CANCER_TYPES}"
echo "============================================"

set -- \
	-c 16 -t ${CANCER_TYPES} \
	--run_cross_depth_eval \
	--cross_depth_source_dir ${SOURCE_DIR} \
	--cross_depth_eval_dirs ${EVAL_DIRS} \
	--features_dir ${FEATURES_DIR} \
	--metadata_file ${METADATA_FILE:-accessory_files/cris_manifest.csv} \
	-e "${EXPERIMENT_NAME}" \
	-o ${OUTPUT_DIR} \
	-s mean_diff
# EXCLUDE_SAMPLES=<file>: the evaluated cohort must equal the source run's
# (model_hpc raises on stored IDs missing from the metadata), so a _nodeep
# source run needs the same exclusion here.
if [ -n "${EXCLUDE_SAMPLES}" ]; then
	set -- "$@" --exclude_samples "${EXCLUDE_SAMPLES}"
	echo "Excluded samples: ${EXCLUDE_SAMPLES}"
fi

apptainer exec --writable-tmpfs --pwd /opt/app --containall \
	--bind src/:/opt/app/src/ \
	--bind features.py:/opt/app/features.py \
	--bind input/:/opt/app/input/ \
	--bind beds/:/opt/app/beds/ \
	--bind accessory_files/:/opt/app/accessory_files/ \
	--bind nested_cv_results/:/opt/app/nested_cv_results/ \
	--bind nested_cv_results_cris/:/opt/app/nested_cv_results_cris/ \
	--bind results/:/opt/app/results/ \
	--bind ${PREPROCESSING_DIR}/cris/:/opt/app/cris/ \
	--bind ${GCFIX_DIR}/:/opt/app/GC \
	--bind ${GCFIX_DIR}/hg38/:/opt/app/hg38 \
	--bind extracted_features/:/opt/app/extracted_features/ \
		${CONTAINER} pixi run python -u src/comp/model_hpc.py "$@"

echo "Finished: $(date)  Exit code: $?"
