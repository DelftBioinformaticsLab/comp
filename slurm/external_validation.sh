#!/bin/sh
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=13107MB
#SBATCH --mail-type=END
#SBATCH --job-name=ext_val
#SBATCH --output=logs/%j_ext_val.out
#SBATCH --error=logs/%j_ext_val.out
. slurm/_site.sh   # site settings: slurm/site.env (scheduler, storage paths, container, Apptainer cache)

# =================================================================================
# External Dataset Validation Script
# =================================================================================
#
# This script validates a model trained on one dataset (e.g., CRIS BRCA+CRC)
# on a completely different external dataset (e.g., Jiang Liver Cancer).
#
# Usage:
#   sbatch slurm/external_validation.sh [FEATURE_TYPE] [TEST_DATASET]
#   sbatch slurm/external_validation.sh dd              # DD on jiang (default)
#   sbatch slurm/external_validation.sh rna              # RNA on jiang
#   sbatch slurm/external_validation.sh atac             # ATAC on jiang
#   sbatch slurm/external_validation.sh dd_cov           # DD + covariate panel (Fig S16)
#   sbatch slurm/external_validation.sh cov              # covariate panel alone (Fig S16)
#   dd_cov / cov deploy the searches of slurm/best_model_search.sh dd_cov|cov
#   (best_model_best_model_search_<type>_<COVARIATE_PANEL><TRAIN_SUFFIX>) with the
#   external cohort's panel matrix extracted_features/scalar_panel_<panel>_<TEST_DATASET>
#   (build_scalar_panel_matrix.py --task <TEST_DATASET> --panel <panel>); COVARIATE_PANEL
#   defaults to transfer5 (pipeline4 + genetic sex).
#
# What it does:
#   1. Loads the best model from CRIS training results
#   2. Applies it to Jiang dataset (never seen during training)
#   3. Generates comprehensive evaluation metrics (all features & gene counts)
#
# =================================================================================

# Configuration
TRAIN_DATASET="cris"
FEATURE_TYPE=${1:-dd}   # dd, rna, or atac
TEST_DATASET=${2:-jiang} # the Jiang et al. cohort
STRATEGY="mean_diff"
# Which best-model search to deploy: best_model_best_model_search_<ft><TRAIN_SUFFIX>.
# Default (empty) = the submitted-era searches; the revision reruns on the manifest
# cohort (which carry OOF deployment thresholds) are submitted with a suffix, e.g.
#   METADATA_FILE=accessory_files/cris_manifest.csv EXPERIMENT_NAME=best_model_search_dd_rev \
#       sbatch slurm/best_model_search.sh cris dd
#   TRAIN_SUFFIX=_rev EXPERIMENT_SUFFIX=_rev sbatch slurm/external_validation.sh dd
TRAIN_SUFFIX=${TRAIN_SUFFIX:-}
EXPERIMENT_SUFFIX=${EXPERIMENT_SUFFIX:-}
# Prefer the cohort manifest (role column -> exact-ID feature matching, deduplicated
# records; built by src/comp/build_<cohort>_manifest.py) over the raw FinaleDB export.
if [ -f "accessory_files/${TEST_DATASET}_manifest.csv" ]; then
    EXTERNAL_METADATA_FILE="accessory_files/${TEST_DATASET}_manifest.csv"
else
    EXTERNAL_METADATA_FILE="accessory_files/${TEST_DATASET}_metadata.csv"
fi

# Set model directory and feature-loading args based on feature type
COVARIATE_PANEL=${COVARIATE_PANEL:-transfer5}   # pipeline4 + genetic sex from the BAM
EXT_PANEL_STEM="extracted_features/scalar_panel_${COVARIATE_PANEL}_${TEST_DATASET}"
FEATURE_TAG=${FEATURE_TYPE}
if [ "${FEATURE_TYPE}" = "rna" ]; then
    FEATURE_ARGS="--features_dir extracted_features/zhu_all_after_lift --no_gene_selection --feature_subset rcov"
    EXTRA_ARGS="--filter_feature rcov"
elif [ "${FEATURE_TYPE}" = "atac" ]; then
    FEATURE_ARGS="--predefined_feature_matrix extracted_features/ulz_atac_tfbs_features_${TEST_DATASET}.npy --predefined_metadata_csv extracted_features/ulz_atac_tfbs_features_${TEST_DATASET}.metadata.csv"
    EXTRA_ARGS=""
elif [ "${FEATURE_TYPE}" = "dd_cov" ]; then
    # the deployed model carries the covariate block; the external cohort's block of
    # the same panel is appended and standardized with the training statistics
    FEATURE_TAG="dd_cov_${COVARIATE_PANEL}"
    FEATURE_ARGS="--features_dir extracted_features/biomart_10kb --augment_covariates ${EXT_PANEL_STEM}"
    EXTRA_ARGS=""
elif [ "${FEATURE_TYPE}" = "cov" ]; then
    FEATURE_TAG="cov_${COVARIATE_PANEL}"
    FEATURE_ARGS="--predefined_feature_matrix ${EXT_PANEL_STEM}.npy --predefined_metadata_csv ${EXT_PANEL_STEM}.metadata.csv --predefined_no_row_standardize"
    EXTRA_ARGS=""
else
    FEATURE_ARGS="--features_dir extracted_features/biomart_10kb"
    EXTRA_ARGS=""
fi
case "${FEATURE_TYPE}" in
    dd_cov|cov) [ -f "${EXT_PANEL_STEM}.npy" ] || { echo "${EXT_PANEL_STEM}.npy missing (python src/comp/build_scalar_panel_matrix.py --task ${TEST_DATASET} --panel ${COVARIATE_PANEL})" >&2; exit 1; } ;;
esac
TRAIN_MODEL_DIR="nested_cv_results_${TRAIN_DATASET}/best_model_best_model_search_${FEATURE_TAG}${TRAIN_SUFFIX}"
EXPERIMENT_NAME="comprehensive_external_${TRAIN_DATASET}_on_${TEST_DATASET}_${FEATURE_TAG}${EXPERIMENT_SUFFIX}"
COMPREHENSIVE="--comprehensive_external_validation"
[ -d "${TRAIN_MODEL_DIR}" ] || { echo "${TRAIN_MODEL_DIR} not found - run slurm/best_model_search.sh cris ${FEATURE_TYPE} first" >&2; exit 1; }
[ -n "$(ls "${TRAIN_MODEL_DIR}"/best_model_GridSearchCV_*.pkl 2>/dev/null)" ] || { echo "${TRAIN_MODEL_DIR} holds no best_model_GridSearchCV_*.pkl yet (search unfinished?)" >&2; exit 1; }

OUTPUT_DIR="nested_cv_results_${TRAIN_DATASET}"

# Job name
JOB_NAME="ext_val_${FEATURE_TYPE}_${TEST_DATASET}"

echo "============================================"
echo "External Dataset Validation"
echo "Training Dataset: ${TRAIN_DATASET}"
echo "Test Dataset: ${TEST_DATASET}"
echo "Feature Type: ${FEATURE_TYPE}"
echo "Model Directory: ${TRAIN_MODEL_DIR}"
echo "Experiment Name: ${EXPERIMENT_NAME}"
echo "Feature Args: ${FEATURE_ARGS}"
echo "============================================"
echo ""

# Execute
apptainer exec --writable-tmpfs --pwd /opt/app --containall \
	--bind src/:/opt/app/src/ \
	--bind features.py:/opt/app/features.py \
	--bind input/:/opt/app/input/ \
	--bind beds/:/opt/app/beds/ \
	--bind accessory_files/:/opt/app/accessory_files/ \
	--bind nested_cv_results/:/opt/app/nested_cv_results/ \
	--bind nested_cv_results_cris/:/opt/app/nested_cv_results_cris/ \
	--bind nested_cv_results_jiang/:/opt/app/nested_cv_results_jiang/ \
	--bind ${PREPROCESSING_DIR}/cris/:/opt/app/cris/ \
	--bind ${PREPROCESSING_DIR}/jiang/:/opt/app/jiang/ \
	--bind extracted_features/:/opt/app/extracted_features/ \
		${CONTAINER} pixi run python -u src/comp/model_hpc.py \
		-c 4 \
		--validate_external \
		${COMPREHENSIVE} \
		--external_model_dir ${TRAIN_MODEL_DIR} \
		--external_metadata_file ${EXTERNAL_METADATA_FILE} \
		${FEATURE_ARGS} \
		${EXTRA_ARGS} \
		-e "${EXPERIMENT_NAME}" \
		-o ${OUTPUT_DIR} \
		-s ${STRATEGY} \
		-t PANCAN
