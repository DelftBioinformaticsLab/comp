#!/bin/sh
#SBATCH --time=11:59:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=40GB
#SBATCH --mail-type=END
#SBATCH --job-name=best_model_search
#SBATCH --output=logs/%j_best_model_search.out
#SBATCH --error=logs/%j_best_model_search.out

# ============================================================================
# Best-model search — trains/saves the single best model per configuration
# ============================================================================
# Produces <output_dir>/best_model_<EXPERIMENT_NAME>/ with
# best_model_GridSearchCV_*.pkl etc., consumed by the external-validation and
# label-shuffle experiments.
#
# Usage (submit from the repo root):
#   sbatch slurm/best_model_search.sh [DATASET] [FEATURE_TYPE]
#   FEATURE_TYPE: dd (biomart TSS) | rna (zhu_all_after_lift) | atac (ulz TFBS)
#                 | dd_cov (TSS + the covariate panel COVARIATE_PANEL appended inside the
#                   model, --augment_covariates; rebuttal Fig S16)
#                 | cov (the covariate panel alone as a predefined matrix, no row
#                   standardization; rebuttal Fig S16)
#   COVARIATE_PANEL (default transfer5 = mean coverage, total fragments, fragment length
#   mean, mean GC and the genetic sex inferred from the BAM: the five covariates our
#   pipeline derives for every cohort; pipeline4 = the same without sex) names the
#   stem extracted_features/scalar_panel_<panel>[_pancan] from build_scalar_panel_matrix.py.
#   The default EXPERIMENT_NAME of these two is best_model_search_<type>_<panel>, and
#   slurm/external_validation.sh <type> deploys them with the external cohort's panel.
#
# Overridable via environment: CANCER_TYPES, EXPERIMENT_NAME, METADATA_FILE, CORES,
# BEST_MODEL_TOP_GENES, REFRESH_DEPLOYMENT_THRESHOLDS, COVARIATE_PANEL.
# REFRESH_DEPLOYMENT_THRESHOLDS=1 re-derives the frozen deployment thresholds of the
# configurations already complete in the output directory from the search's own ten
# folds (10 fits each, no new grid search); rerun slurm/external_validation.sh after.
# Resubmission is safe: configurations whose summary_* already exists in the output
# directory are skipped. The k=15000 configurations of the dd search need more memory
# than the header grants at 64 parallel jobs (they run out of memory); rerun those alone:
#   BEST_MODEL_TOP_GENES=15000 CORES=16 METADATA_FILE=accessory_files/cris_manifest.csv \
#   EXPERIMENT_NAME=best_model_search_dd_rev sbatch --mem=96G slurm/best_model_search.sh cris dd
# NOTE: the atac branch uses the *_pancan* matrix
# (ulz_atac_tfbs_features_pancan.npy), unlike nested_cv_atac.sh, which uses
# ulz_atac_tfbs_features.npy.
# ============================================================================

MODE="best_model_search"
DATASET=${1:-cris}
FEATURE_TYPE=${2:-dd}
JOB_NAME="${MODE}_${FEATURE_TYPE}"
STRATEGY="mean_diff"
COVARIATE_PANEL=${COVARIATE_PANEL:-transfer5}   # pipeline4 + genetic sex from the BAM
# the panel matrices of the PANCAN task carry a _pancan suffix
case " ${CANCER_TYPES:-PANCAN} " in
    *PANCAN*|*pancan*) PANEL_STEM="extracted_features/scalar_panel_${COVARIATE_PANEL}_pancan" ;;
    *)                 PANEL_STEM="extracted_features/scalar_panel_${COVARIATE_PANEL}" ;;
esac
FEATURE_TAG=${FEATURE_TYPE}
case "${FEATURE_TYPE}" in
    dd_cov|cov) FEATURE_TAG="${FEATURE_TYPE}_${COVARIATE_PANEL}"
                [ -f "${PANEL_STEM}.npy" ] || { echo "${PANEL_STEM}.npy missing (python src/comp/build_scalar_panel_matrix.py --panel ${COVARIATE_PANEL}$(case "${PANEL_STEM}" in *_pancan) echo ' --task pancan';; esac))" >&2; exit 1; } ;;
esac
EXPERIMENT_NAME=${EXPERIMENT_NAME:-best_model_search_${FEATURE_TAG}}
MODE_ARGS="--run_best_model_search"
if [ -n "${BEST_MODEL_TOP_GENES}" ]; then
    MODE_ARGS="${MODE_ARGS} --best_model_top_genes ${BEST_MODEL_TOP_GENES}"
fi
if [ -n "${REFRESH_DEPLOYMENT_THRESHOLDS}" ]; then
    MODE_ARGS="${MODE_ARGS} --refresh_deployment_thresholds"
fi
# Add feature-specific arguments
if [ "${FEATURE_TYPE}" = "rna" ]; then
    MODE_ARGS="${MODE_ARGS} --no_gene_selection --features_dir extracted_features/zhu_all_after_lift --feature_subset rcov"
elif [ "${FEATURE_TYPE}" = "atac" ]; then
    PREDEFINED_MATRIX="extracted_features/ulz_atac_tfbs_features_pancan.npy"
    PREDEFINED_META="extracted_features/ulz_atac_tfbs_features_pancan.metadata.csv"
    MODE_ARGS="${MODE_ARGS} --predefined_feature_matrix ${PREDEFINED_MATRIX} --predefined_metadata_csv ${PREDEFINED_META}"
elif [ "${FEATURE_TYPE}" = "dd_cov" ]; then
    # TSS features + the covariate block as passthrough columns of the selector
    MODE_ARGS="${MODE_ARGS} --features_dir extracted_features/biomart_10kb --augment_covariates ${PANEL_STEM}"
elif [ "${FEATURE_TYPE}" = "cov" ]; then
    # the covariate panel alone: heterogeneous scalars, so no per-sample row z-score
    # (the pipeline's StandardScaler column-standardizes on the training data)
    MODE_ARGS="${MODE_ARGS} --predefined_feature_matrix ${PANEL_STEM}.npy --predefined_metadata_csv ${PANEL_STEM}.metadata.csv --predefined_no_row_standardize"
else
    MODE_ARGS="${MODE_ARGS} --features_dir extracted_features/biomart_10kb"
fi

. slurm/_model_hpc_exec.sh
print_job_header

set -- \
	-c ${CORES:-64} -t ${CANCER_TYPES:-PANCAN} \
	${MODE_ARGS} \
	--metadata_file ${METADATA_FILE:-accessory_files/${DATASET}_metadata.csv} \
	-e "${EXPERIMENT_NAME}" \
	-o ${OUTPUT_DIR:-nested_cv_results_${DATASET}} \
	-s ${STRATEGY:-mean_diff}

run_model_hpc "$@"
