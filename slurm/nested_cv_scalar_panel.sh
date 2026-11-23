#!/bin/sh
#SBATCH --time=03:59:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=13107MB
#SBATCH --mail-type=END
#SBATCH --job-name=nested_cv_scalar_panel
#SBATCH --output=logs/%j_nested_cv_scalar_panel.out
#SBATCH --error=logs/%j_nested_cv_scalar_panel.out

# ============================================================================
# Nested CV - scalar-panel baseline (reviewer 2, revision)
# ============================================================================
# Six technical/covariate scalars (mean coverage, total fragments, fragment
# length mean, mean GC, age, sex - NO ichorCNA TF) as one predefined 274x6
# matrix through the same nested CV as the ATAC arm: identical fold generator
# (StratifiedKFold 10, rs=42, on cancer-type strata), all six classifiers,
# classifier chosen post hoc by mean outer AUC (export_fig3_baselines.py).
#
# --predefined_no_row_standardize is REQUIRED here: per-sample z-scoring across
# six heterogeneous scalars mixes units; the pipeline's StandardScaler still
# column-standardizes on training folds.
#
# Build the matrix first (reads the task's *_baseline_scalars.csv):
#   python src/comp/build_scalar_panel_matrix.py [--task pancan]
#
# Task / panel variants (BRCA/CRC full6 is the default). Panel stems:
# scalar_panel_features (full6), scalar_panel_technical3,
# scalar_panel_demographic, scalar_panel_techdemo5 (technical + demographic),
# scalar_panel_biological3, scalar_panel_all8, scalar_panel_complete5,
# scalar_panel_all8_cc, scalar_panel_pipeline4, scalar_panel_transfer5 - each
# with a _pancan twin. EXPERIMENT_NAME must be
# <task>_scalar_panel[_<panel>] (export_fig3_baselines.py reads these dirs):
#   sbatch slurm/nested_cv_scalar_panel.sh
#   EXPERIMENT_NAME=brca_crc_scalar_panel_technical3 \
#     PREDEFINED_STEM=extracted_features/scalar_panel_technical3 \
#     sbatch slurm/nested_cv_scalar_panel.sh
#   EXPERIMENT_NAME=pancan_scalar_panel_all8 CANCER_TYPES=PANCAN \
#     PREDEFINED_STEM=extracted_features/scalar_panel_all8_pancan \
#     sbatch slurm/nested_cv_scalar_panel.sh
# Jiang (HCC vs the 135 non-HCC comparators; only the cohort-transferable
# panels pipeline4 and transfer5 exist for it - build one with
# `python src/comp/build_scalar_panel_matrix.py --task jiang --panel transfer5`
# once export_figS10_jiang.py has written figS10_jiang_scalars.csv):
#   EXPERIMENT_NAME=jiang_scalar_panel_transfer5 \
#     PREDEFINED_STEM=extracted_features/scalar_panel_transfer5_jiang \
#     sbatch slurm/nested_cv_scalar_panel.sh jiang
#
# The matrix is at most 397x10, so this also runs fine on a workstation:
#   python src/comp/model_hpc.py <same args as below>
# (All estimators are seeded, so a container run reproduces a local one.)
# CAUTION: run with pandas < 3. Result pickles written under pandas 3 carry the
# new StringDtype and CANNOT be unpickled by the pandas 2.x environments that
# read them (tss env, container) - the exporter then dies with
# "StringDtype.__init__() takes from 1 to 2 positional arguments".
# ============================================================================

MODE="predefined_scalar_panel"
JOB_NAME="nested_cv_scalar_panel"
DATASET=${1:-cris}
STRATEGY=${2:-mean_diff}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-brca_crc_scalar_panel}
PREDEFINED_STEM=${PREDEFINED_STEM:-extracted_features/scalar_panel_features}
PREDEFINED_MATRIX="${PREDEFINED_STEM}.npy"
PREDEFINED_META="${PREDEFINED_STEM}.metadata.csv"
MODE_ARGS="--run_nested_cv --predefined_feature_matrix ${PREDEFINED_MATRIX} --predefined_metadata_csv ${PREDEFINED_META} --predefined_no_row_standardize --store_fitted_models"

DEFAULT_METADATA="accessory_files/${DATASET}_metadata.csv"
[ "${DATASET}" = "cris" ] && DEFAULT_METADATA="accessory_files/cris_manifest.csv"
[ "${DATASET}" = "jiang" ] && DEFAULT_METADATA="accessory_files/jiang_manifest.csv"

. slurm/_model_hpc_exec.sh
print_job_header

set -- \
	-c 8 -t ${CANCER_TYPES:-"breast colorectal"} \
	${MODE_ARGS} \
	--metadata_file ${METADATA_FILE:-${DEFAULT_METADATA}} \
	-e "${EXPERIMENT_NAME}" \
	-o ${OUTPUT_DIR:-nested_cv_results_${DATASET}} \
	-s ${STRATEGY:-mean_diff}

run_model_hpc "$@"
