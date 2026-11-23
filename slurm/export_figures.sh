#!/bin/sh
#SBATCH --time=03:59:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=52429MB
#SBATCH --mail-type=END
#SBATCH --job-name=export_figures
#SBATCH --output=logs/%j_export_figures.out
#SBATCH --error=logs/%j_export_figures.out
. slurm/_site.sh   # site settings: slurm/site.env (scheduler, storage paths, container, Apptainer cache)

# ============================================================================
# Run a figure exporter (src/figures/export/*.py) on a COMPUTE node, inside
# the same container the models were trained in.
#
# Why: the exporters that refit or reload fold models (export_fig3_baselines.py
# importance + partial dependence) and the ones
# that read the 447 per-sample _BIN.csv files in parallel
# (export_figS10_tss_regression.py, export_figS10_subgroup.py) need several GB
# and several cores. On the login node the kernel kills their joblib workers
# (TerminatedWorkerError, exit SIGKILL(-9)). The container also guarantees the
# library versions of the training runs (scikit-learn / xgboost / pandas as
# pinned in pixi.lock), which the refit-reproduction checks rely on.
#
# Usage (repository root):
#   sbatch slurm/export_figures.sh src/figures/export/export_fig3_baselines.py --task brca_crc
#   COHORT_VARIANT=no_deep sbatch slurm/export_figures.sh src/figures/export/export_fig3_baselines.py --task pancan
#   sbatch slurm/export_figures.sh src/figures/export/export_figS10_tss_regression.py --cores 12 --cache results/figS10_feature_cache.npz
# Resources are overridable on the command line, e.g.
#   sbatch --cpus-per-task=16 --mem=78643M slurm/export_figures.sh <script> [args]
# COHORT_VARIANT (default no_deep; COHORT_VARIANT=all for the cohort with the 30 deep-sequenced controls) is passed into the container; everything else
# the exporters need is bound below (same bind list as slurm/_model_hpc_exec.sh).
# Outputs land in src/figures/data[/no_deep]/ as when run interactively.
# ============================================================================

SCRIPT=$1
[ -n "${SCRIPT}" ] || { echo "usage: sbatch slurm/export_figures.sh src/figures/export/<exporter>.py [args]" >&2; exit 1; }
[ -f "${SCRIPT}" ] || { echo "${SCRIPT}: not found (submit from the repository root)" >&2; exit 1; }
shift

echo "============================================"
echo "Figure export | script: ${SCRIPT} | args: $*"
echo "COHORT_VARIANT=${COHORT_VARIANT:-no_deep} | cores: ${SLURM_CPUS_PER_TASK:-?} | mem: ${SLURM_MEM_PER_NODE:-?} MB"
echo "============================================"

apptainer exec --writable-tmpfs --pwd /opt/app --containall \
	--env "COHORT_VARIANT=${COHORT_VARIANT:-no_deep},PYTHONUTF8=1,MPLBACKEND=Agg" \
	--bind src/:/opt/app/src/ \
	--bind features.py:/opt/app/features.py \
	--bind input/:/opt/app/input/ \
	--bind beds/:/opt/app/beds/ \
	--bind accessory_files/:/opt/app/accessory_files/ \
	--bind nested_cv_results/:/opt/app/nested_cv_results/ \
	--bind nested_cv_results_cris/:/opt/app/nested_cv_results_cris/ \
	--bind nested_cv_results_jiang/:/opt/app/nested_cv_results_jiang/ \
	--bind results/:/opt/app/results/ \
	--bind ${PREPROCESSING_DIR}/cris/:/opt/app/cris/ \
	--bind ${PREPROCESSING_DIR}/jiang/:/opt/app/jiang/ \
	--bind extracted_features/:/opt/app/extracted_features/ \
		${CONTAINER} pixi run python -u "${SCRIPT}" "$@"
