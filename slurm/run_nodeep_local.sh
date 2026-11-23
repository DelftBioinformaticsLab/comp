#!/bin/sh
# ============================================================================
# Every Cristiano model repeated WITHOUT the 30 deep-sequenced healthy samples
# (rebuttal, Fig S10), on a workstation. Each run is the existing wrapper with
#   EXCLUDE_SAMPLES=accessory_files/cris_exclude_deep_healthy.txt
# and an experiment name carrying the _nodeep suffix that the exporters read
# (src/figures/export/_common.py, COHORT_VARIANT=no_deep).
# ============================================================================
# Usage (repository root, `pixi install` done):
#   sh slurm/run_nodeep_local.sh panels      # covariate panels, both tasks      (~1 h)
#   sh slurm/run_nodeep_local.sh comparators # RNA + ATAC nested CV, both tasks  (hours)
#   sh slurm/run_nodeep_local.sh estimate    # cost of the fully nested runs, no fitting
#   sh slurm/run_nodeep_local.sh nested      # fully nested data-driven runs, both tasks (days)
#   sh slurm/run_nodeep_local.sh export      # exporters + notebooks for COHORT=no_deep
#   sh slurm/run_nodeep_local.sh all         # panels, comparators, nested, export
#
# CORES (default 16) is passed to every model_hpc call. The fully nested runs
# can be reduced to the configuration the full-cohort runs selected in 9 of 10
# folds (feature griffin_diff, ranking mean_diff) with
#   NESTED_REDUCED=1 sh slurm/run_nodeep_local.sh nested
# which cuts the cost ~20x; the comparison with the full-cohort run is then a
# fixed-configuration sensitivity analysis, not a repeat of the full selection -
# say so wherever the result is reported.
# ============================================================================

STEP=${1:-all}
export MODEL_HPC_LOCAL=1
export CORES=${CORES:-16}
export EXCLUDE_SAMPLES=accessory_files/cris_exclude_deep_healthy.txt
SUFFIX=_nodeep

cd "$(dirname "$0")/.." || exit 1
[ -x .pixi/envs/default/bin/python ] || { echo "pixi environment missing - run 'pixi install' first" >&2; exit 1; }
[ -f "${EXCLUDE_SAMPLES}" ] || { echo "${EXCLUDE_SAMPLES} missing - run python src/comp/build_deep_healthy_exclusion.py" >&2; exit 1; }
mkdir -p logs

run_panels() {
	# the predefined matrices already exist (build_scalar_panel_matrix.py); model_hpc
	# drops the excluded rows itself, so the full-cohort matrices are reused
	for task in brca_crc pancan; do
		if [ "${task}" = "pancan" ]; then CT=PANCAN; MS=_pancan; else CT="breast colorectal"; MS=""; fi
		for panel in features technical3 demographic techdemo5 biological3 all8 complete5 all8_cc pipeline4 transfer5; do
			name=${panel}; [ "${panel}" = "features" ] && name=""
			[ -n "${name}" ] && name="_${name}"
			exp="${task}_scalar_panel${name}${SUFFIX}"
			if [ -d "nested_cv_results_cris/${exp}" ] && [ -n "$(ls nested_cv_results_cris/${exp}/results_*.pkl 2>/dev/null)" ]; then
				echo "=== ${exp}: results exist, skipped"; continue
			fi
			echo "=== ${exp} ($(date)) ==="
			EXPERIMENT_NAME="${exp}" CANCER_TYPES="${CT}" \
			PREDEFINED_STEM="extracted_features/scalar_panel_${panel}${MS}" \
			sh slurm/nested_cv_scalar_panel.sh > "logs/local_${exp}.out" 2>&1 || { echo "FAILED: see logs/local_${exp}.out"; exit 1; }
		done
	done
}

run_comparators() {
	for task in brca_crc pancan; do
		if [ "${task}" = "pancan" ]; then CT=PANCAN; else CT="breast colorectal"; fi
		for arm in rna atac; do
			exp="${task}_${arm}_rev${SUFFIX}"
			if [ -n "$(ls nested_cv_results_cris/${exp}/results_*.pkl 2>/dev/null)" ]; then
				echo "=== ${exp}: results exist, skipped"; continue
			fi
			echo "=== ${exp} ($(date)) ==="
			EXPERIMENT_NAME="${exp}" CANCER_TYPES="${CT}" \
			sh slurm/nested_cv_${arm}.sh > "logs/local_${exp}.out" 2>&1 || { echo "FAILED: see logs/local_${exp}.out"; exit 1; }
		done
	done
}

nested_env() {
	# $1 = task
	if [ "$1" = "pancan" ]; then
		export CANCER_TYPES=PANCAN EXPERIMENT_NAME="pancan_nested_full${SUFFIX}"
	else
		export CANCER_TYPES="breast colorectal" EXPERIMENT_NAME="brca_crc_nested_full${SUFFIX}"
	fi
	if [ -n "${NESTED_REDUCED}" ]; then
		export RANKING_FUNCTIONS="mean_diff"
		export NESTED_FULL_EXTRA="--nested_full_features griffin_diff"
	fi
}

run_nested() {
	for task in brca_crc pancan; do
		nested_env "${task}"
		echo "=== ${EXPERIMENT_NAME} ($(date)) ==="
		sh slurm/run_nested_full_local.sh full || exit 1
	done
}

run_estimate() {
	for task in brca_crc pancan; do
		nested_env "${task}"
		sh slurm/run_nested_full_local.sh estimate | grep -E "Total samples|TOTAL MODEL FITS|INNER FITS"
	done
}

run_export() {
	export COHORT_VARIANT=no_deep
	P=.pixi/envs/default/bin/python
	${P} src/figures/export/export_fig3.py || exit 1
	${P} src/figures/export/export_fig4.py || exit 1
	${P} src/figures/export/export_fig3_baselines.py --task brca_crc || exit 1
	${P} src/figures/export/export_fig3_baselines.py --task pancan || exit 1
	unset COHORT_VARIANT
	# nbconvert is not in the pixi environment; JUPYTER names one that has it
	# (default: the tss conda env of this workstation, else `jupyter` on PATH)
	JUPYTER=${JUPYTER:-$HOME/miniforge3/envs/tss/bin/jupyter}
	[ -x "${JUPYTER}" ] || JUPYTER=jupyter
	cd src/figures || exit 1
	for nb in fig3 fig4 figS10; do
		echo "=== executing ${nb}.ipynb for cohort no_deep -> output/no_deep/ ==="
		FIG_COHORT=no_deep "${JUPYTER}" nbconvert --to notebook --execute \
			--output "no_deep_${nb}.ipynb" "${nb}.ipynb" || exit 1
	done
	echo "notebooks executed as src/figures/no_deep_<fig>.ipynb (the originals are untouched); figures in src/figures/output/no_deep/"
}

case "${STEP}" in
	panels)      run_panels ;;
	comparators) run_comparators ;;
	estimate)    run_estimate ;;
	nested)      run_nested ;;
	export)      run_export ;;
	all)         run_panels && run_comparators && run_nested && run_export ;;
	*) echo "usage: $0 panels|comparators|estimate|nested|export|all" >&2; exit 1 ;;
esac
