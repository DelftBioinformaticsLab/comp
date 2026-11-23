#!/bin/sh
# ============================================================================
# Fully nested CV on a workstation: the ten outer folds one after another,
# then the aggregation step. Same arguments and output layout as the slurm
# array (slurm/nested_full.sh), so the exporters read the result unchanged.
# ============================================================================
# Usage (from the repository root, pixi environment installed):
#   sh slurm/run_nested_full_local.sh [full|test|estimate|aggregate] [first_fold] [last_fold]
#
# Environment, as for the slurm wrapper: CANCER_TYPES, EXPERIMENT_NAME,
# FEATURE_SET (dd|rna|atac), FEATURES_DIR, RANKING_FUNCTIONS, K_GRID,
# EXCLUDE_SAMPLES. CORES (default 16) replaces the cluster's 32.
#
# The run is resumable: a fold whose partial already exists in
# nested_cv_results_cris/nested_full_<EXPERIMENT_NAME>/ is skipped by
# model_hpc.py, so an interrupted loop can simply be restarted.
#
# Sensitivity runs without the 30 deep-sequenced healthy samples (one call
# per task; each is 10 folds + aggregate):
#   EXCLUDE_SAMPLES=accessory_files/cris_exclude_deep_healthy.txt \
#   EXPERIMENT_NAME=brca_crc_nested_full_nodeep CANCER_TYPES="breast colorectal" \
#   sh slurm/run_nested_full_local.sh full
#   EXCLUDE_SAMPLES=accessory_files/cris_exclude_deep_healthy.txt \
#   EXPERIMENT_NAME=pancan_nested_full_nodeep CANCER_TYPES=PANCAN \
#   sh slurm/run_nested_full_local.sh full
# Folds can also be split over several terminals, e.g. folds 1-5 in one and
# 6-10 in another with CORES=8 each, followed by one "aggregate" call.
# ============================================================================

STEP=${1:-full}
FIRST=${2:-1}
LAST=${3:-10}
export MODEL_HPC_LOCAL=1
export CORES=${CORES:-16}

cd "$(dirname "$0")/.." || exit 1
[ -x .pixi/envs/default/bin/python ] || { echo "pixi environment missing - run 'pixi install' first" >&2; exit 1; }
mkdir -p logs

case "${STEP}" in
	estimate|aggregate)
		sh slurm/nested_full.sh "${STEP}" 2>&1 | tee "logs/local_nested_full_${EXPERIMENT_NAME:-default}_${STEP}.out"
		;;
	full|test)
		SUBSTEP=""
		[ "${STEP}" = "test" ] && SUBSTEP="test"
		f=${FIRST}
		while [ "${f}" -le "${LAST}" ]; do
			echo "=== outer fold ${f}/10 ($(date)) ==="
			FOLD=${f} sh slurm/nested_full.sh ${SUBSTEP} 2>&1 | tee "logs/local_nested_full_${EXPERIMENT_NAME:-default}_fold${f}.out" || exit 1
			f=$((f + 1))
		done
		if [ "${FIRST}" -eq 1 ] && [ "${LAST}" -eq 10 ]; then
			AGG="aggregate"
			[ "${STEP}" = "test" ] && AGG="test_aggregate"
			echo "=== ${AGG} ($(date)) ==="
			sh slurm/nested_full.sh "${AGG}" 2>&1 | tee "logs/local_nested_full_${EXPERIMENT_NAME:-default}_${AGG}.out"
		else
			echo "folds ${FIRST}-${LAST} done; run 'sh slurm/run_nested_full_local.sh aggregate' once all ten exist"
		fi
		;;
	*)
		echo "usage: $0 [full|test|estimate|aggregate] [first_fold] [last_fold]" >&2
		exit 1
		;;
esac
