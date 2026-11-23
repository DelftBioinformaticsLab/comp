#!/bin/sh
# ============================================================================
# Submit the TSS-set arms (rebuttal R2.3 / R3.4) on the no_deep cohort:
# 2 tasks x 2 set types (hallmark, arm) x 2 modes (averaged, pooled) x 5 features
# = 40 nested runs of slurm/nested_cv_tss_sets.sh, results in
#   nested_cv_results_cris/<task>_tss_sets_<type>_<mode>_<feature>_nodeep/
# Matrices that do not exist yet are skipped with a note (the pooled
# griffin_diff / rcov / mwh ones need slurm/extract_tss_sets.sh + the 'pooled'
# step first), so this can be run once after 'averaged' and again after 'pooled'.
# Existing result directories are skipped, not overwritten.
#
# Usage (repository root):  ./slurm/submit_tss_set_panels.sh      (DRY_RUN=1 to list only)
# Optional: MODES="averaged" FEATURES="griffin_diff" SET_TYPES="arm" to restrict.
# ============================================================================

. slurm/_site.sh   # site settings: slurm/site.env (scheduler, storage paths, container, Apptainer cache)
cd "$(dirname "$0")/.." || exit 1
EXCL=accessory_files/cris_exclude_deep_healthy.txt
[ -f "${EXCL}" ] || { echo "${EXCL} missing" >&2; exit 1; }
SET_TYPES=${SET_TYPES:-"hallmark arm"}
MODES=${MODES:-"averaged pooled"}
FEATURES=${FEATURES:-"griffin_diff rcov mwh fslr fslr_coverage"}
mkdir -p logs
n=0
for task in brca_crc pancan; do
	if [ "${task}" = "pancan" ]; then CT=PANCAN; MS=_pancan; else CT="breast colorectal"; MS=""; fi
	for st in ${SET_TYPES}; do
		for mode in ${MODES}; do
			for f in ${FEATURES}; do
				exp="${task}_tss_sets_${st}_${mode}_${f}_nodeep"
				stem="extracted_features/tss_sets/${st}_${mode}_${f}${MS}"
				[ -f "${stem}.npy" ] || { echo "SKIP ${exp}: ${stem}.npy not built yet"; continue; }
				[ -d "nested_cv_results_cris/${exp}" ] && { echo "SKIP ${exp}: result directory exists"; continue; }
				echo ">> ${exp}"
				if [ -z "${DRY_RUN}" ]; then
					EXPERIMENT_NAME=${exp} CANCER_TYPES="${CT}" PREDEFINED_STEM=${stem} EXCLUDE_SAMPLES=${EXCL} \
						sbatch slurm/nested_cv_tss_sets.sh || { echo "FAILED: ${exp}" >&2; exit 1; }
				fi
				n=$((n + 1))
			done
		done
	done
done
echo "${n} run(s) $([ -n "${DRY_RUN}" ] && echo 'listed (dry run)' || echo submitted)"
