#!/bin/sh
# ============================================================================
# Fig S14 - the data-driven model WITH the sample-level covariates of Fig S10,
# as one fully nested model, k-swept (one complete nested selection per k):
#   cov_techdemo5    TSS + the 5 confounders (coverage, fragments, plasma volume, age, sex)
# The reference arm (TSS only) is the existing k-sweep of the same cohort.
# Each arm = slurm/submit_nested_full.sh ksweep with COVARIATE_STEM / COVARIATE_MODE,
# both tasks; the aggregate also derives ksweep_tuned/ (k chosen per fold on inner CV).
#
# Usage (repository root):
#   ./slurm/submit_covariate_arms.sh            # no_deep cohort (default)
#   ./slurm/submit_covariate_arms.sh all        # full cohort (with the 30 deep-sequenced controls)
#   ./slurm/submit_covariate_arms.sh both
#   ARMS="cov_techdemo5" ./slurm/submit_covariate_arms.sh
#   REFERENCE=1 ./slurm/submit_covariate_arms.sh   # also submits the TSS-only k-sweep of the cohort (the reference arm)
#   DRY_RUN=1 ... lists the submissions.
# Results: nested_cv_results_cris/nested_full_<task>_nested_full_<arm>[_nodeep]_ksweep/
# Export at any time (partials are read as they land):
#   python src/figures/export/export_figS14.py      (COHORT_VARIANT defaults to no_deep)
# ============================================================================

COHORTS=${1:-no_deep}
ARMS=${ARMS:-"cov_techdemo5"}
cd "$(dirname "$0")/.." || exit 1
MANIFEST=accessory_files/cris_manifest.csv
EXCL=accessory_files/cris_exclude_deep_healthy.txt

case "${COHORTS}" in
	all) LIST="all" ;; no_deep) LIST="no_deep" ;; both) LIST="all no_deep" ;;
	*) echo "usage: $0 [all|no_deep|both]" >&2; exit 1 ;;
esac

n=0
for cohort in ${LIST}; do
	if [ "${cohort}" = no_deep ]; then
		SUFFIX=_nodeep; EXCLUDE=${EXCL}
		[ -f "${EXCL}" ] || { echo "${EXCL} missing" >&2; exit 1; }
	else
		SUFFIX=""; EXCLUDE=""
	fi
	for task in pancan brca_crc; do   # PANCAN first: its arms are wanted earliest
		if [ "${task}" = pancan ]; then CT=PANCAN; MS=_pancan; else CT="breast colorectal"; MS=""; fi
		if [ -n "${REFERENCE}" ]; then
			# the TSS-only reference arm: the cohort's own data-driven k-sweep
			# (nested_full_<task>_nested_full[_nodeep]_ksweep; also part of submit_nodeep_all.sh phase1)
			EXP="${task}_nested_full${SUFFIX}"
			echo ">> ${cohort} ${task} none (reference): EXPERIMENT_NAME=${EXP}"
			if [ -z "${DRY_RUN}" ]; then
				( export EXCLUDE_SAMPLES="${EXCLUDE}" METADATA_FILE=${MANIFEST} CANCER_TYPES="${CT}" EXPERIMENT_NAME="${EXP}"
				  ./slurm/submit_nested_full.sh ksweep ) || { echo "submission failed: ${EXP}" >&2; exit 1; }
			fi
			n=$((n + 1))
		fi
		for arm in ${ARMS}; do
			case "${arm}" in
				cov_*)   MODE=augment;     panel=${arm#cov_} ;;
				resid_*) MODE=residualize; panel=${arm#resid_} ;;
				*) echo "unknown arm ${arm}" >&2; exit 1 ;;
			esac
			STEM=extracted_features/scalar_panel_${panel}${MS}
			[ -f "${STEM}.npy" ] || { echo "SKIP ${task} ${arm}: ${STEM}.npy missing (python src/comp/build_scalar_panel_matrix.py --panel ${panel}$([ "${task}" = pancan ] && echo ' --task pancan'))"; continue; }
			EXP="${task}_nested_full_${arm}${SUFFIX}"
			echo ">> ${cohort} ${task} ${arm}: EXPERIMENT_NAME=${EXP} COVARIATE_STEM=${STEM} COVARIATE_MODE=${MODE}"
			if [ -z "${DRY_RUN}" ]; then
				( export EXCLUDE_SAMPLES="${EXCLUDE}" METADATA_FILE=${MANIFEST} CANCER_TYPES="${CT}" \
				         EXPERIMENT_NAME="${EXP}" COVARIATE_STEM="${STEM}" COVARIATE_MODE="${MODE}"
				  ./slurm/submit_nested_full.sh ksweep ) || { echo "submission failed: ${EXP}" >&2; exit 1; }
			fi
			n=$((n + 1))
		done
	done
done
echo "${n} k-sweep(s) $([ -n "${DRY_RUN}" ] && echo listed || echo submitted)"
