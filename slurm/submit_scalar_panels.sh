#!/bin/sh
# ============================================================================
# Submit every covariate-panel nested-CV run (reviewer 4 comment 7 / Fig S10)
# to slurm: 10 panels x 2 tasks x the requested cohort(s). Each job is
# slurm/nested_cv_scalar_panel.sh with the panel's predefined matrix; that
# wrapper also stores the extended per-fold records and the
# fitted fold models (model_hpc.py --store_fitted_models), which the deep-dive
# exporter reads.
# ============================================================================
# Usage (repository root, matrices built with src/comp/build_scalar_panel_matrix.py):
#   ./slurm/submit_scalar_panels.sh all        # full cohort with the 30 deep-sequenced controls (274 / 397 samples)
#   ./slurm/submit_scalar_panels.sh no_deep    # without the 30 deep-sequenced healthy
#   ./slurm/submit_scalar_panels.sh both       # both cohorts (40 jobs)
# Optional: PANELS="techdemo5 all8" ./slurm/submit_scalar_panels.sh both
#           (subset of: features technical3 demographic techdemo5 biological3
#            all8 complete5 all8_cc pipeline4 transfer5)
#           DRY_RUN=1 prints the sbatch lines without submitting.
# Build each panel's matrices first (src/comp/build_scalar_panel_matrix.py), e.g.:
#   for p in technical3 techdemo5 all8 complete5; do
#     python src/comp/build_scalar_panel_matrix.py --panel $p
#     python src/comp/build_scalar_panel_matrix.py --panel $p --task pancan
#   done
#   python src/comp/build_scalar_panel_matrix.py --panel all8 --complete_case
#   python src/comp/build_scalar_panel_matrix.py --panel all8 --complete_case --task pancan
#
# An existing result directory of the same name is moved aside to
# <dir>.bak_<timestamp> first, so the rerun never mixes old and new pickles
# (the exporters read the canonical names only). Delete the .bak dirs once
# the new export has been checked.
#
# Afterwards, per cohort:
#   python src/figures/export/export_fig3_baselines.py --task brca_crc
#   python src/figures/export/export_fig3_baselines.py --task pancan
#   COHORT_VARIANT=no_deep python src/figures/export/export_fig3_baselines.py --task brca_crc
#   COHORT_VARIANT=no_deep python src/figures/export/export_fig3_baselines.py --task pancan
# then execute figS10.ipynb. Run the exporters in the environment the panels were trained
# in (the container / pixi env): the importance and partial-dependence steps
# refit or reload the fold models and refuse ones that do not reproduce.
# ============================================================================

. slurm/_site.sh   # site settings: slurm/site.env (scheduler, storage paths, container, Apptainer cache)
COHORTS=${1:-both}
PANELS=${PANELS:-"features technical3 demographic techdemo5 biological3 all8 complete5 all8_cc pipeline4 transfer5"}
STAMP=$(date +%Y%m%d_%H%M)

cd "$(dirname "$0")/.." || exit 1
[ -f slurm/nested_cv_scalar_panel.sh ] || { echo "run from the repository root" >&2; exit 1; }
mkdir -p logs

case "${COHORTS}" in
	all)     LIST="all" ;;
	no_deep) LIST="no_deep" ;;
	both)    LIST="all no_deep" ;;
	*) echo "usage: $0 all|no_deep|both" >&2; exit 1 ;;
esac

n=0
for cohort in ${LIST}; do
	if [ "${cohort}" = "no_deep" ]; then
		SUFFIX=_nodeep
		EXCL=accessory_files/cris_exclude_deep_healthy.txt
		[ -f "${EXCL}" ] || { echo "${EXCL} missing - run python src/comp/build_deep_healthy_exclusion.py" >&2; exit 1; }
	else
		SUFFIX=""
		EXCL=""
	fi
	for task in brca_crc pancan; do
		if [ "${task}" = "pancan" ]; then CT=PANCAN; MS=_pancan; else CT="breast colorectal"; MS=""; fi
		for panel in ${PANELS}; do
			name=${panel}; [ "${panel}" = "features" ] && name=""
			[ -n "${name}" ] && name="_${name}"
			exp="${task}_scalar_panel${name}${SUFFIX}"
			stem="extracted_features/scalar_panel_${panel}${MS}"
			[ -f "${stem}.npy" ] || { echo "SKIP ${exp}: ${stem}.npy missing (build_scalar_panel_matrix.py --panel ${panel}$([ "${task}" = pancan ] && echo ' --task pancan'))"; continue; }
			dir="nested_cv_results_cris/${exp}"
			if [ -d "${dir}" ]; then
				if [ -z "${DRY_RUN}" ]; then
					mv "${dir}" "${dir}.bak_${STAMP}" && echo "moved ${dir} -> ${dir}.bak_${STAMP}"
				else
					echo "(dry run) would move ${dir} -> ${dir}.bak_${STAMP}"
				fi
			fi
			cmd="EXPERIMENT_NAME=${exp} CANCER_TYPES=\"${CT}\" PREDEFINED_STEM=${stem} EXCLUDE_SAMPLES=${EXCL} sbatch slurm/nested_cv_scalar_panel.sh"
			if [ -n "${DRY_RUN}" ]; then
				echo "${cmd}"
			else
				EXPERIMENT_NAME="${exp}" CANCER_TYPES="${CT}" PREDEFINED_STEM="${stem}" EXCLUDE_SAMPLES="${EXCL}" \
					sbatch slurm/nested_cv_scalar_panel.sh || { echo "sbatch failed for ${exp}" >&2; exit 1; }
			fi
			n=$((n + 1))
		done
	done
done
echo "${n} panel job(s) $([ -n "${DRY_RUN}" ] && echo 'listed' || echo 'submitted')"
