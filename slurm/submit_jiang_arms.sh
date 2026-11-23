#!/bin/sh
# ============================================================================
# Jiang generalisation arms (rebuttal Fig S16 / Table S3), the official way: the
# single-CV best-model search on Cristiano PANCAN (GridSearchCV, StratifiedKFold 10,
# every classifier x every k, refit on all samples, deployment thresholds frozen from
# the search's own held-out scores) followed by its comprehensive external validation
# on Jiang (pipeline mode: Cristiano preprocessing and column statistics applied
# unchanged; fixed Cristiano thresholds; HCC vs 32 healthy / 67 CHB / 36 cirrhosis).
#
# Arms (ARMS, default "dd dd_cov cov rna atac"):
#   dd      the TSS model (biomart TSS, 5 feature types x 12 k)
#   dd_cov  the TSS model with the COVARIATE_PANEL block appended INSIDE the model
#           (--augment_covariates; default transfer5 = mean coverage, total fragments,
#           fragment length mean, mean GC and genetic sex from the BAM: the covariates
#           derivable for every cohort)
#   cov     the COVARIATE_PANEL alone (predefined matrix, no row standardization)
#   rna     Zhu RNA regions (rcov, no selection)
#   atac    Ulz ATAC TFBS matrix
# 'comparators' adds the per-classifier nested CV of RNA / ATAC on both tasks
# (<task>_{rna,atac}_rev<suffix>: Fig 3 / Fig 5a), which the no_deep cohort lacks.
#
# Usage (repository root):
#   ./slurm/submit_jiang_arms.sh search        # the best-model searches
#   ./slurm/submit_jiang_arms.sh deploy        # the external validations (+ clinical mode of dd)
#   ./slurm/submit_jiang_arms.sh comparators   # RNA/ATAC nested CV, both tasks
#   ./slurm/submit_jiang_arms.sh all           # search + deploy (each deploy job waits for
#                                              # its search job, --dependency=afterok) + comparators
#   ./slurm/submit_jiang_arms.sh status        # what exists on disk
# Environment: COHORT=no_deep (default; 'all' = the full cohort, whose dd/rna/atac
#   searches and deployments exist already, so ARMS defaults to "dd_cov cov" there),
#   ARMS, COVARIATE_PANEL (transfer5), CLINICAL=0 (skip the clinical-mode deployment of
#   dd, Fig 5k/l), DRY_RUN=1 (print the sbatch lines only).
# Resumable: best_model_search.sh skips every configuration whose summary_*.pkl exists,
# so a search that hit its 12 h walltime is resubmitted with the same 'search' command;
# its deploy job (afterok, never started) is then resubmitted with 'deploy'. External
# validation directories are overwritten by a rerun.
# Afterwards: COHORT_VARIANT=no_deep python src/figures/export/export_figS16_jiang_arms.py --from_runs
# (and export_fig3.py / export_fig4.py / export_fig3_baselines.py for Fig 3 / 5 of the cohort).
# ============================================================================

. slurm/_site.sh   # site settings: slurm/site.env (scheduler, storage paths, container, Apptainer cache)
STEP=${1:-}
COHORT=${COHORT:-no_deep}
COVARIATE_PANEL=${COVARIATE_PANEL:-transfer5}   # pipeline4 + genetic sex from the BAM
export COVARIATE_PANEL
cd "$(dirname "$0")/.." || exit 1
MANIFEST=accessory_files/cris_manifest.csv
case "${COHORT}" in
	no_deep)
		SUFFIX=_rev_nodeep
		EXCL=accessory_files/cris_exclude_deep_healthy.txt
		[ -f "${EXCL}" ] || { echo "${EXCL} missing - run python src/comp/build_deep_healthy_exclusion.py" >&2; exit 1; }
		ARMS=${ARMS:-"dd dd_cov cov rna atac"} ;;
	all)
		SUFFIX=_rev
		EXCL=""
		ARMS=${ARMS:-"dd_cov cov"} ;;
	*) echo "COHORT=${COHORT}: expected no_deep or all" >&2; exit 1 ;;
esac
for f in extracted_features/scalar_panel_${COVARIATE_PANEL}_pancan.npy extracted_features/scalar_panel_${COVARIATE_PANEL}_jiang.npy; do
	[ -f "${f}" ] || { echo "${f} missing - python src/comp/build_scalar_panel_matrix.py --panel ${COVARIATE_PANEL} --task pancan|jiang" >&2; exit 1; }
done
mkdir -p logs
JOBLOG=logs/jiang_arms_${COHORT}_$(date +%Y%m%d_%H%M).jobs

tag() { case "$1" in dd_cov|cov) echo "$1_${COVARIATE_PANEL}" ;; *) echo "$1" ;; esac; }
search_dir() { echo "nested_cv_results_cris/best_model_best_model_search_$(tag "$1")${SUFFIX}"; }
deploy_dir() { echo "nested_cv_results_cris/external_validation_comprehensive_external_cris_on_jiang_$(tag "$1")${SUFFIX}"; }
want_configs() { case "$1" in dd|dd_cov) echo 60 ;; *) echo 1 ;; esac; }   # 5 features x 12 k, or one matrix

run() {   # run <description> <command...>; sets JOBID (the command must print it: sbatch --parsable)
	desc=$1; shift
	echo ">> ${desc}"
	echo "   $*"
	if [ -n "${DRY_RUN}" ]; then JOBID=dry; return 0; fi
	JOBID=$("$@") || { echo "FAILED: ${desc}" >&2; exit 1; }
	JOBID=${JOBID%%;*}
	[ -n "${JOBID}" ] || { echo "FAILED: ${desc} (no job id returned)" >&2; exit 1; }
	echo "   job ${JOBID}"
	echo "${JOBID} ${desc}" >> "${JOBLOG}"
}

submit_search() {   # arm -> SEARCH_JOB
	arm=$1; dir=$(search_dir "${arm}"); exp="best_model_search_$(tag "${arm}")${SUFFIX}"
	case "${arm}" in
		dd|dd_cov) res="--mem=96G"; cores=16 ;;   # the k=15000 configurations are OOM-killed at 40 GB with 64 workers
		cov)       res="--cpus-per-task=8 --mem=8G --time=01:00:00"; cores=8 ;;
		rna|atac)  res=""; cores="" ;;
		*) echo "unknown arm '${arm}' (dd dd_cov cov rna atac)" >&2; exit 1 ;;
	esac
	n=$(ls "${dir}"/summary_*.pkl 2>/dev/null | wc -l)
	[ "${n}" -gt 0 ] && echo "   note: ${dir} holds ${n}/$(want_configs "${arm}") completed configuration(s); the job skips them"
	run "search ${arm} (${exp})" env EXCLUDE_SAMPLES="${EXCL}" METADATA_FILE="${MANIFEST}" CANCER_TYPES=PANCAN \
		EXPERIMENT_NAME="${exp}" CORES="${cores}" sbatch --parsable ${res} slurm/best_model_search.sh cris "${arm}"
	SEARCH_JOB=${JOBID}
}

submit_deploy() {   # arm [job id to wait for]
	arm=$1; dep=$2; dir=$(deploy_dir "${arm}")
	[ -d "${dir}" ] && echo "   note: ${dir} exists and is overwritten"
	run "deploy ${arm} -> ${dir}" env TRAIN_SUFFIX="${SUFFIX}" EXPERIMENT_SUFFIX="${SUFFIX}" \
		sbatch --parsable ${dep:+--dependency=afterok:${dep}} slurm/external_validation.sh "${arm}" jiang
	if [ "${arm}" = dd ] && [ "${CLINICAL:-1}" != 0 ]; then
		run "deploy dd, clinical mode (Fig 5k/l)" env TRAIN_SUFFIX="${SUFFIX}" EXPERIMENT_SUFFIX="${SUFFIX}" \
			sbatch --parsable ${dep:+--dependency=afterok:${dep}} slurm/external_validation_clinical.sh jiang
	fi
}

comparator_state() {   # <experiment> -> "-" | "running (..)" | "done (..)"; the per-classifier
	# results are claimed through <result>.lock files while a job works on them
	d="nested_cv_results_cris/$1"
	n=$(ls "${d}"/results_*.pkl 2>/dev/null | wc -l); l=$(ls "${d}"/*.lock 2>/dev/null | wc -l)
	if [ "${n}" -eq 0 ] && [ "${l}" -eq 0 ]; then echo "-"
	elif [ "${l}" -gt 0 ]; then echo "running (${n} results, ${l} claimed)"
	else echo "done (${n} results)"; fi
}

submit_comparators() {
	for task in pancan brca_crc; do   # PANCAN first
		if [ "${task}" = pancan ]; then CT=PANCAN; else CT="breast colorectal"; fi
		for arm in rna atac; do
			exp="${task}_${arm}${SUFFIX}"
			case "$(comparator_state "${exp}")" in
				done*) echo "   ${exp}: results exist, skipped"; continue ;;
				running*) echo "   note: ${exp} is being worked on (claims present); a second job waits for or takes over stale claims" ;;
			esac
			run "nested CV ${arm} ${task} (${exp})" env EXCLUDE_SAMPLES="${EXCL}" CANCER_TYPES="${CT}" \
				EXPERIMENT_NAME="${exp}" sbatch --parsable "slurm/nested_cv_${arm}.sh"
		done
	done
}

status() {
	echo "cohort ${COHORT} (suffix ${SUFFIX}), panel ${COVARIATE_PANEL}"
	for arm in ${ARMS}; do
		sd=$(search_dir "${arm}"); dd=$(deploy_dir "${arm}")
		n=$(ls "${sd}"/summary_*.pkl 2>/dev/null | wc -l)
		if [ -f "${dd}/comprehensive_validation_summary.pkl" ]; then dep=done; elif [ -d "${dd}" ]; then dep=partial; else dep=-; fi
		printf "  %-7s search %3s/%-3s configs   deploy %-8s %s\n" "${arm}" "${n}" "$(want_configs "${arm}")" "${dep}" "${sd}"
	done
	for task in pancan brca_crc; do for arm in rna atac; do
		exp="${task}_${arm}${SUFFIX}"
		printf "  comparator %-28s %s\n" "${exp}" "$(comparator_state "${exp}")"
	done; done
}

case "${STEP}" in
	search)      for arm in ${ARMS}; do submit_search "${arm}"; done ;;
	deploy)      for arm in ${ARMS}; do submit_deploy "${arm}" ""; done ;;
	comparators) submit_comparators ;;
	all)         for arm in ${ARMS}; do submit_search "${arm}"; submit_deploy "${arm}" "${SEARCH_JOB}"; done
	             submit_comparators ;;
	status)      status; exit 0 ;;
	*) echo "usage: $0 search|deploy|comparators|all|status   (COHORT=no_deep|all ARMS=... DRY_RUN=1)" >&2; exit 1 ;;
esac
echo "done: ${STEP} (${COHORT})$([ -n "${DRY_RUN}" ] && echo ' - dry run' || echo "; job ids in ${JOBLOG}")"
