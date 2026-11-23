#!/bin/sh
# ============================================================================
# Leave-one-cancer-type-out analysis (rebuttal R3.7 / R4.8; replaces Figure 5d-f and the
# fixed-configuration runs of export_figS_loco.py). The same procedure as the Jiang
# transfer (slurm/submit_jiang_arms.sh), once per cancer type of the no_deep pan-cancer task:
#   search  the single-CV best-model search (GridSearchCV, StratifiedKFold 10, every
#           classifier x every k x every feature type, refit on all training samples,
#           thresholds frozen from the search's own held-out scores) on the 130 training
#           healthy donors and the six other cancer types; nothing about the left-out
#           type or the held-out donors enters the configuration, the fit or the threshold
#   deploy  its comprehensive external validation, applied once to the left-out type and
#           the 34 held-out healthy donors (the same donors for every type)
# Splits: python src/comp/build_loco_splits.py -> accessory_files/cris_loco_<type>_{exclude.txt,manifest.csv}
#
# Usage (repository root):
#   ./slurm/submit_loco.sh search     # the seven searches
#   ./slurm/submit_loco.sh deploy     # the seven deployments
#   ./slurm/submit_loco.sh all        # search + deploy (each deploy waits for its search, afterok)
#   ./slurm/submit_loco.sh status     # what exists on disk
# Environment: TYPES (default all seven), DRY_RUN=1 (print the sbatch lines only),
#   ARM=dd (default) | rna | atac: the comparator arms run the same search and
#   deployment on the Zhu RNA regions or the Ulz ATAC matrix (one configuration each; the ATAC
#   test rows come from extracted_features/ulz_atac_tfbs_features_cris_loco_<type>.npy, written
#   by build_loco_splits.py). Results: best_model_best_model_search_<arm>_loco_<type>_rev_nodeep
#   and external_validation_comprehensive_external_cris_on_cris_loco_<type>_<arm>_rev_nodeep.
# Resumable: best_model_search.sh skips every configuration whose summary_*.pkl exists, so a
# search that hit its 12 h walltime is resubmitted with 'search'; its deploy job (afterok,
# never started) is then resubmitted with 'deploy'. Deployment directories are overwritten.
# ============================================================================

. slurm/_site.sh   # site settings: slurm/site.env (scheduler, storage paths, container, Apptainer cache)
STEP=${1:-}
TYPES=${TYPES:-"breast bileduct colorectal gastric lung ovarian pancreatic"}
ARM=${ARM:-dd}
case "${ARM}" in
	dd)       N_CONFIGS=60 ;;
	rna|atac) N_CONFIGS=1 ;;
	*) echo "unknown ARM '${ARM}' (dd rna atac)" >&2; exit 1 ;;
esac
SUFFIX=_rev_nodeep
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs
JOBLOG=logs/loco_${ARM}_$(date +%Y%m%d_%H%M).jobs

search_exp() { echo "best_model_search_${ARM}_loco_$1${SUFFIX}"; }
search_dir() { echo "nested_cv_results_cris/best_model_$(search_exp "$1")"; }
deploy_dir() { echo "nested_cv_results_cris/external_validation_comprehensive_external_cris_on_cris_loco_$1_${ARM}${SUFFIX}"; }

for t in ${TYPES}; do
	for f in "accessory_files/cris_loco_${t}_exclude.txt" "accessory_files/cris_loco_${t}_manifest.csv"; do
		[ -f "${f}" ] || { echo "${f} missing - run python src/comp/build_loco_splits.py" >&2; exit 1; }
	done
	f="extracted_features/ulz_atac_tfbs_features_cris_loco_${t}.npy"
	[ "${ARM}" != atac ] || [ -f "${f}" ] || { echo "${f} missing - run python src/comp/build_loco_splits.py" >&2; exit 1; }
done

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

submit_search() {   # type -> SEARCH_JOB
	t=$1; n=$(ls "$(search_dir "${t}")"/summary_*.pkl 2>/dev/null | wc -l)
	[ "${n}" -gt 0 ] && echo "   note: $(search_dir "${t}") holds ${n}/${N_CONFIGS} completed configurations; the job skips them"
	# dd: 96 GB, 16 workers (the k=15000 configurations are OOM-killed at 40 GB with 64 workers);
	# rna / atac: one small configuration, the script's own resources
	if [ "${ARM}" = dd ]; then cores=16; res="--mem=96G"; else cores=""; res=""; fi
	run "search ${ARM} without ${t}" env EXCLUDE_SAMPLES="accessory_files/cris_loco_${t}_exclude.txt" \
		METADATA_FILE=accessory_files/cris_manifest.csv CANCER_TYPES=PANCAN \
		EXPERIMENT_NAME="$(search_exp "${t}")" CORES="${cores}" \
		sbatch --parsable ${res} --job-name="loco_${ARM}_${t}" slurm/best_model_search.sh cris "${ARM}"
	SEARCH_JOB=${JOBID}
}

submit_deploy() {   # type [job id to wait for]
	t=$1; dep=$2
	[ -d "$(deploy_dir "${t}")" ] && echo "   note: $(deploy_dir "${t}") exists and is overwritten"
	run "deploy ${ARM} on ${t} + held-out healthy" env TRAIN_SUFFIX="_loco_${t}${SUFFIX}" EXPERIMENT_SUFFIX="${SUFFIX}" \
		sbatch --parsable --job-name="loco_dep_${ARM}_${t}" ${dep:+--dependency=afterok:${dep}} \
		slurm/external_validation.sh "${ARM}" "cris_loco_${t}"
}

status() {
	for t in ${TYPES}; do
		n=$(ls "$(search_dir "${t}")"/summary_*.pkl 2>/dev/null | wc -l)
		if [ -f "$(deploy_dir "${t}")/comprehensive_validation_summary.pkl" ]; then d=done
		elif [ -d "$(deploy_dir "${t}")" ]; then d=partial; else d=-; fi
		printf "  %-4s %-11s search %2s/%s configs   deploy %s\n" "${ARM}" "${t}" "${n}" "${N_CONFIGS}" "${d}"
	done
}

case "${STEP}" in
	search) for t in ${TYPES}; do submit_search "${t}"; done ;;
	deploy) for t in ${TYPES}; do submit_deploy "${t}" ""; done ;;
	all)    for t in ${TYPES}; do submit_search "${t}"; submit_deploy "${t}" "${SEARCH_JOB}"; done ;;
	status) status; exit 0 ;;
	*) echo "usage: $0 search|deploy|all|status   (ARM=dd|rna|atac TYPES=... DRY_RUN=1)" >&2; exit 1 ;;
esac
echo "done: ${STEP}$([ -n "${DRY_RUN}" ] && echo ' - dry run' || echo "; job ids in ${JOBLOG}")"
