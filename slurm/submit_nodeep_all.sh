#!/bin/sh
# ============================================================================
# The complete no_deep rerun: every Cristiano-cohort experiment repeated
# without the 30 deep-sequenced healthy samples
# (accessory_files/cris_exclude_deep_healthy.txt), with the directory names
# that src/figures/export/_common.py (COHORT_VARIANT=no_deep) expects.
#
# Three phases, because some runs need the output of others:
#   phase1   everything that only needs features: fully nested dd (both
#            tasks), dd/rna/atac k-sweeps, size sweeps (+ seed replicates),
#            depth arms (1x, 0.1x), rna/atac per-classifier runs, best-model
#            searches (PANCAN dd/rna/atac + BRCA/CRC dd), within-dataset
#            generalization (dd/rna/atac). 33 submissions; the nested
#            data-driven runs take 1-3 days each.
#   phase2   after phase1 has finished: external validation on Jiang (dd/rna/
#            atac + clinical mode) from the no_deep best-model searches,
#            cross-depth evaluation of the no_deep nested runs, label shuffle.
#   phase3   exports for COHORT_VARIANT=no_deep on a compute node
#            (slurm/export_figures.sh), then the notebooks (printed).
# Covariate panels are NOT here: slurm/submit_scalar_panels.sh no_deep.
#
# Usage (repository root):  ./slurm/submit_nodeep_all.sh phase1|phase2|phase3
#   DRY_RUN=1 prints every submission without running it.
# Resumable: nested_full fold partials that exist are skipped by model_hpc.py
# (the local fold 1 of brca_crc_nested_full_nodeep is reused). Result
# directories of the other modes are overwritten by a rerun.
# ============================================================================

. slurm/_site.sh   # site settings: slurm/site.env (scheduler, storage paths, container, Apptainer cache)
PHASE=${1:-}
cd "$(dirname "$0")/.." || exit 1
EXCL=accessory_files/cris_exclude_deep_healthy.txt
[ -f "${EXCL}" ] || { echo "${EXCL} missing - run python src/comp/build_deep_healthy_exclusion.py" >&2; exit 1; }
MANIFEST=accessory_files/cris_manifest.csv
mkdir -p logs
N=0

submit() {   # submit <description> <command...>   (environment set by the caller)
	desc=$1; shift
	N=$((N + 1))
	echo ">> ${desc}"
	echo "   $* | EXPERIMENT_NAME=${EXPERIMENT_NAME:-} CANCER_TYPES=${CANCER_TYPES:-} FEATURE_SET=${FEATURE_SET:-} FEATURES_DIR=${FEATURES_DIR:-}${EXPERIMENT_SUFFIX:+ EXPERIMENT_SUFFIX=${EXPERIMENT_SUFFIX}}${TRAIN_SUFFIX:+ TRAIN_SUFFIX=${TRAIN_SUFFIX}}${SUBSAMPLE_SEED:+ SUBSAMPLE_SEED=${SUBSAMPLE_SEED}}${LABEL_SHUFFLE_MODEL_DIR:+ LABEL_SHUFFLE_MODEL_DIR=${LABEL_SHUFFLE_MODEL_DIR}}"
	if [ -z "${DRY_RUN}" ]; then
		"$@" || { echo "FAILED: ${desc}" >&2; exit 1; }
	fi
}

phase1() {
	export EXCLUDE_SAMPLES=${EXCL} METADATA_FILE=${MANIFEST}
	for task in pancan brca_crc; do   # PANCAN first
		if [ "${task}" = pancan ]; then CT=PANCAN; else CT="breast colorectal"; fi
		# fully nested data-driven run (10 folds + aggregate)
		( export CANCER_TYPES="${CT}" EXPERIMENT_NAME=${task}_nested_full_nodeep
		  submit "nested_full dd ${task}" ./slurm/submit_nested_full.sh full ) || exit 1
		# k-sweeps: data-driven, RNA, ATAC (Fig S13; Fig 4-type curves)
		# (skip with SKIP_DD_KSWEEP=1 when REFERENCE=1 ./slurm/submit_covariate_arms.sh already submitted it)
		[ -n "${SKIP_DD_KSWEEP}" ] || ( export CANCER_TYPES="${CT}" EXPERIMENT_NAME=${task}_nested_full_nodeep
		  submit "ksweep dd ${task}" ./slurm/submit_nested_full.sh ksweep ) || exit 1
		( export FEATURE_SET=rna CANCER_TYPES="${CT}" EXPERIMENT_NAME=${task}_rna_nested_full_nodeep
		  submit "ksweep rna ${task}" ./slurm/submit_nested_full.sh ksweep ) || exit 1
		( export FEATURE_SET=atac CANCER_TYPES="${CT}" EXPERIMENT_NAME=${task}_atac_nested_full_nodeep
		  submit "ksweep atac ${task}" ./slurm/submit_nested_full.sh ksweep ) || exit 1
		# training-cohort size sweep (Fig S9) + the three replicate seeds for the small fractions
		( export CANCER_TYPES="${CT}" EXPERIMENT_NAME=${task}_nested_full_nodeep
		  submit "sizesweep ${task}" ./slurm/submit_nested_full.sh sizesweep ) || exit 1
		for seed in 43 44 45; do
			( export CANCER_TYPES="${CT}" SUBSAMPLE_SEED=${seed} SIZESWEEP_FRACTIONS="0.4 0.3 0.2 0.1" \
			         EXPERIMENT_NAME=${task}_nested_full_seed${seed}_nodeep
			  submit "sizesweep ${task} seed ${seed}" ./slurm/submit_nested_full.sh sizesweep ) || exit 1
		done
		# depth arms (Fig S8): the same nested run on features thinned to 1x and 0.1x
		for depth in 1 0.1; do
			( export CANCER_TYPES="${CT}" FEATURES_DIR=extracted_features/downsampled/cov${depth}x_seed42/biomart_10kb \
			         EXPERIMENT_NAME=${task}_nested_full_cov${depth}x_nodeep
			  submit "depth ${depth}x ${task}" ./slurm/submit_nested_full.sh full ) || exit 1
		done
		# RNA / ATAC per-classifier nested CV (Fig 3 / Fig 5a comparators)
		( export CANCER_TYPES="${CT}" EXPERIMENT_NAME=${task}_rna_rev_nodeep
		  submit "rna ${task}" sbatch slurm/nested_cv_rna.sh ) || exit 1
		( export CANCER_TYPES="${CT}" EXPERIMENT_NAME=${task}_atac_rev_nodeep
		  submit "atac ${task}" sbatch slurm/nested_cv_atac.sh ) || exit 1
	done
	# best-model searches: PANCAN deployment models (Fig 5b/c, j-l, S7) and the BRCA/CRC dd search (label shuffle, Fig S5)
	( export CANCER_TYPES=PANCAN EXPERIMENT_NAME=best_model_search_dd_rev_nodeep CORES=16
	  submit "best_model_search dd PANCAN" sbatch --mem=96G slurm/best_model_search.sh cris dd ) || exit 1
	( export CANCER_TYPES=PANCAN EXPERIMENT_NAME=best_model_search_rna_rev_nodeep
	  submit "best_model_search rna PANCAN" sbatch slurm/best_model_search.sh cris rna ) || exit 1
	( export CANCER_TYPES=PANCAN EXPERIMENT_NAME=best_model_search_atac_rev_nodeep
	  submit "best_model_search atac PANCAN" sbatch slurm/best_model_search.sh cris atac ) || exit 1
	( export CANCER_TYPES="breast colorectal" EXPERIMENT_NAME=best_model_search_brca_crc_rev_nodeep CORES=16
	  submit "best_model_search dd BRCA/CRC" sbatch --mem=96G slurm/best_model_search.sh cris dd ) || exit 1
	# (the within-dataset generalization of the submitted Fig 5d-f was replaced by leave one cancer type out,
	#  slurm/submit_loco.sh; its wrapper is no longer part of the repository)
}

phase2() {
	for d in best_model_best_model_search_dd_rev_nodeep best_model_best_model_search_rna_rev_nodeep \
	         best_model_best_model_search_atac_rev_nodeep best_model_best_model_search_brca_crc_rev_nodeep \
	         nested_full_brca_crc_nested_full_nodeep nested_full_pancan_nested_full_nodeep; do
		[ -d "nested_cv_results_cris/${d}" ] || echo "WARNING: nested_cv_results_cris/${d} not found - phase1 not finished?"
	done
	# external validation on Jiang from the no_deep deployment models (Fig 5b/c, S7, 5j-l)
	for ft in dd rna atac; do
		( export TRAIN_SUFFIX=_rev_nodeep EXPERIMENT_SUFFIX=_rev_nodeep
		  submit "external validation ${ft}" sbatch slurm/external_validation.sh ${ft} jiang ) || exit 1
	done
	( export TRAIN_SUFFIX=_rev_nodeep EXPERIMENT_SUFFIX=_rev_nodeep
	  submit "external validation, clinical mode" sbatch slurm/external_validation_clinical.sh jiang ) || exit 1
	# cross-depth evaluation (Fig S8 dashed arms): the no_deep full-depth models on thinned test folds
	for task in brca_crc pancan; do
		if [ "${task}" = pancan ]; then CT=PANCAN; else CT="breast colorectal"; fi
		( export EXCLUDE_SAMPLES=${EXCL} METADATA_FILE=${MANIFEST} CANCER_TYPES="${CT}" EXPERIMENT_NAME=${task}_nested_full_nodeep
		  submit "cross-depth ${task}" sbatch slurm/cross_depth_eval.sh ) || exit 1
	done
	# (label permutation: the complete nested procedure with LABEL_PERMUTATION_SEED, slurm/submit_nested_full.sh;
	#  the fixed-configuration shuffle it replaced is no longer part of the repository)
}

phase3() {
	E=src/figures/export
	export COHORT_VARIANT=no_deep
	for s in export_fig3.py export_fig5.py export_fig4.py export_figS5_nested.py export_figS8.py export_figS9.py export_figS13.py; do
		submit "export ${s}" sbatch slurm/export_figures.sh ${E}/${s} || exit 1
	done
	for task in brca_crc pancan; do
		submit "export baselines ${task}" sbatch slurm/export_figures.sh ${E}/export_fig3_baselines.py --task ${task} || exit 1
	done
	echo ""
	echo "When the export jobs are done, execute the figure notebooks one at a time from src/figures"
	echo "  (python run_nb.py <notebook>.ipynb; see src/figures/README.md)."
}

case "${PHASE}" in
	phase1) phase1 ;;
	phase2) phase2 ;;
	phase3) phase3 ;;
	*) echo "usage: $0 phase1|phase2|phase3   (DRY_RUN=1 to list only)" >&2; exit 1 ;;
esac
echo "done: ${PHASE}$([ -n "${DRY_RUN}" ] && echo ' (dry run)')"
