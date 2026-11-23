#!/bin/bash
# ============================================================================
# Submit the fully nested CV as a 10-task array + a dependent aggregation job.
# Data-driven TSS features only; all six classifiers (Table S3 + XGBoost) and
# all four ranking functions are searched in the inner loop. Samples come from
# accessory_files/cris_manifest.csv (role == 'model').
# ============================================================================
# Usage:
#   ./slurm/submit_nested_full.sh estimate   # cost estimate only, no fitting (seconds)
#   ./slurm/submit_nested_full.sh test       # reduced grid correctness run (1 feature, 3 k values)
#   ./slurm/submit_nested_full.sh full       # the real run
#   ./slurm/submit_nested_full.sh ksweep     # leakage-free AUROC-vs-k trend (+ ksweep_tuned/)
#   ./slurm/submit_nested_full.sh sizesweep  # sample-size robustness sweep (reviewer)
#
# Comparator region sweep (reviewer: "the same region number sweep should be
# performed for all models"): FEATURE_SET=rna|atac runs the identical fully
# nested procedure, k grid included, on the Zhu RNA regions (792, feature rcov)
# or the Ulz ATAC TF features (504). The ksweep aggregate writes the
# AUROC-vs-k curve and derives ksweep_tuned/, the arm in which k is tuned per
# fold exactly like the data-driven model's. Six array jobs cover the figure:
#   FEATURE_SET=rna  ./slurm/submit_nested_full.sh ksweep
#   FEATURE_SET=atac ./slurm/submit_nested_full.sh ksweep
#   CANCER_TYPES=PANCAN FEATURE_SET=rna  EXPERIMENT_NAME=pancan_rna_nested_full  ./slurm/submit_nested_full.sh ksweep
#   CANCER_TYPES=PANCAN FEATURE_SET=atac EXPERIMENT_NAME=pancan_atac_nested_full ./slurm/submit_nested_full.sh ksweep
#   CANCER_TYPES=PANCAN EXPERIMENT_NAME=pancan_nested_full ./slurm/submit_nested_full.sh ksweep   # data-driven pancan curve
# (the brca_crc data-driven ksweep already exists). Then
#   python src/figures/export/export_figS13.py  ->  src/figures/figS13.ipynb
#
# Task selection via the environment (defaults: brca_crc):
#   BRCA+CRC : ./slurm/submit_nested_full.sh sizesweep
#   PANCAN   : CANCER_TYPES=PANCAN EXPERIMENT_NAME=pancan_nested_full \
#              SIZESWEEP_FRACTIONS="1.0 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1" \
#              ./slurm/submit_nested_full.sh sizesweep
#   Jiang    : trained AND evaluated within the Jiang cohort (HCC vs the 135
#              non-HCC comparators; reviewers 1-3, signal vs domain shift):
#              DATASET=jiang METADATA_FILE=accessory_files/jiang_manifest.csv \
#              EXPERIMENT_NAME=jiang_hcc_nested_full ./slurm/submit_nested_full.sh full
#              Secondary task, HCC vs the 32 healthy only: add CONTROL_GROUPS=Healthy
#              and EXPERIMENT_NAME=jiang_hcc_vs_healthy_nested_full.
#   (PANCAN's 1.0 arm embeds the 100% reference in the sweep itself. The exporter
#    can also use the standalone fully nested PANCAN run as the 100% point, so 1.0 may be dropped from a resubmission UNLESS
#    sizesweep_f100 partials already exist - then keep it so they are resumed
#    rather than orphaned. brca_crc reuses nested_full_brca_crc_nested_full.)
#
# Resubmitting the same array resumes it: a fold (full) or a per-k partial (ksweep,
# sizesweep) that already exists for the same fold, cohort and search grid is skipped
# (model_hpc prints "already exists - skipping"); delete a partial to force it.
# The same experiment may also run on two clusters at once (same storage, one site file
# each): every fold / per-k partial / aggregate is claimed through a lock file next
# to its output, a live claim is skipped (and waited for at the end), a claim without
# heartbeat for 20 min (killed job) is taken over.
# The aggregation job is submitted with --dependency=afterok on the array, so it
# starts by itself once all ten folds have finished.
# ============================================================================

. slurm/_site.sh   # site settings: slurm/site.env (scheduler, storage paths, container, Apptainer cache)
STEP=${1:-test}
case "${FEATURE_SET:-dd}" in
	dd)   DEFAULT_EXPERIMENT=brca_crc_nested_full ;;
	rna)  DEFAULT_EXPERIMENT=brca_crc_rna_nested_full ;;
	atac) DEFAULT_EXPERIMENT=brca_crc_atac_nested_full ;;
	*)    echo "Unknown FEATURE_SET '${FEATURE_SET}' (expected dd | rna | atac)"; exit 1 ;;
esac
RESULTS_ROOT="nested_cv_results_${DATASET:-cris}/nested_full_${EXPERIMENT_NAME:-${DEFAULT_EXPERIMENT}}"

case "${STEP}" in
	estimate)
		echo "Submitting cost-estimate job (no fitting)..."
		JOB=$(sbatch --parsable slurm/nested_full.sh estimate) || { echo "sbatch failed for ${EXPERIMENT_NAME} (see the error above) - nothing submitted" >&2; exit 1; }
		echo "  Job ID: ${JOB}"
		echo "  Output: ${JOB}_*_nested_full.out"
		;;

	test)
		echo "Submitting REDUCED correctness run: griffin_diff only, k in {25,100,250}, all six classifiers."
		ARRAY_JOB=$(sbatch --parsable --array=1-10 slurm/nested_full.sh test) || { echo "sbatch failed for ${EXPERIMENT_NAME} (see the error above) - nothing submitted" >&2; exit 1; }
		echo "  Array job ID: ${ARRAY_JOB} (10 tasks, one per outer fold)"
		AGG_JOB=$(sbatch --parsable --dependency=afterok:${ARRAY_JOB} slurm/nested_full.sh test_aggregate) || { echo "sbatch failed for ${EXPERIMENT_NAME} (see the error above) - nothing submitted" >&2; exit 1; }
		echo "  Aggregation job ID: ${AGG_JOB} (waits for all 10 folds)"
		echo ""
		echo "Results will land in ${RESULTS_ROOT}_test/"
		;;

	full)
		echo "Submitting FULL run: 5 features x 4 ranking functions x 12 k values x 6 classifiers."
		echo "  cohort=${DATASET:-cris} metadata=${METADATA_FILE:-accessory_files/cris_manifest.csv}"
		echo "  experiment=${EXPERIMENT_NAME:-brca_crc_nested_full} control_groups=${CONTROL_GROUPS:-<cohort default>}"
		ARRAY_JOB=$(sbatch --parsable --array=1-10 slurm/nested_full.sh) || { echo "sbatch failed for ${EXPERIMENT_NAME} (see the error above) - nothing submitted" >&2; exit 1; }
		echo "  Array job ID: ${ARRAY_JOB} (10 tasks, one per outer fold)"
		AGG_JOB=$(sbatch --parsable --dependency=afterok:${ARRAY_JOB} slurm/nested_full.sh aggregate) || { echo "sbatch failed for ${EXPERIMENT_NAME} (see the error above) - nothing submitted" >&2; exit 1; }
		echo "  Aggregation job ID: ${AGG_JOB} (waits for all 10 folds)"
		echo ""
		echo "Results will land in ${RESULTS_ROOT}/"
		;;

	ksweep)
		echo "Submitting k-SWEEP: one complete nested selection per k value."
		echo "  region set = ${FEATURE_SET:-dd}, cohort=${DATASET:-cris}, cancer types=${CANCER_TYPES:-breast colorectal}"
		echo "  Feature, ranking function, classifier and hyperparameters are still chosen"
		echo "  inside each fold; only k is held fixed, so every point of the trend is leakage-free."
		ARRAY_JOB=$(sbatch --parsable --array=1-10 slurm/nested_full.sh ksweep) || { echo "sbatch failed for ${EXPERIMENT_NAME} (see the error above) - nothing submitted" >&2; exit 1; }
		echo "  Array job ID: ${ARRAY_JOB} (10 tasks, one per outer fold, each looping all k)"
		AGG_JOB=$(sbatch --parsable --dependency=afterok:${ARRAY_JOB} slurm/nested_full.sh ksweep_aggregate) || { echo "sbatch failed for ${EXPERIMENT_NAME} (see the error above) - nothing submitted" >&2; exit 1; }
		echo "  Aggregation job ID: ${AGG_JOB} (waits for all 10 folds)"
		echo ""
		echo "Results will land in ${RESULTS_ROOT}_ksweep/"
		echo "  per k       : ksweep_k<k>/nested_full_per_fold.csv"
		echo "  the trend   : nested_full_ksweep.csv"
		echo "  k tuned     : ksweep_tuned/ (per fold the k with the best inner score;"
		echo "                = a fully nested run over the whole k grid, no extra compute)"
		;;

	sizesweep)
		echo "Submitting SAMPLE-SIZE SWEEP: one complete fully nested selection per fraction"
		echo "  (${SIZESWEEP_FRACTIONS:-0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1}), each on a nested stratified subset."
		echo "  Feature, ranking function, k, classifier and hyperparameters are all still"
		echo "  chosen inside each outer fold, so every point of the curve is leakage-free."
		echo "  Resubmitting after a partial run is safe: existing (fraction, fold)"
		echo "  partials are detected and skipped, only the missing work is computed."
		ARRAY_JOB=$(sbatch --parsable --array=1-10 slurm/nested_full.sh sizesweep) || { echo "sbatch failed for ${EXPERIMENT_NAME} (see the error above) - nothing submitted" >&2; exit 1; }
		echo "  Array job ID: ${ARRAY_JOB} (10 tasks, one per outer fold, each looping all fractions)"
		# afterany, not afterok: the sizesweep aggregate skips incomplete fractions
		# with a warning, so even if some fold tasks time out it still produces the
		# sweep CSV for the fractions that did finish (inspectable intermediate
		# results; resubmit sizesweep later to fill the gaps).
		AGG_JOB=$(sbatch --parsable --dependency=afterany:${ARRAY_JOB} slurm/nested_full.sh sizesweep_aggregate) || { echo "sbatch failed for ${EXPERIMENT_NAME} (see the error above) - nothing submitted" >&2; exit 1; }
		echo "  Aggregation job ID: ${AGG_JOB} (runs once the array ends, aggregates completed fractions)"
		echo ""
		echo "Results will land in ${RESULTS_ROOT}_sizesweep/"
		echo "  per fraction : sizesweep_f<pct>/nested_full_per_fold.csv (+ oof predictions)"
		echo "  the trend    : nested_full_sizesweep.csv"
		;;

	*)
		echo "Unknown step '${STEP}'. Use: estimate | test | full | ksweep | sizesweep"
		exit 1
		;;
esac

echo ""
echo "Monitor with:  squeue -u \$USER"
