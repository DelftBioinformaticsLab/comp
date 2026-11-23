#!/bin/sh
#SBATCH --time=23:59:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64GB
#SBATCH --mail-type=END
#SBATCH --job-name=nested_full
#SBATCH --output=logs/%A_%a_nested_full.out
#SBATCH --error=logs/%A_%a_nested_full.out

# ============================================================================
# FULLY NESTED CV - one outer fold per array task (data-driven features only)
# ============================================================================
# Feature type, ranking function, top_genes and the classifier hyperparameters are
# ALL chosen inside each outer fold. The ten outer folds are independent, so this
# runs as a 10-task array and a dependent aggregation step.
#
# Usage (normally via ./slurm/submit_nested_full.sh):
#   sbatch --array=1-10 slurm/nested_full.sh          # the ten outer folds
#   sbatch slurm/nested_full.sh aggregate             # combine the partials
#   sbatch slurm/nested_full.sh estimate              # cost estimate, no fitting
#   sbatch --array=1-10 slurm/nested_full.sh test     # reduced grid, correctness check
#   sbatch --array=1-10 slurm/nested_full.sh sizesweep  # sample-size robustness sweep
#   sbatch slurm/nested_full.sh refresh_thresholds    # re-derive stored thresholds + aggregate
#
# Sensitivity variant without the 30 deep-sequenced healthy samples (rebuttal,
# Fig S10): EXCLUDE_SAMPLES=accessory_files/cris_exclude_deep_healthy.txt and an
# EXPERIMENT_NAME ending in _nodeep, e.g.
#   EXCLUDE_SAMPLES=accessory_files/cris_exclude_deep_healthy.txt \
#   EXPERIMENT_NAME=brca_crc_nested_full_nodeep ./slurm/submit_nested_full.sh full
# Workstation (no slurm/apptainer): MODEL_HPC_LOCAL=1 CORES=16 and run this file
# directly with FOLD=<1..10> for each outer fold, then "aggregate" - see
# slurm/run_nested_full_local.sh, which loops over the folds.
#
# $1 : "" (fold, uses SLURM_ARRAY_TASK_ID) | aggregate | estimate | test
#      | ksweep | ksweep_aggregate | sizesweep | sizesweep_aggregate
#      | refresh_thresholds | ksweep_refresh_thresholds | sizesweep_refresh_thresholds
#
# Region set (FEATURE_SET, default dd): the SAME fully nested procedure - k grid
# included - on the comparator methods' own region sets (reviewer: the number of
# TSSs was tuned as a hyperparameter while the comparators used fixed region
# counts). FEATURE_SET=rna runs it on the 792 Zhu et al. regions (feature locked
# to the manuscript's rcov), FEATURE_SET=atac on the 504 Ulz TF-level features;
# the k grid is capped at the region set's width (last value = all regions, i.e.
# the manuscript's fixed-region comparator). Use with ksweep: the aggregate
# writes the AUROC-vs-k curve AND derives the "k tuned per fold" arm
# (ksweep_tuned/), which equals a fully nested run over the whole k grid.
#   FEATURE_SET=rna  ./slurm/submit_nested_full.sh ksweep     # brca_crc
#   FEATURE_SET=atac ./slurm/submit_nested_full.sh ksweep
#   CANCER_TYPES=PANCAN FEATURE_SET=rna  EXPERIMENT_NAME=pancan_rna_nested_full \
#       ./slurm/submit_nested_full.sh ksweep                 # pancan
#
# Other cohorts: the same fully nested procedure trained and evaluated WITHIN the
# Jiang cohort (reviewers 1-3: is the external-cohort gap a lack of signal or a
# domain shift?). The metadata file name selects the cohort branch in model_hpc.py;
# the manifest (role column) drives sample selection with exact-ID file matching:
#   DATASET=jiang METADATA_FILE=accessory_files/jiang_manifest.csv \
#   EXPERIMENT_NAME=jiang_hcc_nested_full ./slurm/submit_nested_full.sh full
# Class 0 defaults to all 135 non-HCC comparators (healthy + CHB + cirrhosis);
# CONTROL_GROUPS="Healthy" restricts it (HCC vs the 32 healthy only).
# ============================================================================

STEP=${1:-fold}
DATASET=${DATASET:-cris}
FEATURE_SET=${FEATURE_SET:-dd}
CANCER_TYPES=${CANCER_TYPES:-"breast colorectal"}
OUTPUT_DIR=${OUTPUT_DIR:-nested_cv_results_${DATASET}}
# Optional class-0 restriction (--control_groups); empty = the cohort's default.
# Word-split into the argument list, so only labels without spaces can be passed
# this way (Healthy, Cirrhosis); the default already includes "Hepatitis B".
CONTROL_GROUPS=${CONTROL_GROUPS:-}

# Data-driven TSS features only. RNA and ATAC are not part of this experiment.
# Override FEATURES_DIR for the depth-downsampling arms, e.g.:
#   FEATURES_DIR=extracted_features/downsampled/cov0.1x_seed42/biomart_10kb \
#   EXPERIMENT_NAME=brca_crc_nested_full_cov0.1x ./slurm/submit_nested_full.sh full
# (sbatch exports the submission environment, so env set at submit time reaches the job)
FEATURES_DIR=${FEATURES_DIR:-extracted_features/biomart_10kb}

# Comparator region sets (FEATURE_SET=rna|atac): the inputs of nested_cv_rna.sh /
# nested_cv_atac.sh, now with region selection inside the nested procedure. The
# k grids end at the region set's width (RNA: 792 Zhu regions; ATAC: 504 TFs),
# so the last point of each sweep is the fixed all-regions comparator; K_GRID
# overrides. RNA_FEATURES widens the RNA arm beyond the manuscript's rcov (e.g.
# "rcov fslr griffin_diff mwh fslr_coverage" gives the Zhu regions the same
# feature-type choice the data-driven arm has).
RNA_FEATURES_DIR=${RNA_FEATURES_DIR:-extracted_features/zhu_all_after_lift}
RNA_FEATURES=${RNA_FEATURES:-rcov}
ATAC_MATRIX=${ATAC_MATRIX:-extracted_features/ulz_atac_tfbs_features.npy}
ATAC_METADATA=${ATAC_METADATA:-extracted_features/ulz_atac_tfbs_features.metadata.csv}
case "${FEATURE_SET}" in
	dd)
		DEFAULT_EXPERIMENT=brca_crc_nested_full
		FEATURE_ARGS="--features_dir ${FEATURES_DIR}"
		TEST_FEATURE_ARGS="--nested_full_features griffin_diff"
		K_GRID=${K_GRID:-}     # empty = model_hpc.py's 12-value grid (10 .. 15000)
		TEST_K_GRID="25 100 250"
		;;
	rna)
		DEFAULT_EXPERIMENT=brca_crc_rna_nested_full
		FEATURE_ARGS="--no_gene_selection --features_dir ${RNA_FEATURES_DIR} --nested_full_features ${RNA_FEATURES}"
		TEST_FEATURE_ARGS=""
		K_GRID=${K_GRID:-"10 25 50 100 250 500 792"}
		TEST_K_GRID="25 100 792"
		;;
	atac)
		DEFAULT_EXPERIMENT=brca_crc_atac_nested_full
		FEATURE_ARGS="--predefined_feature_matrix ${ATAC_MATRIX} --predefined_metadata_csv ${ATAC_METADATA}"
		TEST_FEATURE_ARGS=""
		K_GRID=${K_GRID:-"10 25 50 100 250 504"}
		TEST_K_GRID="25 100 504"
		;;
	*)
		echo "Unknown FEATURE_SET '${FEATURE_SET}' (expected dd | rna | atac)" >&2
		exit 1
		;;
esac
EXPERIMENT_NAME=${EXPERIMENT_NAME:-${DEFAULT_EXPERIMENT}}
# Another region set with the TSS layout: FEATURES_DIR points at its extraction and
# BLACKLIST_INDICES at its own exclusion file.
if [ -n "${BLACKLIST_INDICES}" ]; then
	FEATURE_ARGS="${FEATURE_ARGS} --blacklist_indices ${BLACKLIST_INDICES}"
fi

# Ranking functions searched: the full manuscript set (Table S2). p_value and weighted
# run mannwhitneyu per gene (14,224 calls per ranking fit) and, with the pipeline cache
# disabled, that is recomputed for every candidate sharing it — the reason --time above
# is generous. Trim via the environment for a quick turnaround, e.g.:
#   RANKING_FUNCTIONS="mean_diff fold_change" sbatch --array=1-10 slurm/nested_full.sh
RANKING_FUNCTIONS=${RANKING_FUNCTIONS:-"mean_diff fold_change p_value weighted"}

# Sample-size sweep (reviewer: minimal training-cohort size). Subsets are nested
# prefixes of one seeded per-stratum shuffle, so fractions may be split across
# submissions (e.g. run "0.9 0.8 0.7 0.6 0.5" and "0.4 0.3 0.2 0.1" separately if a
# task would exceed the walltime) and still describe the same cohorts. Include 1.0
# to embed the full-cohort reference in the same sweep (needed for PANCAN, which has
# no standalone fully nested run).
SIZESWEEP_FRACTIONS=${SIZESWEEP_FRACTIONS:-"0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1"}
SUBSAMPLE_SEED=${SUBSAMPLE_SEED:-42}

MODE_ARGS="--run_nested_cv_full ${FEATURE_ARGS}"
MODE_ARGS="${MODE_ARGS} --nested_full_selection_types ${RANKING_FUNCTIONS}"
# Free-form extra arguments (e.g. NESTED_FULL_EXTRA="--nested_full_features griffin_diff"
# for a reduced search; used by slurm/run_nodeep_local.sh NESTED_REDUCED=1).
# label-permutation null of the complete procedure (Fig S5): one seed per experiment name
if [ -n "${LABEL_PERMUTATION_SEED}" ]; then
	MODE_ARGS="${MODE_ARGS} --nested_full_permute_labels ${LABEL_PERMUTATION_SEED}"
fi
if [ -n "${NESTED_FULL_EXTRA}" ]; then
	MODE_ARGS="${MODE_ARGS} ${NESTED_FULL_EXTRA}"
fi
if [ -n "${CONTROL_GROUPS}" ]; then
	MODE_ARGS="${MODE_ARGS} --control_groups ${CONTROL_GROUPS}"
fi
# Covariate block (rebuttal, Fig S14): COVARIATE_STEM=extracted_features/scalar_panel_<panel>[_pancan]
# with COVARIATE_MODE=augment (default; TSS + covariates in one model, covariates bypass the
# ranking) or residualize (covariates regressed out of every TSS inside the training fold).
# Submitted for all arms by slurm/submit_covariate_arms.sh.
if [ -n "${COVARIATE_STEM}" ]; then
	case "${COVARIATE_MODE:-augment}" in
		augment)     MODE_ARGS="${MODE_ARGS} --augment_covariates ${COVARIATE_STEM}" ;;
		residualize) MODE_ARGS="${MODE_ARGS} --residualize_covariates ${COVARIATE_STEM}" ;;
		*) echo "COVARIATE_MODE must be augment or residualize" >&2; exit 1 ;;
	esac
fi

case "${STEP}" in
	estimate)
		MODE_ARGS="${MODE_ARGS} --nested_full_estimate_only"
		;;
	aggregate)
		MODE_ARGS="${MODE_ARGS} --nested_full_aggregate"
		;;
	test)
		# Cheap correctness run: one feature, reduced k grid, still all 4 ranking functions.
		# (The comparator region sets are single-feature already; only the k grid shrinks.)
		EXPERIMENT_NAME="${EXPERIMENT_NAME}_test"
		MODE_ARGS="${MODE_ARGS} --nested_full_only_fold ${SLURM_ARRAY_TASK_ID:-${FOLD:-1}}"
		MODE_ARGS="${MODE_ARGS} ${TEST_FEATURE_ARGS}"
		K_GRID="${TEST_K_GRID}"
		;;
	test_aggregate)
		EXPERIMENT_NAME="${EXPERIMENT_NAME}_test"
		MODE_ARGS="${MODE_ARGS} --nested_full_aggregate"
		MODE_ARGS="${MODE_ARGS} ${TEST_FEATURE_ARGS}"
		K_GRID="${TEST_K_GRID}"
		;;
	ksweep)
		# One complete nested selection per k, so every point of the k trend is
		# leakage-free. The k loop runs inside each task, so the feature matrices are
		# loaded 10 times (once per outer fold) rather than 120 times. The aggregate
		# also derives ksweep_tuned/ (per fold the k with the best inner score), which
		# is the fully nested k-as-hyperparameter arm - for FEATURE_SET=rna|atac the
		# comparator "with the same region-number tuning as the data-driven model".
		EXPERIMENT_NAME="${EXPERIMENT_NAME}_ksweep"
		MODE_ARGS="${MODE_ARGS} --nested_full_ksweep"
		MODE_ARGS="${MODE_ARGS} --nested_full_only_fold ${SLURM_ARRAY_TASK_ID:-${FOLD:-1}}"
		;;
	ksweep_aggregate)
		EXPERIMENT_NAME="${EXPERIMENT_NAME}_ksweep"
		MODE_ARGS="${MODE_ARGS} --nested_full_ksweep --nested_full_aggregate"
		;;
	sizesweep)
		# One complete nested selection per sample fraction (nested stratified
		# subsets). The fraction loop runs inside each task; a task whose fold index
		# exceeds a small fraction's outer fold count skips that fraction.
		EXPERIMENT_NAME="${EXPERIMENT_NAME}_sizesweep"
		MODE_ARGS="${MODE_ARGS} --nested_full_sizesweep"
		MODE_ARGS="${MODE_ARGS} --nested_full_only_fold ${SLURM_ARRAY_TASK_ID:-${FOLD:-1}}"
		MODE_ARGS="${MODE_ARGS} --nested_full_sizesweep_fractions ${SIZESWEEP_FRACTIONS}"
		MODE_ARGS="${MODE_ARGS} --nested_full_subsample_seed ${SUBSAMPLE_SEED}"
		;;
	sizesweep_aggregate)
		EXPERIMENT_NAME="${EXPERIMENT_NAME}_sizesweep"
		MODE_ARGS="${MODE_ARGS} --nested_full_sizesweep --nested_full_aggregate"
		MODE_ARGS="${MODE_ARGS} --nested_full_sizesweep_fractions ${SIZESWEEP_FRACTIONS}"
		MODE_ARGS="${MODE_ARGS} --nested_full_subsample_seed ${SUBSAMPLE_SEED}"
		;;
	refresh_thresholds)
		# Stored fold partials whose operating-point thresholds predate the inner-CV
		# derivation: re-derive them from each fold's stored winning configuration
		# (1 + n_inner fits per fold, no new inner search) and re-aggregate. Use the
		# environment of the original submission (CANCER_TYPES, EXPERIMENT_NAME,
		# FEATURES_DIR, FEATURE_SET, DATASET, ...).
		MODE_ARGS="${MODE_ARGS} --nested_full_refresh_thresholds --nested_full_aggregate"
		;;
	ksweep_refresh_thresholds)
		EXPERIMENT_NAME="${EXPERIMENT_NAME}_ksweep"
		MODE_ARGS="${MODE_ARGS} --nested_full_refresh_thresholds"
		MODE_ARGS="${MODE_ARGS} --nested_full_ksweep --nested_full_aggregate"
		;;
	sizesweep_refresh_thresholds)
		EXPERIMENT_NAME="${EXPERIMENT_NAME}_sizesweep"
		MODE_ARGS="${MODE_ARGS} --nested_full_refresh_thresholds"
		MODE_ARGS="${MODE_ARGS} --nested_full_sizesweep --nested_full_aggregate"
		MODE_ARGS="${MODE_ARGS} --nested_full_sizesweep_fractions ${SIZESWEEP_FRACTIONS}"
		MODE_ARGS="${MODE_ARGS} --nested_full_subsample_seed ${SUBSAMPLE_SEED}"
		;;
	*)
		# One outer fold per array task (FOLD=<n> when run outside slurm).
		MODE_ARGS="${MODE_ARGS} --nested_full_only_fold ${SLURM_ARRAY_TASK_ID:-${FOLD:-1}}"
		# NOTE: no --nested_full_cache_dir here. Pipeline(memory=...) makes joblib hash
		# TopGeneSelector on every fit, and because model_hpc.py runs as a script the
		# class lives in '__main__', which the parallel workers cannot resolve -> every
		# fit fails with PicklingError. The ranking is recomputed per candidate instead
		# (the slow p_value/weighted rankings are memoised in-process by
		# select_top_genes, so this costs ~ms per candidate rather than ~5 s).
		;;
esac
# k grid last, so the test steps' reduced grid replaces the region set's default.
if [ -n "${K_GRID}" ]; then
	MODE_ARGS="${MODE_ARGS} --nested_full_top_genes ${K_GRID}"
fi

echo "============================================"
echo "Fully nested CV | step=${STEP} | array task=${SLURM_ARRAY_TASK_ID:-${FOLD:-n/a}}"
[ -n "${EXCLUDE_SAMPLES}" ] && echo "Excluded samples: ${EXCLUDE_SAMPLES}"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Region set: ${FEATURE_SET} (${FEATURE_ARGS})"
echo "k grid:     ${K_GRID:-<model_hpc default 10 .. 15000>}"
echo "Rankings:   ${RANKING_FUNCTIONS}"
echo "Args: ${MODE_ARGS}"
echo "============================================"

# All six classifiers are searched in the inner loop (Table S3 + XGBoost). Passing them
# explicitly (rather than relying on the all-classifiers default) makes the run FAIL
# LOUDLY if the container predates the xgboost dependency, instead of silently running
# a five-classifier headline experiment.
# The manifest (role column) drives sample selection: role == 'model', exact-ID feature
# matching, pending-GC samples excluded with a warning.
set -- \
	-c 32 -t ${CANCER_TYPES} \
	${MODE_ARGS} \
	--metadata_file ${METADATA_FILE:-accessory_files/cris_manifest.csv} \
	-e "${EXPERIMENT_NAME}" \
	-o ${OUTPUT_DIR} \
	-s mean_diff \
	--nested_full_classifiers "Support Vector Machine" "Random Forest" "K-Nearest Neighbors" "Logistic Regression" "Multi-Layer Perceptron" "XGBoost"

# Execution (container on the cluster, pixi env with MODEL_HPC_LOCAL=1; CORES
# overrides -c 32; EXCLUDE_SAMPLES appends --exclude_samples) is shared with the
# other wrappers - see slurm/_model_hpc_exec.sh.
. slurm/_model_hpc_exec.sh
run_model_hpc "$@"
