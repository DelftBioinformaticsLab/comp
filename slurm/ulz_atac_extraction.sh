#!/bin/bash
#SBATCH --job-name=tfbs
#SBATCH --time=3:38:00      # Request run time (wall-clock). Default is 1 minute
#SBATCH --cpus-per-task=32   # Request number of CPUs (threads) per task. Default is 1 (note: CPUs are always allocated to jobs per 2).
#SBATCH --mem=52429MB          # Request memory (MB) per node. Default is 1024MB (1GB). For multiple tasks, specify --mem-per-cpu instead
#SBATCH --mail-type=END     # Set mail type to 'END' to receive a mail when the job finishes.
#SBATCH --output=logs/%j_ulz_atac.out # Set name of output log. %j is the Slurm jobId
#SBATCH --error=logs/%j_ulz_atac.out # Set name of error log. %j is the Slurm jobId
. slurm/_site.sh   # site settings: slurm/site.env (scheduler, storage paths, container, Apptainer cache)

# ATAC-seq comparator: one feature per TF (Ulz et al. 2019) from the TF-site
# coverage profiles that features.py --aggregate wrote (see slurm/extract_features.sh),
# as one sample x TF matrix for model_hpc.py --predefined_feature_matrix.
# Submit from the repository root:
#   sbatch slurm/ulz_atac_extraction.sh
#   METADATA=accessory_files/jiang_metadata.csv ROLES= OUTPUT=extracted_features/ulz_atac_tfbs_features_jiang.npy sbatch slurm/ulz_atac_extraction.sh
# The Cristiano matrix holds the 397 model rows of the manifest (ROLES=model, the default).
# The Cristiano matrix is read under three names (ulz_atac_tfbs_features{,_brca_crc,_pancan}.npy,
# with their .metadata.csv and .tf_names.csv); they are identical copies.
METADATA=${METADATA:-accessory_files/cris_manifest.csv}
OUTPUT=${OUTPUT:-extracted_features/ulz_atac_tfbs_features.npy}
ROLES=${ROLES-model}
mkdir -p plots logs

echo "=========================================="
echo "ULZ-ATAC Feature Extraction Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Running on node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "=========================================="
echo ""

# Run the extraction script
apptainer exec --writable-tmpfs --pwd /opt/app --containall \
	--bind src/:/opt/app/src/ \
	--bind features.py:/opt/app/features.py \
	--bind input/:/opt/app/input/ \
	--bind beds/:/opt/app/beds/ \
	--bind accessory_files/:/opt/app/accessory_files/ \
    --bind plots/:/opt/app/plots/ \
	--bind nested_cv_results/:/opt/app/nested_cv_results/ \
	--bind nested_cv_results_cris/:/opt/app/nested_cv_results_cris/ \
	--bind nested_cv_results_jiang/:/opt/app/nested_cv_results_jiang/ \
	--bind ${PREPROCESSING_DIR}/cris/:/opt/app/cris/ \
	--bind ${PREPROCESSING_DIR}/jiang/:/opt/app/jiang/ \
	--bind ${GCFIX_DIR}/:/opt/app/GC \
	--bind ${GCFIX_DIR}/hg38/:/opt/app/hg38 \
	--bind extracted_features/:/opt/app/extracted_features/ \
        ${CONTAINER} pixi run python -u src/comp/extract_ulz_atac_features.py \
        --features_dir ./extracted_features \
        --metadata_file "${METADATA}" ${ROLES:+--roles ${ROLES}} \
        --output "${OUTPUT}" \
        --plot_dir "./plots/$(basename "${OUTPUT}" .npy)" \
        --cores ${SLURM_CPUS_PER_TASK} \
        --n_plot_samples 5

echo ""
echo "=========================================="
echo "Job finished: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

