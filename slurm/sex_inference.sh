#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1638MB
#SBATCH --job-name=sex_inference
#SBATCH --output=logs/%A_%a_sex_inference.out
#SBATCH --error=logs/%A_%a_sex_inference.out
. slurm/_site.sh   # site settings: slurm/site.env (scheduler, storage paths, container, Apptainer cache)

# ============================================================================
# Genetic sex per sample: R_Y (Skoglund 2013) and R_X (Mittnik 2016) from MAPQ > 30
# read counts on chr1-22, X, Y (src/comp/compute_sex.py; one pass per BAM, minutes)
# ============================================================================
# One array task per line of input/${DATASET}_bam.txt; results land in
# results/${DATASET}_sex/<sample>.csv. Then summarize (login node is fine):
#   apptainer exec --pwd /opt/app --bind src/:/opt/app/src/ --bind results/:/opt/app/results/ \
#     ${CONTAINER} pixi run python src/comp/compute_sex.py \
#     --summarize results/${DATASET}_sex results/${DATASET}_sex/${DATASET}_sex.csv
#
# Usage:
#   DATASET=jiang sbatch --array=1-249 slurm/sex_inference.sh
#   DATASET=cris  sbatch --array=1-538 slurm/sex_inference.sh
# (array size = number of lines in input/${DATASET}_bam.txt)
# ============================================================================

DATASET=${DATASET:-jiang}
BAM_LIST="input/${DATASET}_bam.txt"
BAM_FILE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${BAM_LIST}" | tr -d '\r')
SAMPLE=$(basename "$BAM_FILE" | sed 's/\.hg38\.frag\.tsv\.bam//')
case "${DATASET}" in
	jiang) BIND="${PREPROCESSING_DIR}/jiang/:/opt/app/jiang/" ;;
	cris)  BIND="${PREPROCESSING_DIR}/cris/:/opt/app/cris/" ;;
	*) echo "unknown DATASET ${DATASET}" >&2; exit 1 ;;
esac

echo "sex inference | dataset=${DATASET} task=${SLURM_ARRAY_TASK_ID} bam=${BAM_FILE} sample=${SAMPLE}"
mkdir -p "results/${DATASET}_sex" logs

apptainer exec --writable-tmpfs --pwd /opt/app --containall \
    --bind src/:/opt/app/src/ \
    --bind input/:/opt/app/input/ \
    --bind results/:/opt/app/results/ \
    --bind "${BIND}" \
    ${CONTAINER} pixi run python -u src/comp/compute_sex.py \
        "/opt/app/${BAM_FILE}" \
        "/opt/app/results/${DATASET}_sex/${SAMPLE}.csv"
echo "exit code: $?"
