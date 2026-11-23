#!/bin/bash
#SBATCH --job-name=jiang_coverage
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=3277MB
#SBATCH --array=1-249
#SBATCH --output=logs/jiang_coverage_%A_%a.out
#SBATCH --error=logs/jiang_coverage_%A_%a.out
. slurm/_site.sh   # site settings: slurm/site.env (scheduler, storage paths, container, Apptainer cache)

echo "=========================================="
echo "Jiang Mean Coverage - Array Task"
echo "Job ID: $SLURM_JOB_ID  Array task: $SLURM_ARRAY_TASK_ID"
echo "Start time: $(date)"
echo "Running on node: $SLURM_NODELIST"
echo "=========================================="

BAM_FILE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" input/jiang_bam.txt)
SAMPLE=$(basename "$BAM_FILE" | sed 's/\.hg38\.frag\.tsv\.bam//')

echo "BAM file : $BAM_FILE"
echo "Sample   : $SAMPLE"
echo ""

mkdir -p results/jiang_coverage

apptainer exec --writable-tmpfs --pwd /opt/app --containall \
    --bind src/:/opt/app/src/ \
    --bind input/:/opt/app/input/ \
    --bind results/:/opt/app/results/ \
    --bind ${PREPROCESSING_DIR}/jiang/:/opt/app/jiang/ \
    ${CONTAINER} pixi run python -u src/comp/compute_mean_coverage.py \
        "/opt/app/${BAM_FILE}" \
        "/opt/app/results/jiang_coverage/${SAMPLE}.csv"

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Finished: $(date)  Exit code: $EXIT_CODE"
echo "=========================================="
