#!/bin/bash
#SBATCH --job-name=cris_coverage
#SBATCH --time=03:59:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=3277MB
#SBATCH --array=1-538
#SBATCH --output=logs/cris_coverage_%A_%a.out
#SBATCH --error=logs/cris_coverage_%A_%a.out
. slurm/_site.sh   # site settings: slurm/site.env (scheduler, storage paths, container, Apptainer cache)

# Native genome-wide mean coverage per Cristiano BAM. Prerequisite for the
# depth-downsampling revision experiment: features.py --target_coverage looks
# up each sample's native coverage in the combined CSV to derive its per-sample
# keep-fraction (see slurm/extract_features_downsampled.sh).
#
# Submit via slurm/submit_cris_coverage.sh, which chains the summary job that
# combines the per-sample CSVs into results/cris_coverage_summary.csv.
# (input/cris_bam.txt has 538 entries; the 10 samples that need the GCfix
# patch are lines 529-538. To cover only those:
#   sbatch --array=529-538 slurm/cris_coverage.sh
# then rerun the summary step.)

echo "=========================================="
echo "Cristiano Mean Coverage - Array Task"
echo "Job ID: $SLURM_JOB_ID  Array task: $SLURM_ARRAY_TASK_ID"
echo "Start time: $(date)"
echo "Running on node: $SLURM_NODELIST"
echo "=========================================="

BAM_FILE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" input/cris_bam.txt)
SAMPLE=$(basename "$BAM_FILE" | sed 's/\.hg38\.frag\.tsv\.bam//')

echo "BAM file : $BAM_FILE"
echo "Sample   : $SAMPLE"
echo ""

mkdir -p results/cris_coverage

apptainer exec --writable-tmpfs --pwd /opt/app --containall \
    --bind src/:/opt/app/src/ \
    --bind input/:/opt/app/input/ \
    --bind results/:/opt/app/results/ \
    --bind ${PREPROCESSING_DIR}/cris/:/opt/app/cris/ \
    ${CONTAINER} pixi run python -u src/comp/compute_mean_coverage.py \
        "/opt/app/${BAM_FILE}" \
        "/opt/app/results/cris_coverage/${SAMPLE}.csv"

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Finished: $(date)  Exit code: $EXIT_CODE"
echo "=========================================="
