#!/bin/bash
#SBATCH --job-name=ds_extract
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=6554MB
#SBATCH --array=1-538%80
#SBATCH --output=logs/ds_extract_%A_%a.out
#SBATCH --error=logs/ds_extract_%A_%a.out
. slurm/_site.sh   # site settings: slurm/site.env (scheduler, storage paths, container, Apptainer cache)

# ============================================================================
# Depth-downsampled BIN feature extraction (reviewer revision: coverage
# robustness). No downsampled BAMs are written: features.py --target_coverage
# thins fragments in code with a seeded read-name hash (util.get_filtered_
# alignments), using per-sample keep-fraction = target / native coverage.
#
#   usage: sbatch slurm/extract_features_downsampled.sh <target_coverage> [seed]
#   e.g.:  sbatch slurm/extract_features_downsampled.sh 0.1
#          sbatch slurm/extract_features_downsampled.sh 1
#
# Prerequisite: results/cris_coverage_summary.csv
#   (bash slurm/submit_cris_coverage.sh)
#
# Each array task processes ONE sample over the two per-sample bed extractions
# the model wrappers consume (same pair as extract_features_recovery.sh):
#   1. beds/biomart_10kb.bed                  -> .../downsampled/cov<T>x_seed<S>/biomart_10kb/
#   2. beds/hypothesis/zhu_all_after_lift.bed -> .../downsampled/cov<T>x_seed<S>/zhu_all_after_lift/
# Full-depth features in extracted_features/ are never touched: features.py
# redirects all downsampled output under extracted_features/downsampled/.
# Per-sample keep-fractions are logged in <run dir>/downsample_log/.
#
# The depth arms then run the fully nested search on the run directory, e.g.
#   FEATURES_DIR=extracted_features/downsampled/cov0.1x_seed42/biomart_10kb \
#   EXPERIMENT_NAME=pancan_nested_full_cov0.1x_nodeep ./slurm/submit_nested_full.sh full
# (slurm/submit_nodeep_all.sh phase1 submits all four).
#
# input/cris_bam.txt has 538 entries (the 10 samples that need the GCfix
# patch are lines 529-538; their native coverage must be in the summary CSV
# before extracting them). The GC correction file is derived from the sample
# name, not from line pairing with input/cris_gc.txt, so a task can never grab
# the wrong one.
#
# NOTE: the medium QOS rejects a 528-element array (QOSMaxSubmitJobPerUserLimit
# counts every pending array element; %80 only limits RUNNING tasks). Submit in
# chunks that fit your remaining medium budget — a command-line --array
# overrides the directive below (re-add the throttle):
#   sbatch --array=1-264%80  slurm/extract_features_downsampled.sh 0.1
#   sbatch --array=265-528%80 slurm/extract_features_downsampled.sh 0.1   # after the first drains
# Check the cap and your current usage with:
#   sacctmgr -n show qos medium format=MaxSubmitJobsPerUser,MaxJobsPerUser,MaxWall
#   squeue -u $USER -h | wc -l
# For 0.1x targets the per-task work is light enough for the short QOS, which
# allows large arrays (the 528-task cris_coverage array went through there):
#   sbatch --time=03:59:00 --array=1-528%80 slurm/extract_features_downsampled.sh 0.1
# Keep the 1x run on medium (larger keep-fractions, longer tasks).
# ============================================================================

TARGET=${1:?usage: sbatch slurm/extract_features_downsampled.sh <target_coverage e.g. 0.1> [seed]}
SEED=${2:-42}

# Normalize the same way python's f"{x:g}" does, so the tag matches features.py
# (e.g. "1.0" -> "1", "0.1" -> "0.1")
TARGET_TAG=$(printf '%g' "$TARGET")
RUN_DIR="extracted_features/downsampled/cov${TARGET_TAG}x_seed${SEED}"

BAM_FILE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" input/cris_bam.txt)
SAMPLE=$(basename "$BAM_FILE" | sed 's/\.hg38\.frag\.tsv\.bam//')
GC_FILE="GC/Output_Bam/${SAMPLE}.hg38.frag.tsv__correction_factors.csv"

echo "=========================================="
echo "Downsampled extraction - Array Task"
echo "Job ID: $SLURM_JOB_ID  Array task: $SLURM_ARRAY_TASK_ID"
echo "Sample : $SAMPLE"
echo "Target : ${TARGET_TAG}x  Seed: $SEED"
echo "Output : $RUN_DIR"
echo "Start time: $(date)  Node: $SLURM_NODELIST"
echo "=========================================="

run_extraction () {
    CMD_ARGS="-I $BAM_FILE --gc_file $GC_FILE -c 1 -F BIN -b $1 \
        --target_coverage $TARGET --coverage_file results/cris_coverage_summary.csv --downsample_seed $SEED"
    echo "=== features.py $CMD_ARGS ==="
    apptainer exec --writable-tmpfs --pwd /opt/app --containall \
        --bind src/:/opt/app/src/ \
        --bind features.py:/opt/app/features.py \
        --bind input/:/opt/app/input/ \
        --bind beds/:/opt/app/beds/ \
        --bind results/:/opt/app/results/ \
        --bind accessory_files/:/opt/app/accessory_files/ \
        --bind ${PREPROCESSING_DIR}/cris/:/opt/app/cris/ \
        --bind ${GCFIX_DIR}/:/opt/app/GC \
        --bind ${GCFIX_DIR}/hg38/:/opt/app/hg38 \
        --bind extracted_features/:/opt/app/extracted_features/ \
            ${CONTAINER} pixi run python features.py $CMD_ARGS
    echo "=== exit code: $? ==="
}

mkdir -p "$RUN_DIR"

run_extraction beds/biomart_10kb.bed
run_extraction beds/hypothesis/zhu_all_after_lift.bed

echo ""
echo "=== post-run check: per-sample files present? ==="
for d in biomart_10kb zhu_all_after_lift; do
    [ -f "${RUN_DIR}/${d}/${SAMPLE}.hg38.frag.tsv_BIN.csv" ] || echo "MISSING: ${d}/${SAMPLE}"
done
echo "check done (silence above = both present)"
echo "Finished: $(date)"
