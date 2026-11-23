#!/bin/sh
#SBATCH --time=23:59:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=80GB
#SBATCH --mail-type=END
#SBATCH --job-name=tss_sets
#SBATCH --output=logs/%A_%a_tss_sets.out
#SBATCH --error=logs/%A_%a_tss_sets.out
. slurm/_site.sh   # site settings: slurm/site.env (scheduler, storage paths, container, Apptainer cache)

# ============================================================================
# Pooled coverage profiles of TSS sets (rebuttal R2.3 / R3.4): for every set
# bed (one Hallmark pathway or one chromosome arm, written by
# `python src/comp/build_tss_set_matrices.py beds`) the coverage over all of the
# set's TSS windows is summed per sample (--aggregate), with the same BAMs and
# GCfix correction as the biomart_10kb TSS features. Output:
#   extracted_features/tss_sets_<type>/<set>/<ID>.hg38.frag.tsv_coverage_aggregate.npy
# then `python src/comp/build_tss_set_matrices.py pooled`.
#
# Usage (repository root), one array per set type, the beds split over the tasks:
#   sbatch --array=0-3 slurm/extract_tss_sets.sh hallmark
#   sbatch --array=0-3 slurm/extract_tss_sets.sh arm
# Without --array the whole bed list runs in one job. Existing valid aggregate
# files are skipped by bin.py, so a resubmission resumes.
# ============================================================================

SET_TYPE=${1:?usage: sbatch [--array=0-N] slurm/extract_tss_sets.sh hallmark|arm}
BEDS=input/tss_sets_${SET_TYPE}_beds.txt
[ -f "${BEDS}" ] || { echo "${BEDS} missing - run python src/comp/build_tss_set_matrices.py beds" >&2; exit 1; }
CMD_ARGS="-I input/cris_model_bam.txt --gc_file input/cris_model_gc.txt -c ${SLURM_CPUS_PER_TASK:-32} -F BIN -b ${BEDS} --aggregate --output_dir extracted_features/tss_sets_${SET_TYPE}"
if [ -n "${SLURM_ARRAY_TASK_ID}" ]; then
	CMD_ARGS="${CMD_ARGS} --job_index ${SLURM_ARRAY_TASK_ID} --total_jobs ${SLURM_ARRAY_TASK_COUNT}"
fi
echo "TSS-set extraction | ${SET_TYPE} | ${CMD_ARGS}"

apptainer exec --writable-tmpfs --pwd /opt/app --containall \
	--bind src/:/opt/app/src/ \
	--bind features.py:/opt/app/features.py \
	--bind input/:/opt/app/input/ \
	--bind beds/:/opt/app/beds/ \
	--bind accessory_files/:/opt/app/accessory_files/ \
	--bind ${PREPROCESSING_DIR}/cris/:/opt/app/cris/ \
	--bind ${GCFIX_DIR}/:/opt/app/GC \
	--bind ${GCFIX_DIR}/hg38/:/opt/app/hg38 \
	--bind extracted_features/:/opt/app/extracted_features/ \
		${CONTAINER} pixi run python features.py ${CMD_ARGS}
echo "Finished: $(date)  Exit code: $?"
