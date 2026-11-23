#!/bin/sh
#SBATCH --job-name=cov
#SBATCH --time=12:38:00      # Request run time (wall-clock). Default is 1 minute
#SBATCH --cpus-per-task=1   # Request number of CPUs (threads) per task. Default is 1 (note: CPUs are always allocated to jobs per 2).
#SBATCH --mem=80GB         # Request memory (MB) per node. Default is 1024MB (1GB). For multiple tasks, specify --mem-per-cpu instead
#SBATCH --mail-type=END     # Set mail type to 'END' to receive a mail when the job finishes.
#SBATCH --output=logs/%j_all.out # Set name of output log. %j is the Slurm jobId
#SBATCH --error=logs/%j_all.out # Set name of error log. %j is the Slurm jobId
. slurm/_site.sh   # site settings: slurm/site.env (scheduler, storage paths, container, Apptainer cache)

# ============================================================================
# Feature extraction (features.py) in the container. All arguments are passed to
# features.py. Submit from the repository root, for example:
#   TSS windows (the data-driven model):
#     sbatch slurm/extract_features.sh -I input/cris_bam.txt --gc_file input/cris_gc.txt -c 64 -F BIN -b beds/biomart_10kb.bed
#   RNA-seq comparator (the 792 regions of Zhu et al. 2021):
#     sbatch slurm/extract_features.sh -I input/cris_bam.txt --gc_file input/cris_gc.txt -c 64 -F BIN -b beds/hypothesis/zhu_all_after_lift.bed
#   ATAC-seq comparator (coverage summed over the top 1,000 sites of each of 504 TFs):
#     sbatch slurm/extract_features.sh -I input/cris_bam.txt --gc_file input/cris_gc.txt -c 64 -F BIN -b beds/hypothesis/GTRD_1000sites_hg38_504 --aggregate
# For the Jiang cohort use input/jiang_bam.txt and input/jiang_gc.txt. Set the
# resources on the command line (sbatch --cpus-per-task=64 --mem=...). As an array
# job the bed files are split over the tasks (--job_index / --total_jobs):
#     sbatch --array=0-7 slurm/extract_features.sh ... -b beds/hypothesis/GTRD_1000sites_hg38_504 --aggregate
# Output: extracted_features/<bed name>/<ID>.hg38.frag.tsv_BIN.csv (one folder per bed).
# ============================================================================
[ $# -gt 0 ] || { echo "usage: sbatch slurm/extract_features.sh <features.py arguments>" >&2; exit 1; }
CMD_ARGS="$*"
if [ -n "${SLURM_ARRAY_TASK_ID}" ]; then
	CMD_ARGS="${CMD_ARGS} --job_index ${SLURM_ARRAY_TASK_ID} --total_jobs ${SLURM_ARRAY_TASK_COUNT}"
fi
echo "features.py ${CMD_ARGS}"

apptainer exec --writable-tmpfs --pwd /opt/app --containall \
	--bind src/:/opt/app/src/ \
	--bind features.py:/opt/app/features.py \
	--bind input/:/opt/app/input/ \
	--bind beds/:/opt/app/beds/ \
	--bind accessory_files/:/opt/app/accessory_files/ \
	--bind nested_cv_results/:/opt/app/nested_cv_results/ \
	--bind ${PREPROCESSING_DIR}/cris/:/opt/app/cris/ \
	--bind ${PREPROCESSING_DIR}/jiang/:/opt/app/jiang/ \
	--bind ${GCFIX_DIR}/:/opt/app/GC \
	--bind ${GCFIX_DIR}/hg38/:/opt/app/hg38 \
	--bind extracted_features/:/opt/app/extracted_features/ \
		${CONTAINER} pixi run python features.py $CMD_ARGS
