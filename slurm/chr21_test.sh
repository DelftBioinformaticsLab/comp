#!/bin/bash
#SBATCH --job-name=chr21_test
#SBATCH --time=03:59:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=16000MB
#SBATCH --output=logs/chr21_test_%j.out
#SBATCH --error=logs/chr21_test_%j.out
. slurm/_site.sh   # site settings: slurm/site.env (scheduler, storage paths, container, Apptainer cache)

# chr21 end-to-end test (tests/chr21/run_chr21_test.sh) in the container. First build the
# 20 chr21 BAMs with finaleDB_extract (sbatch slurm/chr21_test.sbatch there, output
# <PREPROCESSING_DIR>/chr21_test/cris/). Submit from the repository root:
#   sbatch slurm/chr21_test.sh
#   GRID=full sbatch --time=23:59:00 slurm/chr21_test.sh   # all four rankings and six classifiers (hours)
# CHR21_BAM_DIR and CHR21_GENOME override the default input locations.

CHR21_BAM_DIR=${CHR21_BAM_DIR:-${PREPROCESSING_DIR}/chr21_test/cris}
CHR21_GENOME=${CHR21_GENOME:-${PREPROCESSING_DIR}/reference_genome/hg38.analysisSet.fa}
CORES=${SLURM_CPUS_PER_TASK:-4}
mkdir -p results logs

apptainer exec --writable-tmpfs --pwd /opt/app --containall \
	--env GRID="${GRID:-quick}" \
	--bind src/:/opt/app/src/ \
	--bind features.py:/opt/app/features.py \
	--bind beds/:/opt/app/beds/ \
	--bind accessory_files/:/opt/app/accessory_files/ \
	--bind tests/:/opt/app/tests/ \
	--bind results/:/opt/app/results/ \
	--bind "${CHR21_BAM_DIR}/":/opt/app/chr21_bams/ \
	--bind "$(dirname "${CHR21_GENOME}")/":/opt/app/genome/ \
		${CONTAINER} pixi run bash tests/chr21/run_chr21_test.sh \
			/opt/app/chr21_bams "/opt/app/genome/$(basename "${CHR21_GENOME}")" results/chr21_test "${CORES}"
