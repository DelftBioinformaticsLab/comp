#!/bin/sh
#SBATCH --job-name=pon_cov
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=80GB
#SBATCH --mail-type=END
#SBATCH --output=logs/%j_pon_coverage.out
#SBATCH --error=logs/%j_pon_coverage.out
. slurm/_site.sh   # site settings: slurm/site.env (scheduler, storage paths, container, Apptainer cache)

# ============================================================================
# Per-locus coverage arrays (features.py --coverage) for the panel-of-normals
# samples that lack one: Figure 1a (src/figures/export/export_fig1.py) averages
# the TSS coverage profile over the 50 panel participants of
# accessory_files/cris_manifest.csv, and EE87934 (CGPLH203, one of the ten
# samples that need the GCfix patch) was extracted without --coverage.
# Extracts into extracted_features/pon_coverage/biomart_10kb/ (the model's
# BIN.csv in extracted_features/biomart_10kb/ is never touched), then moves only
# the coverage array (19,376 x 11,001 float32, ~0.85 GB) to
# extracted_features/biomart_10kb/EE87934.hg38.frag.tsv_coverage.npy and reports
# whether the side BIN.csv matches the model's one.
# Lists (container paths, LF): input/cris_bam_pon_coverage.txt / input/cris_gc_pon_coverage.txt
# Submit from the repository root:  sbatch slurm/extract_coverage_pon.sh
# ============================================================================
sed -i 's/\r$//' input/cris_bam_pon_coverage.txt input/cris_gc_pon_coverage.txt
CMD_ARGS="-I input/cris_bam_pon_coverage.txt --gc_file input/cris_gc_pon_coverage.txt -c 10 -F BIN --coverage -b beds/biomart_10kb.bed -o extracted_features/pon_coverage"
echo "=== features.py $CMD_ARGS ==="
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
rc=$?
echo "=== exit code: ${rc} ==="
[ ${rc} -eq 0 ] || exit ${rc}
SIDE=extracted_features/pon_coverage/biomart_10kb
DEST=extracted_features/biomart_10kb
for f in "${SIDE}"/*_coverage.npy; do
	[ -f "${f}" ] || { echo "no coverage array in ${SIDE}" >&2; exit 1; }
	b=$(basename "${f}" _coverage.npy)
	if cmp -s "${SIDE}/${b}_BIN.csv" "${DEST}/${b}_BIN.csv"; then echo "${b}: side BIN.csv identical to the model's"; else echo "NOTE ${b}: side BIN.csv differs from the model's (kept apart; the model's one is unchanged)"; fi
	mv "${f}" "${DEST}/" && ls -la "${DEST}/${b}_coverage.npy"
done
