#!/bin/bash
# ============================================================================
# chr21 end-to-end test: feature extraction and the fully nested model search on
# the 20-record chr21 test set of finaleDB_extract, followed by a check that every
# step wrote complete outputs (a smoke test: no numbers are compared with stored values).
#
#   bash tests/chr21/run_chr21_test.sh <chr21 BAM dir> <reference FASTA> [work dir] [cores]
#
#   <chr21 BAM dir>    <ID>.hg38.frag.tsv.bam + .bai of the 20 records, built by
#                      finaleDB_extract: scripts/make_chr21_test.sh -o chr21_test
#                      -> chr21_test/cris/
#   <reference FASTA>  the UCSC hg38 analysis set that finaleDB_extract uses
#                      (hg38.analysisSet.fa, with its .fai)
#   [work dir]         default results/chr21_test (git-ignored)
#   [cores]            default 4
#
# Run from the repository root in the project environment (pixi run bash ...),
# or on a Slurm cluster with slurm/chr21_test.sh.
#
# Differences from the manuscript runs, all to keep the test small:
#   - only chr21 (216 of the 19,376 TSS windows; the TSS filter keeps 160);
#   - no GC-bias correction (features.py --gc switches GCfix off), because GCfix
#     is a separate tool; the manuscript features are GC-corrected;
#   - k grid 10, 25, 50, 100 instead of 10 ... 15,000;
#   - GRID=quick (default): two ranking functions (mean_diff, p_value) and the three
#     fastest classifiers (SVM, k-nearest neighbours, logistic regression) with their
#     full hyperparameter grids. GRID=full: all four ranking functions and all six
#     classifiers, as in the manuscript (about 45 min per outer fold on 8 cores, so
#     several hours).
# Everything else is the manuscript procedure: the same feature code, the five
# feature types, 10 outer and 3 inner folds.
# ============================================================================
set -euo pipefail

BAM_DIR=${1:?usage: run_chr21_test.sh <chr21 BAM dir> <reference FASTA> [work dir] [cores]}
GENOME=${2:?usage: run_chr21_test.sh <chr21 BAM dir> <reference FASTA> [work dir] [cores]}
WORK=${3:-results/chr21_test}
CORES=${4:-4}
PYTHON=${PYTHON:-python}

[ -f features.py ] && [ -d tests/chr21 ] || { echo "run from the repository root" >&2; exit 1; }
[ -f "${GENOME}" ] && [ -f "${GENOME}.fai" ] || { echo "reference FASTA or its .fai not found: ${GENOME}" >&2; exit 1; }

step() { echo "== $1   [$(date +%T), $((SECONDS / 60)) min since start]"; }
step "1/4 inputs (chr21 TSS windows, TSS filter, manifest)"
${PYTHON} tests/chr21/prepare_inputs.py "${WORK}"

: > "${WORK}/bams.txt"
for id in $(grep -v '^#' tests/chr21/chr21_test.csv | tail -n +2 | cut -d, -f1); do
	bam="${BAM_DIR}/${id}.hg38.frag.tsv.bam"
	[ -f "${bam}" ] && [ -f "${bam}.bai" ] || { echo "missing BAM or index: ${bam}" >&2; exit 1; }
	echo "${bam}" >> "${WORK}/bams.txt"
done

step "2/4 feature extraction (BIN features, no GC correction)"
${PYTHON} features.py -I "${WORK}/bams.txt" -o "${WORK}/features" -F BIN \
	-b "${WORK}/biomart_10kb_chr21.bed" -f "${GENOME}" --gc -c "${CORES}"
FEAT="${WORK}/features/biomart_10kb_chr21"
# without GC correction features.py names the files <ID>.hg38.frag.tsv_NOGC_BIN.csv;
# model_hpc.py reads <ID>.hg38.frag.tsv_BIN.csv
for f in "${FEAT}"/*_NOGC_BIN.csv; do
	mv "${f}" "${f%_NOGC_BIN.csv}_BIN.csv"
done
n=$(ls "${FEAT}"/*_BIN.csv | wc -l)
[ "${n}" -eq 20 ] || { echo "expected 20 feature files, found ${n}" >&2; exit 1; }

GRID=${GRID:-quick}
case "${GRID}" in
	quick)
		RANKINGS="mean_diff p_value"
		set -- "Support Vector Machine" "K-Nearest Neighbors" "Logistic Regression" ;;
	full)
		RANKINGS="mean_diff fold_change p_value weighted"
		set -- "Support Vector Machine" "Random Forest" "K-Nearest Neighbors" \
			"Logistic Regression" "Multi-Layer Perceptron" "XGBoost" ;;
	*) echo "GRID must be quick or full" >&2; exit 1 ;;
esac

# With 20 samples the inner training folds hold 12; k-nearest neighbours with 15 or 21 neighbours cannot be fit
# there, and scikit-learn prints a traceback for every such grid point (scored as missing). Silence those warnings.
export PYTHONWARNINGS="ignore::UserWarning"
step "3/4 fully nested model search (BRCA/CRC vs healthy, ${GRID} grid)"
${PYTHON} -u src/comp/model_hpc.py -c "${CORES}" -t breast colorectal \
	--run_nested_cv_full --features_dir "${FEAT}" \
	--blacklist_indices "${WORK}/tss_filter_chr21.npy" \
	--nested_full_selection_types ${RANKINGS} \
	--nested_full_top_genes 10 25 50 100 \
	--nested_full_classifiers "$@" \
	--metadata_file "${WORK}/cris_chr21_manifest.csv" \
	-e "chr21_test_${GRID}" -o "${WORK}/nested" -s mean_diff

step "4/4 output check"
${PYTHON} tests/chr21/check_outputs.py "${WORK}" --grid "${GRID}"
echo "total: $((SECONDS / 60)) min"
