#!/bin/bash
# Submit the Cristiano coverage array job, then submit the summary job with a dependency.
# Usage: bash slurm/submit_cris_coverage.sh
#
# The combined CSV (results/cris_coverage_summary.csv) is written OUTSIDE the
# per-sample directory on purpose: summarize_coverage.py globs *.csv in that
# directory, so a summary placed inside it would be re-ingested on a rerun.
. slurm/_site.sh   # site settings: slurm/site.env (scheduler, storage paths, container, Apptainer cache)

set -e

echo "Submitting Cristiano coverage array job (528 tasks)..."
ARRAY_JOB_ID=$(sbatch --parsable slurm/cris_coverage.sh)
echo "  Array job ID: $ARRAY_JOB_ID"

echo "Submitting summary job (depends on array job completing)..."
SUMMARY_JOB_ID=$(sbatch --parsable \
    --job-name=cris_coverage_summary \
    --time=00:10:00 \
    --cpus-per-task=1 \
    --mem=1638MB \
    --dependency=afterok:${ARRAY_JOB_ID} \
    --output=logs/cris_coverage_summary_%j.out \
    --error=logs/cris_coverage_summary_%j.out \
    --wrap="apptainer exec --writable-tmpfs --pwd /opt/app --containall \
        --bind src/:/opt/app/src/ \
        --bind results/:/opt/app/results/ \
        ${CONTAINER} pixi run python -u src/comp/summarize_coverage.py \
            /opt/app/results/cris_coverage \
            /opt/app/results/cris_coverage_summary.csv")

echo "  Summary job ID: $SUMMARY_JOB_ID"
echo ""
echo "Done. Final CSV will be at: results/cris_coverage_summary.csv"
echo "Summary stats will appear in: logs/cris_coverage_summary_${SUMMARY_JOB_ID}.out"
