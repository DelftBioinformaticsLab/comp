# Shared execution helper for the slurm/ model-experiment wrappers.
# Not an sbatch script — source it from a wrapper, then call run_model_hpc "$@".
#
# Every wrapper runs model_hpc.py through the same apptainer invocation
# (bind list included) defined here.
#
# REQUIREMENT: submit jobs from the repository root (all binds are relative),
# i.e. `sbatch slurm/<wrapper>.sh ...`.
#
# Workstation mode (no slurm, no apptainer): MODEL_HPC_LOCAL=1 runs the same
# model_hpc.py call through the repository's pixi environment (`pixi install`
# once; pixi.lock pins the container's library versions). Every wrapper can be
# executed as a plain shell script this way, e.g.
#   MODEL_HPC_LOCAL=1 CORES=16 EXPERIMENT_NAME=... sh slurm/nested_cv_rna.sh
# CORES replaces the wrapper's -c value (the cluster scripts ask for 32-64).
#
# EXCLUDE_SAMPLES=<file> appends --exclude_samples <file> to every call (the
# sensitivity runs without the 30 deep-sequenced healthy samples:
# EXCLUDE_SAMPLES=accessory_files/cris_exclude_deep_healthy.txt, with an
# EXPERIMENT_NAME carrying the _nodeep suffix the exporters look for).
. slurm/_site.sh   # site settings: slurm/site.env (scheduler, storage paths, container, Apptainer cache)

run_model_hpc() {
	if [ -n "${EXCLUDE_SAMPLES}" ]; then
		set -- "$@" --exclude_samples "${EXCLUDE_SAMPLES}"
	fi
	if [ -n "${CORES}" ]; then
		# replace the wrapper's "-c <n>" (always the first two arguments)
		[ "$1" = "-c" ] && { shift 2; set -- -c "${CORES}" "$@"; }
	fi
	if [ -n "${MODEL_HPC_LOCAL}" ]; then
		echo "[local] pixi run python -u src/comp/model_hpc.py $*"
		pixi run python -u src/comp/model_hpc.py "$@"
		return $?
	fi
	apptainer exec --writable-tmpfs --pwd /opt/app --containall \
		--bind src/:/opt/app/src/ \
		--bind features.py:/opt/app/features.py \
		--bind input/:/opt/app/input/ \
		--bind beds/:/opt/app/beds/ \
		--bind accessory_files/:/opt/app/accessory_files/ \
		--bind nested_cv_results/:/opt/app/nested_cv_results/ \
		--bind nested_cv_results_cris/:/opt/app/nested_cv_results_cris/ \
		--bind nested_cv_results_jiang/:/opt/app/nested_cv_results_jiang/ \
		--bind results/:/opt/app/results/ \
		--bind ${PREPROCESSING_DIR}/cris/:/opt/app/cris/ \
		--bind ${PREPROCESSING_DIR}/jiang/:/opt/app/jiang/ \
		--bind ${GCFIX_DIR}/:/opt/app/GC \
		--bind ${GCFIX_DIR}/hg38/:/opt/app/hg38 \
		--bind extracted_features/:/opt/app/extracted_features/ \
			${CONTAINER} pixi run python -u src/comp/model_hpc.py "$@"
}

# Standard log header shared by the wrappers.
print_job_header() {
	echo "============================================"
	echo "Running job: ${JOB_NAME}"
	echo "Mode: ${MODE}"
	echo "Dataset: ${DATASET:-N/A}"
	echo "Feature Type: ${FEATURE_TYPE:-N/A}"
	echo "Strategy: ${STRATEGY:-N/A}"
	echo "Experiment Name: ${EXPERIMENT_NAME}"
	echo "Mode Arguments: ${MODE_ARGS}"
	if [ ! -z "${CANCER_TYPES}" ]; then
		echo "Cancer Types: ${CANCER_TYPES}"
	fi
	if [ -n "${EXCLUDE_SAMPLES}" ]; then
		echo "Excluded samples: ${EXCLUDE_SAMPLES}"
	fi
	echo "============================================"
	echo ""
}
