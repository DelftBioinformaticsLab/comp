# Loads the site settings (slurm/site.env) for every slurm wrapper and submit script. Not an sbatch script.
# Sourced from the repository root (every wrapper runs there). Stops with a message when site.env is missing.
#
# site.env sets the scheduler account/partition/QoS (SBATCH_*), the storage paths the containers mount
# (PREPROCESSING_DIR, GCFIX_DIR, ICHORCNA_DIR), the container image (CONTAINER) and
# the Apptainer cache folders. Template: slurm/site.env.example.
if [ ! -f slurm/site.env ]; then
	echo "slurm/site.env not found: copy slurm/site.env.example to slurm/site.env and fill in your cluster's values" >&2
	exit 1
fi
. slurm/site.env
for _v in PREPROCESSING_DIR CONTAINER APPTAINER_CACHEDIR APPTAINER_TMPDIR; do
	eval "_val=\${${_v}:-}"
	if [ -z "${_val}" ]; then
		echo "slurm/site.env does not set ${_v}" >&2
		exit 1
	fi
done
mkdir -p "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}" 2>/dev/null
# local folders the wrappers bind into the container (Apptainer stops when a bind source is missing)
mkdir -p logs results extracted_features nested_cv_results nested_cv_results_cris nested_cv_results_jiang 2>/dev/null
