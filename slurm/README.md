# slurm/ — cluster entry points for the model experiments

One sbatch wrapper per experiment; the table below lists them.

**Always submit from the repository root** (`sbatch slurm/<wrapper>.sh ...`):
the apptainer bind paths, the `logs/` output paths, and the sourcing of
`_model_hpc_exec.sh` are all relative to the submission directory.

## Site settings (slurm/site.env)

The wrappers contain nothing specific to one cluster. Copy `site.env.example` to
`site.env` (git-ignored) and fill in:

- `SBATCH_ACCOUNT`, `SBATCH_PARTITION`, `SBATCH_QOS`: sbatch takes these as the
  account, partition and QoS of every job (the wrappers have no such `#SBATCH` lines);
- `PREPROCESSING_DIR` (preprocessed data per cohort, `<dir>/cris`, `<dir>/jiang`),
  `GCFIX_DIR`, `ICHORCNA_DIR`: the folders the
  containers mount;
- `CONTAINER`: the Apptainer image built from `container_pixi.def`;
- `APPTAINER_CACHEDIR`, `APPTAINER_TMPDIR`.

Every job and every `submit_*` script loads it through `_site.sh`, which stops with
a message when `site.env` is missing. For a direct `sbatch`, load it in the shell
first (`. slurm/site.env`, e.g. from `~/.bashrc`). A second cluster gets its own
file on top (`. slurm/site.env; . slurm/site.old.env`).

## Entry points

| Experiment | Entry point | Notes |
|---|---|---|
| Nested CV, data-driven TSS features | `nested_cv_dd.sh [DATASET] [STRATEGY]` | `CANCER_TYPES`/`EXPERIMENT_NAME` via env; historical MS runs: `brca_crc_dd`, `pancan_dd` |
| Nested CV, RNA comparator (no gene selection) | `nested_cv_rna.sh [DATASET] [STRATEGY]` | historical: `brca_crc_rna`, `nested_rna_pancan` |
| Nested CV, ATAC comparator (predefined matrix) | `nested_cv_atac.sh [DATASET] [STRATEGY]` | historical: `brca_crc_atac`, `pancan_atac` |
| Fully nested CV | `./slurm/submit_nested_full.sh estimate\|test\|full\|ksweep\|sizesweep` | 10-task array + dependent aggregation, runs `nested_full.sh`; resumable: resubmitting the same array skips every fold / per-k partial that already exists for the same cohort and grid; `sizesweep` = training-cohort size sweep (env: `CANCER_TYPES`, `EXPERIMENT_NAME`, `SIZESWEEP_FRACTIONS`, `SUBSAMPLE_SEED`). Other cohorts via `DATASET` + `METADATA_FILE` (+ optional `CONTROL_GROUPS`), e.g. trained within Jiang: `DATASET=jiang METADATA_FILE=accessory_files/jiang_manifest.csv EXPERIMENT_NAME=jiang_hcc_nested_full ./slurm/submit_nested_full.sh full` |
| Comparator region-number sweep (reviewer: TSS count tuned vs fixed comparator regions) | `FEATURE_SET=rna\|atac ./slurm/submit_nested_full.sh ksweep` | the identical fully nested procedure, *k* grid included, on the Zhu RNA regions (792, `rcov`) or the Ulz ATAC TF features (504); the ksweep aggregate writes the AUROC-vs-*k* curve and `ksweep_tuned/` (per fold the *k* with the best inner score = the fully nested k-as-hyperparameter arm, no second run needed). Both tasks via `CANCER_TYPES=PANCAN EXPERIMENT_NAME=pancan_{rna,atac}_nested_full`; the data-driven pancan curve via `CANCER_TYPES=PANCAN EXPERIMENT_NAME=pancan_nested_full ./slurm/submit_nested_full.sh ksweep`. Exported by `src/figures/export/export_figS13.py` → `figS13.ipynb` |
| Covariate-augmented TSS models (Fig S14) | `./slurm/submit_covariate_arms.sh [no_deep]` | the fully nested k-sweep with the five confounders appended to the TSS features (`--augment_covariates`); both tasks. Exported by `src/figures/export/export_figS14.py` → `figS14_covariate_augmented.ipynb` |
| Cross-depth evaluation | `sbatch slurm/cross_depth_eval.sh` | single task: refits a completed nested_full run's fold winners on full-depth training data (verified against stored predictions), scores the same test folds on downsampled features; env: `EXPERIMENT_NAME`, `CANCER_TYPES`, `EVAL_DIRS`, `SOURCE_DIR` |
| Best-model search | `best_model_search.sh [DATASET] [dd\|rna\|atac\|dd_cov\|cov]` | atac uses the `_pancan` matrix. `dd_cov` = TSS features + the covariate panel `COVARIATE_PANEL` (default `transfer5` = `pipeline4` + genetic sex from the BAM) appended inside the model (`--augment_covariates`, passthrough of the selector); `cov` = that panel alone (predefined matrix, no row standardization); their default experiment names are `best_model_search_<type>_<panel>` |
| Jiang generalisation arms (rebuttal Fig S16 / Table S3) | `./slurm/submit_jiang_arms.sh search\|deploy\|comparators\|all\|status` | the official deployment protocol for five arms on Cristiano PANCAN: best-model search (dd, dd_cov, cov, rna, atac) then `external_validation.sh <arm> jiang` (`all` chains each deploy to its search with `--dependency=afterok`; dd also in clinical mode), plus the per-classifier RNA/ATAC nested CV of both tasks (`<task>_{rna,atac}_rev<suffix>`). `COHORT=no_deep` (default, suffix `_rev_nodeep`) or `all`; `ARMS`, `COVARIATE_PANEL`, `CLINICAL=0`, `DRY_RUN=1`. Exported by `src/figures/export/export_figS16_jiang_arms.py --from_runs` |
| Threshold refresh (stored runs whose operating thresholds predate the inner-CV derivation) | `sbatch slurm/nested_full.sh refresh_thresholds\|ksweep_refresh_thresholds\|sizesweep_refresh_thresholds`; `REFRESH_DEPLOYMENT_THRESHOLDS=1 sbatch slurm/best_model_search.sh [DATASET] [dd\|rna\|atac]` | single task each, same environment variables as the original submission: refits every stored fold winner (must reproduce its stored test predictions), re-derives the thresholds from its scores on the inner validation folds (1 + 3 fits per fold, no new search) and re-aggregates; previous thresholds are kept in the pickles and listed in `threshold_refresh_log.csv`. Rerun cross-depth, external validation and the exporters afterwards |
| External validation (comprehensive) | `external_validation.sh [dd\|rna\|atac\|dd_cov\|cov] [TEST]` | swept by `submit_external_val_all.sh`. `dd_cov` / `cov` deploy the matching searches with the external cohort's panel matrix `extracted_features/scalar_panel_<panel>_<TEST>` (the block is appended in the model's column position and standardized with the training statistics); the job aborts before the container starts when the search directory is missing |
| External validation (clinical) | `external_validation_clinical.sh [TEST] [single\|comprehensive]` | pancan best model, DD only |
| External validation, all three feature types | `./slurm/submit_external_val_all.sh [TEST] [type]` | |

`_model_hpc_exec.sh` is not a job: it holds the shared apptainer invocation and the common
log header, and is sourced by the wrappers above and by `nested_full.sh`;
`external_validation.sh` has its own apptainer blocks.

## Workstation mode and the no-deep cohort variant

Three environment variables, honoured by every wrapper that sources
`_model_hpc_exec.sh`:

| Variable | Effect |
|---|---|
| `MODEL_HPC_LOCAL=1` | run `model_hpc.py` through the repository's pixi environment (`pixi install` once; `pixi.lock` pins the container's versions) instead of apptainer - the wrappers then work as plain shell scripts, no slurm |
| `CORES=<n>` | replaces the wrapper's `-c` value (cluster scripts ask for 32-64) |
| `EXCLUDE_SAMPLES=<file>` | appends `--exclude_samples <file>` (IDs dropped from both classes before fold generation; also removed from predefined-matrix rows) |

The sensitivity variant without the 30 deep-sequenced healthy samples
(`accessory_files/cris_exclude_deep_healthy.txt`, written by
`src/comp/build_deep_healthy_exclusion.py`) uses experiment names with the
suffix `_nodeep`; `src/figures/export/_common.py` maps every Cristiano result
directory to that twin and writes the CSVs to `src/figures/data/no_deep/` when
`COHORT_VARIANT=no_deep` is set, and the notebooks `fig3`, `fig4`, `figS10` read
them with `FIG_COHORT=no_deep`. Drivers:

| Entry point | What it runs |
|---|---|
| `run_nested_full_local.sh [full\|test\|estimate\|aggregate] [first] [last]` | the ten outer folds of `nested_full.sh` one after another (`FOLD=<n>` replaces the array index) plus the aggregate; resumable |
| `run_nodeep_local.sh panels\|comparators\|estimate\|nested\|export\|all` | every Cristiano model without the 30 samples: covariate panels, RNA/ATAC nested CV, fully nested data-driven runs (both tasks), then the exporters and notebooks for the variant. `NESTED_REDUCED=1` restricts the nested search to `griffin_diff` x `mean_diff` |

## Resuming and running on several clusters

`model_hpc.py` claims every unit of work (a fold partial, a per-k or per-fraction partial,
a covariate-panel result pickle, an aggregate) through a lock file next to its output
(`<output>.lock`, heartbeat every minute). A worker that meets a live claim skips the unit
and waits for its result at the end; a claim without heartbeat for 20 min (killed job) is
taken over; outputs are renamed into place, never half-written; complete units are never
recomputed (delete the file to force). A resubmission therefore resumes, and the same
experiment can be submitted on two clusters that mount the same storage, each with its own
site file (`. slurm/site.env; . slurm/site.<other>.env`). Tunables (seconds):
`NESTED_FULL_LOCK_STALE_SECONDS` (1200), `NESTED_FULL_LOCK_POLL_SECONDS` (60),
`NESTED_FULL_LOCK_MAX_WAIT_SECONDS` (43200).

On clusters where jobs read the storage through Kerberos tickets, make sure the tickets
outlive the jobs: an expired ticket stops running jobs with a bus error (exit 135).

## Feature extraction / QC stage

The upstream wrappers also live here (formerly `run_*.sh` at the repo root):

| Entry point | What it runs |
|---|---|
| `extract_features.sh <features.py arguments>` | `features.py` feature extraction in the container; the TSS, RNA and ATAC commands are in its header and in the top-level README, section 5. As an array job the bed files are split over the tasks |
| `ulz_atac_extraction.sh` | builds the Ulz TFBS feature matrix (`src/comp/extract_ulz_atac_features.py`); `METADATA`, `ROLES`, `OUTPUT` select the cohort |
| `jiang_coverage.sh` + `submit_jiang_coverage.sh` | per-sample genome-wide mean coverage for the Jiang cohort (array job + summary) |
| `cris_coverage.sh` + `submit_cris_coverage.sh` | per-sample genome-wide mean coverage for the Cristiano cohort (array job + summary -> `results/cris_coverage_summary.csv`); prerequisite for the downsampled extraction |
| `extract_features_downsampled.sh <target_cov> [seed]` | revision experiment: depth-downsampled BIN extraction (528-task array, one sample per task, both model bed sets). In-code fragment thinning via `features.py --target_coverage` — no downsampled BAMs are written; output under `extracted_features/downsampled/cov<T>x_seed<S>/` |

ichorCNA itself is run with the user's own installation (settings in the top-level README, section 6);
`create_ichor_pon.sh` builds the panel of normals it uses.
