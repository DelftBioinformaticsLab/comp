# comp — data-driven TSS fragmentomics for cancer detection in cfDNA

Code for the analysis in:

> Bram Pronk, Stavros Makrodimitris, Saskia M. Wilting, Marcel J. T. Reinders. *Genome-wide Cell-free DNA
> Fragmentation Profiling at Transcription Start Sites Detects Cancer Without Cancer-Type-Specific Region Selection.*
> NAR Genomics and Bioinformatics (submitted).

Delft Bioinformatics Lab, Delft University of Technology; Swammerdam Institute for Life Sciences, University of
Amsterdam; Department of Medical Oncology, Erasmus MC Cancer Institute. Contact: Bram Pronk (I.B.Pronk@tudelft.nl).

The code extracts five fragmentomic features at 14,221 transcription start sites (TSSs) from shallow whole-genome
sequencing of plasma cell-free DNA, selects the TSSs, feature, ranking function, number of TSSs and classifier in a
fully nested cross-validation, and compares the resulting model with hypothesis-driven region sets (ATAC-seq,
RNA-seq) and ichorCNA. Every release is archived on Zenodo; its DOI is linked from the release page.

## Contents
- [Reproducing the figures](#reproducing-the-figures)
1. [Environment](#1-environment)
2. [Data: download and preprocessing](#2-data-download-and-preprocessing)
3. [Cohorts](#3-cohorts)
4. [TSS windows and the quality filter](#4-tss-windows-and-the-quality-filter)
5. [Feature extraction](#5-feature-extraction)
6. [Per-sample measures](#6-per-sample-measures)
7. [Models](#7-models)
8. [Figures and tables](#8-figures-and-tables)
9. [Test data](#9-test-data)
10. [Repository layout](#10-repository-layout)
- [Data availability](#data-availability) · [Citation](#citation) · [License](#license)

The analysis is a sequence of scripts, run from the repository root; there is no command-line tool to install. Heavy
steps run on a Slurm cluster through the wrappers in `slurm/`, each of which calls one script inside the container.

## Reproducing the figures

Every data figure of the manuscript can be redrawn on a workstation, without the cluster or the sequencing data:
the notebooks read only the source data in `src/figures/data/no_deep/`, which hold the out-of-fold score, the outer
fold and the operating threshold of every participant and method.

```bash
pixi install --locked                                    # pixi >= 0.74
cd src/figures && pixi run python run_nb.py fig3.ipynb   # one notebook per figure, one at a time
```

`src/figures/README.md` lists the notebook of each figure. Recomputing the source data from the public records is
the full pipeline of Sections 1-8; Section 9 is a small end-to-end test of it.

## 1. Environment

The locked environment is `pixi.lock` (linux-64, osx-arm64; lock-file format v7, so pixi 0.74 or newer; older
versions re-solve the environment and rewrite the lock, which `--locked` prevents):

```bash
pixi install                      # runtime environment
pixi install -e dev               # + pytest, pre-commit, mkdocs
pixi run -e dev pytest tests/     # every pipeline script imports
```

On a cluster, the same environment runs inside an Apptainer container built from `container_pixi.def`:

```bash
apptainer build comp.sif container_pixi.def
```

The Slurm wrappers read every site-specific setting (account, partition, storage paths, container) from
`slurm/site.env`: copy `slurm/site.env.example` and fill it in (see `slurm/README.md`).

## 2. Data: download and preprocessing

Both cohorts are public on [FinaleDB](http://finaledb.research.cchmc.org/) as hg38 fragment files:

| Cohort | Study | Samples analysed | Record IDs |
|---|---|---|---|
| Cristiano et al. 2019 (EGA EGAS00001003611) | development cohort, both tasks | 367 (164 healthy, 203 cancer) + 50 panel of normals | `accessory_files/cris_manifest.csv` |
| Jiang et al. 2015 | additional cohort (HCC vs non-HCC) | 225 | `accessory_files/jiang_manifest.csv` |

Preprocessing, per record: download the FinaleDB hg38 fragment file and convert it to a BAM with a modified
Fragmentstein (v2023.4a; Balázs et al. 2024). Both steps are released in the companion repository
[finaleDB_extract](https://github.com/DelftBioinformaticsLab/finaleDB_extract) (release v1.0.0). It takes the `ID` column of the
manifests above and writes `<PREPROCESSING_DIR>/cris/<ID>.hg38.frag.tsv.bam` and
`<PREPROCESSING_DIR>/jiang/<ID>.hg38.frag.tsv.bam` with their indexes (`PREPROCESSING_DIR` in `slurm/site.env`); its
README gives the versions, the runtime and disk use, and a chr21 test set (Section 9).

Two third-party tools are used as published, not supplied here. ichorCNA 0.3.2 with HMMcopy `readCounter` gives the
tumour fraction (Section 6). GCfix (Rahman et al. 2025, *Bioinformatics* 41:btaf293) gives the GC-bias correction
factors: commit `b0c7c0e` (14 March 2025) of [GCfix_Software](https://github.com/Rafeed-bot/GCfix_Software), with
the hg38 reference files of its Zenodo record (13924466) and default parameters:

```bash
python GCfix_run.py --input_folder <BAM folder> --output_folder Output_Bam --CPU 32 --ref hg38/ref.fa \
    --ref_genome_GC hg38/ref_genome_GC.npy --GC_estimation_regions hg38/correction_factor_estimation_bin_locations.csv
```

It writes `<GCFIX_DIR>/Output_Bam/<ID>.hg38.frag.tsv__correction_factors.csv`, which `features.py` reads (the lists
`input/<cohort>_gc.txt`). We changed GCfix in one place, `accessory_files/gcfix_b0c7c0e_local_change.patch`
(`git apply` in the GCfix folder): when a fragment-length group has no usable GC bins, the released version stops
with an error; the change returns a neutral correction factor of 1 for that group instead. It changes nothing for a
sample that the released version processes, and was needed for 10 of the 538 Cristiano runs.

## 3. Cohorts

The manifests define every sample's role (model, panel of normals, excluded with a reason), task membership and,
after the models have run, its outer fold and out-of-fold prediction (Supplementary Table S1).

```bash
python src/comp/build_cris_manifest.py              # accessory_files/cris_manifest.csv + input/cris_pon_wigs.txt
python src/comp/build_jiang_manifest.py             # accessory_files/jiang_manifest.csv
python src/comp/build_deep_healthy_exclusion.py     # the 30 high-coverage healthy controls left out of every model
```

## 4. TSS windows and the quality filter

10 kb windows around the TSSs of protein-coding genes (`beds/biomart_10kb.bed`). The filter removes windows overlapping
the exclusion lists, chrX, windows closer than 5 kb to another TSS and windows with highly variable coverage across
the panel of normals (Supplementary Figure S2). It keeps 14,226 windows; the model then drops the columns that
duplicate another window in every sample, leaving 14,221 TSSs:

```bash
python src/comp/derive_tss_filter.py                # accessory_files/final_tss_indices_to_remove_hg38.npy
```

## 5. Feature extraction

Five features per TSS and sample (GD, FSLR, FSLRC, RCOV, MWH; Supplementary Table S3), from the GC-corrected BAMs.
`slurm/extract_features.sh` runs `features.py` in the container with the arguments it is given; output goes to
`extracted_features/<bed name>/<ID>.hg38.frag.tsv_BIN.csv`:

```bash
sbatch slurm/extract_features.sh -I input/cris_bam.txt --gc_file input/cris_gc.txt -c 64 -F BIN -b beds/biomart_10kb.bed
sbatch slurm/extract_features.sh -I input/jiang_bam.txt --gc_file input/jiang_gc.txt -c 64 -F BIN -b beds/biomart_10kb.bed
```

Comparators, for each cohort (`cris` shown; `jiang` the same with its lists):

```bash
# RNA-seq: the 792 blood-specific regions of Zhu et al. 2021 -> extracted_features/zhu_all_after_lift/
sbatch slurm/extract_features.sh -I input/cris_bam.txt --gc_file input/cris_gc.txt -c 64 -F BIN -b beds/hypothesis/zhu_all_after_lift.bed
# ATAC-seq: coverage summed over the top 1,000 sites of each of 504 TFs, then one feature per TF (Ulz et al. 2019)
sbatch slurm/extract_features.sh -I input/cris_bam.txt --gc_file input/cris_gc.txt -c 64 -F BIN -b beds/hypothesis/GTRD_1000sites_hg38_504 --aggregate
sbatch slurm/ulz_atac_extraction.sh                 # -> extracted_features/ulz_atac_tfbs_features.npy (397 model rows)
METADATA=accessory_files/jiang_metadata.csv ROLES= OUTPUT=extracted_features/ulz_atac_tfbs_features_jiang.npy sbatch slurm/ulz_atac_extraction.sh
cp extracted_features/ulz_atac_tfbs_features.npy extracted_features/ulz_atac_tfbs_features_pancan.npy   # and its .metadata.csv, .tf_names.csv
# TSS sets (Hallmark pathways, chromosome arms, all TSSs)
python src/comp/build_tss_set_matrices.py beds
python src/comp/build_tss_set_matrices.py averaged
sbatch --array=0-3 slurm/extract_tss_sets.sh hallmark; sbatch --array=0-3 slurm/extract_tss_sets.sh arm; sbatch slurm/extract_tss_sets.sh genome
python src/comp/build_tss_set_matrices.py pooled
```

The models read the pan-cancer ATAC matrix as `ulz_atac_tfbs_features_pancan.npy` (an identical copy). For the
depth analysis, `slurm/extract_features_downsampled.sh <target coverage>` extracts the TSS and RNA features from
fragments thinned in code (no downsampled BAMs; needs `results/cris_coverage_summary.csv` from
`slurm/submit_cris_coverage.sh`), into `extracted_features/downsampled/cov<T>x_seed42/`.

## 6. Per-sample measures

Coverage (`slurm/submit_cris_coverage.sh`, `slurm/submit_jiang_coverage.sh`), sex from the reads
(`slurm/sex_inference.sh`, `src/comp/compute_sex.py`), ichorCNA tumour fraction with a panel of normals of the 50
healthy samples (`slurm/create_ichor_pon.sh`, `src/comp/collect_ichorcna_tf.py`), and the covariate panels
(`src/comp/build_scalar_panel_matrix.py`).

ichorCNA is not wrapped here; run it with your own installation. We used ichorCNA 0.3.2: HMMcopy `readCounter
--window 1000000 --quality 10` on chr1-22 and chrX of each BAM, then `runIchorCNA.R` with the hg38 1 Mb GC and
mappability wigs and the centromere file shipped with ichorCNA, `--normalPanel results/ichorCNA_pon/pon_cris50_median.rds`,
`--chrTrain 'c(1:18,20:22)'`, `--genomeBuild hg38 --ploidy 'c(2)' --estimateScPrevalence FALSE`, and the output in
`results/ichorCNA/<ID>.hg38.frag.tsv/`, which `collect_ichorcna_tf.py` reads.

## 7. Models

All models run through `src/comp/model_hpc.py`; every outer fold is one array task. The fully nested cross-validation
of the data-driven model (10 outer folds; feature, ranking function, number of TSSs, classifier and hyperparameters
selected in the inner loop):

```bash
# BRCA/CRC versus healthy, and pan-cancer versus healthy, without the 30 high-coverage controls
EXCLUDE_SAMPLES=accessory_files/cris_exclude_deep_healthy.txt EXPERIMENT_NAME=brca_crc_nested_full_nodeep ./slurm/submit_nested_full.sh full
CANCER_TYPES=PANCAN EXCLUDE_SAMPLES=accessory_files/cris_exclude_deep_healthy.txt EXPERIMENT_NAME=pancan_nested_full_nodeep ./slurm/submit_nested_full.sh full
```

The same submit script runs the comparators (`FEATURE_SET=atac|rna`), the sweep over the number of regions
(`ksweep`), the training-cohort size (`sizesweep`) and the label permutation (`LABEL_PERMUTATION_SEED`). Other analyses:
the Jiang cohort (`slurm/best_model_search.sh`, `slurm/external_validation.sh`), leave one cancer type out
(`slurm/submit_loco.sh`), the covariate panels and covariate-augmented models (`slurm/submit_scalar_panels.sh`,
`slurm/submit_covariate_arms.sh`), sequencing depth (`slurm/cross_depth_eval.sh`). `slurm/README.md` lists every
wrapper.

Every Cristiano run of the manuscript leaves out the 30 high-coverage controls: `EXCLUDE_SAMPLES=accessory_files/
cris_exclude_deep_healthy.txt METADATA_FILE=accessory_files/cris_manifest.csv`, and experiment names end in `_nodeep`
(the names the exporters expect). Most runs are submitted by `./slurm/submit_nodeep_all.sh phase1`, then `phase2`
once phase 1 has finished (`DRY_RUN=1` lists every submission); the others are below. Per figure:

| Figure | Runs | Submitted by |
|---|---|---|
| Figure 3 | fully nested BRCA/CRC (`brca_crc_nested_full_nodeep`); RNA, ATAC (`brca_crc_{rna,atac}_rev_nodeep`); ichorCNA and covariate baselines | `submit_nodeep_all.sh phase1`; `submit_scalar_panels.sh no_deep` |
| Figure 4a | fully nested pan-cancer (`pancan_nested_full_nodeep`), RNA, ATAC (`pancan_{rna,atac}_rev_nodeep`) | `submit_nodeep_all.sh phase1` |
| Figure 4b-c | pan-cancer models deployed on Jiang (best-model search, then external validation, also per k); the same procedure trained within Jiang | `submit_nodeep_all.sh phase1` and `phase2` (`submit_jiang_arms.sh all` runs the same deployments plus the covariate arms of the response letter); `DATASET=jiang METADATA_FILE=accessory_files/jiang_manifest.csv EXPERIMENT_NAME=jiang_hcc_nested_full ./slurm/submit_nested_full.sh full` (and `ksweep`) |
| Figure 4d-e | leave one cancer type out: one search per left-out type, applied to that type and 34 held-out donors | `python src/comp/build_loco_splits.py`, then `./slurm/submit_loco.sh all` with `ARM=dd`, `rna` and `atac` |
| Figure 5a-b, Suppl. model analysis | every configuration scored on the outer folds (`pancan_dd_rev_nodeep`, `brca_crc_dd_rev_nodeep`) | `EXPERIMENT_NAME=pancan_dd_rev_nodeep sbatch slurm/nested_cv_dd.sh` (empty `CANCER_TYPES` = pan-cancer; `CANCER_TYPES="breast colorectal"` for BRCA/CRC) |
| Figure 5c, Suppl. model analysis | label permutation of the complete procedure (`pancan_nested_full_perm1_nodeep`, `brca_crc_nested_full_perm1_nodeep`) | `LABEL_PERMUTATION_SEED=1 CANCER_TYPES=PANCAN EXPERIMENT_NAME=pancan_nested_full_perm1_nodeep ./slurm/submit_nested_full.sh full` |
| Figure 5d | TSS sets | `submit_tss_set_panels.sh` |
| Suppl. region-number sweep | k-sweeps of TSSs, RNA, ATAC | `submit_nodeep_all.sh phase1` |
| Suppl. depth and cohort size | depth arms (1x, 0.1x), cross-depth evaluation, size sweeps | `submit_nodeep_all.sh phase1` and `phase2` |
| Suppl. covariates | covariate panels; covariates inside the model | `submit_scalar_panels.sh no_deep`; `submit_covariate_arms.sh` |
| Figure 6, Suppl. TSS ranking, gene sets | no model runs: the exporters compute the rankings from the features | |

## 8. Figures and tables

Each figure is drawn by a notebook in `src/figures/` from source-data files in `src/figures/data/no_deep/`, written by
the exporters in `src/figures/export/` from the model results:

```bash
python src/figures/export/export_fig3.py      # one or more exporters per figure
cd src/figures && python run_nb.py fig3.ipynb  # one notebook at a time
```

`src/figures/README.md` maps every manuscript figure, supplementary figure and table, and every figure of the
response to the reviewers, to its notebook and exporters, and gives the order in which to run the exporters. The
Excel supplementary tables are built by `src/figures/export/build_supplementary_tables.py`. Figure 2 (overview) is
drawn by hand (`src/figures/drawn/fig2_revised.pptx`), as is the sample-flow figure.

## 9. Test data

An end-to-end test on chromosome 21 of 20 Cristiano records (10 healthy, 5 breast cancer, 5 colorectal cancer): feature
extraction, then the fully nested model search. It is a smoke test: it checks that every step runs and writes complete
outputs, not that it reproduces stored numbers.

1. Build the 20 chr21 BAMs with finaleDB_extract (its Section 9; about 600 MB, rebuilt from FinaleDB in a few
   minutes): `pixi run bash scripts/make_chr21_test.sh -g reference_genome/hg38.analysisSet.fa -o chr21_test -j 4`.
2. In this repository:

```bash
pixi run bash tests/chr21/run_chr21_test.sh <finaleDB_extract>/chr21_test/cris <finaleDB_extract>/reference_genome/hg38.analysisSet.fa
sbatch slurm/chr21_test.sh          # the same in the container on a Slurm cluster
```

The script writes its inputs (the 216 chr21 TSS windows, the TSS filter restricted to them, a 20-record manifest),
extracts the BIN features of the five feature types, runs the fully nested search (BRCA/CRC vs healthy, 10 outer and 3
inner folds) and prints `chr21 test PASSED` when every record has all 216 windows and the five feature types, every
outer fold has a selected configuration, and every record has one held-out probability. Work files go to `results/chr21_test/`. To keep the test small it differs from the manuscript
runs: only chr21, no GC-bias correction (GCfix is a separate tool; `features.py --gc` switches it off), k from 10 to
100, and by default two ranking functions (mean difference, p-value) and three classifiers (SVM, k-nearest neighbours,
logistic regression) with their full hyperparameter grids. In the container on a 16-core cluster node the whole
test takes under 3 minutes, almost all of it the model search (25 minutes on a 4-core workstation). Building the BAM
files beforehand with finaleDB_extract needs internet access (run it on a login node).
`GRID=full` runs all four ranking functions and all six classifiers as in the manuscript (about 45 minutes per outer
fold on 8 cores). With 20 samples and one chromosome the AUROC carries no information.

## 10. Repository layout

```
src/comp/            pipeline scripts (features, cohorts, filter, models: model_hpc.py)
features.py          feature extraction entry point
slurm/               cluster wrappers (site settings in slurm/site.env)
beds/                TSS windows, region sets, exclusion lists
accessory_files/     manifests and filters
src/figures/         figure notebooks, exporters (export/), source data (data/no_deep/)
tests/               import test of every pipeline script; chr21/ end-to-end smoke test
container_pixi.def   Apptainer container
pixi.lock            locked environment
```

## Data availability

- **Sequencing data:** both cohorts are public on [FinaleDB](http://finaledb.research.cchmc.org/): Cristiano et al.
  2019 (EGA EGAS00001003611) and Jiang et al. 2015. The record IDs are in `accessory_files/cris_manifest.csv` and
  `accessory_files/jiang_manifest.csv`; Section 2 describes the download and preprocessing.
- **Download and BAM conversion:** [finaleDB_extract](https://github.com/DelftBioinformaticsLab/finaleDB_extract)
  (release v1.0.0).
- **Code and source data:** this repository, release v1.0.0, archived on Zenodo (DOI on the release page). Supplementary Table S1 lists
  every participant with their outer fold and out-of-fold prediction.

## Citation

If you use this code, please cite the article (also in [CITATION.cff](CITATION.cff), which GitHub shows as "Cite this
repository"):

> Pronk B, Makrodimitris S, Wilting SM, Reinders MJT. Genome-wide Cell-free DNA Fragmentation Profiling at
> Transcription Start Sites Detects Cancer Without Cancer-Type-Specific Region Selection. NAR Genomics and
> Bioinformatics (submitted).

Questions about the code: open an issue or contact the corresponding author (I.B.Pronk@tudelft.nl).

## License

GNU General Public License v3.0 ([LICENSE](LICENSE)).
