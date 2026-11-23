# src/figures: figures and tables

Every figure and supplementary table of the manuscript, and every figure in the response to the reviewers, is made in
two steps:

1. an **exporter** in `export/` reads the model results, manifests and features, checks its numbers against the
   stored pipeline results (recomputed AUROCs and confusion counts must equal the stored ones), and writes
   source-data files to `data/no_deep/` (to `data/` for the three figures drawn from the panel of normals only);
2. a **notebook** in this folder reads only those source-data files and draws the figure into `output/no_deep/`
   (`output/` for the panel-of-normals figures).

The notebooks need no result pickles, cluster paths or network access, so they can be rerun from the committed
source data alone. `output/` is git-ignored; the committed notebooks carry their rendered output.

## Running

Exporters run from the repository root, notebooks from this folder:

```bash
python src/figures/export/export_fig3.py
cd src/figures && python run_nb.py fig3.ipynb
```

`run_nb.py` executes a notebook in place (`jupyter nbconvert --execute --inplace`) and lets only one notebook run at a
time, because concurrent writes to a network drive fail. The exporters that refit fold models or read the per-sample
feature files need several cores and GB; on a Slurm cluster run them with
`sbatch slurm/export_figures.sh src/figures/export/<exporter>.py [options]`, which uses the container of the model
runs.

**Cohort.** All models and figures use the Cristiano et al. cohort without the 30 high-coverage healthy controls
(`accessory_files/cris_exclude_deep_healthy.txt`; Supplementary Figure "sample flow"). This is the default
(`COHORT_VARIANT=no_deep` for the exporters, `COHORT = "no_deep"` in each notebook's setup cell).
`COHORT_VARIANT=all` gives the cohort with them, which the manuscript does not use.

## Manuscript figures

| Figure | Notebook | Exporters (source data) |
|---|---|---|
| Figure 1 (TSS features and gene expression) | `fig1.ipynb` | `export_fig1.py` |
| Figure 2 (overview) | drawn by hand: `drawn/fig2_revised.pptx` | |
| Figure 3 (BRCA/CRC versus healthy) | `fig3.ipynb` | `export_fig3.py`, `export_method_comparisons.py` |
| Figure 4 (pan-cancer, Jiang cohort, leave one cancer type out) | `fig4.ipynb` | `export_fig4.py`, `export_fig4_loco.py`, `export_fig4_loco_rankings.py`, `export_figS16_jiang_arms.py --from_runs` |
| Figure 5 (analysis of the pan-cancer model) | `fig5.ipynb` | `export_fig5.py`, `export_figS5_nested.py` |
| Figure 6 (TSS ranking) | `fig6_tss_interpretation.ipynb` | `export_figS17_cohen_d.py`, `export_figS18_gene_sets.py`, `export_figS18_matched_10k.py`, `export_figS_loco_consistency.py`, `export_figS_loco_consistency_boot.py` |

## Supplementary figures

| Supplementary figure | Notebook | Exporters (source data) |
|---|---|---|
| Sample flow | drawn by hand: `drawn/sample_flow.drawio` (draw.io) | |
| Cristiano versus Jiang cohort | `figS_cohort_comparison.ipynb` | `export_figS12_cohorts.py`, `export_fig3.py`, `export_fig4.py`; `figS10.ipynb` (fragment-length halves) |
| TSS quality filter | `figS2.ipynb` | `export_figS2.py` |
| TSS ranking detail | `figS_tss_ranking_detail.ipynb` | `export_figS17.py`, `export_figS17_cohen_d.py`, `export_figS_loco_consistency.py` |
| Gene expression and sequencing depth | `fig1_depth.ipynb` | `export_fig1_depth.py` |
| Cancer type, stage and tumour fraction | `figS_strata.ipynb` | `export_fig3_baselines.py`, `export_figS13.py`; `figS10.ipynb` (stage and type table) |
| Covariates | `figS_covariates.ipynb` | `export_fig3_baselines.py`, `export_fig4.py`, `export_figS10_tss_regression.py`, `export_figS14.py`, `export_figS16_jiang_arms.py`; `figS10.ipynb` (correlations, model versus panels, measures) |
| Sequencing depth and cohort size | `figS_depth_size.ipynb` | `export_figS8.py`, `export_figS9.py` |
| Model analysis, BRCA/CRC | `figS_model_analysis.ipynb` | `export_fig5.py`, `export_figS5_nested.py` |
| Region-number sweep | `figS13.ipynb` | `export_figS13.py`, `export_fig3.py`, `export_fig4.py` |
| Gene sets | `figS_gene_sets.ipynb` | `export_figS18_gene_sets.py`, `export_figS18_matched_10k.py`, `export_figS18_zhu_es.py` |
| Pooled out-of-fold ROC and precision-recall curves | `figS_pooled_curves.ipynb` | `export_fig3.py`, `export_fig4.py`, `export_method_comparisons.py` |

Where a notebook is listed among the sources, it writes source data that a later notebook reads: run `figS10.ipynb`
before `figS_cohort_comparison`, `figS_strata` and `figS_covariates`, and `figS_strata.ipynb` (which writes
`fig3b_calls.csv`) before `fig3.ipynb`.

## Supplementary tables

| Table | Made by |
|---|---|
| Participants (Excel) | `export_tableS_folds_counts.py`, then `build_supplementary_tables.py` |
| Cristiano versus Jiang cohort characteristics (printed) | `export_figS12_cohorts.py` (`tableS_cohorts.tex`) |
| Features, data parameters, classifiers and hyperparameter grids (printed) | LaTeX tables in the manuscript; the grids are `CLF_PARAM_GRIDS` and `DATA_PARAM_GRIDS` in `src/comp/model_hpc.py` |
| C8 cell-type signatures (Excel) | `export_tableS_c8_classes.py`, then `build_supplementary_tables.py` |
| Selected configuration and confusion counts per outer fold (Excel) | `export_tableS_folds_counts.py`, then `build_supplementary_tables.py` |

`build_supplementary_tables.py` writes the Excel files and their LaTeX entries to `submission/supplementary_tables/`
(git-ignored, rebuilt from the CSVs).
The statistical comparisons between methods quoted in the text (corrected resampled *t*-test, DeLong, bootstrap,
Holm) are in `data/no_deep/stats_method_comparisons.csv`, from `export_method_comparisons.py`.

## Figures shown only in the response to the reviewers

| Figure (comment) | Notebook | Exporters |
|---|---|---|
| Leave one cancer type out (R3.7) | `fig4_loco.ipynb` | `export_fig4_loco.py`, `export_fig4_loco_rankings.py` |
| Jiang application per comparison group (R4.4) | `fig4.ipynb` (writes `figS16.png`) | `export_fig4.py`, `export_figS16_jiang_arms.py --from_runs` (also writes `figS16_jiang_arms.png`) |
| Jiang comparison groups, tests (R3.3, R4.4) | `figS16_jiang_groups.ipynb` | `export_fig4.py`, `export_figS10_jiang.py --scalars_only`, `export_figS16_jiang_arms.py` |
| Sequencing depth (R1.3) | `figS8.ipynb` | `export_figS8.py` |
| Training-cohort size (R1.4) | `figS9.ipynb` | `export_figS9.py` |
| Covariates and the model (R1.3, R4.1, R4.7) | `figS10.ipynb` | `export_fig3_baselines.py --task brca_crc` and `--task pancan`, `export_figS10_jiang.py`, `export_figS10_subgroup.py` (`COHORT_VARIANT=all`: it describes the 30 controls), `export_figS14.py`, `export_figS8.py` |
| Cohort comparison (R1.2, R3.2) | `figS12.ipynb` | `export_figS12_cohorts.py` |
| Covariates inside the model (R4.7) | `figS14_covariate_augmented.ipynb` | `export_figS14.py` |
| Per-TSS covariate regression (R4.7) | `figS_tss_regression.ipynb` | `export_figS10_tss_regression.py`, `export_fig3_baselines.py`, `export_figS17.py` |
| TSS ranking and its stability (R2.1, R4.7) | `figS17_tss_ranking.ipynb` | `export_figS17.py`, `export_figS17_cohen_d.py` |
| Gene sets (R2.1, R3.1) | `figS18_gene_sets.ipynb` | `export_figS18_gene_sets.py`, `export_figS18_matched_10k.py`, `export_figS18_zhu_es.py`, `export_figS18_reactome_healthy.py` |
| Agreement between cancer types (R3.1) | `figS_loco_consistency.ipynb` | `export_figS_loco_consistency.py` |
| TSS sets: Hallmark and chromosome arms (R2.3, R3.4) | `figS_tss_sets.ipynb` | `export_figS_tss_sets.py`, `export_fig5.py` |

`export_cohen_d_vs_rankings.py` is a check quoted in the letter (how close the interpretation ranking is to the four
ranking functions); it draws no figure.

## Order of the exporters

Some exporters read the source data of others. From the repository root, after the model runs:

```bash
E=src/figures/export
# panel-of-normals figures (data/; COHORT_VARIANT=all writes there)
COHORT_VARIANT=all python $E/export_fig1.py
COHORT_VARIANT=all python $E/export_fig1_depth.py
COHORT_VARIANT=all python $E/export_figS2.py
# predictions of the fully nested models and the baselines
python $E/export_fig3.py; python $E/export_fig4.py
python $E/export_fig3_baselines.py --task brca_crc; python $E/export_fig3_baselines.py --task pancan
python $E/export_figS13.py; python $E/export_figS8.py; python $E/export_figS9.py; python $E/export_figS5_nested.py
python $E/export_tableS_folds_counts.py             # also tableS_participants.csv, read by the next three
python $E/export_fig5.py; python $E/export_fig4_loco.py; python $E/export_fig4_loco_rankings.py
# cohorts and covariates
python $E/export_figS12_cohorts.py
python $E/export_figS10_jiang.py; python $E/export_figS10_tss_regression.py; python $E/export_figS14.py
# full-cohort inputs (data/): the 30 deep-sequenced controls and the companion of export_figS18_gene_sets.py
COHORT_VARIANT=all python $E/export_fig3_baselines.py --task brca_crc; COHORT_VARIANT=all python $E/export_fig3_baselines.py --task pancan
COHORT_VARIANT=all python $E/export_figS10_subgroup.py
COHORT_VARIANT=all python $E/export_figS10_tss_regression.py
python $E/export_figS16_jiang_arms.py --from_runs
python $E/export_figS_tss_sets.py
python $E/export_method_comparisons.py              # needs fig3, fig4a and figS16 source data
# TSS ranking and gene sets (the GD matrix is cached at the first call and reused)
python $E/export_figS17.py --gd_cache results/gd_cache.npz
python $E/export_figS_loco.py                       # caches the GD matrix for the next two
python $E/export_figS17_cohen_d.py; python $E/export_figS_loco_consistency.py
python $E/export_figS_loco_consistency_boot.py      # bootstrap intervals of Figure 6c
python $E/export_figS18_gene_sets.py --gd_cache results/gd_cache.npz --pon_cache results/pon_cache.npz
python $E/export_figS18_matched_10k.py; python $E/export_figS18_zhu_es.py
python $E/export_figS18_reactome_healthy.py
# supplementary tables
python $E/export_tableS_c8_classes.py; python $E/build_supplementary_tables.py
python $E/export_cohen_d_vs_rankings.py             # letter check
```

Then run the notebooks, `figS10.ipynb` first (see above). The gene-set exporters need `gseapy` (the published results
were computed with 1.3.1; the locked environment has 1.3.2) and the MSigDB files in
`accessory_files/annotations/msigdb/` (see `accessory_files/annotations/README.md`). Each exporter's docstring
gives its inputs, outputs and runtime.

## Notes

- **Operating thresholds.** Every 95%-specificity threshold is the 0.95 quantile of the healthy training samples'
  scores in the inner-loop predictions of the selected configuration (for a deployed model: of its own ten search
  folds), applied unchanged to the outer test fold or the external cohort
  (`model_hpc.inner_cv_operating_thresholds`). The exporters print `!!` for a result written before this rule
  (no `threshold_source` field); `slurm/nested_full.sh refresh_thresholds` updates such a run without repeating the
  search.
- **Check cells.** Several notebooks end with a cell that compares computed numbers with those quoted in the
  manuscript; a moved number shows up there.
- `_figstyle.py` sets the journal style of the main figures (178 mm wide, 7 pt).
