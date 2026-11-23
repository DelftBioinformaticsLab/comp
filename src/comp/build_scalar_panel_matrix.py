#!/usr/bin/env python
"""Build the scalar-panel baseline feature matrix (reviewer 2, revision).

Packages the six technical/covariate scalars - mean_coverage, total_fragments,
fragment_length_mean, mean_gc, age, sex_female - as a predefined feature matrix
for model_hpc.py's --predefined_feature_matrix path, so the panel baseline runs
through the identical nested CV as the ATAC arm (same fold generator, same six
classifiers and grids, classifier chosen per outer fold on inner-CV scores).
ichorCNA tumor fraction is deliberately EXCLUDED: it is its own baseline, not a
property of the sequencing data.

Input : src/figures/data/fig3_baseline_scalars.csv   (--task brca_crc, default)
        src/figures/data/fig4_baseline_scalars.csv   (--task pancan)
        (the audited per-sample scalar tables written by
        src/figures/export/export_fig3_baselines.py - single source of truth
        for scalar provenance; rows are in manifest order, which reproduces the
        frozen outer-fold partition)
Output: extracted_features/<panel stem>[_pancan].npy   (n x k, float)
        + .metadata.csv (sample_id, disease) + .tf_names.csv (feature names)
        (model_hpc strips the _pancan suffix when deriving the feature name,
        so both tasks report the panel stem as feature name). Panel stems and
        compositions: see PANELS below (--panel full6|technical3|demographic|
        techdemo5|biological3|all8|complete5|pipeline4|transfer5).

Missing values (a few ages) are imputed with the cohort-wide median - a
label-free constant, unlike the per-fold train medians of the univariate
baselines (which need no NaN-free matrix). Documented in the notebooks.

Run from the repository root:
    python src/comp/build_scalar_panel_matrix.py [--task pancan]
Then:
    sbatch slurm/nested_cv_scalar_panel.sh          (or run the same call locally)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]

TASKS = {
    "brca_crc": dict(csv="fig3_baseline_scalars.csv", suffix="", n=274),
    "pancan": dict(csv="fig4_baseline_scalars.csv", suffix="_pancan", n=397),
    # Jiang HCC vs non-HCC (export_figS10_jiang.py writes the table; only the
    # JIANG_PANELS are defined there)
    "jiang": dict(csv="figS10_jiang_scalars.csv", suffix="_jiang", n=225),
}

# Panel compositions. ichorcna_tf is never included - it is its own baseline,
# not a property of the sequencing data. 'full6' is the original panel (kept
# under its original matrix stem so the existing runs stay valid); the
# category-pure panels decompose the signal by covariate class (Fig S10).
PANELS = {
    "full6": dict(stem="scalar_panel_features",
                  cols=["mean_coverage", "total_fragments", "fragment_length_mean",
                        "mean_gc", "age", "sex_female"]),
    "technical3": dict(stem="scalar_panel_technical3",
                       cols=["mean_coverage", "total_fragments", "plasma_volume"]),
    "demographic": dict(stem="scalar_panel_demographic",
                        cols=["age", "sex_female"]),
    # The confounder set: technical / pre-analytical plus
    # demographic - the covariates a tumour cannot cause, in one model.
    "techdemo5": dict(stem="scalar_panel_techdemo5",
                      cols=["mean_coverage", "total_fragments", "plasma_volume",
                            "age", "sex_female"]),
    # every reported covariate (no ichorCNA TF)
    "all8": dict(stem="scalar_panel_all8",
                 cols=["mean_coverage", "total_fragments", "plasma_volume", "age",
                       "sex_female", "fragment_length_mean", "mean_gc", "cfdna_extracted"]),
    # Sensitivity panel for informative missingness: cohort-median imputation
    # writes one exact value into samples whose missingness is class-skewed
    # (age for 9-15 cancers), which a tree model can exploit. 'complete5' uses
    # only covariates with no missing values in either task (no imputation at
    # all); --complete_case drops every sample with any missing panel covariate
    # instead of imputing.
    "complete5": dict(stem="scalar_panel_complete5",
                      cols=["mean_coverage", "total_fragments", "sex_female",
                            "fragment_length_mean", "mean_gc"]),
    # Cohort-transferable panel: the five covariates our pipeline derives from
    # the BAM / fragment file with one definition in every cohort. 'pipeline4' (defined for
    # --task jiang as well); 'biological3' the cancer-associated measurements.
    "pipeline4": dict(stem="scalar_panel_pipeline4",
                      cols=["mean_coverage", "total_fragments", "fragment_length_mean", "mean_gc"]),
    "biological3": dict(stem="scalar_panel_biological3",
                        cols=["fragment_length_mean", "mean_gc", "cfdna_extracted"]),
    # Genetic sex inferred from the BAM (slurm/sex_inference.sh) exists for
    # both cohorts with one definition, so the cohort-transferable set is pipeline4 +
    # sex_female_bam (1 female / 0 male; unassigned calls resolved by the exporters,
    # _common.resolve_sex_call, so the column is complete). Defined for --task jiang as
    # well; the panel deployed Cristiano -> Jiang (Fig S16) and trained within Jiang.
    "transfer5": dict(stem="scalar_panel_transfer5",
                      cols=["mean_coverage", "total_fragments", "fragment_length_mean",
                            "mean_gc", "sex_female_bam"]),
}
JIANG_PANELS = ("pipeline4", "transfer5")   # the covariates that exist for Jiang



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(TASKS), default="brca_crc")
    ap.add_argument("--panel", choices=list(PANELS), default="full6")
    ap.add_argument("--complete_case", action="store_true",
                    help="Drop every sample with a missing value in the panel's covariates instead "
                         "of imputing (no imputation at all; reduced cohort, own outer folds). "
                         "Stem gets a _cc suffix.")
    a = ap.parse_args()
    cfg, panel = TASKS[a.task], PANELS[a.panel]
    PANEL = panel["cols"]
    SCALARS_CSV = REPO / "src" / "figures" / "data" / cfg["csv"]
    stem = panel["stem"] + ("_cc" if a.complete_case else "") + cfg["suffix"]
    OUT_STEM = REPO / "extracted_features" / stem

    t = pd.read_csv(SCALARS_CSV)
    assert len(t) == cfg["n"], f"expected the {cfg['n']}-sample task table, got {len(t)} rows"
    assert t["sample_id"].is_unique
    if a.task == "jiang" and a.panel not in JIANG_PANELS:
        raise SystemExit(f"--task jiang defines only --panel {'|'.join(JIANG_PANELS)} (no age/plasma/cfDNA/QC covariates exist for Jiang)")
    if a.complete_case:
        miss = t[PANEL].isna().any(axis=1)
        print(f"complete-case: dropping {int(miss.sum())} samples with any missing panel covariate "
              f"({int((miss & (t['y_true'] == 0)).sum())} healthy, {int((miss & (t['y_true'] == 1)).sum())} cancer)")
        t = t[~miss].reset_index(drop=True)

    mat = t[PANEL].to_numpy(dtype=float)
    for j, col in enumerate(PANEL):
        n_nan = int(np.isnan(mat[:, j]).sum())
        if n_nan:
            assert col != "sex_female_bam", "sex_female_bam has missing values - rerun the exporter (calls are resolved there)"
            med = np.nanmedian(mat[:, j])
            mat[np.isnan(mat[:, j]), j] = med
            print(f"{col}: imputed {n_nan} missing value(s) with cohort median {med:g}")
    assert not np.isnan(mat).any()

    np.save(OUT_STEM.with_suffix(".npy"), mat)
    t[["sample_id", "disease"]].to_csv(
        OUT_STEM.parent / (OUT_STEM.name + ".metadata.csv"), index=False)
    pd.DataFrame({"feature": PANEL}).to_csv(
        OUT_STEM.parent / (OUT_STEM.name + ".tf_names.csv"), index=False)

    print(f"wrote {OUT_STEM}.npy {mat.shape}, .metadata.csv, .tf_names.csv")
    print(t["disease"].value_counts().to_string())


if __name__ == "__main__":
    main()
