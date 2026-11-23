"""Source data for the scalar covariate baselines (--task brca_crc: Figure 3; --task pancan:
Figure 4; both: Fig S10): univariate scalar arms and multivariate covariate-panel arms
('bl_panel_*', see PANELS) on the outer folds of the fully nested run.

Outputs (data[/no_deep]/; prefix fig3_ for brca_crc, fig4_ for pancan):
  *_baseline_scalars.csv      one row per task sample: fold, label, disease, stage, raw scalars
  *_baseline_predictions.csv  per-sample out-of-fold scores (schema of fig3_predictions.csv plus
                              orientation, n_imputed_train); for pancan also the data_driven arm
  *_fold_configs.csv          winning configuration per outer fold and arm
  *_panel_importance.csv      permutation importance of every panel covariate
  *_panel_{metrics,inner_cv,model_summary,train_scores,partial_dependence,decision_maps}.csv
  tableS3_baselines.csv / tableS3_baselines_pancan.csv  per-approach AUROC and 95%-spec metrics

Inputs: accessory_files/cris_manifest.csv, results/cris_coverage_summary.csv,
accessory_files/cristiano_metadata.xlsx, ichorCNA/ichorCNA_summary.tsv, results/cris_sex/cris_sex.csv,
extracted_features/biomart_10kb/, the fully nested run and the scalar-panel runs
(src/comp/build_scalar_panel_matrix.py + slurm/nested_cv_scalar_panel.sh).

A scalar is evaluated per outer fold with train-median imputation, a train-derived sign and a
train-derived 95%-specificity threshold; panel classifiers are chosen per fold on inner-CV scores.
mean_gc is GCfix-weighted, not the raw library GC content. Permutation importances are valid only
if the refit fold models reproduce the stored predictions (same library versions as the runs).

Run from the repository root (mean_gc reads every _BIN.csv unless a previous export is reused):
    python src/figures/export/export_fig3_baselines.py [--task pancan]
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from glob import glob
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, os.path.dirname(__file__))
from _common import (  # noqa: E402
    DATA_DIR,
    ICHOR_AGG_CSV,
    NESTED_FULL_DIR,
    PANCAN_NESTED_FULL_DIR,
    assert_predictions_match_stored,
    best_mean_auc_config,
    load_results,
    rp,
    IS_VARIANT,
    REPO_ROOT,
    excluded_sample_ids,
    resolve_sex_call,
    variant_dir,
)

MANIFEST_CSV = "accessory_files/cris_manifest.csv"
COVERAGE_CSV = "results/cris_coverage_summary.csv"
CRISTIANO_XLSX = "accessory_files/cristiano_metadata.xlsx"
BIOMART_DIR = "extracted_features/biomart_10kb"

PANEL_KEEP = [
    "feature", "classifier", "top_genes", "fold", "auc_full_model",
    "test_sample_ids", "y_pred_proba", "y_test",
    "full_model_thresholds", "confusion_matrix_full_model", "best_params",
]
# extended per-fold records; absent from older pickles (load_results keeps only existing columns)
PANEL_EXTENDED_KEEP = [
    "train_sample_ids", "test_metrics", "operating_point_metrics", "train_metrics",
    "y_train", "inner_cv_train_scores", "resubstitution_train_scores",
    "inner_cv_results", "fitted_model_summary", "feature_names",
]

# Univariate baselines by category (presentation only; the evaluation is the same for all).
CATEGORIES = {
    "mean_coverage": "technical",
    "total_fragments": "technical",
    "plasma_volume": "technical",
    "age": "demographic",
    "sex_female": "demographic",
    "fragment_length_mean": "biological",
    "mean_gc": "biological",
    "cfdna_extracted": "biological",
    "ichorcna_tf": "biological",
}
SCALARS = list(CATEGORIES)

TASKS = {
    "brca_crc": dict(
        in_task_col="in_task_brca_crc",
        fold_col="fold_brca_crc",
        scalars_csv="fig3_baseline_scalars.csv",
        pred_csv="fig3_baseline_predictions.csv",
        table_csv="tableS3_baselines.csv",
        configs_csv="fig3_fold_configs.csv",
        importance_csv="fig3_panel_importance.csv",
        deep_dive_prefix="fig3_panel",
        nested_full_dir=NESTED_FULL_DIR,
        panel_prefix="nested_cv_results_cris/brca_crc_scalar_panel",
        matrix_suffix="",
        cancer_labels=["Breast cancer", "Colorectal cancer"], control_labels=["Healthy"],
        n_samples=274, n_cancer=80,
        headline_auc=0.9703,
    ),
    "pancan": dict(
        in_task_col="in_task_pancan",
        fold_col=None,  # manifest fold_pancan not filled; folds from the nested_full run
        nested_full_dir=PANCAN_NESTED_FULL_DIR,
        scalars_csv="fig4_baseline_scalars.csv",
        pred_csv="fig4_baseline_predictions.csv",
        table_csv="tableS3_baselines_pancan.csv",
        configs_csv="fig4_fold_configs.csv",
        importance_csv="fig4_panel_importance.csv",
        deep_dive_prefix="fig4_panel",
        panel_prefix="nested_cv_results_cris/pancan_scalar_panel",
        matrix_suffix="_pancan",
        cancer_labels=None, control_labels=["Healthy"],  # cancer = everything not Healthy
        n_samples=397, n_cancer=203,
        headline_auc=0.9539,
    ),
    # Jiang cohort (HCC vs non-HCC): only the panel helpers, used by export_figS10_jiang.py.
    "jiang": dict(
        scalars_csv="figS10_jiang_scalars.csv",
        panel_prefix="nested_cv_results_jiang/jiang_scalar_panel",
        matrix_suffix="_jiang",
        cancer_labels=["Liver cancer"], control_labels=["Healthy", "Cirrhosis", "Hepatitis B"],
        n_samples=225, n_cancer=90,
    ),
}

# Multivariate panel arms (src/comp/build_scalar_panel_matrix.py --panel ...):
# result dir = <panel_prefix><dir_suffix>; feature = the matrix stem as model_hpc names it.
PANELS = {
    "bl_panel_full6": dict(dir_suffix="", feature="scalar_panel_features",
                           category="panel (pipeline + demographics)",
                           cols=["mean_coverage", "total_fragments", "fragment_length_mean",
                                 "mean_gc", "age", "sex_female"]),
    "bl_panel_technical3": dict(dir_suffix="_technical3", feature="scalar_panel_technical3",
                                category="panel (technical)",
                                cols=["mean_coverage", "total_fragments", "plasma_volume"]),
    "bl_panel_demographic": dict(dir_suffix="_demographic", feature="scalar_panel_demographic",
                                 category="panel (demographic)",
                                 cols=["age", "sex_female"]),
    # the confounder set: technical + demographic in one model
    "bl_panel_techdemo5": dict(dir_suffix="_techdemo5", feature="scalar_panel_techdemo5",
                               category="panel (technical + demographic)",
                               cols=["mean_coverage", "total_fragments", "plasma_volume",
                                     "age", "sex_female"]),
    # every reported covariate (no ichorCNA TF)
    "bl_panel_all8": dict(dir_suffix="_all8", feature="scalar_panel_all8",
                          category="panel (all covariates)",
                          cols=["mean_coverage", "total_fragments", "plasma_volume", "age",
                                "sex_female", "fragment_length_mean", "mean_gc", "cfdna_extracted"]),
    # informative-missingness sensitivity panels (see build_scalar_panel_matrix.py)
    "bl_panel_complete5": dict(dir_suffix="_complete5", feature="scalar_panel_complete5",
                               category="panel (no imputation)",
                               cols=["mean_coverage", "total_fragments", "sex_female",
                                     "fragment_length_mean", "mean_gc"]),
    # complete-case all-covariate panel: no imputation at all (samples with any
    # missing covariate dropped -> reduced cohort, own outer folds)
    "bl_panel_all8_cc": dict(dir_suffix="_all8_cc", feature="scalar_panel_all8_cc",
                             category="panel (all covariates, complete cases only)",
                             frozen_folds=False,
                             cols=["mean_coverage", "total_fragments", "plasma_volume", "age",
                                   "sex_female", "fragment_length_mean", "mean_gc", "cfdna_extracted"]),
    # 'pipeline4': covariates our pipeline computes the same way from any fragment file;
    # 'biological3': the cancer-associated measurements.
    "bl_panel_pipeline4": dict(dir_suffix="_pipeline4", feature="scalar_panel_pipeline4",
                               category="panel (pipeline, cohort-transferable)",
                               cols=["mean_coverage", "total_fragments", "fragment_length_mean", "mean_gc"]),
    "bl_panel_biological3": dict(dir_suffix="_biological3", feature="scalar_panel_biological3",
                                 category="panel (cancer-associated measurements)",
                                 cols=["fragment_length_mean", "mean_gc", "cfdna_extracted"]),
    # cohort-transferable panel: pipeline4 + genetic sex from the BAM (src/comp/compute_sex.py)
    "bl_panel_transfer5": dict(dir_suffix="_transfer5", feature="scalar_panel_transfer5",
                               category="panel (cohort-transferable, 5)",
                               cols=["mean_coverage", "total_fragments", "fragment_length_mean",
                                     "mean_gc", "sex_female_bam"]),
}
# columns exported alongside SCALARS but not evaluated as univariate arms
EXTRA_COLS = ["sex_female_bam"]
CRIS_SEX_CSV = "results/cris_sex/cris_sex.csv"


# --------------------------------------------------------------------------
# assembly of the raw per-sample scalar table
# --------------------------------------------------------------------------

def load_pancan_nested_full() -> pd.DataFrame:
    with open(rp(TASKS["pancan"]["nested_full_dir"], "nested_full_per_fold.pkl"), "rb") as fh:
        per_fold = pickle.load(fh)
    return per_fold.rename(columns={"test_auc": "auc_full_model", "n_top_genes": "top_genes"})


def nested_run_available(task: str) -> bool:
    return os.path.exists(rp(TASKS[task]["nested_full_dir"], "nested_full_per_fold.pkl"))


def folds_from_panel_run(task: str) -> dict:
    """Outer folds of a frozen-fold panel run, for a cohort variant whose fully nested
    run is not on disk (same samples, order and StratifiedKFold seed)."""
    for method in ["bl_panel_all8", "bl_panel_technical3", "bl_panel_techdemo5", "bl_panel_full6",
                   "bl_panel_demographic",
                   "bl_panel_complete5", "bl_panel_transfer5", "bl_panel_pipeline4", "bl_panel_biological3"]:
        panel_dir = variant_dir(TASKS[task]["panel_prefix"] + PANELS[method]["dir_suffix"])
        paths = sorted(glob(rp(panel_dir, "results*.pkl")))
        if paths:
            with open(paths[0], "rb") as fh:
                df = pickle.load(fh)
            print(f"  {task}: nested run {TASKS[task]['nested_full_dir']} not on disk - outer folds taken "
                  f"from {panel_dir} (data-driven arm omitted)")
            return {sid: int(r["fold"]) for _, r in df.iterrows() for sid in r["test_sample_ids"]}
    raise FileNotFoundError(f"{task}: neither the nested run nor any frozen-fold panel run exists for this cohort variant")


def load_task_cohort(task: str) -> pd.DataFrame:
    cfg = TASKS[task]
    mf = pd.read_csv(rp(MANIFEST_CSV))
    rows = mf[mf[cfg["in_task_col"]] == True].copy()  # noqa: E712
    excl = set(excluded_sample_ids())
    if excl:   # cohort variant: the excluded samples leave the task cohort
        n0 = len(rows)
        rows = rows[~rows["ID"].isin(excl)]
        print(f"  cohort variant: {n0 - len(rows)} excluded sample(s) removed from the {task} cohort")
    out = pd.DataFrame({
        "sample_id": rows["ID"],
        "sample_name": rows["sample_name"],
        "y_true": (rows["disease"] != "Healthy").astype(int),
        "disease": rows["disease"],
        "total_fragments": rows["fragNum"].astype(float),
    })

    if task == "brca_crc" and not IS_VARIANT:
        assert rows["fold_brca_crc"].notna().all(), "task samples without fold assignment"
        out["fold"] = rows["fold_brca_crc"].astype(int).to_numpy()
        # frozen-fold cross-check against the exported headline predictions
        fig3_pred = pd.read_csv(os.path.join(DATA_DIR, "fig3_predictions.csv"))
        dd = fig3_pred[fig3_pred["method"] == "data_driven"]
        dd_fold = dict(zip(dd["sample_id"], dd["fold"].astype(int)))
        assert set(out["sample_id"]) == set(dd_fold), "cohort differs from fig3_predictions.csv"
        mismatch = [s for s, f in zip(out["sample_id"], out["fold"]) if dd_fold[s] != f]
        assert not mismatch, f"fold assignment differs from fig3_predictions.csv: {mismatch[:5]}"
    elif IS_VARIANT and not nested_run_available(task):
        fold_of = folds_from_panel_run(task)
        assert set(out["sample_id"]) == set(fold_of), f"{task} cohort differs from the panel run's test partition"
        out["fold"] = out["sample_id"].map(fold_of)
    elif task == "brca_crc":
        # cohort variant: the manifest's fold column belongs to the full cohort;
        # folds come from the variant's own fully nested run
        with open(rp(cfg["nested_full_dir"], "nested_full_per_fold.pkl"), "rb") as fh:
            per_fold = pickle.load(fh)
        fold_of = {sid: int(r["fold"]) for _, r in per_fold.iterrows() for sid in r["test_sample_ids"]}
        assert set(out["sample_id"]) == set(fold_of), (
            f"{task} cohort differs from the test partition of {cfg['nested_full_dir']}")
        out["fold"] = out["sample_id"].map(fold_of)
    else:
        per_fold = load_pancan_nested_full()
        fold_of = {sid: int(r["fold"]) for _, r in per_fold.iterrows()
                   for sid in r["test_sample_ids"]}
        assert set(out["sample_id"]) == set(fold_of), (
            "manifest pancan cohort differs from the nested_full run's test partition")
        out["fold"] = out["sample_id"].map(fold_of)
    return out


def add_coverage(t: pd.DataFrame) -> None:
    cov = pd.read_csv(rp(COVERAGE_CSV))
    m = dict(zip(cov["sample_name"], cov["mean_coverage"]))
    t["mean_coverage"] = t["sample_id"].map(m)


def add_fragment_length_mean(t: pd.DataFrame) -> None:
    lengths = np.arange(100, 221, dtype=float)
    vals = []
    for sid in t["sample_id"]:
        path = rp(BIOMART_DIR, f"{sid}.hg38.frag.tsv_gc_fragment_length_distribution.npy")
        a = np.load(path)
        assert a.shape == (101, 121), f"{sid}: unexpected histogram shape {a.shape}"
        hist = a.sum(axis=0)
        assert hist.sum() > 0, f"{sid}: empty length histogram"
        vals.append(float((hist * lengths).sum() / hist.sum()))
    t["fragment_length_mean"] = vals
    assert t["fragment_length_mean"].between(100, 220).all()


def _read_csv_retry(path: str, attempts: int = 3, **kwargs) -> pd.DataFrame:
    """pd.read_csv with retries: streaming a large CSV from the network share
    occasionally dies mid-read with a spurious FileNotFoundError/OSError."""
    import time
    for attempt in range(1, attempts + 1):
        try:
            return pd.read_csv(path, **kwargs)
        except (FileNotFoundError, OSError) as e:
            if attempt == attempts or not os.path.exists(path):
                raise
            print(f"  transient read failure on {os.path.basename(path)} "
                  f"(attempt {attempt}/{attempts}): {e}; retrying")
            time.sleep(5 * attempt)


def add_mean_gc(t: pd.DataFrame, cache_csv: str | None = None) -> None:
    """Count-weighted mean_fragment_gc_content per sample from the _BIN.csv files (slow).
    Reuses a previously exported scalars table that covers every sample, unless
    --recompute_gc is given."""
    if cache_csv and not os.path.exists(cache_csv) and IS_VARIANT:
        # cohort variant: per-sample values are the same, reuse the full-cohort table
        cache_csv = os.path.join(os.path.dirname(os.path.dirname(cache_csv)), os.path.basename(cache_csv))
    if cache_csv and os.path.exists(cache_csv):
        old = pd.read_csv(cache_csv)
        if "mean_gc" in old.columns and set(t["sample_id"]) <= set(old["sample_id"]):
            t["mean_gc"] = t["sample_id"].map(dict(zip(old["sample_id"], old["mean_gc"])))
            print(f"  mean_gc: reused for {len(t)} samples from {os.path.basename(cache_csv)} "
                  "(pass --recompute_gc to re-read the _BIN.csv files)")
            assert t["mean_gc"].between(20, 70).all()
            return
    vals = []
    for i, sid in enumerate(t["sample_id"], 1):
        path = rp(BIOMART_DIR, f"{sid}.hg38.frag.tsv_BIN.csv")
        d = _read_csv_retry(path, usecols=["mean_fragment_gc_content", "absolute_fragment_counts"])
        w = d["absolute_fragment_counts"]
        vals.append(float((d["mean_fragment_gc_content"] * w).sum() / w.sum()))
        if i % 50 == 0:
            print(f"  mean_gc: {i}/{len(t)} samples read")
    t["mean_gc"] = vals
    assert t["mean_gc"].between(20, 70).all(), "mean GC outside plausible percent range"


def add_age_sex(t: pd.DataFrame) -> None:
    s1 = pd.read_excel(rp(CRISTIANO_XLSX), sheet_name="1", header=1)
    per_patient = (
        s1.groupby("Patient")[["Age at Diagnosis", "Gender", "Volume of Plasma (ml)",
                               "cfDNA Extracted (ng/ml)"]]
        .agg(lambda c: c.dropna().iloc[0] if c.notna().any() else np.nan)
    )
    dup_conflicts = (
        s1.dropna(subset=["Age at Diagnosis"]).groupby("Patient")["Age at Diagnosis"].nunique()
    )
    assert (dup_conflicts <= 1).all(), "conflicting ages across timepoint rows"
    t["age"] = t["sample_name"].map(per_patient["Age at Diagnosis"]).astype(float)
    gender = t["sample_name"].map(per_patient["Gender"])
    assert gender.dropna().isin(["F", "M"]).all(), f"unexpected gender codes: {gender.unique()}"
    t["sex_female"] = gender.map({"F": 1.0, "M": 0.0})
    # Pre-analytical covariates from the same sheet (cfDNA Input duplicates Extracted and is not used).
    t["plasma_volume"] = t["sample_name"].map(per_patient["Volume of Plasma (ml)"]).astype(float)
    t["cfdna_extracted"] = t["sample_name"].map(per_patient["cfDNA Extracted (ng/ml)"]).astype(float)
    n_unmatched = t["sample_name"].isin(per_patient.index).eq(False).sum()
    print(f"  age/sex: {n_unmatched} task samples unmatched in Cristiano S1, "
          f"{t['age'].isna().sum()} missing age, {t['sex_female'].isna().sum()} missing sex, "
          f"{t['plasma_volume'].isna().sum()} missing plasma volume, "
          f"{t['cfdna_extracted'].isna().sum()} missing cfDNA concentration")


def add_bam_sex(t: pd.DataFrame, csv: str = CRIS_SEX_CSV) -> None:
    """Genetic sex from the BAM (src/comp/compute_sex.py, slurm/sex_inference.sh; unassigned
    calls resolved with _common.resolve_sex_call): sex_female_bam = 1 female, 0 male.
    Asserted identical to Cristiano S1 sex wherever S1 has a value."""
    if not os.path.exists(rp(csv)):
        print(f"  BAM sex: {csv} not found (slurm/sex_inference.sh) - column left empty")
        t["sex_female_bam"] = np.nan
        return
    s = resolve_sex_call(pd.read_csv(rp(csv))).set_index("sample_name")
    call = t["sample_id"].map(s["sex_call_resolved"])
    assert call.notna().all(), f"BAM sex: {int(call.isna().sum())} task samples without a call"
    t["sex_female_bam"] = call.map({"female": 1.0, "male": 0.0}).astype(float)
    assert t["sex_female_bam"].notna().all(), f"unresolved sex calls: {sorted(call.unique())}"
    ok = t["sex_female"].notna()
    agree = int((t.loc[ok, "sex_female_bam"] == t.loc[ok, "sex_female"]).sum())
    how = t["sample_id"].map(s["sex_call_how"]).value_counts().to_dict()
    print(f"  BAM sex: calls {how}; agrees with Cristiano S1 for {agree} of {int(ok.sum())} samples")
    assert agree == int(ok.sum()), "BAM-derived sex disagrees with Cristiano S1"


def add_rerun_ichorcna_tf(t: pd.DataFrame) -> None:
    """Tumour fractions from the ichorCNA run with the cris50 panel of normals."""
    rerun = pd.read_csv(rp("ichorCNA/ichorCNA_summary.tsv"), sep="\t")
    t["ichorcna_tf"] = t["sample_id"].map(dict(zip(rerun["sample"], rerun["tumor_fraction"])))
    assert t["ichorcna_tf"].notna().all(), "task samples missing from the ichorCNA rerun summary"


def add_ichorcna_tf(t: pd.DataFrame) -> None:
    ichor = pd.read_csv(rp(ICHOR_AGG_CSV), index_col=0)
    dup = ichor[ichor.duplicated(subset="ID", keep=False)]
    if len(dup):
        for sid, grp in dup.groupby("ID"):
            assert (grp.nunique() <= 1).all(), f"duplicate rows for {sid} are not identical"
        ichor = ichor.drop_duplicates(subset="ID")
        print(f"  ichorCNA: deduplicated {len(dup)} duplicate rows (pre-rerun table)")
    add_rerun_ichorcna_tf(t)   # the aggregate table supplies only the stage
    t["stage"] = t["sample_id"].map(dict(zip(ichor["ID"], ichor["Stage"])))
    print(f"  ichorCNA: {t['ichorcna_tf'].isna().sum()} task samples without TF "
          "(train-median imputed at evaluation)")


# --------------------------------------------------------------------------
# fold-wise evaluation of one scalar
# --------------------------------------------------------------------------

def spec95_threshold(y: np.ndarray, s: np.ndarray) -> float:
    """Lowest cutoff with specificity >= 0.95 (same convention as fig3.ipynb)."""
    fpr, _, thr = roc_curve(y, s)
    idx = np.where(1 - fpr >= 0.95)[0]
    return float(thr[idx[-1]])


def evaluate_scalar(t: pd.DataFrame, col: str) -> pd.DataFrame:
    rows = []
    for f in sorted(t["fold"].unique()):
        train, test = t[t["fold"] != f], t[t["fold"] == f]
        med = train[col].median()
        assert np.isfinite(med), f"{col}: no finite training values in fold {f}"
        x_tr = train[col].fillna(med).to_numpy(dtype=float)
        x_te = test[col].fillna(med).to_numpy(dtype=float)
        y_tr = train["y_true"].to_numpy(dtype=int)
        orientation = 1.0 if roc_auc_score(y_tr, x_tr) >= 0.5 else -1.0
        thr = spec95_threshold(y_tr, orientation * x_tr)
        for sid, y, s in zip(test["sample_id"], test["y_true"], orientation * x_te):
            rows.append({
                "sample_id": sid, "method": f"bl_{col}", "fold": int(f),
                "y_true": int(y), "score": float(s), "threshold_95spec": thr,
                "orientation": orientation,
                "n_imputed_train": int(train[col].isna().sum()),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# per-method summary metrics (Table S3)
# --------------------------------------------------------------------------

def binary_metrics(y: np.ndarray, y_hat: np.ndarray) -> dict:
    tn = int(np.sum((y == 0) & (y_hat == 0)))
    fp = int(np.sum((y == 0) & (y_hat == 1)))
    fn = int(np.sum((y == 1) & (y_hat == 0)))
    tp = int(np.sum((y == 1) & (y_hat == 1)))
    m = {
        "sensitivity": tp / (tp + fn) if tp + fn else 0.0,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
    }
    if tp + fp:
        m["precision"] = tp / (tp + fp)
        m["f1"] = (2 * m["precision"] * m["sensitivity"] / (m["precision"] + m["sensitivity"])
                   if m["precision"] + m["sensitivity"] else 0.0)
    return m


def summarize_method(df: pd.DataFrame, method: str, protocol: str,
                     fold_thresholds: dict | None = None) -> dict:
    """Per-fold AUROC and operating-point metrics for one method's OOF rows.

    `df` must carry sample-level rows with fold, y_true, score and (unless
    `fold_thresholds` overrides them) a per-fold threshold_95spec column.
    """
    fold_aucs, fold_metrics = [], []
    for f, g in df.groupby("fold"):
        y, s = g["y_true"].to_numpy(dtype=int), g["score"].to_numpy(dtype=float)
        fpr, tpr, _ = roc_curve(y, s)
        fold_aucs.append(float(np.trapezoid(tpr, fpr)))
        thr = fold_thresholds[f] if fold_thresholds else g["threshold_95spec"].iloc[0]
        fold_metrics.append(binary_metrics(y, (s >= thr).astype(int)))
    row = {
        "method": method,
        "protocol": protocol,
        "n_folds": df["fold"].nunique(),
        "n_samples": len(df),
        "auc_pooled_oof": roc_auc_score(df["y_true"], df["score"]),
        "auc_fold_mean": float(np.mean(fold_aucs)),
        "auc_fold_sd": float(np.std(fold_aucs)),
    }
    for k in ["sensitivity", "specificity", "precision", "f1", "accuracy"]:
        vals = [m[k] for m in fold_metrics if k in m]
        row[f"{k}_95spec_mean"] = float(np.mean(vals)) if vals else np.nan
        row[f"{k}_95spec_sd"] = float(np.std(vals)) if vals else np.nan
    return row


def rows_from_per_fold(sel: pd.DataFrame, method: str) -> pd.DataFrame:
    """Per-sample prediction rows out of a per-fold results table (stored
    scores + train-derived 95%-spec thresholds)."""
    records = []
    for _, row in sel.iterrows():
        thr = float(row["full_model_thresholds"]["95_percent_specificity"])
        for sid, y, s in zip(row["test_sample_ids"], row["y_test"], row["y_pred_proba"]):
            records.append({
                "sample_id": sid, "method": method, "fold": int(row["fold"]),
                "y_true": int(y), "score": float(s), "threshold_95spec": thr,
                "orientation": np.nan, "n_imputed_train": 0,
            })
    return pd.DataFrame(records)


def export_panel_arm(t: pd.DataFrame, task: str, method: str) -> tuple[pd.DataFrame, dict] | None:
    """Load one scalar-panel nested-CV result set, pick the classifier per outer fold by the best
    inner-CV AUROC, validate against the frozen folds and stored numbers, and return
    (per-sample prediction rows, table row, per-fold configurations, selected rows)."""
    cfg = PANELS[method]
    panel_dir = variant_dir(TASKS[task]["panel_prefix"] + cfg["dir_suffix"])
    if not os.path.isdir(rp(panel_dir)):
        print("\n" + "!" * 78)
        print(f"WARNING: {panel_dir} not found - the {method} nested CV has not run yet")
        print("(build_scalar_panel_matrix.py --panel ... + nested_cv_scalar_panel.sh).")
        print(f"Exporting WITHOUT the {method} arm.")
        print("!" * 78)
        return None

    panel = load_results(f"{panel_dir}/results*.pkl", PANEL_KEEP + ["inner_cv_best_score"] + PANEL_EXTENDED_KEEP)
    assert (panel["feature"] == cfg["feature"]).all(), (
        f"{method}: unexpected feature name {panel['feature'].unique()}")
    assert panel["inner_cv_best_score"].notna().all(), (
        f"{method}: pickles lack inner_cv_best_score - fully nested selection impossible")
    # fully nested classifier selection: per outer fold, the best inner-CV AUROC;
    # the post-hoc best-mean-outer-AUC choice is only a printed diagnostic
    sel = panel.loc[panel.groupby("fold")["inner_cv_best_score"].idxmax()].sort_values("fold")
    assert len(sel) == 10 and sorted(sel["fold"]) == list(range(1, 11)), f"{method}: expected 10 folds"
    posthoc = best_mean_auc_config(panel, ["classifier"])
    chosen = sel["classifier"].value_counts().to_dict()
    print(f"\n{method}: {panel['classifier'].nunique()} classifiers; fully nested per-fold choice "
          f"{chosen}; mean outer AUC {sel['auc_full_model'].mean():.4f} "
          f"(post-hoc best-of-{panel['classifier'].nunique()} would be {posthoc['classifier']} "
          f"at {posthoc['mean_auc']:.4f}, optimism {posthoc['mean_auc'] - sel['auc_full_model'].mean():+.4f})")

    if cfg.get("frozen_folds", True):
        # frozen-fold identity with the headline run
        task_fold = dict(zip(t["sample_id"], t["fold"]))
        for _, row in sel.iterrows():
            folds_seen = {task_fold[s] for s in row["test_sample_ids"]}
            assert folds_seen == {int(row["fold"])}, (
                f"{method} fold {row['fold']} test samples span task folds {folds_seen}; "
                "outer partition differs from the headline run")
        fold_note = "fold partition matches the frozen folds"
    else:
        n_cov = sum(len(r["test_sample_ids"]) for _, r in sel.iterrows())
        assert set(s for _, r in sel.iterrows() for s in r["test_sample_ids"]) <= set(t["sample_id"])
        fold_note = f"reduced cohort ({n_cov} samples), own outer folds (sensitivity arm)"
    assert_predictions_match_stored(sel, method)
    print(f"{method}: {fold_note}; stored AUCs and "
          "95%-spec confusion matrices reproduce from the exported scores")

    pred = rows_from_per_fold(sel, method)
    configs = [{"method": method, "fold": int(r["fold"]), "feature": cfg["feature"],
                "selection_type": "none", "top_genes": "all",
                "classifier": r["classifier"], "hyperparameters": json.dumps(
                    {k.replace("clf__", ""): (v if isinstance(v, (int, float, str, bool)) else str(v))
                     for k, v in r["best_params"].items()}, sort_keys=True),
                "inner_cv_auc": float(r["inner_cv_best_score"]),
                "test_auc": float(r["auc_full_model"])} for _, r in sel.iterrows()]
    table_row = summarize_method(pred, method, "nested_cv")
    table_row["selected_classifier"] = "per-fold (inner CV): " + ", ".join(
        f"{k} x{v}" for k, v in sorted(chosen.items(), key=lambda kv: -kv[1]))
    table_row["posthoc_best_classifier"] = posthoc["classifier"]
    table_row["posthoc_auc_fold_mean"] = posthoc["mean_auc"]
    table_row["panel_features"] = "+".join(cfg["cols"])
    return pred, table_row, configs, sel


# --------------------------------------------------------------------------
# permutation importance of the panel covariates
# --------------------------------------------------------------------------
# Each fold's winning model is refit (seeded) and must reproduce the stored held-out
# predictions; importance = AUROC drop when one covariate is shuffled on the held-out fold.

def _fresh_classifier(name):
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    if name == "Support Vector Machine":
        return SVC(probability=True, random_state=42)
    if name == "Random Forest":
        return RandomForestClassifier(random_state=42)
    if name == "K-Nearest Neighbors":
        return KNeighborsClassifier()
    if name == "Logistic Regression":
        return LogisticRegression(random_state=42, max_iter=10000)
    if name == "Multi-Layer Perceptron":
        return MLPClassifier(random_state=42, early_stopping=True, validation_fraction=0.1)
    if name == "XGBoost":
        from xgboost import XGBClassifier
        return XGBClassifier(random_state=42, n_jobs=1, tree_method="hist", eval_metric="logloss")
    raise ValueError(name)


def load_panel_matrix(task: str, method: str):
    """Rebuild the exact design matrix model_hpc used: healthy rows first, then
    cancer rows, each in metadata order (the predefined-matrix path)."""
    cfg, tcfg = PANELS[method], TASKS[task]
    stem = cfg["feature"] + tcfg["matrix_suffix"]
    X = np.load(rp("extracted_features", stem + ".npy")).astype(float)
    meta = pd.read_csv(rp("extracted_features", stem + ".metadata.csv"))
    names = pd.read_csv(rp("extracted_features", stem + ".tf_names.csv")).iloc[:, 0].tolist()
    assert len(meta) == X.shape[0] and len(names) == X.shape[1]
    excl = set(excluded_sample_ids())
    if excl:   # cohort variant: model_hpc dropped these rows from the predefined matrix
        keep = ~meta["sample_id"].isin(excl).to_numpy()
        X, meta = X[keep], meta[keep].reset_index(drop=True)
    healthy = meta["disease"].isin(tcfg["control_labels"]).to_numpy()
    if tcfg["cancer_labels"] is None:
        cancer = ~healthy
    else:
        cancer = meta["disease"].isin(tcfg["cancer_labels"]).to_numpy()
    assert (healthy | cancer).all(), f"{task}: matrix rows outside the task's classes"
    order = np.concatenate([np.where(healthy)[0], np.where(cancer)[0]])
    X = X[order]
    y = np.concatenate([np.zeros(healthy.sum(), int), np.ones(cancer.sum(), int)])
    ids = meta["sample_id"].to_numpy()[order]
    return X, y, ids, names


def panel_importance(task: str, method: str, sel: pd.DataFrame, n_repeats: int = 30) -> list[dict]:
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    X, y, ids, names = load_panel_matrix(task, method)
    pos = {s: i for i, s in enumerate(ids)}
    rng = np.random.default_rng(0)
    base_oof = np.full(len(y), np.nan)
    perm_oof = np.full((n_repeats, len(names), len(y)), np.nan)
    per_fold_drop = []
    reproduced = []
    for _, row in sel.sort_values("fold").iterrows():
        test_idx = np.array(sorted(pos[s] for s in row["test_sample_ids"]))
        train_idx = np.setdiff1d(np.arange(len(y)), test_idx)
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", _fresh_classifier(row["classifier"]))])
        pipe.set_params(**row["best_params"]).fit(X[train_idx], y[train_idx])
        p_test = pipe.predict_proba(X[test_idx])[:, 1]
        stored = np.asarray(row["y_pred_proba"], float)
        stored_order = np.array([pos[s] for s in row["test_sample_ids"]])
        stored_aligned = stored[np.argsort(stored_order)]
        ok = np.allclose(p_test, stored_aligned, atol=1e-6)
        reproduced.append(ok)
        base_oof[test_idx] = p_test
        base_auc = roc_auc_score(y[test_idx], p_test)
        drops = np.zeros((n_repeats, len(names)))
        for r in range(n_repeats):
            for j in range(len(names)):
                Xp = X[test_idx].copy()
                Xp[:, j] = rng.permutation(Xp[:, j])
                pp = pipe.predict_proba(Xp)[:, 1]
                perm_oof[r, j, test_idx] = pp
                drops[r, j] = base_auc - roc_auc_score(y[test_idx], pp)
        per_fold_drop.append(drops.mean(axis=0))
    per_fold_drop = np.array(per_fold_drop)
    pooled_base = roc_auc_score(y, base_oof)
    pooled_drop = np.array([[pooled_base - roc_auc_score(y, perm_oof[r, j]) for j in range(len(names))]
                            for r in range(n_repeats)])
    n_ok = int(sum(reproduced))
    print(f"  importance {method}: {n_ok}/10 fold models reproduce the stored held-out predictions "
          f"(pooled OOF AUROC {pooled_base:.4f})")
    if n_ok < 10:
        import sklearn
        try:
            import xgboost
            xgb_v = xgboost.__version__
        except ImportError:
            xgb_v = "absent"
        print("!" * 78)
        print(f"WARNING: {method}: only {n_ok}/10 refit fold models reproduce the stored predictions - the "
              f"importances written for this panel are NOT those of the nested models. This happens when the "
              f"exporter runs under different library versions than the nested runs (here scikit-learn "
              f"{sklearn.__version__}, xgboost {xgb_v}; the runs were made with scikit-learn 1.9.1, xgboost 3.4.1). "
              "Re-run this exporter in that environment; figS10's check cell refuses non-reproducing importances.")
        print("!" * 78)
    return [{"method": method, "covariate": nm,
             "pooled_oof_auc_drop": float(pooled_drop[:, j].mean()),
             "pooled_oof_auc_drop_sd": float(pooled_drop[:, j].std()),
             "per_fold_auc_drop_mean": float(per_fold_drop[:, j].mean()),
             "per_fold_auc_drop_sd": float(per_fold_drop[:, j].std()),
             "pooled_oof_auc": pooled_base, "n_folds_reproduced": n_ok}
            for j, nm in enumerate(names)]


# --------------------------------------------------------------------------
# deep dive into the covariate panels
# --------------------------------------------------------------------------
# From the extended per-fold records and the fitted fold models (--store_fitted_models):
#   <prefix>_metrics.csv             per fold: winner, test and training-side metrics
#   <prefix>_inner_cv.csv            every hyperparameter candidate with its inner-CV score
#   <prefix>_model_summary.csv       per covariate of the winning model: coefficient, importance, scaler
#   <prefix>_train_scores.csv        per training sample: inner-CV and resubstitution score
#   <prefix>_partial_dependence.csv  1-D partial dependence of every covariate
#   <prefix>_decision_maps.csv       2-D response over the two most important covariates
# Panels whose pickles lack the extended records are skipped.

OP_NAMES = {"default_0.5": "default", "95_percent_specificity": "spec95",
            "90_percent_specificity": "spec90", "youden_optimal": "youden"}
PD_GRID_Q = np.linspace(0.025, 0.975, 21)
MAP_GRID = 25


def _has_extended(sel: pd.DataFrame) -> bool:
    return "test_metrics" in sel.columns and sel["test_metrics"].map(lambda v: isinstance(v, dict)).all()


def _load_fold_models(task: str, method: str, sel: pd.DataFrame) -> dict:
    """{fold: fitted pipeline} of each fold's winning classifier, from the
    models_<classifier>_<feature>_all_features.pkl files (--store_fitted_models).
    Empty when the files are absent or a classifier's library is missing."""
    cfg = PANELS[method]
    panel_dir = variant_dir(TASKS[task]["panel_prefix"] + cfg["dir_suffix"])
    out, cache = {}, {}
    for _, r in sel.iterrows():
        clf_safe = r["classifier"].replace(" ", "_").lower()
        path = rp(panel_dir, f"models_{clf_safe}_{cfg['feature']}_all_features.pkl")
        if path not in cache:
            try:
                with open(path, "rb") as fh:
                    cache[path] = pickle.load(fh)["models"]
            except FileNotFoundError:
                cache[path] = None
            except Exception as e:  # e.g. xgboost not importable in this environment
                print(f"  {method}: cannot load {os.path.basename(path)} ({type(e).__name__}: {e}) - "
                      "partial dependence skipped for those folds")
                cache[path] = None
        if cache[path] is not None and int(r["fold"]) in cache[path]:
            out[int(r["fold"])] = cache[path][int(r["fold"])]
    return out


def export_panel_deep_dive(task: str, panel_outs: list, imp_rows: list[dict]) -> None:
    cfg_t = TASKS[task]
    prefix = cfg_t["deep_dive_prefix"]
    metrics, landscape, model_rows, train_rows, pd_rows, map_rows = [], [], [], [], [], []
    imp = pd.DataFrame(imp_rows) if imp_rows else pd.DataFrame(columns=["method", "covariate", "pooled_oof_auc_drop"])
    for pred, table_row, configs, sel in panel_outs:
        method = table_row["method"]
        if not _has_extended(sel):
            print(f"  deep dive: {method} pickles carry no extended per-fold records (run predates "
                  "the extended reporting) - rerun the panel to include it")
            continue
        sel = sel.sort_values("fold")
        names = None
        for _, r in sel.iterrows():
            f = int(r["fold"])
            fm = r["fitted_model_summary"] if isinstance(r["fitted_model_summary"], dict) else {}
            names = r["feature_names"] if isinstance(r["feature_names"], list) else PANELS[method]["cols"]
            row = {"method": method, "fold": f, "classifier": r["classifier"],
                   "hyperparameters": json.dumps({k.replace("clf__", ""): (v if isinstance(v, (int, float, str, bool)) else str(v))
                                                  for k, v in r["best_params"].items()}, sort_keys=True),
                   "inner_cv_auc": float(r["inner_cv_best_score"]), "test_auc_stored": float(r["auc_full_model"]),
                   "classifier_class": fm.get("classifier_class"), "n_iter": fm.get("n_iter"),
                   "n_support": sum(fm["n_support"]) if "n_support" in fm else np.nan}
            for k, v in r["test_metrics"].items():
                row[f"test_{k}"] = v
            tm = r["train_metrics"]
            for k, v in tm["inner_cv"].items():
                row[f"train_inner_{k}"] = v
            for k, v in tm["resubstitution"].items():
                row[f"train_resub_{k}"] = v
            for op, m in r["operating_point_metrics"].items():
                for k, v in m.items():
                    row[f"op_{OP_NAMES.get(op, op)}_{k}"] = v
            metrics.append(row)
            # fitted model per covariate
            coef = fm.get("coef"); impf = fm.get("feature_importances")
            smean, sscale = fm.get("scaler_mean"), fm.get("scaler_scale")
            for j, cov in enumerate(names):
                model_rows.append({"method": method, "fold": f, "classifier": r["classifier"], "covariate": cov,
                                   "coef_per_sd": coef[j] if coef is not None and len(coef) == len(names) else np.nan,
                                   "intercept": fm["intercept"][0] if fm.get("intercept") else np.nan,
                                   "feature_importance": impf[j] if impf is not None and len(impf) == len(names) else np.nan,
                                   "scaler_mean": smean[j] if smean else np.nan,
                                   "scaler_scale": sscale[j] if sscale else np.nan})
            # training-side scores per sample
            for sid, yv, a, b in zip(r["train_sample_ids"], r["y_train"], r["inner_cv_train_scores"], r["resubstitution_train_scores"]):
                train_rows.append({"method": method, "fold": f, "sample_id": sid, "y_true": int(yv),
                                   "inner_cv_score": float(a), "resubstitution_score": float(b)})
        # the whole inner-CV landscape (all classifiers, all candidates, all folds)
        panel_dir = variant_dir(cfg_t["panel_prefix"] + PANELS[method]["dir_suffix"])
        allc = load_results(f"{panel_dir}/results*.pkl", ["classifier", "fold", "inner_cv_best_score", "inner_cv_results"])
        winner_of = dict(zip(sel["fold"].astype(int), sel["classifier"]))
        for _, r in allc.iterrows():
            if not isinstance(r.get("inner_cv_results"), list):
                continue
            f = int(r["fold"])
            for c in r["inner_cv_results"]:
                landscape.append({"method": method, "fold": f, "classifier": r["classifier"],
                                  "params": json.dumps(c["params"], sort_keys=True),
                                  "mean_inner_auc": c["mean_test_score"], "sd_inner_auc": c["std_test_score"],
                                  "rank_in_classifier": c["rank_test_score"],
                                  **{f"split{i}_inner_auc": v for i, v in enumerate(c["split_test_scores"])},
                                  "mean_fit_time_s": c["mean_fit_time"],
                                  "is_best_of_classifier": bool(c["is_best"]),
                                  "is_fold_winner": bool(c["is_best"] and winner_of.get(f) == r["classifier"])})
        # partial dependence + 2-D decision map from the stored fold models
        models = _load_fold_models(task, method, sel)
        if not models:
            print(f"  deep dive: {method}: no stored fold models (--store_fitted_models) - partial dependence skipped")
            continue
        X, y, ids, mnames = load_panel_matrix(task, method)
        pos = {s_: i for i, s_ in enumerate(ids)}
        top2 = None
        mi = imp[imp["method"] == method] if len(imp) else imp
        if len(mi) >= 2 and len(mnames) >= 2:
            top2 = list(mi.sort_values("pooled_oof_auc_drop", ascending=False)["covariate"].head(2))
        for _, r in sel.iterrows():
            f = int(r["fold"])
            if f not in models:
                continue
            pipe = models[f]
            test_idx = np.array(sorted(pos[s_] for s_ in r["test_sample_ids"]))
            train_idx = np.setdiff1d(np.arange(len(y)), test_idx)
            Xtr = X[train_idx]
            # the stored model must reproduce the stored held-out predictions
            p_test = pipe.predict_proba(X[test_idx])[:, 1]
            stored = np.asarray(r["y_pred_proba"], float)[np.argsort([pos[s_] for s_ in r["test_sample_ids"]])]
            if not np.allclose(p_test, stored, atol=1e-6):
                print(f"  deep dive: {method} fold {f}: stored model does NOT reproduce the stored predictions - skipped")
                continue
            for j, cov in enumerate(mnames):
                col = Xtr[:, j]
                grid = np.array([0.0, 1.0]) if set(np.unique(col)) <= {0.0, 1.0} else np.unique(np.quantile(col, PD_GRID_Q))
                for g in grid:
                    Xm = Xtr.copy(); Xm[:, j] = g
                    pp = pipe.predict_proba(Xm)[:, 1]
                    pd_rows.append({"method": method, "fold": f, "covariate": cov, "grid_value": float(g),
                                    "pd_mean": float(pp.mean()), "ice_p10": float(np.percentile(pp, 10)),
                                    "ice_p90": float(np.percentile(pp, 90))})
            if top2 and all(t in mnames for t in top2):
                jx, jy = mnames.index(top2[0]), mnames.index(top2[1])
                base = np.median(Xtr, axis=0)
                gx = np.linspace(*np.quantile(Xtr[:, jx], [0.025, 0.975]), MAP_GRID)
                gy = np.linspace(*np.quantile(Xtr[:, jy], [0.025, 0.975]), MAP_GRID)
                GX, GY = np.meshgrid(gx, gy)
                Xm = np.tile(base, (GX.size, 1)); Xm[:, jx] = GX.ravel(); Xm[:, jy] = GY.ravel()
                pp = pipe.predict_proba(Xm)[:, 1]
                for xv, yv, pv in zip(GX.ravel(), GY.ravel(), pp):
                    map_rows.append({"method": method, "fold": f, "cov_x": top2[0], "cov_y": top2[1],
                                     "x": float(xv), "y": float(yv), "p": float(pv)})
    for name, rows in [("metrics", metrics), ("inner_cv", landscape), ("model_summary", model_rows),
                       ("train_scores", train_rows), ("partial_dependence", pd_rows), ("decision_maps", map_rows)]:
        out = os.path.join(DATA_DIR, f"{prefix}_{name}.csv")
        if rows:
            pd.DataFrame(rows).to_csv(out, index=False)
            print(f"wrote {out} ({len(rows)} rows)")
        else:
            print(f"  deep dive: no rows for {name} - {os.path.basename(out)} not written")


def dd_fold_configs(per_fold: pd.DataFrame) -> list[dict]:
    """Per-fold configuration of a fully nested data-driven run."""
    out = []
    for _, r in per_fold.sort_values("fold").iterrows():
        hp = r["clf_hyperparams"] if isinstance(r["clf_hyperparams"], dict) else {}
        out.append({"method": "data_driven", "fold": int(r["fold"]), "feature": r["feature"],
                    "selection_type": r["selection_type"], "top_genes": int(r["top_genes"]),
                    "classifier": r["classifier"],
                    "hyperparameters": json.dumps({k.replace("clf__", ""): (v if isinstance(v, (int, float, str, bool)) else str(v))
                                                   for k, v in hp.items()}, sort_keys=True),
                    "inner_cv_auc": float(r["inner_best_score"]), "test_auc": float(r["auc_full_model"])})
    return out


def legacy_ichor_thresholds(fig3_pred: pd.DataFrame) -> dict:
    """Per-fold thresholds of the legacy ichorCNA arm exactly as fig3.ipynb
    derives them (duplicate-row weighting, out-of-task rows in the train side)."""
    ichor_all = fig3_pred[fig3_pred["method"] == "ichorcna"]
    expand = lambda d: d.loc[d.index.repeat(d["n_rows_in_source"])]  # noqa: E731
    out = {}
    for f in sorted(ichor_all["fold"].dropna().unique()):
        train = ichor_all[(ichor_all["fold"] != f) | ichor_all["fold"].isna()]
        train = expand(train).dropna(subset=["score"])
        out[f] = spec95_threshold(train["y_true"].to_numpy(int), train["score"].to_numpy(float))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", choices=["brca_crc", "pancan"], default="brca_crc")
    ap.add_argument("--recompute_gc", action="store_true",
                    help="Re-read every _BIN.csv for mean_gc instead of reusing the previously exported values")
    ap.add_argument("--scalars_only", action="store_true",
                    help="Rewrite the per-sample scalar table and stop: reuse the previously exported "
                         "columns and re-apply the BAM-derived sex from its summary "
                         "CSVs - e.g. after slurm/sex_inference.sh (no panel refits, runs anywhere)")
    args = ap.parse_args()
    task, cfg = args.task, TASKS[args.task]

    print(f"assembling the {cfg['n_samples']}-sample scalar table ({task}) ...")
    t = load_task_cohort(task)
    full_table = os.path.join(os.path.dirname(DATA_DIR), cfg["scalars_csv"]) if IS_VARIANT else None
    if args.scalars_only and not IS_VARIANT:
        full_table = os.path.join(DATA_DIR, cfg["scalars_csv"])   # reuse the table's own previous export
    if full_table and os.path.exists(full_table) and not args.recompute_gc:
        # per-sample scalars equal those of the full-cohort table; reuse them (some raw
        # inputs exist only on the cluster)
        full = pd.read_csv(full_table).set_index("sample_id")
        missing = set(t["sample_id"]) - set(full.index)
        assert not missing, f"{len(missing)} variant samples absent from {full_table}"
        reuse = ["stage"] + SCALARS + [c for c in EXTRA_COLS if c in full.columns]
        for col in reuse:
            t[col] = t["sample_id"].map(full[col])
        print(f"  scalars: {len(reuse) - 1} per-sample columns reused from "
              f"{os.path.relpath(full_table, REPO_ROOT)}")
        add_rerun_ichorcna_tf(t)   # the reused table can predate the rerun
    else:
        add_coverage(t)
        add_fragment_length_mean(t)
        add_age_sex(t)
        add_ichorcna_tf(t)
        add_mean_gc(t, None if args.recompute_gc else os.path.join(DATA_DIR, cfg["scalars_csv"]))
    add_bam_sex(t)   # always from the summary CSV: cheap, and fills tables exported before the inference ran

    n_expected = cfg["n_samples"] - len(excluded_sample_ids())
    assert len(t) == n_expected and t["y_true"].sum() == cfg["n_cancer"], \
        f"unexpected task cohort: {len(t)} samples (expected {n_expected}), {t['y_true'].sum()} cancer"
    assert sorted(t["fold"].unique()) == list(range(1, 11))
    for col in ["mean_coverage", "total_fragments", "fragment_length_mean", "mean_gc"]:
        assert t[col].notna().all(), f"{col}: unexpected missing values"

    cols = ["sample_id", "sample_name", "fold", "y_true", "disease", "stage"] + SCALARS + EXTRA_COLS
    scalars_out = os.path.join(DATA_DIR, cfg["scalars_csv"])
    t[cols].to_csv(scalars_out, index=False)
    print(f"wrote {scalars_out} ({len(t)} rows)")
    if args.scalars_only:
        return

    print("\nper-fold evaluation (train-fold orientation + threshold) ...")
    pred_frames = [evaluate_scalar(t, col) for col in SCALARS]
    bl_pred = pd.concat(pred_frames, ignore_index=True)

    panel_outs = [export_panel_arm(t, task, method) for method in PANELS]
    panel_outs = [p for p in panel_outs if p is not None]
    if panel_outs:
        bl_pred = pd.concat([bl_pred] + [p[0] for p in panel_outs], ignore_index=True)
    fold_configs = [c for p in panel_outs for c in p[2]]
    # permutation importance for the frozen-fold panels (models refit per fold)
    imp_rows = []
    for p_out in panel_outs:
        method = p_out[1]["method"]
        if PANELS[method].get("frozen_folds", True):
            imp_rows += panel_importance(task, method, p_out[3])
    imp_out = os.path.join(DATA_DIR, cfg["importance_csv"])
    pd.DataFrame(imp_rows).to_csv(imp_out, index=False)
    print(f"wrote {imp_out} ({len(imp_rows)} covariate x panel rows)")
    print("\ndeep dive (extended per-fold records + stored fold models) ...")
    export_panel_deep_dive(task, panel_outs, imp_rows)

    rows = []
    if IS_VARIANT and not nested_run_available(task):
        print(f"  {task}: fully nested run pending - data_driven / rna / atac / ichorCNA rows omitted "
              "from this export (rerun once nested_cv_results_cris/" + os.path.basename(cfg["nested_full_dir"]) + " exists)")
    elif task == "brca_crc":
        with open(rp(cfg["nested_full_dir"], "nested_full_per_fold.pkl"), "rb") as fh:
            dd_sel_b = pickle.load(fh).rename(columns={"test_auc": "auc_full_model", "n_top_genes": "top_genes"})
        fold_configs = dd_fold_configs(dd_sel_b) + fold_configs
        fig3_pred = pd.read_csv(os.path.join(DATA_DIR, "fig3_predictions.csv"))
        for method in ["data_driven", "rna", "atac"]:
            df = fig3_pred[(fig3_pred["method"] == method) & fig3_pred["fold"].notna()]
            rows.append(summarize_method(df, method, "nested_cv"))
        ichor_in_task = fig3_pred[(fig3_pred["method"] == "ichorcna") & fig3_pred["fold"].notna()]
        rows.append(summarize_method(ichor_in_task, "ichorcna", "raw_score_legacy",
                                     fold_thresholds=legacy_ichor_thresholds(fig3_pred)))
    else:
        # pancan: the data_driven arm from the fully nested run, written with the baselines
        dd_sel = load_pancan_nested_full()
        fold_configs = dd_fold_configs(dd_sel) + fold_configs
        assert len(dd_sel) == 10
        assert_predictions_match_stored(dd_sel, "pancan data_driven")
        dd_pred = rows_from_per_fold(dd_sel, "data_driven")
        dd_fold = dict(zip(t["sample_id"], t["fold"]))
        assert all(dd_fold[r.sample_id] == r.fold for r in dd_pred.itertuples()), \
            "pancan data_driven fold mismatch (internal)"
        bl_pred = pd.concat([dd_pred, bl_pred], ignore_index=True)
        rows.append(summarize_method(dd_pred, "data_driven", "nested_cv"))

    configs_out = os.path.join(DATA_DIR, cfg["configs_csv"])
    pd.DataFrame(fold_configs).to_csv(configs_out, index=False)
    print(f"wrote {configs_out} ({len(fold_configs)} per-fold configurations)")
    pred_out = os.path.join(DATA_DIR, cfg["pred_csv"])
    bl_pred.to_csv(pred_out, index=False)
    print(f"wrote {pred_out} ({len(bl_pred)} rows, {bl_pred['method'].nunique()} methods)")

    print("\nTable summary ...")
    rows.extend(p[1] for p in panel_outs)
    for col in SCALARS:
        df = bl_pred[bl_pred["method"] == f"bl_{col}"]
        rows.append(summarize_method(df, f"bl_{col}", "scalar_train_oriented"))
    table = pd.DataFrame(rows)
    method_cat = {"data_driven": "model", "rna": "model", "atac": "model",
                  "ichorcna": "competing_method"}
    method_cat.update({m: cfg["category"] for m, cfg in PANELS.items()})
    method_cat.update({f"bl_{c}": cat for c, cat in CATEGORIES.items()})
    table.insert(2, "category", table["method"].map(method_cat))
    table_out = os.path.join(DATA_DIR, cfg["table_csv"])
    table.to_csv(table_out, index=False)
    print(f"wrote {table_out}")

    with pd.option_context("display.width", 160):
        print(table[["method", "protocol", "auc_fold_mean", "auc_fold_sd",
                     "auc_pooled_oof", "sensitivity_95spec_mean"]].round(3).to_string(index=False))

    # sanity: the headline nested-CV AUC must reproduce the stored run
    if not (table["method"] == "data_driven").any():
        return
    dd = table[table["method"] == "data_driven"].iloc[0]
    if IS_VARIANT:
        print(f"cohort variant: data_driven per-fold AUC {dd['auc_fold_mean']:.4f} "
              f"(full-cohort headline {cfg['headline_auc']} not asserted)")
    else:
        assert abs(dd["auc_fold_mean"] - cfg["headline_auc"]) < 5e-4, (
            f"data_driven per-fold AUC {dd['auc_fold_mean']:.4f} drifted from the "
            f"headline {cfg['headline_auc']} - wrong source vintage?")


if __name__ == "__main__":
    main()
