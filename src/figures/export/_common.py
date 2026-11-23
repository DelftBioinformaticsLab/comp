"""Shared helpers for the figure source-data exporters.

These exporters read the raw experiment outputs (result pickles under
nested_cv_results_cris/, results/, and accessory_files/) and write the
per-figure source-data CSVs in src/figures/data/. They are the ONLY layer
allowed to touch result pickles or machine-specific paths; the figure
notebooks in src/figures/ read the CSVs alone.

Run from the repository root, e.g.:
    python src/figures/export/export_fig3.py

Exporters validate what they write against the numbers stored in the pickles
(AUCs, confusion matrices).
"""

from __future__ import annotations

import os
import pickle
import re
from glob import glob

import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA_DIR = os.path.join(REPO_ROOT, "src", "figures", "data")

# ---------------------------------------------------------------------------
# Result directories (repository-root relative; the cohort-variant suffix is
# applied below). NESTED_FULL_DIR: the fully nested BRCA/CRC run, with feature,
# ranking, k and classifier selected inside each outer fold.
# ---------------------------------------------------------------------------
NESTED_FULL_DIR = "nested_cv_results_cris/nested_full_brca_crc_nested_full"
# Fully nested PANCAN run (also the full-depth reference of the depth arms and size sweep).
PANCAN_NESTED_FULL_DIR = "nested_cv_results_cris/nested_full_pancan_nested_full"
# Depth arms: fully nested runs on features downsampled to a uniform depth
# (extracted_features/downsampled/cov<T>x_seed42, slurm/extract_features_downsampled.sh).
# Cross-depth evaluations (slurm/cross_depth_eval.sh): full-depth models evaluated on
# downsampled test folds; export_figS8 appends them as xd_cov1x / xd_cov0.1x when present.
CROSS_DEPTH_DIRS = {
    "brca_crc": "nested_cv_results_cris/cross_depth_brca_crc_nested_full",
    "pancan": "nested_cv_results_cris/cross_depth_pancan_nested_full",
}
DEPTH_ARM_DIRS = {
    "brca_crc": {
        "full": NESTED_FULL_DIR,
        "cov1x": "nested_cv_results_cris/nested_full_brca_crc_nested_full_cov1x",
        "cov0.1x": "nested_cv_results_cris/nested_full_brca_crc_nested_full_cov0.1x",
    },
    "pancan": {
        "full": PANCAN_NESTED_FULL_DIR,
        "cov1x": "nested_cv_results_cris/nested_full_pancan_nested_full_cov1x",
        "cov0.1x": "nested_cv_results_cris/nested_full_pancan_nested_full_cov0.1x",
    },
}
# Training-cohort size sweeps: one fully nested selection per sample fraction on
# nested stratified subsets (slurm/nested_full.sh sizesweep). The 100% point is
# NESTED_FULL_DIR for brca_crc and, if the sweep lacks it, PANCAN_NESTED_FULL_DIR.
SIZESWEEP_DIRS = {
    "brca_crc": "nested_cv_results_cris/nested_full_brca_crc_nested_full_sizesweep",
    "pancan": "nested_cv_results_cris/nested_full_pancan_nested_full_sizesweep",
}
# Seed replicates of the size sweep (other --nested_full_subsample_seed values; submit with
#   SUBSAMPLE_SEED=<s> EXPERIMENT_NAME=<task>_nested_full_seed<s>); export_figS9.py turns
# them into an across-seed band, reading the seed from each sweep CSV.
SIZESWEEP_SEED_REPLICATE_GLOBS = {
    "brca_crc": "nested_cv_results_cris/nested_full_brca_crc_nested_full_seed*_sizesweep",
    "pancan": "nested_cv_results_cris/nested_full_pancan_nested_full_seed*_sizesweep",
}
# Experiments that export_fig4 / export_figS5 reach outside the constants above
# (all produced by slurm/submit_nodeep_all.sh):
#   nested_cv_results_cris/best_model_best_model_search_{dd,rna,atac}_rev[_nodeep]
#   nested_cv_results_cris/external_validation_comprehensive_external_cris_on_jiang_{dd,rna,atac}_rev[_nodeep]
#   (Jiang arms of Fig S16, slurm/submit_jiang_arms.sh: the same two patterns with the
#    feature tags dd_cov_pipeline4 and cov_pipeline4; export_figS16_jiang_arms.py --from_runs)
#   nested_cv_results_cris/external_validation_comprehensive_clinical_cris_on_jiang_rev[_nodeep]
#   nested_cv_results_cris/cross_depth_{brca_crc,pancan}_nested_full[_nodeep]
# Fully nested runs trained and evaluated within the Jiang cohort (same procedure as
# NESTED_FULL_DIR; accessory_files/jiang_manifest.csv; slurm/submit_nested_full.sh full
# with DATASET=jiang). hcc_vs_healthy uses CONTROL_GROUPS=Healthy. export_fig4 skips
# tasks not on disk.
JIANG_NESTED_FULL_DIRS = {
    "hcc_vs_nonhcc": "nested_cv_results_jiang/nested_full_jiang_hcc_nested_full",
    "hcc_vs_healthy": "nested_cv_results_jiang/nested_full_jiang_hcc_vs_healthy_nested_full",
}
# k-sweep of the Jiang-trained fully nested run (submit_nested_full.sh ksweep with
# DATASET=jiang): one complete nested selection per k on the same folds, giving the
# within-Jiang AUROC-vs-k curve for fig 5j. Exported when present.
JIANG_KSWEEP_DIR = "nested_cv_results_jiang/nested_full_jiang_hcc_nested_full_ksweep"
DD_RESULT_DIRS = {
    "mean_diff": "nested_cv_results_cris/brca_crc_dd_rev",
    "random": "nested_cv_results_cris/brca_crc_dd_rev",
    "p_value": "nested_cv_results_cris/brca_crc_dd_rev",
    "fold_change": "nested_cv_results_cris/brca_crc_dd_rev",
    "weighted": "nested_cv_results_cris/brca_crc_dd_rev",
}
# Per-configuration sweep per task (slurm/nested_cv_dd.sh, one job per ranking function,
# all writing into one directory); DD_RESULT_DIRS above is its BRCA/CRC-only form.
DD_SWEEP_DIRS = {
    "brca_crc": "nested_cv_results_cris/brca_crc_dd_rev",
    "pancan": "nested_cv_results_cris/pancan_dd_rev",
}
DD_SWEEP_RANKINGS = ["mean_diff", "random", "p_value", "fold_change", "weighted"]
RNA_RESULT_DIR = "nested_cv_results_cris/brca_crc_rna_rev"
ATAC_RESULT_DIR = "nested_cv_results_cris/brca_crc_atac_rev"
# Region-number sweeps, one per (task, method) (slurm/nested_full.sh FEATURE_SET=dd|rna|atac
# ksweep): a fully nested selection at each fixed k on that method's region set
# (data_driven: 14,224 TSSs; rna: 792 Zhu regions, rcov; atac: 504 Ulz TF features).
# Each directory holds ksweep_k<k>/ per k, the trend CSV and ksweep_tuned/ (per fold the
# k with the best inner score, i.e. k tuned as a hyperparameter).
REGION_SWEEP_DIRS = {
    "brca_crc": {
        "data_driven": "nested_cv_results_cris/nested_full_brca_crc_nested_full_ksweep",
        "rna": "nested_cv_results_cris/nested_full_brca_crc_rna_nested_full_ksweep",
        "atac": "nested_cv_results_cris/nested_full_brca_crc_atac_nested_full_ksweep",
    },
    "pancan": {
        "data_driven": "nested_cv_results_cris/nested_full_pancan_nested_full_ksweep",
        "rna": "nested_cv_results_cris/nested_full_pancan_rna_nested_full_ksweep",
        "atac": "nested_cv_results_cris/nested_full_pancan_atac_nested_full_ksweep",
    },
}
# Standalone fully nested runs over the same k grid: for data_driven the main arms; for
# the comparators optional cross-checks of the sweep-derived tuned arm.
REGION_SWEEP_STANDALONE_DIRS = {
    "brca_crc": {
        "data_driven": NESTED_FULL_DIR,
        "rna": "nested_cv_results_cris/nested_full_brca_crc_rna_nested_full",
        "atac": "nested_cv_results_cris/nested_full_brca_crc_atac_nested_full",
    },
    "pancan": {
        "data_driven": PANCAN_NESTED_FULL_DIR,
        "rna": "nested_cv_results_cris/nested_full_pancan_rna_nested_full",
        "atac": "nested_cv_results_cris/nested_full_pancan_atac_nested_full",
    },
}
# Fixed-region comparator arms (all regions, no selection; classifier chosen per outer
# fold on inner-CV scores).
FIXED_COMPARATOR_DIRS = {
    "brca_crc": {"rna": RNA_RESULT_DIR, "atac": ATAC_RESULT_DIR},
    "pancan": {"rna": "nested_cv_results_cris/pancan_rna_rev",
               "atac": "nested_cv_results_cris/pancan_atac_rev"},
}
ICHOR_AGG_CSV = "accessory_files/cris_stage_vaf_coverage_ichor_aggregate.csv"
ICHOR_CSV = "accessory_files/cris_stage_vaf_coverage_ichor.csv"
# The two tables above are used for their stage and disease columns only; tumour fractions
# come from the ichorCNA run with the cohort panel of normals:
ICHOR_RERUN_TSV = "ichorCNA/ichorCNA_summary.tsv"


def apply_rerun_ichorcna_tf(ichor: pd.DataFrame, id_col: str = "ID") -> pd.DataFrame:
    """Replace `ichorcna_tf` by the value of the ichorCNA rerun and keep only the samples the
    rerun covers, so that no tumour fraction estimated with the old panel of normals is left
    (rows of other samples would be on a different scale)."""
    rerun = pd.read_csv(rp(ICHOR_RERUN_TSV), sep="\t").set_index("sample")["tumor_fraction"]
    out = ichor[ichor[id_col].isin(rerun.index)].copy()
    out["ichorcna_tf"] = out[id_col].map(rerun).to_numpy()
    assert out["ichorcna_tf"].notna().all(), "sample without a tumour fraction in the ichorCNA rerun"
    print(f"  ichorCNA tumour fractions from {ICHOR_RERUN_TSV}: {len(out)} samples kept, "
          f"{len(ichor) - len(out)} without a rerun value dropped")
    return out

# ---------------------------------------------------------------------------
# Cohort variant. The default COHORT_VARIANT=no_deep uses the runs without the 30
# deep-sequenced healthy samples (model_hpc.py --exclude_samples
# accessory_files/cris_exclude_deep_healthy.txt): Cristiano result directories carry
# VARIANT_SUFFIX on the experiment name (e.g. brca_crc_rna_rev_nodeep) and CSVs go to
# src/figures/data/no_deep/. COHORT_VARIANT=all is the full cohort (src/figures/data/).
# Jiang-trained runs keep their paths. Full-cohort headline assertions are skipped
# under a variant (IS_VARIANT).
COHORT_VARIANT = os.environ.get("COHORT_VARIANT", "no_deep")
VARIANT_SUFFIXES = {"all": "", "no_deep": "_nodeep"}
if COHORT_VARIANT not in VARIANT_SUFFIXES:
    raise SystemExit(f"COHORT_VARIANT={COHORT_VARIANT!r}: expected one of {sorted(VARIANT_SUFFIXES)}")
VARIANT_SUFFIX = VARIANT_SUFFIXES[COHORT_VARIANT]
IS_VARIANT = COHORT_VARIANT != "all"
EXCLUDE_SAMPLES_FILES = {"no_deep": "accessory_files/cris_exclude_deep_healthy.txt"}


# Step suffixes that slurm/nested_full.sh appends AFTER the experiment name
# (nested_full_<EXPERIMENT_NAME>_ksweep, _sizesweep, _test): the variant suffix
# belongs to the experiment name, i.e. in front of them.
_STEP_SUFFIXES = ("_ksweep", "_sizesweep", "_test")


def variant_dir(path: str) -> str:
    """The variant's twin of a Cristiano result directory: the variant suffix
    appended to the experiment-name component (in front of a trailing
    nested_full step suffix). Jiang / other-cohort paths are returned unchanged; under
    COHORT_VARIANT=all every path is returned unchanged."""
    if not VARIANT_SUFFIX:
        return path
    if not path.startswith("nested_cv_results_cris/"):
        return path
    head, _, name = path.partition("/")
    first, sep, rest = name.partition("/")
    for step in _STEP_SUFFIXES:
        if first.endswith(step):
            first = first[: -len(step)] + VARIANT_SUFFIX + step
            break
    else:
        first = first + VARIANT_SUFFIX
    return f"{head}/{first}{sep}{rest}" if sep else f"{head}/{first}"


def _apply_variant(obj):
    if isinstance(obj, str):
        return variant_dir(obj)
    if isinstance(obj, dict):
        return {k: _apply_variant(v) for k, v in obj.items()}
    return obj


if IS_VARIANT:
    DATA_DIR = os.path.join(DATA_DIR, COHORT_VARIANT)
    for _name in ["NESTED_FULL_DIR", "PANCAN_NESTED_FULL_DIR", "CROSS_DEPTH_DIRS", "DEPTH_ARM_DIRS",
                  "SIZESWEEP_DIRS", "SIZESWEEP_SEED_REPLICATE_GLOBS", "DD_RESULT_DIRS", "DD_SWEEP_DIRS", "RNA_RESULT_DIR",
                  "ATAC_RESULT_DIR", "REGION_SWEEP_DIRS", "REGION_SWEEP_STANDALONE_DIRS", "FIXED_COMPARATOR_DIRS"]:
        globals()[_name] = _apply_variant(globals()[_name])
    print(f"[_common] COHORT_VARIANT={COHORT_VARIANT}: result dirs get the suffix {VARIANT_SUFFIX!r}, "
          f"CSVs go to {os.path.relpath(DATA_DIR, REPO_ROOT)}/")


def headline_nested_dir(path: str) -> str:
    """The directory that holds a task's headline fully nested run (aggregate files
    nested_full_per_fold.pkl / nested_full_oof_predictions.csv / nested_full_summary.json):
    `path` if it has them, else the k-sweep's ksweep_tuned/ (per fold the k with the best
    inner score) when that has them, else `path` unchanged."""
    if os.path.exists(os.path.join(REPO_ROOT, path, "nested_full_oof_predictions.csv")):
        return path
    tuned = f"{path}_ksweep/ksweep_tuned"
    if (os.path.exists(os.path.join(REPO_ROOT, tuned, "nested_full_oof_predictions.csv"))
            and os.path.exists(os.path.join(REPO_ROOT, tuned, "nested_full_per_fold.pkl"))):
        print(f"[_common] {path}: no aggregate; using the k-sweep's tuned arm {tuned}")
        return tuned
    return path


if IS_VARIANT:
    NESTED_FULL_DIR = headline_nested_dir(NESTED_FULL_DIR)
    PANCAN_NESTED_FULL_DIR = headline_nested_dir(PANCAN_NESTED_FULL_DIR)
    DEPTH_ARM_DIRS["brca_crc"]["full"] = NESTED_FULL_DIR
    DEPTH_ARM_DIRS["pancan"]["full"] = PANCAN_NESTED_FULL_DIR


def excluded_sample_ids() -> list[str]:
    """IDs excluded under the current cohort variant (empty for 'all')."""
    f = EXCLUDE_SAMPLES_FILES.get(COHORT_VARIANT)
    if not f:
        return []
    with open(rp(f)) as fh:
        ids = [ln.split("#", 1)[0].strip() for ln in fh]
    return sorted({s for s in ids if s})


def rp(*parts: str) -> str:
    """Path relative to the repo root."""
    return os.path.join(REPO_ROOT, *parts)


def load_results(pattern: str, keep_cols: list[str], extra: dict | None = None) -> pd.DataFrame:
    """Load all result pickles matching `pattern` (repo-root relative), keep
    only `keep_cols`, and concatenate."""
    paths = sorted(glob(rp(pattern)))
    if not paths:
        raise FileNotFoundError(f"No result pickles match {pattern!r}")
    frames = []
    for p in paths:
        with open(p, "rb") as fh:
            df = pickle.load(fh)
        df = df[[c for c in keep_cols if c in df.columns]]
        if extra:
            for k, v in extra.items():
                df[k] = v
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def best_mean_auc_config(df: pd.DataFrame, group_cols: list[str]) -> dict:
    """Configuration (group_cols) with the highest mean auc_full_model.

    Same selection as find_best_datadriven_config / find_best_profile_classifier
    in src/comp/hypo_nestedcv_plots.ipynb.
    """
    grouped = df.groupby(group_cols)["auc_full_model"].mean().reset_index()
    best = grouped.loc[grouped["auc_full_model"].idxmax()]
    out = {c: best[c] for c in group_cols}
    out["mean_auc"] = float(best["auc_full_model"])
    return out


# model_hpc.THRESHOLD_SOURCE_INNER_CV: thresholds read off the selected
# configuration's scores on the inner validation folds of the nested procedure.
THRESHOLD_SOURCE_INNER_CV = "inner_cv_selected_config"


def warn_if_superseded_thresholds(thresholds, label: str) -> bool:
    """True (and a printed warning) when any full_model_thresholds dict was not derived
    from the inner validation folds. Such runs need slurm/nested_full.sh
    refresh_thresholds or a rerun before their operating points are reported."""
    stale = sum(1 for t in thresholds
                if not isinstance(t, dict) or t.get("threshold_source") != THRESHOLD_SOURCE_INNER_CV)
    if stale:
        print(f"  !! {label}: {stale} fold(s) carry operating thresholds from the superseded "
              "separate 5-fold pass, not from the inner validation folds - refresh or rerun "
              "this experiment before quoting sensitivity / specificity from it")
    return bool(stale)


def assert_predictions_match_stored(rows: pd.DataFrame, label: str) -> None:
    """For each fold row of a results DataFrame, recompute AUC and the
    95%-specificity confusion matrix from (y_test, y_pred_proba) and assert
    they equal the stored values. Guarantees the exported per-sample scores
    reproduce the pipeline numbers."""
    from sklearn.metrics import roc_auc_score

    warn_if_superseded_thresholds(list(rows["full_model_thresholds"]), label)
    for _, row in rows.iterrows():
        y = np.asarray(row["y_test"], dtype=int)
        s = np.asarray(row["y_pred_proba"], dtype=float)
        auc_re = roc_auc_score(y, s)
        assert np.isclose(auc_re, row["auc_full_model"], atol=1e-10), (
            f"{label} fold {row['fold']}: recomputed AUC {auc_re} != stored {row['auc_full_model']}"
        )
        thr = row["full_model_thresholds"]["95_percent_specificity"]
        cm_stored = row["confusion_matrix_full_model"]["95_percent_specificity"]
        cm_stored = np.asarray(cm_stored, dtype=int)
        y_pred = (s >= thr).astype(int)
        tn = int(np.sum((y == 0) & (y_pred == 0)))
        fp = int(np.sum((y == 0) & (y_pred == 1)))
        fn = int(np.sum((y == 1) & (y_pred == 0)))
        tp = int(np.sum((y == 1) & (y_pred == 1)))
        cm_re = np.array([[tn, fp], [fn, tp]])
        assert (cm_re == cm_stored).all(), (
            f"{label} fold {row['fold']}: confusion matrix at 95% specificity does not "
            f"reproduce from scores + stored threshold ({cm_re.tolist()} vs {cm_stored.tolist()}); "
            "check >= convention / threshold provenance"
        )


# ---------------------------------------------------------------------------
# Genetic sex from the BAM (src/comp/compute_sex.py, slurm/sex_inference.sh):
# results/<dataset>_sex/<dataset>_sex.csv carries R_Y (Skoglund 2013) and R_X
# (Mittnik 2016) with their own calls, and sex_call = the call when both agree.
# Samples the published thresholds leave unassigned are resolved (a third value
# could leak the label):
#   1. both metrics agree                  -> that call            ('both')
#   2. exactly one metric makes a call      -> that call            ('one metric')
#   3. neither does                         -> male if R_X < 0.8    ('r_x side')
#   (two metrics calling opposite sexes stay unassigned)
SEX_RX_SPLIT = 0.8


def resolve_sex_call(s: pd.DataFrame) -> pd.DataFrame:
    """sex_call_resolved (female / male / unassigned) and sex_call_how per row of a
    compute_sex.py summary table (columns sex_call, r_y_call, r_x_call, r_x)."""
    def one(r):
        if r["sex_call"] != "unassigned":
            return r["sex_call"], "both"
        calls = {r["r_y_call"], r["r_x_call"]} - {"unassigned"}
        if len(calls) == 1:
            return calls.pop(), "one metric"
        if len(calls) == 2:
            return "unassigned", "contradictory"
        return ("female" if r["r_x"] >= SEX_RX_SPLIT else "male"), "r_x side"
    out = s.apply(lambda r: pd.Series(one(r), index=["sex_call_resolved", "sex_call_how"]), axis=1)
    return pd.concat([s, out], axis=1)
