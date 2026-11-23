"""Source data for Figure 4 and its supplementary figures: the pan-cancer nested CV and the
application to the Jiang cohort (HCC vs non-HCC comparators).
  fig4a_*            pan-cancer nested CV, ROC per method and per cancer type;
  fig4bc_*           Jiang: ROC of the deployed Cristiano-trained models, sensitivity at the
                     frozen 95%-specificity threshold;
  fig4gi_*           Jiang subgroups and the fully nested run trained within Jiang;
  fig4j_*, fig4kl_*  external AUROC vs number of TSSs, pipeline vs clinical standardization.

Inputs: PANCAN_NESTED_FULL_DIR and the pancan RNA / ATAC runs, cris_stage_vaf_coverage_ichor.csv,
the external_validation_comprehensive_external_cris_on_jiang_{dd,rna,atac} runs with their
training summaries, JIANG_NESTED_FULL_DIRS and JIANG_KSWEEP_DIR (_common.py). Missing runs are
skipped with a notice.

Writes to src/figures/data/<COHORT_VARIANT>/: fig4a_*, fig4bc_*, fig4gi_*, fig4j_*, fig4kl_*
and figS7_auc_vs_k.csv.

Run from the repository root:
    python src/figures/export/export_fig4.py
"""

import json
import os
import pickle
import sys
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, os.path.dirname(__file__))
from _common import (DATA_DIR, JIANG_KSWEEP_DIR, JIANG_NESTED_FULL_DIRS,  # noqa: E402
                     PANCAN_NESTED_FULL_DIR, THRESHOLD_SOURCE_INNER_CV, apply_rerun_ichorcna_tf, best_mean_auc_config,
                     excluded_sample_ids, load_results, rp, variant_dir)


class _TolerantUnpickler(pickle.Unpickler):
    """Unpickler for summaries that reference estimator classes not installed here (xgboost):
    unknown classes become inert placeholders; plain dict/float fields load unchanged."""

    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ImportError, AttributeError):
            return type(name, (), {"__setstate__": lambda self, state: None,
                                   "__reduce__": lambda self: (object, ())})

# Pan-cancer sources (variant-mapped: COHORT_VARIANT=no_deep uses the *_nodeep directories).
# SUBMITTED_DIRS is provenance only.
PANCAN_DD_DIR = PANCAN_NESTED_FULL_DIR   # variant-mapped in _common
PANCAN_RNA_DIR = variant_dir("nested_cv_results_cris/pancan_rna_rev")
PANCAN_ATAC_DIR = variant_dir("nested_cv_results_cris/pancan_atac_rev")
SUBMITTED_DIRS = {"dd": "nested_cv_results_cris/pancan_dd",
                  "rna": "nested_cv_results_cris/nested_rna_pancan",
                  "atac": "nested_cv_results_cris/pancan_atac - Copy"}
ICHOR_CSV = "accessory_files/cris_stage_vaf_coverage_ichor.csv"
# External application to Jiang: the *_rev directories (slurm/external_validation.sh with
# TRAIN_SUFFIX=_rev EXPERIMENT_SUFFIX=_rev) take precedence when present.
def _ext_dir(ft: str) -> str:
    base = f"nested_cv_results_cris/external_validation_comprehensive_external_cris_on_jiang_{ft}"
    d = f"{base}_rev" if os.path.isdir(rp(f"{base}_rev")) else base
    return variant_dir(d)   # cohort variant: deployment models trained without the excluded samples


EXT_DIRS = {ft: _ext_dir(ft) for ft in ["dd", "rna", "atac"]}
KEEP = ["feature", "classifier", "top_genes", "fold", "auc_full_model",
        "auc_per_cancer_type", "test_sample_ids", "y_pred_proba", "y_test"]


def nested_classifier_selection(cand: pd.DataFrame, locked_feature: str, label: str):
    """Per outer fold, the classifier with the best inner-CV AUROC on the locked feature.
    Returns (per-fold rows, summary dict incl. the post-hoc alternative)."""
    cand = cand[cand["feature"] == locked_feature]
    assert len(cand) and cand["inner_cv_best_score"].notna().all(), (
        f"{label}: pickles lack inner_cv_best_score (rerun the nested CV with the current model_hpc.py)")
    sel = cand.loc[cand.groupby("fold")["inner_cv_best_score"].idxmax()].sort_values("fold")
    posthoc = best_mean_auc_config(cand, ["classifier"])
    info = {"per_fold": sel["classifier"].value_counts().to_dict(),
            "modal_classifier": sel["classifier"].mode().iloc[0],
            "mean_auc": float(sel["auc_full_model"].mean()),
            "posthoc_classifier": posthoc["classifier"], "posthoc_mean_auc": posthoc["mean_auc"]}
    print(f"  {label}: fully nested per-fold classifier choice {info['per_fold']}; mean outer AUC "
          f"{info['mean_auc']:.4f} (post-hoc best-of-{cand['classifier'].nunique()} "
          f"{posthoc['classifier']} would give {posthoc['mean_auc']:.4f})")
    return sel, info


def export_fig4a() -> None:
    # data-driven: the fully nested pan-cancer run
    with open(rp(PANCAN_DD_DIR, "nested_full_per_fold.pkl"), "rb") as fh:
        dd_sel = pickle.load(fh).rename(columns={"test_auc": "auc_full_model",
                                                 "n_top_genes": "top_genes"})
    dd_sel = dd_sel.sort_values("fold")
    dd_modal = {"feature": dd_sel["feature"].mode().iloc[0],
                "classifier": dd_sel["classifier"].mode().iloc[0],
                "weighting_function": dd_sel["selection_type"].mode().iloc[0],
                "top_genes": int(dd_sel["top_genes"].mode().iloc[0])}
    best_dd = dict(dd_modal, mean_auc=float(dd_sel["auc_full_model"].mean()))
    n_distinct = len(dd_sel[["feature", "selection_type", "top_genes", "classifier"]].drop_duplicates())

    rna = load_results(f"{PANCAN_RNA_DIR}/results*.pkl", KEEP + ["inner_cv_best_score"])
    atac = load_results(f"{PANCAN_ATAC_DIR}/results*.pkl", KEEP + ["inner_cv_best_score"])
    atac_feat = "ulz_atac_tfbs_features"
    rna_sel, best_rna = nested_classifier_selection(rna, "rcov", "rna")
    atac_sel, best_atac = nested_classifier_selection(atac, atac_feat, "atac")

    print("PANCAN selected configs:")
    print(f"  data_driven: fully nested per fold, modal {dd_modal} ({n_distinct} distinct), "
          f"mean AUC {best_dd['mean_auc']:.4f}")
    print(f"  rna:         {best_rna}")
    print(f"  atac:        {best_atac}")
    for name, sel in [("dd", dd_sel), ("rna", rna_sel), ("atac", atac_sel)]:
        assert len(sel) == 10, f"{name}: expected 10 folds, got {len(sel)}"

    def fold_map(sel):
        return {int(r["fold"]): frozenset(r["test_sample_ids"]) for _, r in sel.iterrows()}

    assert fold_map(dd_sel) == fold_map(rna_sel) == fold_map(atac_sel), \
        "PANCAN fold partitions differ between arms"

    ichor = pd.read_csv(rp(ICHOR_CSV), index_col=0)
    mult = ichor.groupby("ID").size()
    dup_ids = set(mult[mult > 1].index)
    for sid in dup_ids:
        grp = ichor[ichor["ID"] == sid]
        assert (grp.nunique() <= 1).all(), f"duplicate rows for {sid} differ"
    ichor = ichor.drop_duplicates(subset="ID")
    excl = set(excluded_sample_ids())   # cohort variant: excluded samples leave the ichorCNA arm too
    if excl:
        ichor = ichor[~ichor["ID"].isin(excl)]
        print(f"  ichorCNA table: {len(excl)} excluded sample(s) removed (cohort variant)")
    ichor = apply_rerun_ichorcna_tf(ichor)   # tumour fractions of the rerun (cohort panel of normals)
    dup_ids &= set(ichor["ID"])
    print(f"  ichorCNA table: {len(ichor)} unique samples ({len(dup_ids)} duplicated)")
    ann = ichor.set_index("ID")

    # validate that per-sample scores reproduce stored full-model and
    # per-cancer-type AUCs (the latter is what Figure S6 recomputes)
    for name, sel in [("data_driven", dd_sel), ("rna", rna_sel), ("atac", atac_sel)]:
        for _, row in sel.iterrows():
            y = np.asarray(row["y_test"], dtype=int)
            s = np.asarray(row["y_pred_proba"], dtype=float)
            assert np.isclose(roc_auc_score(y, s), row["auc_full_model"], atol=1e-10)
            if "auc_per_cancer_type" not in row.index or not isinstance(row["auc_per_cancer_type"], dict):
                continue  # the fully nested per-fold table stores no per-type AUCs (S6 recomputes them)
            disease = pd.Series(row["test_sample_ids"]).map(ann["disease"]).to_numpy()
            for ct, auc_stored in row["auc_per_cancer_type"].items():
                m = (disease == "Healthy") | (disease == ct)
                if len(np.unique(y[m])) < 2:
                    continue
                auc_re = roc_auc_score(y[m], s[m])
                assert np.isclose(auc_re, auc_stored, atol=1e-10), (
                    f"{name} fold {row['fold']} {ct}: {auc_re} != stored {auc_stored}")
        print(f"  {name}: stored full-model (and, where stored, per-cancer-type) AUCs reproduced")

    records = []
    for method, sel in [("data_driven", dd_sel), ("rna", rna_sel), ("atac", atac_sel)]:
        for _, row in sel.iterrows():
            for sid, y, s in zip(row["test_sample_ids"], row["y_test"], row["y_pred_proba"]):
                records.append({"sample_id": sid, "method": method,
                                "fold": int(row["fold"]), "y_true": int(y),
                                "score": float(s)})
    pred = pd.DataFrame(records)

    dd_folds = fold_map(dd_sel)
    sample_to_fold = {sid: f for f, ids in dd_folds.items() for sid in ids}
    ichor_rows = pd.DataFrame({
        "sample_id": ichor["ID"], "method": "ichorcna",
        "fold": ichor["ID"].map(sample_to_fold),
        "y_true": (ichor["disease"] != "Healthy").astype(int),
        "score": ichor["ichorcna_tf"],
    })
    pred = pd.concat([pred, ichor_rows], ignore_index=True)
    pred["n_rows_in_source"] = np.where(
        (pred["method"] == "ichorcna") & pred["sample_id"].isin(dup_ids), 2, 1)
    pred["disease"] = pred["sample_id"].map(ann["disease"])

    out = os.path.join(DATA_DIR, "fig4a_predictions.csv")
    pred.to_csv(out, index=False)
    print(f"wrote {out} ({len(pred)} rows; "
          f"{pred[pred.method == 'data_driven']['sample_id'].nunique()} task samples)")

    cfg = pd.DataFrame([
        {"method": "data_driven", **{k: best_dd[k] for k in
         ["feature", "classifier", "weighting_function", "top_genes"]},
         "source_dir": PANCAN_DD_DIR, "selection_mean_auc": best_dd["mean_auc"],
         "note": f"fully nested: config chosen per fold; modal shown ({n_distinct} distinct)"},
        {"method": "rna", "feature": "rcov", "classifier": best_rna["modal_classifier"],
         "weighting_function": "", "top_genes": "", "source_dir": PANCAN_RNA_DIR,
         "selection_mean_auc": best_rna["mean_auc"],
         "note": f"fully nested: classifier per fold on inner CV {best_rna['per_fold']}; modal shown "
                 f"(post-hoc best-of-6 {best_rna['posthoc_classifier']} would give {best_rna['posthoc_mean_auc']:.4f})"},
        {"method": "atac", "feature": atac_feat, "classifier": best_atac["modal_classifier"],
         "weighting_function": "", "top_genes": "", "source_dir": PANCAN_ATAC_DIR,
         "selection_mean_auc": best_atac["mean_auc"],
         "note": f"fully nested: classifier per fold on inner CV {best_atac['per_fold']}; modal shown "
                 f"(post-hoc best-of-6 {best_atac['posthoc_classifier']} would give {best_atac['posthoc_mean_auc']:.4f})"},
        {"method": "ichorcna", "feature": "ichorcna_tf", "classifier": "",
         "weighting_function": "", "top_genes": "", "source_dir": ICHOR_CSV,
         "selection_mean_auc": ""},
    ])
    cfg.to_csv(os.path.join(DATA_DIR, "fig4a_config.csv"), index=False)
    print("wrote fig4a_config.csv")


def _train_threshold_95spec(best_model_path: str) -> tuple[float | None, str]:
    """The deployed model's 95%-specificity threshold frozen on the training cohort, and its
    source: "training_inner_cv" (deployment_thresholds from the search's own folds; expected),
    "training_oof" (separate 10-fold pass; flagged) or "training_insample_legacy" (in-sample
    ROC of old summaries; can be infinite)."""
    summary_path = Path(rp(str(best_model_path).replace("best_model_GridSearchCV_", "summary_")))
    if not summary_path.exists():
        # the stored path may be absolute from the cluster; retry inside the repo
        parts = str(best_model_path).replace("\\", "/").split("/")
        if "nested_cv_results_cris" in parts:
            rel = "/".join(parts[parts.index("nested_cv_results_cris"):])
            summary_path = Path(rp(rel.replace("best_model_GridSearchCV_", "summary_")))
    if not summary_path.exists():
        print(f"  ! training summary not found for {best_model_path}")
        return None, "missing"
    with open(summary_path, "rb") as fh:
        summ = _TolerantUnpickler(fh).load()
    dep = summ.get("deployment_thresholds")
    if dep and np.isfinite(dep.get("95_percent_specificity", np.nan)):
        if dep.get("threshold_source") == THRESHOLD_SOURCE_INNER_CV:
            return float(dep["95_percent_specificity"]), "training_inner_cv"
        print(f"  !! {summary_path.name}: frozen threshold comes from the superseded separate "
              "10-fold pass, not from the search's own folds - refresh the best-model search "
              "(REFRESH_DEPLOYMENT_THRESHOLDS=1) and rerun the external validation")
        return float(dep["95_percent_specificity"]), "training_oof"
    roc_ct = summ["full_dataset_evaluation"]["roc_per_cancer_type"]
    thrs = [v["thresholds"]["95_percent_specificity"] for v in roc_ct.values()
            if "thresholds" in v and "95_percent_specificity" in v["thresholds"]]
    print("  ! no deployment_thresholds in the training summary (pre-revision best-model "
          "search): falling back to the in-sample threshold — rerun the best-model search")
    return (float(np.mean(thrs)) if thrs else None), "training_insample_legacy"


def _flatten_fixed_threshold(method: str, fte: dict) -> list[dict]:
    """fixed_threshold_evaluation (model_hpc._fixed_threshold_evaluation) -> rows."""
    rows = []
    for op, entry in (fte or {}).items():
        for group, c in [("all_non_hcc", entry["overall"])] + list(entry["per_control_group"].items()):
            rows.append({"method": method, "operating_point": op, "threshold": entry["threshold"],
                         "comparison_group": group,
                         "tp": c["tp"], "fn": c["fn"], "tn": c["tn"], "fp": c["fp"],
                         "sensitivity": c["sensitivity"],
                         "sensitivity_ci_low": c["sensitivity_ci95"][0],
                         "sensitivity_ci_high": c["sensitivity_ci95"][1],
                         "specificity": c["specificity"],
                         "specificity_ci_low": c["specificity_ci95"][0],
                         "specificity_ci_high": c["specificity_ci95"][1]})
    return rows


def export_fig4bc() -> None:
    manifest = pd.read_csv(rp("accessory_files/jiang_manifest.csv")).set_index("ID")
    pred_rows, cfg_rows, fixed_rows = [], [], []
    for ft, ext_dir in EXT_DIRS.items():
        with open(rp(f"{ext_dir}/comprehensive_validation_summary.pkl"), "rb") as fh:
            summary = pickle.load(fh)

        # deployed model = best feature by training CV AUC
        fr ={k: v for k, v in summary["feature_results"].items() if v is not None}
        best_feat = max(fr, key=lambda k: fr[k].get("training_cv_auc", 0))

        detail = {}
        for p in sorted(glob(rp(f"{ext_dir}/external_validation_*_results.pkl"))):
            with open(p, "rb") as fh:
                d = pickle.load(fh)
            detail[d.get("best_model_info", {}).get("feature", os.path.basename(p))] = d
        bp = detail[best_feat]

        y = np.asarray(bp["y_true"], dtype=int)
        s = np.asarray(bp["y_pred_proba"], dtype=float)
        # stored AUC uses y_scores_for_roc (decision_function when probabilities are
        # degenerate); the exported score is predict_proba, the scale of the thresholds
        s_roc = np.asarray(bp.get("y_scores_for_roc", s), dtype=float)
        assert np.isclose(roc_auc_score(y, s_roc), bp["overall_auc"], atol=1e-9), \
            f"{ft}: recomputed external AUC != stored"
        sample_ids = list(bp.get("sample_ids", [None] * len(y)))
        subgroups = list(bp.get("stratification_labels", [None] * len(y)))

        # subgroups must match the manifest and the published counts 32 / 67 / 36 / 90
        assert set(sample_ids) == set(manifest.index), f"{ft}: sample set != manifest"
        assert all(manifest.loc[sid, "disease"] == g for sid, g in zip(sample_ids, subgroups)), \
            f"{ft}: stratification labels disagree with the manifest"
        assert pd.Series(subgroups).value_counts().to_dict() == {
            "Liver cancer": 90, "Hepatitis B": 67, "Cirrhosis": 36, "Healthy": 32}, \
            f"{ft}: comparison-group counts differ from Jiang et al. 2015"

        for sid, yi, si, sr, g in zip(sample_ids, y, s, s_roc, subgroups):
            pred_rows.append({"method": ft, "sample_id": sid, "y_true": int(yi),
                              "subgroup": g,
                              "control_group": manifest.loc[sid, "control_group"] if yi == 0 else "",
                              "score": float(si), "score_for_roc": float(sr)})

        thr_train, thr_source = _train_threshold_95spec(bp.get("best_model_path", ""))
        fixed_rows += _flatten_fixed_threshold(ft, bp.get("fixed_threshold_evaluation"))
        cfg_rows.append({
            "method": ft, "feature": best_feat,
            "classifier": bp.get("best_model_info", {}).get("classifier", ""),
            "n_genes": bp.get("best_model_info", {}).get("n_genes", ""),
            "train_cv_auc": bp.get("best_model_info", {}).get("cv_auc", np.nan),
            "external_auc": bp.get("overall_auc", np.nan),
            "threshold_95spec_external": bp.get("threshold_95spec", np.nan),
            "threshold_95spec_train": thr_train if thr_train is not None else np.nan,
            "threshold_95spec_train_source": thr_source,
            "proba_degenerate": bool(bp.get("proba_degenerate", False)),
            "source_dir": ext_dir,
        })
        print(f"  {ft}: best_feat={best_feat}, external AUC={bp['overall_auc']:.3f}, "
              f"n={len(y)} samples, thr_train={thr_train} ({thr_source})")

    pd.DataFrame(pred_rows).to_csv(
        os.path.join(DATA_DIR, "fig4bc_external_predictions.csv"), index=False)
    pd.DataFrame(cfg_rows).to_csv(
        os.path.join(DATA_DIR, "fig4bc_config.csv"), index=False)
    print("wrote fig4bc_external_predictions.csv + fig4bc_config.csv")
    fixed_cols = ["method", "operating_point", "threshold", "comparison_group",
                  "tp", "fn", "tn", "fp", "sensitivity", "sensitivity_ci_low",
                  "sensitivity_ci_high", "specificity", "specificity_ci_low",
                  "specificity_ci_high"]
    pd.DataFrame(fixed_rows, columns=fixed_cols).to_csv(
        os.path.join(DATA_DIR, "fig4bc_fixed_threshold.csv"), index=False)
    if fixed_rows:
        print(f"wrote fig4bc_fixed_threshold.csv ({len(fixed_rows)} rows)")
    else:
        print("wrote fig4bc_fixed_threshold.csv (EMPTY: the external runs predate "
              "fixed_threshold_evaluation — rerun slurm/external_validation.sh after the "
              "best-model search; the notebook derives 5c from the frozen threshold instead)")

    # S7: external AUC vs number of genes for the DD models
    with open(rp(f"{EXT_DIRS['dd']}/comprehensive_validation_summary.pkl"), "rb") as fh:
        summary = pickle.load(fh)
    rows = []
    for k, r in sorted(summary.get("gene_count_results", {}).items()):
        if not r:
            continue
        rows.append({"n_genes": int(k), "external_auc": r.get("auc", np.nan),
                     "training_cv_auc": r.get("training_cv_auc", np.nan),
                     "feature": r.get("feature", ""), "classifier": r.get("classifier", "")})
    pd.DataFrame(rows).to_csv(os.path.join(DATA_DIR, "figS7_auc_vs_k.csv"), index=False)
    print(f"wrote figS7_auc_vs_k.csv ({len(rows)} gene counts)")


def _clinical_dir() -> str | None:
    """Clinical-mode run (StandardScaler refit on Jiang; slurm/external_validation_clinical.sh),
    or None. Only the _rev run is used; the run without _rev used the 249-row Jiang metadata."""
    d = variant_dir("nested_cv_results_cris/external_validation_comprehensive_clinical_cris_on_jiang_rev")
    return d if os.path.isdir(rp(d)) else None


def export_fig4jkl_kcurve_and_modes() -> None:
    """Panel j: external AUROC vs number of TSSs (per-k winners of the DD search on Jiang),
    pipeline and, when present, clinical mode. Panels k/l: per-sample scores of the deployed
    DD model in pipeline vs clinical mode. Per-k sample scores only when the run stored them."""
    manifest = pd.read_csv(rp("accessory_files/jiang_manifest.csv")).set_index("ID")
    modes = {"pipeline": EXT_DIRS["dd"]}
    clin = _clinical_dir()
    if clin:
        modes["clinical"] = clin
    else:
        print("  clinical-mode revision run not on disk (TRAIN_SUFFIX=_rev EXPERIMENT_SUFFIX=_rev "
              "sbatch slurm/external_validation_clinical.sh jiang) — k-curve and mode comparison "
              "exported for pipeline mode only")

    k_rows, k_pred_rows, mode_pred_rows, mode_cfg_rows = [], [], [], []
    for mode, d in modes.items():
        with open(rp(f"{d}/comprehensive_validation_summary.pkl"), "rb") as fh:
            summary = pickle.load(fh)
        for k, r in sorted(summary.get("gene_count_results", {}).items()):
            if not r:
                continue
            has_scores = r.get("y_pred_proba") is not None and r.get("y_true") is not None
            k_rows.append({"mode": mode, "n_genes": int(k), "feature": r.get("feature", ""),
                           "classifier": r.get("classifier", ""),
                           "training_cv_auc": r.get("training_cv_auc", np.nan),
                           "external_auc": r.get("auc", np.nan),
                           "sens_at_95spec_external_roc": r.get("sens_at_95spec", np.nan),
                           "per_sample_scores": has_scores, "source_dir": d})
            if has_scores:
                y = np.asarray(r["y_true"], dtype=int)
                s = np.asarray(r["y_pred_proba"], dtype=float)
                s_roc = np.asarray(r.get("y_scores_for_roc", s), dtype=float)
                assert np.isclose(roc_auc_score(y, s_roc), r["auc"], atol=1e-9), f"{mode} k={k}"
                for sid, yi, si, sr, g in zip(r["sample_ids"], y, s, s_roc, r["stratification_labels"]):
                    k_pred_rows.append({"mode": mode, "n_genes": int(k), "sample_id": sid,
                                        "y_true": int(yi), "subgroup": g,
                                        "score": float(si), "score_for_roc": float(sr)})

        # deployed model per mode = best feature by TRAINING CV AUC (as in fig 5b)
        fr = {f: v for f, v in summary["feature_results"].items() if v is not None}
        best_feat = max(fr, key=lambda f: fr[f].get("training_cv_auc", 0))
        with open(rp(f"{d}/external_validation_{best_feat}_results.pkl"), "rb") as fh:
            bp = pickle.load(fh)
        if bp.get("y_true") is None:
            print(f"  {mode}: per-feature pickle predates per-sample export; mode comparison skipped")
            continue
        y = np.asarray(bp["y_true"], dtype=int)
        s = np.asarray(bp["y_pred_proba"], dtype=float)
        s_roc = np.asarray(bp.get("y_scores_for_roc", s), dtype=float)
        assert np.isclose(roc_auc_score(y, s_roc), bp["overall_auc"], atol=1e-9), f"{mode}: AUC mismatch"
        assert set(bp["sample_ids"]) == set(manifest.index), f"{mode}: sample set != manifest"
        for sid, yi, si, sr, g in zip(bp["sample_ids"], y, s, s_roc, bp["stratification_labels"]):
            mode_pred_rows.append({"mode": mode, "sample_id": sid, "y_true": int(yi), "subgroup": g,
                                   "score": float(si), "score_for_roc": float(sr)})
        thr_train, thr_source = _train_threshold_95spec(bp.get("best_model_path", ""))
        mode_cfg_rows.append({"mode": mode, "feature": best_feat,
                              "classifier": bp.get("best_model_info", {}).get("classifier", ""),
                              "n_genes": bp.get("best_model_info", {}).get("n_genes", ""),
                              "train_cv_auc": bp.get("best_model_info", {}).get("cv_auc", np.nan),
                              "external_auc": bp.get("overall_auc", np.nan),
                              "threshold_95spec_train": thr_train if thr_train is not None else np.nan,
                              "threshold_95spec_train_source": thr_source,
                              "proba_degenerate": bool(bp.get("proba_degenerate", False)),
                              "source_dir": d})
        print(f"  {mode}: deployed {best_feat} k={bp.get('best_model_info', {}).get('n_genes')} "
              f"{bp.get('best_model_info', {}).get('classifier')} -> Jiang AUC {bp['overall_auc']:.3f}; "
              f"k-curve {len([r for r in k_rows if r['mode'] == mode])} points"
              f"{' with per-sample scores' if any(r['per_sample_scores'] for r in k_rows if r['mode'] == mode) else ''}")

    pd.DataFrame(k_rows, columns=["mode", "n_genes", "feature", "classifier", "training_cv_auc",
                                  "external_auc", "sens_at_95spec_external_roc", "per_sample_scores",
                                  "source_dir"]).to_csv(os.path.join(DATA_DIR, "fig4j_auc_vs_k.csv"), index=False)
    pd.DataFrame(k_pred_rows, columns=["mode", "n_genes", "sample_id", "y_true", "subgroup", "score",
                                       "score_for_roc"]).to_csv(
        os.path.join(DATA_DIR, "fig4j_auc_vs_k_predictions.csv"), index=False)
    pd.DataFrame(mode_pred_rows, columns=["mode", "sample_id", "y_true", "subgroup", "score", "score_for_roc"]).to_csv(
        os.path.join(DATA_DIR, "fig4kl_external_modes_predictions.csv"), index=False)
    pd.DataFrame(mode_cfg_rows, columns=["mode", "feature", "classifier", "n_genes", "train_cv_auc",
                                         "external_auc", "threshold_95spec_train",
                                         "threshold_95spec_train_source", "proba_degenerate", "source_dir"]).to_csv(
        os.path.join(DATA_DIR, "fig4kl_external_modes_config.csv"), index=False)
    print(f"wrote fig4j_auc_vs_k.csv ({len(k_rows)} rows), fig4j_auc_vs_k_predictions.csv ({len(k_pred_rows)} rows), "
          f"fig4kl_external_modes_predictions.csv ({len(mode_pred_rows)} rows) + config")


def export_fig4j_jiang_ksweep() -> None:
    """Within-Jiang AUROC vs k (JIANG_KSWEEP_DIR): per-k summary and per-sample OOF scores.
    Skipped with a notice when the sweep is missing."""
    manifest = pd.read_csv(rp("accessory_files/jiang_manifest.csv")).set_index("ID")
    trend_path = rp(JIANG_KSWEEP_DIR, "nested_full_ksweep.csv")
    rows, pred_rows = [], []
    if not os.path.exists(trend_path):
        print(f"  {JIANG_KSWEEP_DIR}/nested_full_ksweep.csv not found — Jiang k-sweep pending "
              "(DATASET=jiang METADATA_FILE=accessory_files/jiang_manifest.csv "
              "EXPERIMENT_NAME=jiang_hcc_nested_full ./slurm/submit_nested_full.sh ksweep); skipped")
    else:
        trend = pd.read_csv(trend_path)
        for r in trend.itertuples(index=False):
            k = int(r.k)
            kdir = rp(JIANG_KSWEEP_DIR, f"ksweep_k{k:05d}")
            oof = pd.read_csv(os.path.join(kdir, "nested_full_oof_predictions.csv"))
            assert oof["sample_id"].is_unique and set(oof["sample_id"]) <= set(manifest.index), f"k={k}"
            assert (oof["stratification_label"].to_numpy()
                    == manifest.loc[oof["sample_id"], "disease"].to_numpy()).all(), f"k={k}: labels"
            pooled = roc_auc_score(oof["y_true"], oof["y_pred_proba"])
            assert np.isclose(pooled, r.oof_auc_pooled, atol=1e-9), f"k={k}: pooled OOF AUC mismatch"
            rows.append({"k": k, "mean_test_auc": r.mean_test_auc, "std_test_auc": r.std_test_auc,
                         "oof_auc_pooled": r.oof_auc_pooled, "mean_inner_score": r.mean_inner_score,
                         "modal_feature": r.modal_feature, "modal_selection_type": r.modal_selection_type,
                         "n_distinct_configs": int(r.n_distinct_configs),
                         "mean_num_genes_used": r.mean_num_genes_used, "source_dir": JIANG_KSWEEP_DIR})
            for o in oof.itertuples(index=False):
                pred_rows.append({"k": k, "sample_id": o.sample_id, "y_true": int(o.y_true),
                                  "subgroup": o.stratification_label, "outer_fold": int(o.outer_fold),
                                  "score": float(o.y_pred_proba)})
        print(f"  Jiang k-sweep: {len(rows)} k values; pooled OOF AUROC "
              f"{rows[0]['oof_auc_pooled']:.3f} (k={rows[0]['k']}) -> {rows[-1]['oof_auc_pooled']:.3f} (k={rows[-1]['k']})")
    pd.DataFrame(rows, columns=["k", "mean_test_auc", "std_test_auc", "oof_auc_pooled", "mean_inner_score",
                                "modal_feature", "modal_selection_type", "n_distinct_configs",
                                "mean_num_genes_used", "source_dir"]).to_csv(
        os.path.join(DATA_DIR, "fig4j_jiang_ksweep.csv"), index=False)
    pd.DataFrame(pred_rows, columns=["k", "sample_id", "y_true", "subgroup", "outer_fold", "score"]).to_csv(
        os.path.join(DATA_DIR, "fig4j_jiang_ksweep_predictions.csv"), index=False)
    print(f"wrote fig4j_jiang_ksweep.csv ({len(rows)} rows) + fig4j_jiang_ksweep_predictions.csv ({len(pred_rows)} rows)")


def export_fig4gi_jiang_nested() -> None:
    """Fully nested runs trained within Jiang: pooled OOF predictions with subgroup labels and
    the per-fold configurations. Missing runs are skipped."""
    manifest = pd.read_csv(rp("accessory_files/jiang_manifest.csv")).set_index("ID")
    oof_rows, cfg_rows = [], []
    for task, d in JIANG_NESTED_FULL_DIRS.items():
        oof_path = rp(d, "nested_full_oof_predictions.csv")
        if not os.path.exists(oof_path):
            print(f"  {task}: {d}/nested_full_oof_predictions.csv not found — run pending, skipped")
            continue
        oof = pd.read_csv(oof_path)
        per_fold = pd.read_csv(rp(d, "nested_full_per_fold.csv"))
        with open(rp(d, "nested_full_summary.json")) as fh:
            summ = json.load(fh)

        assert oof["sample_id"].is_unique, f"{task}: duplicated sample in the OOF table"
        assert set(oof["sample_id"]) <= set(manifest.index), f"{task}: samples outside the manifest"
        assert (oof["stratification_label"].to_numpy()
                == manifest.loc[oof["sample_id"], "disease"].to_numpy()).all(), \
            f"{task}: OOF subgroup labels disagree with the manifest"
        pooled = roc_auc_score(oof["y_true"], oof["y_pred_proba"])
        assert np.isclose(pooled, summ["oof_auc_pooled"], atol=1e-9), \
            f"{task}: pooled OOF AUC {pooled} != summary {summ['oof_auc_pooled']}"
        assert np.isclose(per_fold["test_auc"].mean(), summ["mean_test_auc"], atol=1e-9)

        for r in oof.itertuples(index=False):
            oof_rows.append({"task": task, "sample_id": r.sample_id, "y_true": int(r.y_true),
                             "subgroup": r.stratification_label,
                             "control_group": manifest.loc[r.sample_id, "control_group"]
                             if int(r.y_true) == 0 else "",
                             "outer_fold": int(r.outer_fold), "score": float(r.y_pred_proba)})
        modal = {c: per_fold[c].mode().iloc[0]
                 for c in ["feature", "selection_type", "n_top_genes", "classifier"]}
        n_distinct = len(per_fold[["feature", "selection_type", "n_top_genes", "classifier"]]
                         .drop_duplicates())
        cfg_rows.append({"task": task, "n_samples": int(summ["n_samples"]),
                         "n_hcc": int((oof["y_true"] == 1).sum()),
                         "n_control": int((oof["y_true"] == 0).sum()),
                         "control_groups": " | ".join(sorted(
                             oof.loc[oof["y_true"] == 0, "stratification_label"].unique())),
                         "mean_test_auc": summ["mean_test_auc"], "std_test_auc": summ["std_test_auc"],
                         "oof_auc_pooled": summ["oof_auc_pooled"],
                         "modal_feature": modal["feature"], "modal_selection_type": modal["selection_type"],
                         "modal_n_top_genes": int(modal["n_top_genes"]),
                         "modal_classifier": modal["classifier"], "n_distinct_configs": n_distinct,
                         "source_dir": d})
        print(f"  {task}: n={summ['n_samples']}, outer AUC {summ['mean_test_auc']:.3f} ± "
              f"{summ['std_test_auc']:.3f}, pooled OOF {summ['oof_auc_pooled']:.3f}, modal {modal}")

    oof_cols = ["task", "sample_id", "y_true", "subgroup", "control_group", "outer_fold", "score"]
    cfg_cols = ["task", "n_samples", "n_hcc", "n_control", "control_groups", "mean_test_auc",
                "std_test_auc", "oof_auc_pooled", "modal_feature", "modal_selection_type",
                "modal_n_top_genes", "modal_classifier", "n_distinct_configs", "source_dir"]
    pd.DataFrame(oof_rows, columns=oof_cols).to_csv(
        os.path.join(DATA_DIR, "fig4gi_jiang_nested_oof.csv"), index=False)
    pd.DataFrame(cfg_rows, columns=cfg_cols).to_csv(
        os.path.join(DATA_DIR, "fig4gi_jiang_nested_config.csv"), index=False)
    print(f"wrote fig4gi_jiang_nested_oof.csv ({len(oof_rows)} rows) + fig4gi_jiang_nested_config.csv"
          + (" (no Jiang-trained run on disk yet)" if not cfg_rows else ""))


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    print("=== Fig 5a / S6 (PANCAN) ===")
    export_fig4a()
    print("\n=== Fig 5b/c / S7 (external validation, Jiang) ===")
    if all(os.path.isdir(rp(d)) for d in EXT_DIRS.values()):
        export_fig4bc()
    else:
        # the notebook then falls back to the full-cohort CSVs
        print("  deployment runs missing for this cohort variant "
              f"({[d for d in EXT_DIRS.values() if not os.path.isdir(rp(d))]}) - skipped")
    print("\n=== Fig 5g-i (fully nested, trained within Jiang) ===")
    export_fig4gi_jiang_nested()
    print("\n=== Fig 5j-l (AUROC vs k on Jiang; pipeline vs clinical standardization) ===")
    if all(os.path.isdir(rp(d)) for d in EXT_DIRS.values()):
        export_fig4jkl_kcurve_and_modes()   # per-k deployment of the variant's best-model search
    else:
        print("  deployment runs missing for this cohort - 5j-l skipped (the notebook falls back to the full-cohort CSVs)")
    # cohort-independent, but written to every variant's data directory
    export_fig4j_jiang_ksweep()
