"""Source data for Figure S14: the fully nested data-driven model with the sample-level
covariates of Figure S10 added, per number of TSSs (k).

Arms (slurm/submit_covariate_arms.sh; model_hpc.py --augment_covariates):
  none             TSS only (the k-sweep of the same cohort)
  cov_techdemo5    TSS + coverage, fragments, plasma volume, age and sex; the covariates
                   bypass the TSS ranking

Reads the per-fold, per-k pickles (ksweep_k<k>/nested_full_fold_<f>.pkl), so it also runs on
unfinished sweeps; every table records the folds available per (arm, k). The tuned k comes
from ksweep_tuned/ when present, else from the available k values.

Outputs (DATA_DIR, i.e. src/figures/data[/no_deep]/):
  figS14_status.csv       folds done per (task, arm, k)
  figS14_folds.csv        per (task, arm, k, fold): winning configuration, inner-CV and
                          held-out AUROC, number of TSSs selected
  figS14_predictions.csv  held-out score of every sample per (task, arm, k)
  figS14_tuned.csv        per (task, arm, fold): the k chosen on inner CV and its AUROC
  figS14_blocks.csv       per (task, arm, k, fold): block permutation importance, weight sums
  figS14_covariates.csv   per covariate: weight, rank, permutation drop
  figS14_references.csv   fold-mean AUROC of the covariate-only panels (Table S3)

Run from the repository root (COHORT_VARIANT defaults to no_deep, see _common.py):
    python src/figures/export/export_figS14.py [--arm_dir task:arm=dir ...]
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from glob import glob

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from _common import DATA_DIR, REGION_SWEEP_DIRS, COHORT_VARIANT, rp, variant_dir  # noqa: E402

ARMS = ["none", "cov_techdemo5"]
ARM_PANEL = {"cov_techdemo5": "bl_panel_techdemo5"}
TASKS = {"brca_crc": "tableS3_baselines.csv", "pancan": "tableS3_baselines_pancan.csv"}
K_GRID = [10, 25, 50, 100, 250, 500, 1000, 1500, 2500, 5000, 10000, 15000]
N_FOLDS = 10


def arm_dir(task: str, arm: str) -> str:
    if arm == "none":
        return REGION_SWEEP_DIRS[task]["data_driven"]
    return variant_dir(f"nested_cv_results_cris/nested_full_{task}_nested_full_{arm}_ksweep")


def read_partials(d: str) -> dict:
    """{k: {fold: fold_record}} from the per-k partial pickles of a sweep directory."""
    out = {}
    for sub in sorted(glob(rp(d, "ksweep_k*"))):
        k = int(os.path.basename(sub).replace("ksweep_k", ""))
        for p in sorted(glob(os.path.join(sub, "nested_full_fold_*.pkl"))):
            try:
                with open(p, "rb") as fh:
                    rec = pickle.load(fh)["fold_record"]
            except Exception as e:  # a partial being written right now
                print(f"  unreadable partial skipped: {p} ({type(e).__name__})")
                continue
            out.setdefault(k, {})[int(rec["fold"])] = rec
    return out


def hp_json(d) -> str:
    return json.dumps({k.replace("clf__", ""): (v if isinstance(v, (int, float, str, bool)) else str(v))
                       for k, v in (d or {}).items()}, sort_keys=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm_dir", action="append", default=[], metavar="task:arm=dir",
                    help="override a sweep directory (testing); never commit CSVs made with overrides")
    args = ap.parse_args()
    overrides = {}
    for spec in args.arm_dir:
        key, _, d = spec.partition("=")
        task, _, arm = key.partition(":")
        overrides[(task, arm)] = d
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"cohort variant: {COHORT_VARIANT} -> {DATA_DIR}/")

    status, folds, preds, tuned, blocks, covs, refs = [], [], [], [], [], [], []
    for task, table_csv in TASKS.items():
        # reference levels: the covariate-only panels of Table S3 (same cohort)
        tpath = os.path.join(DATA_DIR, table_csv)
        # fold-mean AUPR of the panels (cancers positive), reference lines of the AUPR panel
        panel_aupr = {}
        ppath = os.path.join(DATA_DIR, "fig3_baseline_predictions.csv" if task == "brca_crc" else "fig4_baseline_predictions.csv")
        if os.path.exists(ppath):
            pp = pd.read_csv(ppath)
            for panel in ARM_PANEL.values():
                q = pp[(pp["method"] == panel) & pp["score"].notna()]
                if len(q):
                    panel_aupr[panel] = float(np.mean([average_precision_score(g["y_true"], g["score"] * (g["orientation"].fillna(1.0).iloc[0]))
                                                       for _, g in q.groupby("fold") if g["y_true"].nunique() == 2]))
        if os.path.exists(tpath):
            t = pd.read_csv(tpath)
            for arm, panel in ARM_PANEL.items():
                r = t[t["method"] == panel]
                if len(r):
                    refs.append({"task": task, "arm": arm, "reference": panel,
                                 "auc_fold_mean": float(r["auc_fold_mean"].iloc[0]),
                                 "auc_fold_sd": float(r["auc_fold_sd"].iloc[0]),
                                 "sensitivity_95spec_mean": float(r["sensitivity_95spec_mean"].iloc[0]),
                                 "aupr_fold_mean": panel_aupr.get(panel, np.nan)})
        for arm in ARMS:
            d = overrides.get((task, arm), arm_dir(task, arm))
            if not os.path.isdir(rp(d)):
                print(f"  {task} {arm}: {d} not on disk - pending")
                status.append({"task": task, "arm": arm, "k": np.nan, "n_folds_done": 0,
                               "n_folds_expected": N_FOLDS, "source_dir": d, "state": "pending"})
                continue
            per_k = read_partials(d)
            if not per_k:
                print(f"  {task} {arm}: no per-k partials yet in {d}")
                status.append({"task": task, "arm": arm, "k": np.nan, "n_folds_done": 0,
                               "n_folds_expected": N_FOLDS, "source_dir": d, "state": "pending"})
                continue
            ks = sorted(per_k)
            print(f"  {task} {arm}: {d} - k values {ks}; folds per k: "
                  + ", ".join(f"{k}:{len(per_k[k])}" for k in ks))
            for k in ks:
                status.append({"task": task, "arm": arm, "k": k, "n_folds_done": len(per_k[k]),
                               "n_folds_expected": N_FOLDS, "source_dir": d,
                               "state": "complete" if len(per_k[k]) == N_FOLDS else "partial"})
                for f, rec in sorted(per_k[k].items()):
                    y = np.asarray(rec["y_test"], int)
                    s = np.asarray(rec["y_pred_proba"], float)
                    auc_re = roc_auc_score(y, s)
                    assert np.isclose(auc_re, rec["test_auc"], atol=1e-9), (task, arm, k, f)
                    cb = rec.get("covariate_block") or {}
                    n_cov = int(cb.get("n_covariates", 0))
                    n_tss = int(cb.get("n_tss_selected", rec["num_genes_used"]))
                    thr = rec["full_model_thresholds"]["95_percent_specificity"]
                    folds.append({"task": task, "arm": arm, "k": k, "fold": f, "feature": rec["feature"],
                                  "selection_type": rec["selection_type"], "classifier": rec["classifier"],
                                  "hyperparameters": hp_json(rec.get("clf_hyperparams")),
                                  "inner_cv_auc": float(rec["inner_best_score"]), "test_auc": float(rec["test_auc"]),
                                  # precision-recall: cancers positive; chance level = prevalence_test
                                  "test_aupr": float(average_precision_score(y, s)), "prevalence_test": float(y.mean()),
                                  "n_tss_selected": n_tss, "n_covariates": n_cov,
                                  "n_tss_available": int(max(rec.get("n_features_available", {rec["feature"]: 0}).values()) - n_cov)
                                  if rec.get("n_features_available") else np.nan,
                                  "threshold_95spec": float(thr), "n_test": int(len(y)),
                                  "elapsed_s": float(rec.get("fold_elapsed_seconds", np.nan))})
                    for sid, yi, si in zip(rec["test_sample_ids"], y, s):
                        preds.append({"task": task, "arm": arm, "k": k, "fold": f, "sample_id": sid,
                                      "y_true": int(yi), "score": float(si), "threshold_95spec": float(thr)})
                    if cb:
                        b = {"task": task, "arm": arm, "k": k, "fold": f, "mode": cb["mode"],
                             "classifier": rec["classifier"], "weight_kind": cb.get("weight_kind"),
                             "n_tss_selected": n_tss, "n_covariates": n_cov}
                        for key in ["perm_drop_covariates_mean", "perm_drop_covariates_sd", "perm_drop_tss_mean",
                                    "perm_drop_tss_sd", "tss_abs_weight_median", "tss_abs_weight_max",
                                    "tss_abs_weight_sum", "covariate_abs_weight_sum", "residual_r2_median",
                                    "residual_r2_p90", "residual_r2_max", "residual_r2_frac_gt_0.1",
                                    "residual_r2_frac_gt_0.25"]:
                            b[key] = cb.get(key, np.nan)
                        blocks.append(b)
                        if cb["mode"] == "augment":
                            names = cb["names"]
                            for j, name in enumerate(names):
                                covs.append({"task": task, "arm": arm, "k": k, "fold": f, "covariate": name,
                                             "weight": (cb.get("covariate_weights") or [np.nan] * n_cov)[j],
                                             "abs_rank": (cb.get("covariate_abs_rank") or [np.nan] * n_cov)[j],
                                             "quantile_vs_tss": (cb.get("covariate_abs_quantile_vs_tss") or [np.nan] * n_cov)[j],
                                             "perm_drop": (cb.get("perm_drop_per_covariate") or [np.nan] * n_cov)[j],
                                             "weight_kind": cb.get("weight_kind"), "classifier": rec["classifier"]})
            # tuned k: ksweep_tuned/ when present, else derived from the available k values
            tuned_pkl = rp(d, "ksweep_tuned", "nested_full_per_fold.pkl")
            grid = sorted(set(K_GRID) | set(ks))
            if os.path.exists(tuned_pkl):
                with open(tuned_pkl, "rb") as fh:
                    tp = pickle.load(fh)
                for _, r in tp.sort_values("fold").iterrows():
                    tuned.append({"task": task, "arm": arm, "fold": int(r["fold"]), "k_selected": int(r["ksweep_k_selected"]),
                                  "inner_cv_auc": float(r["inner_best_score"]), "test_auc": float(r["test_auc"]),
                                  "test_aupr": float(average_precision_score(np.asarray(r["y_test"], int), np.asarray(r["y_pred_proba"], float))),
                                  "classifier": r["classifier"], "feature": r["feature"], "n_k_available": len(ks),
                                  "complete": True, "source": "ksweep_tuned"})
            else:
                for f in range(1, N_FOLDS + 1):
                    avail = [k for k in ks if f in per_k[k]]
                    if not avail:
                        continue
                    best = max(avail, key=lambda k: (per_k[k][f]["inner_best_score"], -k))
                    rec = per_k[best][f]
                    tuned.append({"task": task, "arm": arm, "fold": f, "k_selected": best,
                                  "inner_cv_auc": float(rec["inner_best_score"]), "test_auc": float(rec["test_auc"]),
                                  "test_aupr": float(average_precision_score(np.asarray(rec["y_test"], int), np.asarray(rec["y_pred_proba"], float))),
                                  "classifier": rec["classifier"], "feature": rec["feature"], "n_k_available": len(avail),
                                  "complete": len(avail) >= len(grid), "source": "derived_from_partials"})

    HEADERS = {"references": ["task", "arm", "reference", "auc_fold_mean", "auc_fold_sd", "sensitivity_95spec_mean", "aupr_fold_mean"],
               "blocks": ["task", "arm", "k", "fold", "mode"], "covariates": ["task", "arm", "k", "fold", "covariate"],
               "tuned": ["task", "arm", "fold", "k_selected"], "folds": ["task", "arm", "k", "fold"],
               "predictions": ["task", "arm", "k", "fold", "sample_id"], "status": ["task", "arm", "k"]}
    for name, rows in [("status", status), ("folds", folds), ("predictions", preds), ("tuned", tuned),
                       ("blocks", blocks), ("covariates", covs), ("references", refs)]:
        out = os.path.join(DATA_DIR, f"figS14_{name}.csv")
        (pd.DataFrame(rows) if rows else pd.DataFrame(columns=HEADERS[name])).to_csv(out, index=False)
        print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
