"""Source data for Figure S13: region-number (k) sweep for every method.

Each method's region set goes through the same fully nested procedure, one complete
nested selection per fixed k (slurm/nested_full.sh FEATURE_SET=dd|rna|atac ksweep):
data_driven (14,224 TSSs x 5 features), rna (792 Zhu et al. regions, rcov) and atac
(504 Ulz TF-level features). Per task and method three arms are exported:
ksweep (every k), tuned (k chosen per outer fold on the inner-CV score; from
ksweep_tuned/ or the standalone nested run) and fixed (all regions, comparators only).

Writes to src/figures/data/: figS13_ksweep_folds.csv, figS13_predictions.csv,
figS13_arms.csv, figS13_fold_configs.csv. Missing runs are skipped with a NOTE.

Run from the repository root:
    python src/figures/export/export_figS13.py [--skip-fixed]
--sweep task/method=DIR overrides a sweep directory (testing only; do not commit its CSVs).
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from _common import (  # noqa: E402
    DATA_DIR,
    FIXED_COMPARATOR_DIRS,
    REGION_SWEEP_DIRS,
    REGION_SWEEP_STANDALONE_DIRS,
    assert_predictions_match_stored,
    best_mean_auc_config,
    load_results,
    rp,
)

METHODS = ["data_driven", "rna", "atac"]
LOCKED_FEATURE = {"rna": "rcov", "atac": "ulz_atac_tfbs_features"}
# Region-set sizes before model_hpc drops duplicate columns; n_regions_total in the
# CSVs is the unique-column width k selects from.
NOMINAL_REGIONS = {"data_driven": 14224, "rna": 792, "atac": 504}
KEEP_FIXED = [
    "feature", "classifier", "top_genes", "fold", "auc_full_model",
    "test_sample_ids", "y_pred_proba", "y_test",
    "full_model_thresholds", "confusion_matrix_full_model",
    "selection_type", "inner_cv_best_score",
]


# --------------------------------------------------------------------- loading
def _rename(per_fold: pd.DataFrame) -> pd.DataFrame:
    return per_fold.rename(columns={"test_auc": "auc_full_model",
                                    "n_top_genes": "top_genes"})


def load_nested_dir(d: str, label: str):
    """Aggregated per-fold table of a nested_full-style directory (schema
    aligned with assert_predictions_match_stored), its OOF CSV and summary."""
    with open(rp(d, "nested_full_per_fold.pkl"), "rb") as fh:
        per_fold = _rename(pickle.load(fh)).sort_values("fold").reset_index(drop=True)
    oof = pd.read_csv(rp(d, "nested_full_oof_predictions.csv"))
    with open(rp(d, "nested_full_summary.json")) as fh:
        summary = json.load(fh)
    assert_predictions_match_stored(per_fold, label)
    oof_map = dict(zip(oof["sample_id"], oof["y_pred_proba"]))
    assert len(oof) == sum(len(r) for r in per_fold["test_sample_ids"]), \
        f"{label}: OOF rows do not match the per-fold tables"
    for _, r in per_fold.iterrows():
        for sid, s in zip(r["test_sample_ids"], r["y_pred_proba"]):
            assert np.isclose(oof_map[sid], s), f"{label}: OOF mismatch for {sid}"
    return per_fold, oof, summary


def load_sweep(sweep_dir: str, label: str):
    """{k: (per_fold, oof)} for every aggregated ksweep_k<k>/ subdirectory."""
    out = {}
    for name in sorted(os.listdir(rp(sweep_dir))):
        if not name.startswith("ksweep_k"):
            continue
        sub = os.path.join(sweep_dir, name)
        if not os.path.exists(rp(sub, "nested_full_per_fold.pkl")):
            print(f"NOTE: {label}: {name} not aggregated - skipped.")
            continue
        k = int(name.replace("ksweep_k", ""))
        per_fold, oof, _ = load_nested_dir(sub, f"{label} k={k}")
        out[k] = (per_fold, oof)
    return out


def derive_tuned(per_k: dict) -> pd.DataFrame:
    """Per outer fold, the k with the best inner-CV score (ties -> smallest k), the
    rule model_hpc's ksweep aggregate uses for ksweep_tuned/."""
    ks = sorted(per_k)
    rows = []
    for fold in sorted(per_k[ks[0]][0]["fold"].astype(int)):
        inner = {k: float(per_k[k][0].loc[per_k[k][0]["fold"].astype(int) == fold,
                                          "inner_best_score"].iloc[0]) for k in ks}
        best = max(ks, key=lambda k: (inner[k], -k))
        row = per_k[best][0][per_k[best][0]["fold"].astype(int) == fold].iloc[0].copy()
        row["ksweep_k_selected"] = int(best)
        row["ksweep_inner_scores"] = inner
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


def load_fixed(d: str, method: str, label: str):
    """The manuscript's fixed-region comparator: locked feature, all regions,
    classifier chosen per outer fold on inner-CV scores (export_fig3 logic)."""
    cand = load_results(f"{d}/results*.pkl", KEEP_FIXED)
    locked = LOCKED_FEATURE[method]
    cand = cand[cand["feature"] == locked].copy()
    assert len(cand), f"{label}: no results for feature {locked} in {d}"
    if "inner_cv_best_score" not in cand.columns or cand["inner_cv_best_score"].isna().any():
        scores_path = rp(d, "inner_cv_scores.csv")
        assert os.path.exists(scores_path), (
            f"{label}: pickles lack inner_cv_best_score and {scores_path} is missing - "
            f"rerun the nested CV with the current model_hpc.py")
        sc = pd.read_csv(scores_path)
        sc = sc[sc["feature"] == locked][["classifier", "fold", "inner_cv_auc"]]
        cand = cand.drop(columns=[c for c in ["inner_cv_best_score"] if c in cand.columns])
        cand = cand.merge(sc.rename(columns={"inner_cv_auc": "inner_cv_best_score"}),
                          on=["classifier", "fold"], how="left")
        assert cand["inner_cv_best_score"].notna().all(), f"{label}: inner scores incomplete"
    sel = cand.loc[cand.groupby("fold")["inner_cv_best_score"].idxmax()] \
        .sort_values("fold").reset_index(drop=True)
    assert len(sel) == 10, f"{label}: expected 10 folds, got {len(sel)}"
    assert_predictions_match_stored(sel, label)
    posthoc = best_mean_auc_config(cand, ["classifier"])
    print(f"  {label}: per-fold classifier choice {sel['classifier'].value_counts().to_dict()}, "
          f"mean AUROC {sel['auc_full_model'].mean():.4f} (post-hoc best-of-"
          f"{cand['classifier'].nunique()} {posthoc['classifier']} {posthoc['mean_auc']:.4f})")
    return sel


# --------------------------------------------------------------------- helpers
def fold_map(per_fold: pd.DataFrame) -> dict:
    return {int(r["fold"]): frozenset(r["test_sample_ids"]) for _, r in per_fold.iterrows()}


def hp_json(d) -> str:
    if not isinstance(d, dict):
        return ""
    return json.dumps({k.replace("clf__", ""): (v.item() if hasattr(v, "item") else v)
                       for k, v in sorted(d.items())}, default=str)


def n_regions_total(per_fold: pd.DataFrame, per_k: dict | None) -> int:
    """Width of the region set k was selected from (n_features_available, else the
    regions used at the largest k)."""
    if "n_features_available" in per_fold.columns and per_fold["n_features_available"].notna().all():
        return int(max(max(d.values()) for d in per_fold["n_features_available"]))
    if per_k:
        top = per_k[max(per_k)][0]
        return int(top["num_genes_used"].max())
    return int(per_fold["num_genes_used"].max())


def summarise(per_fold: pd.DataFrame, oof_scores: pd.DataFrame | None = None) -> dict:
    aucs = per_fold["auc_full_model"].to_numpy(dtype=float)
    if oof_scores is None:
        y = np.concatenate([np.asarray(r["y_test"]) for _, r in per_fold.iterrows()])
        s = np.concatenate([np.asarray(r["y_pred_proba"]) for _, r in per_fold.iterrows()])
    else:
        y, s = oof_scores["y_true"], oof_scores["y_pred_proba"]
    cfg_cols = [c for c in ["feature", "selection_type", "top_genes", "classifier"]
                if c in per_fold.columns]
    return {
        "n_folds": int(len(per_fold)),
        "mean_test_auc": float(aucs.mean()),
        "std_test_auc": float(aucs.std(ddof=0)),
        "oof_auc_pooled": float(roc_auc_score(y, s)),
        "modal_feature": per_fold["feature"].mode().iloc[0],
        "modal_selection_type": (per_fold["selection_type"].mode().iloc[0]
                                 if "selection_type" in per_fold.columns
                                 and per_fold["selection_type"].notna().any() else "none"),
        "modal_classifier": per_fold["classifier"].mode().iloc[0],
        "n_distinct_configs": int(len(per_fold[cfg_cols].drop_duplicates())),
    }


def prediction_rows(task, method, arm, per_fold, k_of_fold, strat_map):
    rows = []
    for _, r in per_fold.iterrows():
        thr = float(r["full_model_thresholds"]["95_percent_specificity"])
        k = int(k_of_fold(r))
        for sid, y, s in zip(r["test_sample_ids"], r["y_test"], r["y_pred_proba"]):
            rows.append({"task": task, "method": method, "arm": arm, "k": k,
                         "fold": int(r["fold"]), "sample_id": sid, "y_true": int(y),
                         "score": float(s), "threshold_95spec": thr,
                         "stratification_label": strat_map.get(sid)})
    return rows


def compare_arms(a: pd.DataFrame, b: pd.DataFrame, label_a: str, label_b: str) -> list[bool]:
    """Fold-by-fold identity of two tuned arms (k, configuration, scores)."""
    same = []
    for (_, ra), (_, rb) in zip(a.sort_values("fold").iterrows(), b.sort_values("fold").iterrows()):
        ok = (int(ra["top_genes"]) == int(rb["top_genes"])
              and ra["feature"] == rb["feature"] and ra["classifier"] == rb["classifier"]
              and ra["selection_type"] == rb["selection_type"]
              and np.allclose(ra["y_pred_proba"], rb["y_pred_proba"]))
        same.append(bool(ok))
        if not ok:
            print(f"  !! fold {int(ra['fold'])}: {label_a} k={ra['top_genes']} {ra['classifier']} "
                  f"vs {label_b} k={rb['top_genes']} {rb['classifier']}")
    n_ok = sum(same)
    print(f"  {label_a} vs {label_b}: {n_ok}/{len(same)} folds identical"
          + ("" if n_ok == len(same) else "  <-- MISMATCH (tie-break or grid difference?)"))
    return same


# ------------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sweep", action="append", default=[], metavar="TASK/METHOD=DIR",
                    help="override a sweep directory (testing only; repeatable)")
    ap.add_argument("--skip-fixed", action="store_true",
                    help="do not export the fixed-region manuscript arms")
    args = ap.parse_args()

    sweep_dirs = {t: dict(m) for t, m in REGION_SWEEP_DIRS.items()}
    overridden = False
    for spec in args.sweep:
        key, _, d = spec.partition("=")
        task, _, method = key.partition("/")
        if task not in sweep_dirs or method not in METHODS or not d:
            raise SystemExit(f"--sweep expects <task>/<method>=<dir>, got {spec!r}")
        sweep_dirs[task][method] = d
        overridden = True
    if overridden:
        print("!! source-directory overrides active - do NOT commit the resulting CSVs\n")

    os.makedirs(DATA_DIR, exist_ok=True)
    ksweep_rows, pred_rows, arm_rows, cfg_rows = [], [], [], []
    n_sweeps = 0

    for task in sweep_dirs:
        print(f"\n=== task {task} ===")
        partitions = {}       # label -> fold_map, for the cross-arm partition check
        strat_map = {}

        def check_partition(label, per_fold):
            fm = fold_map(per_fold)
            if partitions:
                ref_label, ref = next(iter(partitions.items()))
                assert fm == ref, (f"{task}: outer-fold partition of {label} differs from "
                                   f"{ref_label}; the arms are not comparable fold by fold")
            partitions[label] = fm

        for method in METHODS:
            sweep_dir = sweep_dirs[task][method]
            per_k = {}
            if os.path.isdir(rp(sweep_dir)):
                per_k = load_sweep(sweep_dir, f"{task}/{method}")
            if not per_k:
                print(f"NOTE: {task}/{method}: no aggregated k-sweep at {sweep_dir} - "
                      f"curve skipped.")
            else:
                n_sweeps += 1
                for k, (pf, oof) in per_k.items():
                    check_partition(f"{method} ksweep k={k}", pf)
                    if "stratification_label" in oof.columns:
                        strat_map.update(dict(zip(oof["sample_id"], oof["stratification_label"])))

            # ---------------------------------------------------- tuned arm
            standalone_dir = REGION_SWEEP_STANDALONE_DIRS[task][method]
            standalone = None
            if os.path.exists(rp(standalone_dir, "nested_full_per_fold.pkl")):
                standalone, s_oof, _ = load_nested_dir(standalone_dir, f"{task}/{method} standalone")
                if "stratification_label" in s_oof.columns:
                    strat_map.update(dict(zip(s_oof["sample_id"], s_oof["stratification_label"])))
            derived_dir = os.path.join(sweep_dir, "ksweep_tuned")
            derived = None
            if per_k and os.path.exists(rp(derived_dir, "nested_full_per_fold.pkl")):
                derived, _, _ = load_nested_dir(derived_dir, f"{task}/{method} ksweep_tuned")
                derived_src, derived_prov = derived_dir, "ksweep_tuned/ (model_hpc ksweep aggregate)"
            elif per_k:
                derived = derive_tuned(per_k)
                derived_src = sweep_dir
                derived_prov = "derived in exporter from the ksweep partials (per fold: best inner score)"
                print(f"NOTE: {task}/{method}: no ksweep_tuned/ (sweep aggregated before "
                      f"the tuned-arm step) - tuned arm derived here with the same rule.")

            tuned, tuned_src, tuned_prov, matches = None, None, None, None
            if standalone is not None:
                tuned, tuned_src = standalone, standalone_dir
                tuned_prov = "standalone fully nested run over the whole k grid"
                if derived is not None:
                    print(f"  {task}/{method}: cross-check of the sweep-derived tuned arm "
                          f"against the standalone run")
                    matches = compare_arms(standalone, derived, "standalone", "ksweep-derived")
                    if method == "data_driven" and not all(matches):
                        raise AssertionError(
                            f"{task}/{method}: the sweep-derived arm must reproduce the "
                            f"manuscript's fully nested run fold for fold")
            elif derived is not None:
                tuned, tuned_src, tuned_prov = derived, derived_src, derived_prov
            if tuned is None:
                print(f"NOTE: {task}/{method}: no tuned arm (neither {standalone_dir} nor a "
                      f"k-sweep) - skipped.")
            else:
                check_partition(f"{method} tuned", tuned)

            # ---------------------------------------------------- fixed arm
            fixed = None
            fixed_dir = FIXED_COMPARATOR_DIRS[task].get(method)   # none for data_driven
            if fixed_dir and not args.skip_fixed:
                if os.path.isdir(rp(fixed_dir)):
                    fixed = load_fixed(fixed_dir, method, f"{task}/{method} fixed")
                    check_partition(f"{method} fixed", fixed)
                else:
                    print(f"NOTE: {task}/{method}: fixed-region run {fixed_dir} missing.")

            n_total = None
            if per_k or tuned is not None:
                n_total = n_regions_total(tuned if tuned is not None else per_k[max(per_k)][0],
                                          per_k or None)

            # ---------------------------------------------------- export rows
            if per_k:
                print(f"  {task}/{method}: k-sweep over {sorted(per_k)} "
                      f"(region set width {n_total:,})")
                for k in sorted(per_k):
                    pf, oof = per_k[k]
                    summ = summarise(pf, oof)
                    print(f"    k={k:>6}: AUROC {summ['mean_test_auc']:.4f} +/- "
                          f"{summ['std_test_auc']:.4f} (pooled {summ['oof_auc_pooled']:.4f}), "
                          f"modal {summ['modal_feature']}/{summ['modal_classifier']}")
                    arm_rows.append({"task": task, "method": method, "arm": "ksweep", "k": k,
                                     "n_regions_total": n_total,
                                     "n_regions_nominal": NOMINAL_REGIONS[method], **summ,
                                     "k_votes": "",
                                     "source_dir": os.path.join(sweep_dir, f"ksweep_k{k:05d}"),
                                     "provenance": "complete nested selection at fixed k"})
                    for _, r in pf.iterrows():
                        ksweep_rows.append({
                            "task": task, "method": method, "k": k, "fold": int(r["fold"]),
                            "n_regions_total": n_total,
                            "num_regions_used": int(r["num_genes_used"]),
                            "feature": r["feature"], "selection_type": r["selection_type"],
                            "classifier": r["classifier"],
                            "hyperparameters": hp_json(r["clf_hyperparams"]),
                            "inner_cv_auc": float(r["inner_best_score"]),
                            "test_auc": float(r["auc_full_model"]),
                            "source_dir": sweep_dir})
                    pred_rows += prediction_rows(task, method, "ksweep", pf, lambda r, k=k: k,
                                                 strat_map)

            if tuned is not None:
                summ = summarise(tuned)
                votes = tuned["top_genes"].astype(int).value_counts().sort_index(ascending=False)
                print(f"  {task}/{method}: k TUNED per fold: AUROC {summ['mean_test_auc']:.4f} "
                      f"+/- {summ['std_test_auc']:.4f}; k votes "
                      f"{'; '.join(f'{k}={v}' for k, v in votes.items())} [{tuned_prov}]")
                arm_rows.append({"task": task, "method": method, "arm": "tuned", "k": np.nan,
                                 "n_regions_total": n_total,
                                 "n_regions_nominal": NOMINAL_REGIONS[method], **summ,
                                 "k_votes": "; ".join(f"{k}={v}" for k, v in votes.items()),
                                 "source_dir": tuned_src, "provenance": tuned_prov
                                 + ("" if matches is None else
                                    f"; sweep-derived arm identical in {sum(matches)}/{len(matches)} folds")})
                pred_rows += prediction_rows(task, method, "tuned", tuned,
                                             lambda r: int(r["top_genes"]), strat_map)
                argmax = None
                if derived is not None:
                    argmax = dict(zip(derived["fold"].astype(int), derived["top_genes"].astype(int)))
                for i, (_, r) in enumerate(tuned.sort_values("fold").iterrows()):
                    cfg_rows.append({
                        "task": task, "method": method, "arm": "tuned", "fold": int(r["fold"]),
                        "feature": r["feature"], "selection_type": r["selection_type"],
                        "k": int(r["top_genes"]), "classifier": r["classifier"],
                        "hyperparameters": hp_json(r["clf_hyperparams"]),
                        "inner_cv_auc": float(r["inner_best_score"]),
                        "test_auc": float(r["auc_full_model"]),
                        "k_argmax_ksweep": (argmax.get(int(r["fold"])) if argmax else np.nan),
                        "matches_standalone": (matches[i] if matches is not None else np.nan)})

            if fixed is not None:
                summ = summarise(fixed)
                n_fixed = int(n_total) if n_total is not None else np.nan
                print(f"  {task}/{method}: FIXED regions (manuscript): AUROC "
                      f"{summ['mean_test_auc']:.4f} +/- {summ['std_test_auc']:.4f}")
                arm_rows.append({"task": task, "method": method, "arm": "fixed", "k": np.nan,
                                 "n_regions_total": n_fixed,
                                 "n_regions_nominal": NOMINAL_REGIONS[method], **summ,
                                 "k_votes": "",
                                 "source_dir": FIXED_COMPARATOR_DIRS[task][method],
                                 "provenance": "manuscript comparator: all regions, no selection, "
                                               "classifier per fold on inner-CV scores"})
                pred_rows += prediction_rows(task, method, "fixed", fixed,
                                             lambda r: n_fixed if n_fixed == n_fixed else -1,
                                             strat_map)
                for _, r in fixed.iterrows():
                    cfg_rows.append({
                        "task": task, "method": method, "arm": "fixed", "fold": int(r["fold"]),
                        "feature": r["feature"], "selection_type": "none",
                        "k": n_fixed, "classifier": r["classifier"],
                        "hyperparameters": "",
                        "inner_cv_auc": float(r["inner_cv_best_score"]),
                        "test_auc": float(r["auc_full_model"]),
                        "k_argmax_ksweep": np.nan, "matches_standalone": np.nan})

    if n_sweeps == 0:
        print("\nNo region-number sweep found for any task/method. Submit them first:")
        print("  FEATURE_SET=rna  ./slurm/submit_nested_full.sh ksweep")
        print("  FEATURE_SET=atac ./slurm/submit_nested_full.sh ksweep")
        print("  CANCER_TYPES=PANCAN FEATURE_SET=rna  EXPERIMENT_NAME=pancan_rna_nested_full  "
              "./slurm/submit_nested_full.sh ksweep")
        print("  CANCER_TYPES=PANCAN FEATURE_SET=atac EXPERIMENT_NAME=pancan_atac_nested_full "
              "./slurm/submit_nested_full.sh ksweep")
        print("  CANCER_TYPES=PANCAN EXPERIMENT_NAME=pancan_nested_full ./slurm/submit_nested_full.sh ksweep")
        raise SystemExit(1)

    KS_COLS = ["task", "method", "k", "fold", "n_regions_total", "num_regions_used", "feature",
               "selection_type", "classifier", "hyperparameters", "inner_cv_auc", "test_auc",
               "source_dir"]
    PRED_COLS = ["task", "method", "arm", "k", "fold", "sample_id", "y_true", "score",
                 "threshold_95spec", "stratification_label"]
    ARM_COLS = ["task", "method", "arm", "k", "n_regions_total", "n_regions_nominal",
                "n_folds", "mean_test_auc",
                "std_test_auc", "oof_auc_pooled", "modal_feature", "modal_selection_type",
                "modal_classifier", "n_distinct_configs", "k_votes", "source_dir", "provenance"]
    CFG_COLS = ["task", "method", "arm", "fold", "feature", "selection_type", "k", "classifier",
                "hyperparameters", "inner_cv_auc", "test_auc", "k_argmax_ksweep",
                "matches_standalone"]
    for name, rows, cols in [("figS13_ksweep_folds.csv", ksweep_rows, KS_COLS),
                             ("figS13_predictions.csv", pred_rows, PRED_COLS),
                             ("figS13_arms.csv", arm_rows, ARM_COLS),
                             ("figS13_fold_configs.csv", cfg_rows, CFG_COLS)]:
        out = os.path.join(DATA_DIR, name)
        pd.DataFrame(rows, columns=cols).to_csv(out, index=False)
        print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
