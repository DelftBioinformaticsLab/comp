"""Source data for Figure 3 and Figure S3: the data-driven model against the RNA and ATAC
comparators and ichorCNA on healthy vs BRCA/CRC (Cristiano cohort).

Inputs (result directories in _common.py, cohort variant from COHORT_VARIANT):
NESTED_FULL_DIR (fully nested data-driven run), RNA_RESULT_DIR (Zhu regions, rcov),
ATAC_RESULT_DIR (Ulz TFBS) and accessory_files/cris_stage_vaf_coverage_ichor_aggregate.csv.
RNA and ATAC classifiers are chosen per outer fold on inner-CV scores.

Writes to src/figures/data/<COHORT_VARIANT>/: fig3_predictions.csv (out-of-fold score, fold,
label and 95%-specificity threshold per method and sample; ichorCNA rows outside the task have
fold = NA), fig3_config.csv (configuration per method) and fig3_model_selection.csv (per-fold
selection table for the QC section of fig3.ipynb).

Run from the repository root:
    python src/figures/export/export_fig3.py
"""

import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from _common import (  # noqa: E402
    ATAC_RESULT_DIR,
    DATA_DIR,
    ICHOR_AGG_CSV,
    NESTED_FULL_DIR,
    RNA_RESULT_DIR,
    apply_rerun_ichorcna_tf,
    assert_predictions_match_stored,
    best_mean_auc_config,
    excluded_sample_ids,
    load_results,
    rp,
)

KEEP = [
    "feature", "classifier", "top_genes", "fold", "auc_full_model",
    "test_sample_ids", "y_pred_proba", "y_test",
    "full_model_thresholds", "confusion_matrix_full_model",
    # optional (absent in older pickles): inner-CV score of the
    # winning hyperparameters — load_results keeps them only when present
    "selection_type", "inner_cv_best_score",
]


def nested_classifier_selection(cand: pd.DataFrame, locked_feature: str,
                                result_dir: str, label: str) -> tuple[dict, dict]:
    """Per outer fold, the classifier with the best inner-CV AUROC on a locked feature.
    Returns ({'sel': per-fold rows, 'candidates': candidates with inner scores},
    post-hoc best-mean-outer-AUC config as a diagnostic)."""
    cand = cand[cand["feature"] == locked_feature].copy()
    if "inner_cv_best_score" not in cand.columns or cand["inner_cv_best_score"].isna().any():
        scores_path = rp(result_dir, "inner_cv_scores.csv")
        assert os.path.exists(scores_path), (
            f"{label}: pickles lack inner_cv_best_score and {scores_path} is missing - rerun the "
            f"nested CV of {result_dir} with the current model_hpc.py")
        sc = pd.read_csv(scores_path)
        sc = sc[sc["feature"] == locked_feature][["classifier", "fold", "inner_cv_auc"]]
        cand = cand.drop(columns=[c for c in ["inner_cv_best_score"] if c in cand.columns])
        cand = cand.merge(sc.rename(columns={"inner_cv_auc": "inner_cv_best_score"}),
                          on=["classifier", "fold"], how="left")
        assert cand["inner_cv_best_score"].notna().all(), f"{label}: inner scores incomplete"
        print(f"  {label}: inner-CV scores taken from {os.path.basename(result_dir)}/inner_cv_scores.csv")
    sel = cand.loc[cand.groupby("fold")["inner_cv_best_score"].idxmax()].sort_values("fold")
    posthoc = best_mean_auc_config(cand, ["classifier"])
    chosen = sel["classifier"].value_counts().to_dict()
    posthoc["per_fold"] = chosen
    print(f"  {label}: fully nested per-fold classifier choice {chosen}; mean outer AUC "
          f"{sel['auc_full_model'].mean():.4f} (post-hoc best-of-{cand['classifier'].nunique()} "
          f"{posthoc['classifier']} would give {posthoc['mean_auc']:.4f}, "
          f"optimism {posthoc['mean_auc'] - sel['auc_full_model'].mean():+.4f})")
    return {"sel": sel, "candidates": cand}, posthoc


def load_nested_full() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-fold table (columns renamed to the load_results schema) and the run's OOF CSV."""
    with open(rp(NESTED_FULL_DIR, "nested_full_per_fold.pkl"), "rb") as fh:
        per_fold = pickle.load(fh)
    oof = pd.read_csv(rp(NESTED_FULL_DIR, "nested_full_oof_predictions.csv"))
    per_fold = per_fold.rename(columns={"test_auc": "auc_full_model", "n_top_genes": "top_genes"})
    return per_fold, oof


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

    # ------------------------------------------------------------------ load
    dd_sel, dd_oof = load_nested_full()
    rna = load_results(f"{RNA_RESULT_DIR}/results*.pkl", KEEP)
    atac = load_results(f"{ATAC_RESULT_DIR}/results*.pkl", KEEP)
    print(f"loaded: dd nested_full {len(dd_sel)} folds, rna {len(rna)} rows, atac {len(atac)} rows")

    # ------------------------------------------------------- model selection
    # Data-driven: chosen inside each fold. RNA / ATAC: classifier chosen per outer fold on
    # inner-CV scores (from the pickles, else inner_cv_scores.csv).
    rna_sel, best_rna = nested_classifier_selection(rna, "rcov", RNA_RESULT_DIR, "rna")
    atac_sel, best_atac = nested_classifier_selection(atac, "ulz_atac_tfbs_features",
                                                      ATAC_RESULT_DIR, "atac")
    rna = rna_sel.pop("candidates")
    atac = atac_sel.pop("candidates")
    rna_sel, atac_sel = rna_sel["sel"], atac_sel["sel"]

    for name, sel in [("data_driven", dd_sel), ("rna", rna_sel), ("atac", atac_sel)]:
        assert len(sel) == 10, f"{name}: expected 10 folds, got {len(sel)}"

    dd_modal = {
        "feature": dd_sel["feature"].mode().iloc[0],
        "classifier": dd_sel["classifier"].mode().iloc[0],
        "weighting_function": dd_sel["selection_type"].mode().iloc[0],
        "top_genes": int(dd_sel["top_genes"].mode().iloc[0]),
    }
    n_distinct = len(dd_sel[["feature", "selection_type", "top_genes", "classifier"]].drop_duplicates())
    print("selected configs:")
    print(f"  data_driven: chosen per fold (fully nested), modal = {dd_modal}, "
          f"{n_distinct} distinct config(s); mean AUC "
          f"{dd_sel['auc_full_model'].mean():.4f} ± {dd_sel['auc_full_model'].std(ddof=0):.4f}")
    print(f"  rna:         {best_rna}")
    print(f"  atac:        {best_atac}")

    # --------------------------------------------------- fold-partition check
    def fold_map(sel: pd.DataFrame) -> dict[int, frozenset]:
        return {int(r["fold"]): frozenset(r["test_sample_ids"]) for _, r in sel.iterrows()}

    dd_folds, rna_folds, atac_folds = fold_map(dd_sel), fold_map(rna_sel), fold_map(atac_sel)
    assert dd_folds == rna_folds == atac_folds, (
        "outer-fold partitions differ between the three arms; the ichorCNA fold "
        "assignment below would be ambiguous")
    print("fold partitions identical across data_driven / rna / atac")

    # cross-check the per-fold table against the run's own OOF CSV
    oof_map = dict(zip(dd_oof["sample_id"], dd_oof["y_pred_proba"]))
    for _, row in dd_sel.iterrows():
        for sid, s in zip(row["test_sample_ids"], row["y_pred_proba"]):
            assert np.isclose(oof_map[sid], s), f"dd OOF mismatch for {sid}"
    print("data_driven per-fold scores match nested_full_oof_predictions.csv")

    # ------------------------------------------------ score/threshold checks
    for name, sel in [("data_driven", dd_sel), ("rna", rna_sel), ("atac", atac_sel)]:
        assert_predictions_match_stored(sel, name)
        has_train_fields = all("train_fpr_at_95spec" in r["full_model_thresholds"]
                               for _, r in sel.iterrows())
        print(f"  {name}: stored AUC + 95%-spec confusion matrices reproduced; "
              f"train-derived threshold fields present: {has_train_fields}")

    # ----------------------------------------------------------- assemble
    records = []
    for method, sel in [("data_driven", dd_sel), ("rna", rna_sel), ("atac", atac_sel)]:
        for _, row in sel.iterrows():
            thr = float(row["full_model_thresholds"]["95_percent_specificity"])
            for sid, y, s in zip(row["test_sample_ids"], row["y_test"], row["y_pred_proba"]):
                records.append({"sample_id": sid, "method": method,
                                "fold": int(row["fold"]), "y_true": int(y),
                                "score": float(s), "threshold_95spec": thr})
    pred = pd.DataFrame(records)

    # ichorCNA: every sample, with its test fold (NA outside the task; those rows only feed
    # the per-fold ichorCNA thresholds).
    ichor = pd.read_csv(rp(ICHOR_AGG_CSV), index_col=0)
    multiplicity = ichor.groupby("ID").size()
    dup_ids = multiplicity[multiplicity > 1]
    if len(dup_ids):
        # duplicated rows must be identical; tumour fractions are replaced below
        for sid in dup_ids.index:
            grp = ichor[ichor["ID"] == sid]
            assert (grp.nunique() <= 1).all(), f"duplicate rows for {sid} are not identical"
        ichor = ichor.drop_duplicates(subset="ID")
    # samples excluded by the cohort variant also leave the ichorCNA arm
    excl = set(excluded_sample_ids())
    if excl:
        n_before = len(ichor)
        ichor = ichor[~ichor["ID"].isin(excl)]
        print(f"ichorCNA CSV: {n_before - len(ichor)} excluded sample(s) removed (cohort variant)")
    ichor = apply_rerun_ichorcna_tf(ichor)   # tumour fractions of the rerun (cohort panel of normals)
    print(f"ichorCNA CSV: {len(ichor)} unique samples")

    sample_to_fold = {sid: f for f, ids in dd_folds.items() for sid in ids}
    ichor_rows = pd.DataFrame({
        "sample_id": ichor["ID"],
        "method": "ichorcna",
        "fold": ichor["ID"].map(sample_to_fold),
        "y_true": (ichor["disease"] != "Healthy").astype(int),
        "score": ichor["ichorcna_tf"],
        "threshold_95spec": np.nan,
    })
    pred = pd.concat([pred, ichor_rows], ignore_index=True)
    # every source row is unique; column kept for the notebook
    pred["n_rows_in_source"] = 1

    # stage + disease annotations (Cristiano et al. publication metadata)
    ann = ichor.set_index("ID")
    # strip: one stage I patient is stored as "I "
    pred["stage"] = pred["sample_id"].map(ann["Stage"].astype("string").str.strip())
    pred["disease"] = pred["sample_id"].map(ann["disease"])
    task_cancer = pred[(pred["method"] == "data_driven") & (pred["y_true"] == 1)]
    n_nostage = int(task_cancer["stage"].isna().sum())
    if n_nostage:
        print(f"WARNING: {n_nostage} task cancer sample(s) lack a Stage annotation "
              f"({sorted(task_cancer.loc[task_cancer['stage'].isna(), 'sample_id'])[:5]} ...)")

    n_task = pred[pred["method"] == "data_driven"]["sample_id"].nunique()
    print(f"task samples: {n_task}; ichorCNA rows: {len(ichor_rows)} "
          f"({int(ichor_rows['fold'].notna().sum())} inside the task folds)")

    out = os.path.join(DATA_DIR, "fig3_predictions.csv")
    pred.to_csv(out, index=False)
    print(f"wrote {out} ({len(pred)} rows)")

    cfg = pd.DataFrame([
        {"method": "data_driven", "feature": dd_modal["feature"],
         "classifier": dd_modal["classifier"], "weighting_function": dd_modal["weighting_function"],
         "top_genes": 14224 if dd_modal["top_genes"] >= 14224 else dd_modal["top_genes"],
         "source_dir": NESTED_FULL_DIR,
         "selection_mean_auc": float(dd_sel["auc_full_model"].mean()),
         "note": f"fully nested: config chosen per fold; modal shown ({n_distinct} distinct)"},
        {"method": "rna", "feature": "rcov",
         "classifier": rna_sel["classifier"].mode().iloc[0],
         "weighting_function": "", "top_genes": "", "source_dir": RNA_RESULT_DIR,
         "selection_mean_auc": float(rna_sel["auc_full_model"].mean()),
         "note": f"fully nested: classifier chosen per fold on inner CV {best_rna['per_fold']}; "
                 f"modal shown (post-hoc best-of-6 {best_rna['classifier']} would give {best_rna['mean_auc']:.4f})"},
        {"method": "atac", "feature": "ulz_atac_tfbs_features",
         "classifier": atac_sel["classifier"].mode().iloc[0],
         "weighting_function": "", "top_genes": "", "source_dir": ATAC_RESULT_DIR,
         "selection_mean_auc": float(atac_sel["auc_full_model"].mean()),
         "note": f"fully nested: classifier chosen per fold on inner CV {best_atac['per_fold']}; "
                 f"modal shown (post-hoc best-of-6 {best_atac['classifier']} would give {best_atac['mean_auc']:.4f})"},
        {"method": "ichorcna", "feature": "ichorcna_tf", "classifier": "",
         "weighting_function": "", "top_genes": "", "source_dir": ICHOR_AGG_CSV,
         "selection_mean_auc": "", "note": ""},
    ])
    out_cfg = os.path.join(DATA_DIR, "fig3_config.csv")
    cfg.to_csv(out_cfg, index=False)
    print(f"wrote {out_cfg}")

    # ------------------------------------------- per-fold selection audit trail
    # data_driven: per-fold winner plus inner scores of the other features;
    # rna/atac: every candidate classifier on the locked feature
    sel_rows = []
    for _, r in dd_sel.iterrows():
        sel_rows.append({"method": "data_driven", "fold": int(r["fold"]),
                         "feature": r["feature"], "classifier": r["classifier"],
                         "selection_type": r["selection_type"], "top_genes": r["top_genes"],
                         "inner_cv_auc": float(r["inner_best_score"]),
                         "test_auc": float(r["auc_full_model"]), "chosen": True})
        for feat, sc in r["inner_best_score_per_feature"].items():
            if feat == r["feature"]:
                continue
            sel_rows.append({"method": "data_driven", "fold": int(r["fold"]),
                             "feature": feat, "classifier": np.nan,
                             "selection_type": np.nan, "top_genes": np.nan,
                             "inner_cv_auc": float(sc), "test_auc": np.nan, "chosen": False})
    for method, cand, locked, sel in [
            ("rna", rna, "rcov", rna_sel),
            ("atac", atac, "ulz_atac_tfbs_features", atac_sel)]:
        chosen_pairs = set(zip(sel["fold"].astype(int), sel["classifier"]))
        for _, r in cand[cand["feature"] == locked].iterrows():
            sel_rows.append({"method": method, "fold": int(r["fold"]),
                             "feature": r["feature"], "classifier": r["classifier"],
                             "selection_type": r.get("selection_type", "none"),
                             "top_genes": r["top_genes"],
                             "inner_cv_auc": float(r["inner_cv_best_score"]),
                             "test_auc": float(r["auc_full_model"]),
                             "chosen": (int(r["fold"]), r["classifier"]) in chosen_pairs})
    out_sel = os.path.join(DATA_DIR, "fig3_model_selection.csv")
    pd.DataFrame(sel_rows).to_csv(out_sel, index=False)
    print(f"wrote {out_sel} ({len(sel_rows)} rows)")


if __name__ == "__main__":
    main()
