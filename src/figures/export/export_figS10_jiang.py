"""Source data for the Jiang panels of Fig S10: sample-level covariate baselines on the
Jiang et al. 2015 HCC cohort (90 HCC vs 135 non-HCC controls).

Within Jiang: the five 'transfer5' covariates (mean coverage, total fragments, mean
fragment length, mean fragment GC, BAM-inferred sex) as univariate baselines and as one
fully nested panel (nested_cv_results_jiang/jiang_scalar_panel_transfer5), on the outer
folds of the within-Jiang nested run (asserted identical when that run exists).
Cristiano -> Jiang: univariate baselines with sign and 95%-specificity threshold learned
on all Cristiano samples, and the Cristiano transfer5 panel deployed as its ten refit
outer-fold models (ensemble = mean posterior).

Inputs: accessory_files/jiang_manifest.csv, results/jiang_coverage/,
results/jiang_sex/jiang_sex.csv, the Cristiano baseline tables (export_fig3_baselines.py,
both tasks) and the transfer5 nested runs (src/comp/build_scalar_panel_matrix.py
--panel transfer5 --task {brca_crc,pancan,jiang} + slurm/nested_cv_scalar_panel.sh).
Writes to src/figures/data/: figS10_jiang_scalars.csv, figS10_jiang_predictions.csv,
figS10_jiang_fold_configs.csv, figS10_jiang_panel_importance.csv,
figS10_jiang_transfer.csv, tableS3_baselines_jiang.csv.

Run from the repository root:
    python src/figures/export/export_figS10_jiang.py [--gc_cache CSV] [--scalars_only]
--scalars_only rewrites only figS10_jiang_scalars.csv (reusing fragment length and GC
from the previous table). mean_gc is a slow network read; it is cached in the scalars CSV
or --gc_cache.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(__file__))
from _common import DATA_DIR, JIANG_NESTED_FULL_DIRS, assert_predictions_match_stored, resolve_sex_call, rp  # noqa: E402
from export_fig3_baselines import (  # noqa: E402
    PANELS,
    TASKS,
    _fresh_classifier,
    add_fragment_length_mean,
    add_mean_gc,
    binary_metrics,
    evaluate_scalar,
    export_panel_arm,
    load_panel_matrix,
    panel_importance,
    rows_from_per_fold,
    spec95_threshold,
    summarize_method,
)

JIANG_MANIFEST = "accessory_files/jiang_manifest.csv"
JIANG_COVERAGE_DIR = "results/jiang_coverage"
JIANG_SEX_CSV = "results/jiang_sex/jiang_sex.csv"
# columns of the BAM sex call (src/comp/compute_sex.py) written next to the covariates
SEX_COLS = ["sex_bam", "sex_call_how", "r_y", "r_x"]
FIG5BC_EXTERNAL = "fig4bc_external_predictions.csv"
METHOD = "bl_panel_transfer5"
COVARIATES = PANELS[METHOD]["cols"]
UNIVARIATE = list(COVARIATES)   # every single-covariate arm (within Jiang and transferred)
CRIS_TABLES = {"brca_crc": "fig3_baseline_scalars.csv", "pancan": "fig4_baseline_scalars.csv"}
GROUPS = ["all_non_hcc", "Healthy", "Chronic hepatitis B", "HBV-associated cirrhosis"]


# --------------------------------------------------------------------------
# cohort + covariates
# --------------------------------------------------------------------------

def jiang_cohort() -> pd.DataFrame:
    """The 225 model-role samples in model_hpc row order (controls in manifest
    order, then HCC) with the outer folds model_hpc's nested CV draws."""
    cfg = TASKS["jiang"]
    mf = pd.read_csv(rp(JIANG_MANIFEST))
    model = mf[(mf["role"] == "model") & ~mf["gcfix_pending"].fillna(False).astype(bool)]
    model = model.drop_duplicates(subset="ID")
    ctrl = model[model["disease"].isin(cfg["control_labels"])]
    hcc = model[model["disease"].isin(cfg["cancer_labels"])]
    rows = pd.concat([ctrl, hcc], ignore_index=True)
    assert len(rows) == cfg["n_samples"] and len(hcc) == cfg["n_cancer"], (len(rows), len(hcc))
    t = pd.DataFrame({
        "sample_id": rows["ID"], "sample_name": rows["sample_name"],
        "y_true": rows["disease"].isin(cfg["cancer_labels"]).astype(int),
        "disease": rows["disease"], "control_group": rows["control_group"].fillna(""),
        "total_fragments": rows["fragNum"].astype(float),
    })
    # model_hpc: StratifiedKFold(10, shuffle, rs=42).split(X, disease strata)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    fold = np.zeros(len(t), int)
    for k, (_, te) in enumerate(skf.split(np.zeros(len(t)), t["disease"].to_numpy()), 1):
        fold[te] = k
    t["fold"] = fold
    return t


def assert_folds_match_nested_full(t: pd.DataFrame) -> pd.DataFrame | None:
    d = JIANG_NESTED_FULL_DIRS["hcc_vs_nonhcc"]
    path = rp(d, "nested_full_per_fold.pkl")
    if not os.path.exists(path):
        print(f"  within-Jiang nested TSS run not on disk ({d}); folds reproduced from the "
              "fold generator, data_driven arm skipped")
        return None
    with open(path, "rb") as fh:
        per_fold = pickle.load(fh).rename(columns={"test_auc": "auc_full_model", "n_top_genes": "top_genes"})
    fold_of = dict(zip(t["sample_id"], t["fold"]))
    assert set(fold_of) == {s for _, r in per_fold.iterrows() for s in r["test_sample_ids"]}, \
        "within-Jiang nested run cohort differs from the manifest cohort"
    for _, r in per_fold.iterrows():
        seen = {fold_of[s] for s in r["test_sample_ids"]}
        assert seen == {int(r["fold"])}, f"fold {r['fold']}: reproduced partition differs ({seen})"
    print("  within-Jiang nested TSS run found: outer partition identical to the reproduced folds")
    return per_fold


def add_coverage(t: pd.DataFrame) -> None:
    vals = []
    for sid in t["sample_id"]:
        c = pd.read_csv(rp(JIANG_COVERAGE_DIR, f"{sid}.csv"))
        assert c["sample_name"].iloc[0] == sid and len(c) == 1, sid
        vals.append(float(c["mean_coverage"].iloc[0]))
    t["mean_coverage"] = vals


def add_bam_sex(t: pd.DataFrame) -> list[str]:
    """Resolved R_Y / R_X sex call per sample (sex_bam, sex_female_bam = 1 female / 0 male);
    returns the extra columns written next to the covariates."""
    assert os.path.exists(rp(JIANG_SEX_CSV)), f"{JIANG_SEX_CSV} not found (DATASET=jiang slurm/sex_inference.sh + --summarize)"
    s = resolve_sex_call(pd.read_csv(rp(JIANG_SEX_CSV))).set_index("sample_name")
    t["sex_bam"] = t["sample_id"].map(s["sex_call_resolved"])
    assert t["sex_bam"].isin(["female", "male"]).all(), f"Jiang samples without a resolved sex call: {t['sex_bam'].value_counts(dropna=False).to_dict()}"
    t["sex_female_bam"] = t["sex_bam"].map({"female": 1.0, "male": 0.0})
    t["sex_call_how"] = t["sample_id"].map(s["sex_call_how"])
    t["r_y"] = t["sample_id"].map(s["r_y"])
    t["r_x"] = t["sample_id"].map(s["r_x"])
    print("  sex (R_Y + R_X): " + ", ".join(f"{k} {v}" for k, v in t["sex_bam"].value_counts().items())
          + f"; how: {t['sex_call_how'].value_counts().to_dict()}")
    return SEX_COLS


# --------------------------------------------------------------------------
# Cristiano -> Jiang transfer
# --------------------------------------------------------------------------

def group_aucs(df: pd.DataFrame, score_col: str = "score") -> dict:
    """AUROC of HCC vs each comparison group (and vs all non-HCC)."""
    out = {}
    hcc = df[df["y_true"] == 1]
    for g in GROUPS:
        ctrl = df[df["y_true"] == 0] if g == "all_non_hcc" else df[(df["y_true"] == 0) & (df["control_group"] == g)]
        sub = pd.concat([hcc, ctrl])
        out[f"auc_hcc_vs_{g}"] = roc_auc_score(sub["y_true"], sub[score_col])
    return out


def transfer_univariate(cris: pd.DataFrame, jiang: pd.DataFrame, col: str, task: str) -> pd.DataFrame:
    """Sign + 95%-spec threshold from ALL Cristiano task samples, applied to Jiang."""
    x = cris[col].fillna(cris[col].median()).to_numpy(float)
    y = cris["y_true"].to_numpy(int)
    orientation = 1.0 if roc_auc_score(y, x) >= 0.5 else -1.0
    thr = spec95_threshold(y, orientation * x)
    return pd.DataFrame({
        "method": f"bl_{col}", "source_task": task, "sample_id": jiang["sample_id"],
        "y_true": jiang["y_true"], "subgroup": jiang["disease"], "control_group": jiang["control_group"],
        "score": orientation * jiang[col].to_numpy(float), "threshold_95spec": thr,
        "orientation": orientation, "fold_model": "all-Cristiano",
    })


def transfer_panel(cris: pd.DataFrame, jiang: pd.DataFrame, task: str) -> tuple[pd.DataFrame, dict] | None:
    """Cristiano transfer5 panel deployed on Jiang as its ten refit outer-fold models.
    Returns per-sample rows (one block per fold model plus the ensemble) and a summary."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    out = export_panel_arm(cris, task, METHOD)
    if out is None:
        return None
    sel = out[3]
    X, y, ids, names = load_panel_matrix(task, METHOD)
    assert names == COVARIATES, names
    Xj = jiang[names].to_numpy(float)
    pos = {s: i for i, s in enumerate(ids)}
    blocks, per_model = [], []
    for _, row in sel.sort_values("fold").iterrows():
        test_idx = np.array(sorted(pos[s] for s in row["test_sample_ids"]))
        train_idx = np.setdiff1d(np.arange(len(y)), test_idx)
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", _fresh_classifier(row["classifier"]))])
        pipe.set_params(**row["best_params"]).fit(X[train_idx], y[train_idx])
        p_test = pipe.predict_proba(X[test_idx])[:, 1]
        stored = np.asarray(row["y_pred_proba"], float)[np.argsort([pos[s] for s in row["test_sample_ids"]])]
        assert np.allclose(p_test, stored, atol=1e-6), f"{task} fold {row['fold']}: refit does not reproduce"
        pj = pipe.predict_proba(Xj)[:, 1]
        thr = float(row["full_model_thresholds"]["95_percent_specificity"])
        blocks.append(pd.DataFrame({
            "method": METHOD, "source_task": task, "sample_id": jiang["sample_id"], "y_true": jiang["y_true"],
            "subgroup": jiang["disease"], "control_group": jiang["control_group"], "score": pj,
            "threshold_95spec": thr, "orientation": np.nan, "fold_model": f"fold{int(row['fold'])}",
        }))
        m = binary_metrics(jiang["y_true"].to_numpy(int), (pj >= thr).astype(int))
        per_model.append({"fold": int(row["fold"]), "classifier": row["classifier"],
                          "auc_jiang": roc_auc_score(jiang["y_true"], pj), **m})
    per_model = pd.DataFrame(per_model)
    ens = np.mean([b["score"].to_numpy() for b in blocks], axis=0)
    blocks.append(pd.DataFrame({
        "method": METHOD, "source_task": task, "sample_id": jiang["sample_id"], "y_true": jiang["y_true"],
        "subgroup": jiang["disease"], "control_group": jiang["control_group"], "score": ens,
        "threshold_95spec": np.nan, "orientation": np.nan, "fold_model": "ensemble",
    }))
    summary = {
        "method": METHOD, "protocol": f"transfer_{task}", "n_samples": len(jiang),
        "auc_pooled_oof": roc_auc_score(jiang["y_true"], ens),
        "auc_fold_mean": per_model["auc_jiang"].mean(), "auc_fold_sd": per_model["auc_jiang"].std(ddof=0),
        "sensitivity_95spec_mean": per_model["sensitivity"].mean(), "sensitivity_95spec_sd": per_model["sensitivity"].std(ddof=0),
        "specificity_95spec_mean": per_model["specificity"].mean(), "specificity_95spec_sd": per_model["specificity"].std(ddof=0),
        "selected_classifier": out[1]["selected_classifier"], "panel_features": "+".join(names),
        **group_aucs(blocks[-1]),
    }
    print(f"  transfer {task} -> Jiang: ensemble AUROC {summary['auc_pooled_oof']:.3f} "
          f"(ten fold models {summary['auc_fold_mean']:.3f} +/- {summary['auc_fold_sd']:.3f}; "
          f"sens {summary['sensitivity_95spec_mean']:.2f} / spec {summary['specificity_95spec_mean']:.2f} at "
          "each model's Cristiano threshold)")
    return pd.concat(blocks, ignore_index=True), summary


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gc_cache", default=None,
                    help="CSV with sample_id, mean_gc to reuse (default: the previously exported scalars table)")
    ap.add_argument("--scalars_only", action="store_true",
                    help="Write figS10_jiang_scalars.csv and stop (no panel refits or evaluation)")
    args = ap.parse_args()
    cfg = TASKS["jiang"]
    scalars_out = os.path.join(DATA_DIR, cfg["scalars_csv"])

    print("assembling the 225-sample Jiang covariate table ...")
    t = jiang_cohort()
    add_coverage(t)
    if args.scalars_only and os.path.exists(scalars_out):
        # the fragment-length .npy files sit on the cluster share; reuse the audited values
        prev = pd.read_csv(scalars_out).set_index("sample_id")
        t["fragment_length_mean"] = t["sample_id"].map(prev["fragment_length_mean"])
        print(f"  fragment_length_mean reused from {scalars_out}")
    else:
        add_fragment_length_mean(t)
    add_mean_gc(t, args.gc_cache or scalars_out)
    sex_cols = add_bam_sex(t)
    for col in UNIVARIATE:
        assert t[col].notna().all(), f"{col}: missing values"
    cols = ["sample_id", "sample_name", "fold", "y_true", "disease", "control_group"] + UNIVARIATE + sex_cols
    t[cols].to_csv(scalars_out, index=False)
    print(f"wrote {scalars_out} ({len(t)} rows)")
    if args.scalars_only:
        return
    per_fold_dd = assert_folds_match_nested_full(t)

    # ---- within Jiang -------------------------------------------------------
    print("\nwithin-Jiang evaluation on the frozen folds ...")
    bl_pred = pd.concat([evaluate_scalar(t, col) for col in UNIVARIATE], ignore_index=True)
    rows, configs, imp_rows = [], [], []
    if per_fold_dd is not None:
        assert_predictions_match_stored(per_fold_dd, "jiang data_driven")
        dd_pred = rows_from_per_fold(per_fold_dd, "data_driven")
        bl_pred = pd.concat([dd_pred, bl_pred], ignore_index=True)
        rows.append(summarize_method(dd_pred, "data_driven", "nested_cv"))
    panel = export_panel_arm(t, "jiang", METHOD)
    if panel is not None:
        bl_pred = pd.concat([bl_pred, panel[0]], ignore_index=True)
        rows.append(panel[1])
        configs = panel[2]
        imp_rows = panel_importance("jiang", METHOD, panel[3])
    for col in UNIVARIATE:
        rows.append(summarize_method(bl_pred[bl_pred["method"] == f"bl_{col}"], f"bl_{col}", "scalar_train_oriented"))
    for r in rows:  # AUROC vs each comparison group on the pooled OOF scores
        d = bl_pred[bl_pred["method"] == r["method"]].merge(t[["sample_id", "control_group"]], on="sample_id")
        r.update(group_aucs(d))
    pred_out = os.path.join(DATA_DIR, "figS10_jiang_predictions.csv")
    bl_pred.to_csv(pred_out, index=False)
    print(f"wrote {pred_out} ({len(bl_pred)} rows, {bl_pred['method'].nunique()} methods)")
    pd.DataFrame(configs).to_csv(os.path.join(DATA_DIR, "figS10_jiang_fold_configs.csv"), index=False)
    pd.DataFrame(imp_rows).to_csv(os.path.join(DATA_DIR, "figS10_jiang_panel_importance.csv"), index=False)

    # ---- Cristiano -> Jiang ---------------------------------------------------
    print("\nCristiano -> Jiang transfer ...")
    transfer = []
    for task, csv in CRIS_TABLES.items():
        cris = pd.read_csv(os.path.join(DATA_DIR, csv))
        for col in UNIVARIATE:
            assert col in cris.columns and cris[col].notna().all(), (
                f"{csv} lacks {col} - rerun export_fig3_baselines.py --task {task} (--scalars_only is enough)")
        for col in UNIVARIATE:
            tr = transfer_univariate(cris, t, col, task)
            transfer.append(tr)
            m = binary_metrics(tr["y_true"].to_numpy(int), (tr["score"] >= tr["threshold_95spec"]).astype(int))
            rows.append({"method": f"bl_{col}", "protocol": f"transfer_{task}", "n_samples": len(tr),
                         "auc_pooled_oof": roc_auc_score(tr["y_true"], tr["score"]),
                         "sensitivity_95spec_mean": m["sensitivity"], "specificity_95spec_mean": m["specificity"],
                         "orientation": tr["orientation"].iloc[0], **group_aucs(tr)})
        tp = transfer_panel(cris, t, task)
        if tp is not None:
            transfer.append(tp[0])
            rows.append(tp[1])
    # the deployed TSS models (export_fig4.py) on the same comparison groups
    ext_path = os.path.join(DATA_DIR, FIG5BC_EXTERNAL)
    if os.path.exists(ext_path):
        ext = pd.read_csv(ext_path)
        for method, g in ext.groupby("method"):
            g = g.merge(t[["sample_id"]], on="sample_id")
            assert len(g) == len(t), f"{FIG5BC_EXTERNAL} {method}: cohort differs"
            rows.append({"method": {"dd": "data_driven"}.get(method, method), "protocol": "deployed_cristiano_model (fig4bc)",
                         "n_samples": len(g), "auc_pooled_oof": roc_auc_score(g["y_true"], g["score"]), **group_aucs(g)})
    transfer = pd.concat(transfer, ignore_index=True)
    tr_out = os.path.join(DATA_DIR, "figS10_jiang_transfer.csv")
    transfer.to_csv(tr_out, index=False)
    print(f"wrote {tr_out} ({len(transfer)} rows)")

    table = pd.DataFrame(rows)
    table_out = os.path.join(DATA_DIR, "tableS3_baselines_jiang.csv")
    table.to_csv(table_out, index=False)
    print(f"wrote {table_out}")
    with pd.option_context("display.width", 200):
        print(table[["method", "protocol", "auc_pooled_oof", "auc_fold_mean", "auc_fold_sd",
                     "auc_hcc_vs_Healthy", "auc_hcc_vs_Chronic hepatitis B", "auc_hcc_vs_HBV-associated cirrhosis",
                     "sensitivity_95spec_mean", "specificity_95spec_mean"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
