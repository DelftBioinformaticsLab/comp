"""Fig S16: models trained on the Cristiano PANCAN task applied once to the Jiang et al.
2015 HCC cohort (225 samples), with preprocessing and the 95%-specificity threshold fixed
on Cristiano.

Arms: tss (TSS model), tss_cov5 (TSS model plus the five 'transfer5' covariates as extra
columns), cov5 (covariates alone), rna and atac (comparators, --from_runs only), bl_<cov>
(each covariate alone) and bl_ichorcna_tf (ichorCNA tumour fraction).

Modes: --from_runs reads the deployments of slurm/submit_jiang_arms.sh
(nested_cv_results_cris/external_validation_comprehensive_external_cris_on_jiang_
{dd, dd_cov_transfer5, cov_transfer5, rna, atac}_rev[_nodeep]/; per arm the feature type
with the best training CV AUROC). --gd_cache <npz> is an exploratory quick refit on a
cached Griffin-dip matrix; its outputs carry the suffix _quickfit.

Inputs: data[/no_deep]/fig4_baseline_scalars.csv and data/figS10_jiang_scalars.csv
(export_figS10_jiang.py). Writes data[/no_deep]/figS16_jiang_scores[_quickfit].csv,
figS16_jiang_summary[_quickfit].csv and output[/no_deep]/figS16_jiang_arms[_quickfit].png.
Run from the repository root:
    COHORT_VARIANT=no_deep python src/figures/export/export_figS16_jiang_arms.py --from_runs
    python src/figures/export/export_figS16_jiang_arms.py --gd_cache <npz> [--cores 8]
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, os.path.dirname(__file__))
from _common import COHORT_VARIANT, DATA_DIR, IS_VARIANT, rp, variant_dir  # noqa: E402

# the 'transfer5' panel (build_scalar_panel_matrix.py); sex_female_bam = genetic sex from the BAM
COV5 = ["mean_coverage", "total_fragments", "fragment_length_mean", "mean_gc", "sex_female_bam"]
GROUPS = ["all_non_hcc", "Healthy", "Chronic hepatitis B", "HBV-associated cirrhosis"]
CV = StratifiedKFold(3, shuffle=True, random_state=42)
# the model's own logistic-regression grid (model_hpc CLF_PARAM_GRIDS): elastic net, saga
LR_GRID = {"clf__C": list(np.logspace(-3, 1, 5)), "clf__penalty": ["elasticnet"], "clf__solver": ["saga"], "clf__l1_ratio": [0.1, 0.5, 0.9]}
PANEL_GRIDS = {
    "Logistic Regression": (LogisticRegression(max_iter=10000, random_state=42), {"clf__C": [0.01, 0.1, 1.0, 10.0]}),
    "Random Forest": (RandomForestClassifier(random_state=42), {"clf__n_estimators": [100, 300], "clf__max_depth": [3, 5, None]}),
    "Support Vector Machine": (SVC(probability=True, random_state=42), {"clf__C": [0.1, 1.0, 10.0], "clf__kernel": ["rbf", "linear"]}),
    "K-Nearest Neighbors": (KNeighborsClassifier(), {"clf__n_neighbors": [5, 11, 21]}),
}
try:
    from xgboost import XGBClassifier
    PANEL_GRIDS["XGBoost"] = (XGBClassifier(random_state=42, verbosity=0), {"clf__n_estimators": [100, 300], "clf__max_depth": [2, 4], "clf__learning_rate": [0.05, 0.2]})
except Exception:  # pragma: no cover
    pass

# official runs (slurm/submit_jiang_arms.sh): arm -> feature-type tag of the deployment directory
RUN_TAGS = {"tss": "dd", "tss_cov5": "dd_cov_transfer5", "cov5": "cov_transfer5", "rna": "rna", "atac": "atac"}
LABELS = {"tss": "TSS model", "tss_cov5": "TSS + 5 transferable measures", "cov5": "5 transferable measures alone",
          "rna": "RNA regions (Zhu)", "atac": "ATAC TFBS (Ulz)", **{f"bl_{c}": c.replace("_", " ") for c in COV5},
          "bl_sex_female_bam": "sex (from the BAM)", "bl_ichorcna_tf": "ichorCNA tumour fraction"}
COLORS = {"tss": "#0173B2", "tss_cov5": "#56B4E9", "cov5": "#D55E00", "rna": "#029E73", "atac": "#CC78BC",
          **{f"bl_{c}": "0.6" for c in COV5}, "bl_ichorcna_tf": "#6A3D9A"}
# ichorCNA per cohort, both with the Cristiano 50-sample panel of normals
ICHOR_TSV = {"cris": "ichorCNA/ichorCNA_summary.tsv", "jiang": "ichorCNA_jiang/ichorCNA_summary.tsv"}


def spec95_threshold(y, s):
    fpr, _, thr = roc_curve(y, s)
    idx = np.where(1 - fpr >= 0.95)[0]
    return float(thr[idx[-1]])


def boot_ci(y, s, n=2000, seed=0):
    rng = np.random.default_rng(seed); y = np.asarray(y); s = np.asarray(s); out = []
    for _ in range(n):
        b = rng.choice(len(y), len(y), replace=True)
        if y[b].min() != y[b].max():
            out.append(roc_auc_score(y[b], s[b]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def group_rows(arm, jiang, score, thr, config):
    rows = []
    hcc = jiang["y_true"].to_numpy() == 1
    for g in GROUPS:
        ctrl = (jiang["y_true"].to_numpy() == 0) if g == "all_non_hcc" else ((jiang["y_true"].to_numpy() == 0) & (jiang["control_group"].to_numpy() == g))
        m = hcc | ctrl
        auc = roc_auc_score(jiang["y_true"][m], score[m]); lo, hi = boot_ci(jiang["y_true"][m], score[m])
        rows.append(dict(arm=arm, comparison=g, n_hcc=int(hcc.sum()), n_ctrl=int(ctrl.sum()), auroc=auc, ci_lo=lo, ci_hi=hi,
                         sensitivity_at_cris_thr=float((score[hcc] >= thr).mean()) if np.isfinite(thr) else np.nan,
                         specificity_at_cris_thr=float((score[ctrl] < thr).mean()) if np.isfinite(thr) else np.nan,
                         threshold=thr, config=config))
    return rows


def _json_safe(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, dict):
        return {k: _json_safe(x) for k, x in v.items()}
    return v


def load_run_arm(arm: str, jiang: pd.DataFrame):
    """(scores aligned to `jiang`, Cristiano threshold, config JSON) of an official
    deployment's best feature type; None when the run is absent."""
    d = variant_dir(f"nested_cv_results_cris/external_validation_comprehensive_external_cris_on_jiang_{RUN_TAGS[arm]}_rev")
    summ = rp(d, "comprehensive_validation_summary.pkl")
    if not os.path.isfile(summ):
        print(f"  {arm:9s} run not on disk ({d}) - skipped (slurm/submit_jiang_arms.sh)")
        return None
    with open(summ, "rb") as fh:
        fr = {k: v for k, v in pickle.load(fh)["feature_results"].items() if v is not None}
    best_feat = max(fr, key=lambda k: fr[k].get("training_cv_auc", 0))
    detail = {}
    for p in sorted(glob(rp(d, "external_validation_*_results.pkl"))):
        with open(p, "rb") as fh:
            r = pickle.load(fh)
        detail[r.get("best_model_info", {}).get("feature", os.path.basename(p))] = r
    r = detail[best_feat]
    info = r["best_model_info"]
    s = pd.Series(np.asarray(r["y_pred_proba"], dtype=float), index=[str(x) for x in r["sample_ids"]])
    assert set(s.index) == set(jiang["sample_id"].astype(str)), f"{arm}: Jiang sample set differs from the scalars table"
    y_run = pd.Series(np.asarray(r["y_true"], dtype=int), index=s.index).reindex(jiang["sample_id"].astype(str))
    assert (y_run.to_numpy() == jiang["y_true"].to_numpy()).all(), f"{arm}: labels differ from the scalars table"
    score = s.reindex(jiang["sample_id"].astype(str)).to_numpy()
    assert np.isclose(roc_auc_score(jiang["y_true"], score), r["overall_auc"], atol=1e-6) or bool(r.get("proba_degenerate")), \
        f"{arm}: recomputed AUROC differs from the stored one"
    fte = (r.get("fixed_threshold_evaluation") or {}).get("95_percent_specificity") or {}
    thr = fte.get("threshold", (r.get("deployment_thresholds") or {}).get("95_percent_specificity", np.nan))
    thr = float(thr) if thr is not None else np.nan
    cfg = json.dumps(_json_safe({"classifier": info.get("classifier"), "feature": info.get("feature"), "n_genes": info.get("n_genes"),
                                 **(info.get("best_params") or {}), "train_cv_auc": round(float(info.get("cv_auc", np.nan)), 4),
                                 "proba_degenerate": bool(r.get("proba_degenerate", False)),   # constant scores on Jiang
                                 "source_dir": d}))
    print(f"  {arm:9s} {best_feat} k={info.get('n_genes')} {info.get('classifier')}: training CV AUROC {info.get('cv_auc', np.nan):.3f}, "
          f"Jiang AUROC {roc_auc_score(jiang['y_true'], score):.3f}, Cristiano threshold {thr:.4f}")
    return score, thr, cfg


def quickfit_arms(args, cris, jiang, y_c, y_j, scores, summary):
    z = np.load(args.gd_cache, allow_pickle=True); ids = z["ids"]; n_cris = int(z["n_cris"]); G = z["gd"].astype(float)
    assert list(ids[:n_cris]) == cris["sample_id"].tolist() and list(ids[n_cris:]) == jiang["sample_id"].tolist()
    # ---- TSS preprocessing defined on Cristiano, applied to Jiang -------------------------
    remove = np.load(rp("accessory_files/final_tss_indices_to_remove_hg38.npy")); keep = np.setdiff1d(np.arange(G.shape[1]), remove)
    X = G[:, keep]
    mu = np.nanmean(X, axis=1, keepdims=True); sd = np.nanstd(X, axis=1, keepdims=True); X = np.nan_to_num((X - mu) / sd)
    Xc, Xj = X[:n_cris], X[n_cris:]
    _, uniq = np.unique(Xc, axis=1, return_index=True); uniq = np.sort(uniq); Xc, Xj = Xc[:, uniq], Xj[:, uniq]
    nan_j = int(np.isnan(G[n_cris:][:, keep]).any(axis=1).sum())
    print(f"TSS columns after blacklist {len(keep)}, after duplicate removal {Xc.shape[1]}; Jiang samples with NaN windows (set to 0 after z-score): {nan_j}")
    Cc = cris[COV5].to_numpy(float).copy(); Cj = jiang[COV5].to_numpy(float).copy()
    for j in range(len(COV5)):
        med = np.nanmedian(Cc[:, j]); Cc[np.isnan(Cc[:, j]), j] = med; Cj[np.isnan(Cj[:, j]), j] = med

    def fit_eval(arm, Xtr, Xte, grid_items, config_note=""):
        best = None
        for name, (clf, grid) in grid_items:
            pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
            gs = GridSearchCV(pipe, grid, cv=CV, scoring="roc_auc", n_jobs=args.cores).fit(Xtr, y_c)
            if best is None or gs.best_score_ > best[1]:
                best = (name, gs.best_score_, gs.best_estimator_, gs.best_params_)
        name, inner, est, params = best
        oof = cross_val_predict(est, Xtr, y_c, cv=CV, method="predict_proba")[:, 1]
        thr = spec95_threshold(y_c, oof)
        s = est.predict_proba(Xte)[:, 1]
        cfg = json.dumps({"classifier": name, **{k.replace("clf__", ""): _json_safe(v) for k, v in params.items()}, "inner_cv_auc": round(float(inner), 4), "cris_cv_auc": round(float(roc_auc_score(y_c, oof)), 4)})
        print(f"  {arm:10s} {name}: inner-CV AUROC {inner:.3f}, Cristiano 3-fold CV AUROC {roc_auc_score(y_c, oof):.3f}, Jiang AUROC {roc_auc_score(y_j, s):.3f}  [{cfg}]")
        scores[arm] = s; summary.extend(group_rows(arm, jiang, s, thr, cfg + config_note))

    lr = [("Logistic Regression", (LogisticRegression(max_iter=10000, random_state=42), LR_GRID))]
    print("\nquick refit on all Cristiano samples, applied to Jiang:")
    fit_eval("tss", Xc, Xj, lr)
    fit_eval("tss_cov5", np.hstack([Xc, Cc]), np.hstack([Xj, Cj]), lr, " +5 covariates")
    fit_eval("cov5", Cc, Cj, list(PANEL_GRIDS.items()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from_runs", action="store_true", help="read the official best-model-search deployments (slurm/submit_jiang_arms.sh)")
    ap.add_argument("--gd_cache", default=None, help="quick refit: .npz with ids, n_cris, gd (Cristiano PANCAN rows, then the 225 Jiang rows)")
    ap.add_argument("--cores", type=int, default=4)
    args = ap.parse_args()
    if args.from_runs == bool(args.gd_cache):
        ap.error("give exactly one of --from_runs and --gd_cache")
    suffix = "" if args.from_runs else "_quickfit"
    cris = pd.read_csv(os.path.join(DATA_DIR, "fig4_baseline_scalars.csv")); jiang = pd.read_csv(rp("src/figures/data/figS10_jiang_scalars.csv"))
    y_c = cris["y_true"].to_numpy(int); y_j = jiang["y_true"].to_numpy(int)
    print(f"Cristiano {COHORT_VARIANT} PANCAN: {len(cris)} ({y_c.sum()} cancers); Jiang: {len(jiang)} ({y_j.sum()} HCC; {jiang['control_group'].value_counts().to_dict()})")
    scores, summary = {}, []

    if args.from_runs:
        print("\nofficial deployments (best-model search on Cristiano -> comprehensive external validation on Jiang):")
        for arm in RUN_TAGS:
            got = load_run_arm(arm, jiang)
            if got is None:
                continue
            s, thr, cfg = got
            scores[arm] = s; summary.extend(group_rows(arm, jiang, s, thr, cfg))
        if not scores:
            raise SystemExit("no deployment run found - nothing to export")
    else:
        quickfit_arms(args, cris, jiang, y_c, y_j, scores, summary)

    # ---- each covariate alone, oriented and thresholded on Cristiano ----------------------
    for c in COV5:
        assert c in cris.columns and c in jiang.columns, f"{c} missing from the scalar tables - rerun export_fig3_baselines.py / export_figS10_jiang.py --scalars_only"
    Cc = cris[COV5].to_numpy(float).copy(); Cj = jiang[COV5].to_numpy(float).copy()
    for j, c in enumerate(COV5):
        med = np.nanmedian(Cc[:, j]); Cc[np.isnan(Cc[:, j]), j] = med; Cj[np.isnan(Cj[:, j]), j] = med
        orient = 1.0 if roc_auc_score(y_c, Cc[:, j]) >= 0.5 else -1.0
        s = orient * Cj[:, j]; thr = spec95_threshold(y_c, orient * Cc[:, j])
        scores[f"bl_{c}"] = s; summary.extend(group_rows(f"bl_{c}", jiang, s, thr, json.dumps({"orientation": orient})))
        print(f"  bl_{c:22s} orientation {orient:+.0f}: Jiang AUROC {roc_auc_score(y_j, s):.3f}")

    # ---- ichorCNA tumour fraction alone (higher = cancer; threshold at 95% specificity on Cristiano)
    tf = {}
    for path in ICHOR_TSV.values():
        ich = pd.read_csv(rp(path), sep="\t")
        tf.update(zip(ich["sample"], ich["tumor_fraction"]))
    tf_c, tf_j = cris["sample_id"].map(tf).to_numpy(float), jiang["sample_id"].map(tf).to_numpy(float)
    assert not np.isnan(tf_c).any() and not np.isnan(tf_j).any(), "ichorCNA TF missing for a sample"
    thr = spec95_threshold(y_c, tf_c)
    scores["bl_ichorcna_tf"] = tf_j
    summary.extend(group_rows("bl_ichorcna_tf", jiang, tf_j, thr, json.dumps({"orientation": 1.0, "threshold_cristiano": thr})))
    print(f"  bl_ichorcna_tf           threshold {thr:.4f} (Cristiano 95% spec): Jiang AUROC {roc_auc_score(y_j, tf_j):.3f}")

    out_scores = pd.DataFrame({"sample_id": jiang["sample_id"], "y_true": y_j, "control_group": jiang["control_group"], **scores})
    out_scores.to_csv(os.path.join(DATA_DIR, f"figS16_jiang_scores{suffix}.csv"), index=False)
    pd.DataFrame(summary).to_csv(os.path.join(DATA_DIR, f"figS16_jiang_summary{suffix}.csv"), index=False)
    print(f"\nwrote {DATA_DIR}/figS16_jiang_scores{suffix}.csv and figS16_jiang_summary{suffix}.csv")
    S = pd.DataFrame(summary)
    print(S.pivot(index="arm", columns="comparison", values="auroc").round(3).to_string())

    # ---- figure: AUROC per arm and comparison group (CI bars) + pooled ROC vs all non-HCC ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_dir = rp("src/figures/output", COHORT_VARIANT) if IS_VARIANT else rp("src/figures/output"); os.makedirs(out_dir, exist_ok=True)
    model_arms = [a for a in ["tss", "tss_cov5", "cov5", "rna", "atac"] if a in scores]
    arms = model_arms + [f"bl_{c}" for c in COV5] + ["bl_ichorcna_tf"]

    def is_degenerate(arm):
        """Constant scores on Jiang (the stored run's proba_degenerate flag): the AUROC is the ordering of numerical noise."""
        try:
            return bool(json.loads(S[S["arm"] == arm]["config"].iloc[0]).get("proba_degenerate", False))
        except Exception:
            return False

    degenerate = [a for a in model_arms if is_degenerate(a)]
    n_ctrl = {g: int(S[(S["arm"] == model_arms[0]) & (S["comparison"] == g)]["n_ctrl"].iloc[0]) for g in GROUPS}
    n_hcc = int(S["n_hcc"].iloc[0])
    group_label = {"all_non_hcc": f"HCC vs all {n_ctrl['all_non_hcc']} non-HCC", "Healthy": f"vs {n_ctrl['Healthy']} healthy",
                   "Chronic hepatitis B": f"vs {n_ctrl['Chronic hepatitis B']} chronic hepatitis B", "HBV-associated cirrhosis": f"vs {n_ctrl['HBV-associated cirrhosis']} cirrhosis"}

    def ttl(ax, letter, text):
        ax.set_title(f"$\\bf{{{letter}}}$   {text}", loc="left", fontsize=11, pad=8)

    fig, axes = plt.subplots(1, 2, figsize=(17, 5.8), gridspec_kw={"width_ratios": [1.7, 1]})
    ax = axes[0]; x = np.arange(len(GROUPS)); w = 0.8 / len(arms)
    for i, arm in enumerate(arms):
        r = S[S["arm"] == arm].set_index("comparison").loc[GROUPS]
        deg = arm in degenerate
        ax.bar(x + (i - len(arms) / 2 + 0.5) * w, r["auroc"], w, yerr=None if deg else [r["auroc"] - r["ci_lo"], r["ci_hi"] - r["auroc"]],
               color="white" if deg else COLORS[arm], edgecolor=COLORS[arm] if deg else "none", hatch="////" if deg else None,
               capsize=2, label=LABELS[arm], alpha=0.9 if not arm.startswith("bl_") else 0.5)
    ax.axhline(0.5, color="0.3", linestyle=":", linewidth=1); ax.set_ylim(0.2, 1.0); ax.set_xticks(x)
    ax.set_xticklabels([group_label[g] for g in GROUPS], fontsize=9.5)
    ax.set_ylabel(f"AUROC on the Jiang cohort ({n_hcc} HCC; bars = 95% bootstrap interval)"); ax.grid(axis="y", alpha=0.25); ax.set_axisbelow(True)
    ax.legend(fontsize=8, frameon=False, ncol=2, loc="upper right")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ttl(ax, "a", "Models trained on the Cristiano cohort, applied once to Jiang with a fixed preprocessing and threshold")
    ax = axes[1]
    for arm in model_arms:
        if arm in degenerate:
            continue
        fpr, tpr, _ = roc_curve(y_j, scores[arm]); ax.plot(fpr, tpr, color=COLORS[arm], linewidth=2, label=f"{LABELS[arm]}: {roc_auc_score(y_j, scores[arm]):.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3); ax.set_xlabel(f"false positive rate ({n_ctrl['all_non_hcc']} non-HCC)"); ax.set_ylabel(f"true positive rate ({n_hcc} HCC)")
    ax.legend(fontsize=8.5, frameon=False, loc="lower right")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ttl(ax, "b", "ROC on Jiang, HCC vs all non-HCC")
    if degenerate:
        fig.text(0.01, -0.03, "Hatched, " + ", ".join(LABELS[a] for a in degenerate) + ": the stored deployment returns one score for every "
                 "Jiang sample; its AUROC is the ordering of numerical noise and is not reported.", fontsize=8.5, color="0.3")
    plt.tight_layout(); fig.savefig(os.path.join(out_dir, f"figS16_jiang_arms{suffix}.png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, f"figS16_jiang_arms{suffix}.pdf"), bbox_inches="tight"); plt.close(fig)
    print("wrote", os.path.join(out_dir, f"figS16_jiang_arms{suffix}.png"), "and .pdf" + (f"; degenerate arms: {degenerate}" if degenerate else ""))


if __name__ == "__main__":
    main()
