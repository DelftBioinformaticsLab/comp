"""Source data for the per-TSS covariate regression (companion of Fig S10; notebooks
src/figures/figS_tss_regression.ipynb and figS_covariates.ipynb).

One OLS per TSS, feature value ~ 1 + class + covariates, on the features as the classifier sees
them (TSS filter, per-sample z-score, duplicate columns dropped), for both tasks; a diagnostic of
covariate influence, not a ranking for the model. Adjustment sets (ADJ_SETS): 'confounders' (log10
coverage, plasma volume, age, sex), 'all' (plus fragment length, mean GC, log10 cfDNA
concentration; these carry tumour signal, so the class effect under 'all' is a lower bound) and two
splits of the confounders. Scopes: pooled, healthy_only, cancer_only. Covariates are median-imputed
and z-scored; the reference profile is the mean z-scored TSS vector of the 50 panel-of-normals samples.

Outputs (data[/no_deep]/):
  figS10_tss_regression_samples.csv  per PANCAN sample: class, task membership, profile correlation
  figS10_tss_regression_summary.csv  per (task, feature, sample_set, scope, model, term): TSSs at
                                     BH q < 0.05, median |t|, VIF, agreement statistics
  figS10_tss_regression_r2.csv       adjusted and partial R2, quantiles over TSSs
  figS10_tss_regression_tss.csv      per TSS (griffin_diff, pooled): class beta/t/q, covariate t, R2

Inputs: fig3_/fig4_baseline_scalars.csv, extracted_features/biomart_10kb/ and the TSS filter.
Under the default COHORT_VARIANT=no_deep only griffin_diff is analysed; COHORT_VARIANT=all runs
all five features and adds the sample set 'excl_deep'.

Run from the repository root (after export_fig3_baselines.py for both tasks):
    python src/figures/export/export_figS10_tss_regression.py [--cores 8]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from _common import DATA_DIR, rp  # noqa: E402
from _common import IS_VARIANT, COHORT_VARIANT  # noqa: E402

BIOMART_DIR = "extracted_features/biomart_10kb"
MANIFEST_CSV = "accessory_files/cris_manifest.csv"
TSS_FILTER = "accessory_files/final_tss_indices_to_remove_hg38.npy"
PRIMARY = "griffin_diff"
FEATURES = [PRIMARY] if IS_VARIANT else ["griffin_diff", "rcov", "mwh", "fslr", "fslr_coverage"]   # DATA_PARAM_GRIDS['feature']
N_UNIQUE_PRIMARY = 14221   # columns the fully nested runs report for griffin_diff (both tasks)
TASKS = {"brca_crc": "fig3_baseline_scalars.csv", "pancan": "fig4_baseline_scalars.csv"}
DEEP_COVERAGE_X = 6.0

# covariate -> (source column, log10 transform)
COVARIATES = {
    "mean_coverage": ("mean_coverage", True),
    "plasma_volume": ("plasma_volume", False),
    "age": ("age", False),
    "sex_female": ("sex_female", False),
    "fragment_length_mean": ("fragment_length_mean", False),
    "mean_gc": ("mean_gc", False),
    "cfdna_extracted": ("cfdna_extracted", True),
}
ADJ_SETS = {
    "confounders": ["mean_coverage", "plasma_volume", "age", "sex_female"],
    "all": ["mean_coverage", "plasma_volume", "age", "sex_female",
            "fragment_length_mean", "mean_gc", "cfdna_extracted"],
    # the confounders split: which of them moves the class effect
    "coverage_only": ["mean_coverage"],
    "no_coverage": ["plasma_volume", "age", "sex_female"],
}
TOP_KS = [100, 500, 1000]


# --------------------------------------------------------------------------
# features, as the classifier sees them
# --------------------------------------------------------------------------

def _read_one(sid: str):
    d = pd.read_csv(rp(BIOMART_DIR, f"{sid}.hg38.frag.tsv_BIN.csv"), usecols=["gene_name"] + FEATURES)
    return d["gene_name"].to_numpy(), {f: d[f].to_numpy(dtype=float) for f in FEATURES}


def load_features(ids: list[str], cores: int):
    """TSS filter, then the pipeline's per-sample row z-score
    (normalize_feature_arrays). Returns (gene names, original window indices,
    {feature: n x n_kept matrix}, {feature: n x 2 raw per-sample mean and SD over TSSs,
    before the z-score})."""
    remove = np.load(rp(TSS_FILTER))
    rows = Parallel(n_jobs=cores)(delayed(_read_one)(s) for s in ids)
    genes = None
    out = {f: [] for f in FEATURES}
    raw = {f: [] for f in FEATURES}
    for sid, (g, feats) in zip(ids, rows):
        if genes is None:
            genes, kept = np.delete(g, remove), np.delete(np.arange(len(g)), remove)
        else:
            assert np.array_equal(np.delete(g, remove), genes), f"{sid}: TSS order differs"
        for f in FEATURES:
            x = np.delete(feats[f], remove)
            mu, sd = np.nanmean(x), np.nanstd(x)
            out[f].append(np.nan_to_num((x - mu) / (sd if sd > 0 else 1.0)))
            raw[f].append((mu, sd))
    return genes, kept, {f: np.vstack(v) for f, v in out.items()}, {f: np.array(v) for f, v in raw.items()}


def unique_columns(X: np.ndarray) -> np.ndarray:
    """Column positions model_hpc keeps: the first occurrence of every distinct
    column, in the original order."""
    _, idx = np.unique(X, axis=1, return_index=True)
    return np.sort(idx)


# --------------------------------------------------------------------------
# one OLS per TSS, shared design matrix
# --------------------------------------------------------------------------

def bh(p: np.ndarray) -> np.ndarray:
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if ok.any():
        q[ok] = stats.false_discovery_control(p[ok], method="bh")
    return q


def ols(X: np.ndarray, Y: np.ndarray) -> dict:
    """Y (n x m) on X (n x p, intercept included): coefficients, t, two-sided
    p, residual sum of squares. Constant TSS columns give NaN statistics."""
    n, p = X.shape
    xtx_inv = np.linalg.inv(X.T @ X)
    B = xtx_inv @ X.T @ Y
    rss = ((Y - X @ B) ** 2).sum(axis=0)
    dof = n - p
    sigma2 = rss / dof
    with np.errstate(divide="ignore", invalid="ignore"):
        t = B / np.sqrt(np.outer(np.diag(xtx_inv), sigma2))
    t[:, sigma2 <= 1e-20] = np.nan
    return {"beta": B, "t": t, "p": 2 * stats.t.sf(np.abs(t), dof), "rss": rss, "dof": dof}


def adj_r2(rss: np.ndarray, tss: np.ndarray, n: int, p: int) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1.0 - (rss / (n - p)) / (tss / (n - 1))


def vif(Z: np.ndarray) -> np.ndarray:
    """Variance inflation factor of every column of Z (no intercept column)."""
    if Z.shape[1] < 2:
        return np.ones(Z.shape[1])
    out = []
    for j in range(Z.shape[1]):
        others = np.column_stack([np.ones(len(Z)), np.delete(Z, j, axis=1)])
        resid = Z[:, j] - others @ np.linalg.lstsq(others, Z[:, j], rcond=None)[0]
        out.append(Z[:, j].var() / max(resid.var(), 1e-300))
    return np.array(out)


def design_covariates(sc: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, dict]:
    """Transformed, median-imputed, z-scored covariate block and the number of
    imputed values per covariate."""
    Z, n_imp = [], {}
    for c in cols:
        src, log = COVARIATES[c]
        v = sc[src].to_numpy(dtype=float)
        assert np.nanmin(v) > 0 or not log, f"{c}: non-positive values under log10"
        v = np.log10(v) if log else v.copy()
        miss = np.isnan(v)
        n_imp[c] = int(miss.sum())
        v[miss] = np.nanmedian(v)
        assert v.std() > 0, f"{c}: constant in this sample set"
        Z.append((v - v.mean()) / v.std())
    return np.column_stack(Z), n_imp


def top_k_jaccard(a: np.ndarray, b: np.ndarray, k: int) -> float:
    ta, tb = set(np.argsort(-np.abs(a))[:k]), set(np.argsort(-np.abs(b))[:k])
    return len(ta & tb) / len(ta | tb)


def analyse(task: str, feature: str, sample_set: str, sc: pd.DataFrame, Y: np.ndarray, ref: np.ndarray):
    """All models for one (task, feature, sample set). `ref` is the
    panel-of-normals mean profile on the same TSS columns. Returns summary
    rows, r2 rows and a dict of per-TSS columns."""
    n, m = Y.shape
    y = sc["y_true"].to_numpy(dtype=float)
    ones = np.ones(n)
    tss = ((Y - Y.mean(axis=0)) ** 2).sum(axis=0)
    ok = np.isfinite(tss) & (tss > 1e-20)
    base = {"task": task, "feature": feature, "sample_set": sample_set, "n_tss": int(ok.sum())}
    summary, r2rows, per_tss = [], [], {}

    def term_row(scope, model, term, t, q, n_samples, vif_val=np.nan, n_imputed=0, **extra):
        fin = np.isfinite(t)
        extra["spearman_t_vs_pon_profile"] = float(stats.spearmanr(t[fin], ref[fin])[0])
        return {**base, "scope": scope, "model": model, "term": term, "n_samples": n_samples,
                "n_sig_q05": int(np.nansum(q < 0.05)), "frac_sig_q05": float(np.nansum(q < 0.05) / ok.sum()),
                "median_abs_t": float(np.nanmedian(np.abs(t))), "vif": float(vif_val),
                "n_imputed": int(n_imputed), **extra}

    def quantiles(scope, model, quantity, v):
        v = v[ok]
        return {**base, "scope": scope, "model": model, "quantity": quantity,
                "mean": float(np.mean(v)), "q10": float(np.quantile(v, 0.10)), "q25": float(np.quantile(v, 0.25)),
                "median": float(np.median(v)), "q75": float(np.quantile(v, 0.75)), "q90": float(np.quantile(v, 0.90))}

    # ---- pooled, unadjusted: value ~ class
    f0 = ols(np.column_stack([ones, y]), Y)
    t0, q0 = f0["t"][1], bh(f0["p"][1])
    summary.append(term_row("pooled", "unadjusted", "class", t0, q0, n))
    r2_class = adj_r2(f0["rss"], tss, n, 2)
    r2rows.append(quantiles("pooled", "unadjusted", "adj_r2_class_only", r2_class))
    per_tss.update({"beta_class_unadj": f0["beta"][1], "t_class_unadj": t0, "q_class_unadj": q0,
                    "adj_r2_class_only": r2_class})

    for adj, cols in ADJ_SETS.items():
        Z, n_imp = design_covariates(sc, cols)
        # ---- pooled, covariates only and class + covariates
        fc = ols(np.column_stack([ones, Z]), Y)
        f1 = ols(np.column_stack([ones, y, Z]), Y)
        v = vif(np.column_stack([y, Z]))
        t1, q1 = f1["t"][1], bh(f1["p"][1])
        both = np.isfinite(t0) & np.isfinite(t1)
        rho = stats.spearmanr(t0[both], t1[both])[0]
        slope = float((t0[both] * t1[both]).sum() / (t0[both] ** 2).sum())
        s0, s1 = q0 < 0.05, q1 < 0.05
        summary.append(term_row(
            "pooled", adj, "class", t1, q1, n, v[0],
            spearman_t_vs_unadjusted=float(rho), slope_t_vs_unadjusted=slope,
            n_sig_lost=int(np.sum(s0 & ~s1)), n_sig_gained=int(np.sum(~s0 & s1)),
            n_sign_flips_among_sig=int(np.sum(s0 & s1 & (np.sign(t0) != np.sign(t1)))),
            **{f"jaccard_top{k}": top_k_jaccard(t0[both], t1[both], k) for k in TOP_KS}))
        for j, c in enumerate(cols):
            tj, qj = f1["t"][2 + j], bh(f1["p"][2 + j])
            okj = np.isfinite(tj) & np.isfinite(t0)
            summary.append(term_row("pooled", adj, c, tj, qj, n, v[1 + j], n_imp[c],
                                    spearman_t_vs_class_unadjusted=float(stats.spearmanr(tj[okj], t0[okj])[0])))
            per_tss[f"t_{c}_{adj}"] = tj
        p_cov, p_full = 1 + Z.shape[1], 2 + Z.shape[1]
        r2_cov, r2_full = adj_r2(fc["rss"], tss, n, p_cov), adj_r2(f1["rss"], tss, n, p_full)
        with np.errstate(divide="ignore", invalid="ignore"):
            pr2_class = (fc["rss"] - f1["rss"]) / fc["rss"]          # class | covariates
            pr2_cov = (f0["rss"] - f1["rss"]) / f0["rss"]            # covariates | class
        for name, val in [("adj_r2_covariates_only", r2_cov), ("adj_r2_class_plus_covariates", r2_full),
                          ("partial_r2_class_given_covariates", pr2_class),
                          ("partial_r2_covariates_given_class", pr2_cov)]:
            r2rows.append(quantiles("pooled", adj, name, val))
        per_tss.update({f"beta_class_{adj}": f1["beta"][1], f"t_class_{adj}": t1, f"q_class_{adj}": q1,
                        f"adj_r2_covariates_only_{adj}": r2_cov, f"adj_r2_full_{adj}": r2_full,
                        f"partial_r2_class_{adj}": pr2_class, f"partial_r2_covariates_{adj}": pr2_cov})

        # ---- within one class: value ~ covariates (no class contrast in the data)
        for scope, cls in [("healthy_only", 0), ("cancer_only", 1)]:
            sel = (y == cls)
            Zs, n_imp_s = design_covariates(sc[sel], cols)
            Ys = Y[sel]
            fs = ols(np.column_stack([np.ones(sel.sum()), Zs]), Ys)
            vs = vif(Zs)
            tss_s = ((Ys - Ys.mean(axis=0)) ** 2).sum(axis=0)
            for j, c in enumerate(cols):
                tj, qj = fs["t"][1 + j], bh(fs["p"][1 + j])
                okj = np.isfinite(tj) & np.isfinite(t0)
                summary.append(term_row(scope, adj, c, tj, qj, int(sel.sum()), vs[j], n_imp_s[c],
                                        spearman_t_vs_class_unadjusted=float(stats.spearmanr(tj[okj], t0[okj])[0])))
                if scope == "healthy_only":
                    per_tss[f"t_{c}_{adj}_healthy_only"] = tj
            r2rows.append(quantiles(scope, adj, "adj_r2_covariates_only",
                                    adj_r2(fs["rss"], tss_s, int(sel.sum()), 1 + Zs.shape[1])))
    return summary, r2rows, per_tss


def main() -> None:
    if IS_VARIANT and COHORT_VARIANT != "no_deep":
        print(f"export_figS10_tss_regression.py: not applicable under COHORT_VARIANT={COHORT_VARIANT} - nothing written")
        return
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cores", type=int, default=8)
    ap.add_argument("--cache", default=None,
                    help="Optional .npz path to cache the loaded feature matrices (reused when present)")
    args = ap.parse_args()

    tables = {task: pd.read_csv(os.path.join(DATA_DIR, csv)) for task, csv in TASKS.items()}
    pan = tables["pancan"]
    assert (len(pan), len(tables["brca_crc"])) == ((367, 244) if IS_VARIANT else (397, 274))
    assert set(tables["brca_crc"]["sample_id"]) <= set(pan["sample_id"])
    ids = pan["sample_id"].tolist()

    if args.cache and os.path.exists(args.cache):
        z = np.load(args.cache, allow_pickle=True)
        assert list(z["ids"]) == ids, "cache holds a different sample list"
        genes, kept, X = z["genes"], z["kept"], {f: z[f] for f in FEATURES}
        RAW = None   # the cache holds the z-scored matrices only
        print(f"features: reused cache {args.cache}")
    else:
        print(f"loading {len(FEATURES)} features for {len(ids)} samples ...")
        genes, kept, X, RAW = load_features(ids, args.cores)
        if args.cache:
            np.savez(args.cache, ids=np.array(ids), genes=genes, kept=kept, **{f: X[f] for f in FEATURES})  # float64: duplicate-column detection is precision-sensitive
    print(f"  {X[PRIMARY].shape[0]} samples x {X[PRIMARY].shape[1]} TSS windows after the TSS filter")
    row_of = {s: i for i, s in enumerate(ids)}

    # reference profile: mean z-scored TSS vector of the panel of normals
    mf = pd.read_csv(rp(MANIFEST_CSV))
    pon_ids = mf.loc[mf["role"] == "panel_of_normals", "ID"].tolist()
    assert len(pon_ids) == 50 and not set(pon_ids) & set(ids), "panel of normals must be 50 samples outside the tasks"
    pon_cache = args.cache + ".pon.npz" if args.cache else None
    if pon_cache and os.path.exists(pon_cache):
        zp = np.load(pon_cache, allow_pickle=True)
        assert list(zp["ids"]) == pon_ids and np.array_equal(zp["kept"], kept)
        P = {f: zp[f] for f in FEATURES}
    else:
        print(f"loading the {len(pon_ids)} panel-of-normals samples ...")
        genes_p, kept_p, P, _ = load_features(pon_ids, args.cores)
        assert np.array_equal(kept_p, kept) and np.array_equal(genes_p, genes)
        if pon_cache:
            np.savez(pon_cache, ids=np.array(pon_ids), kept=kept, **{f: P[f] for f in FEATURES})
    pon_mean = {f: P[f].astype(float).mean(axis=0) for f in FEATURES}

    samples = pan[["sample_id", "sample_name", "y_true", "disease"]].copy()
    samples["in_task_brca_crc"] = samples["sample_id"].isin(set(tables["brca_crc"]["sample_id"]))
    samples["deep_subgroup"] = (pan["mean_coverage"] > DEEP_COVERAGE_X).to_numpy()
    for f in FEATURES:
        Xf, ref_f = X[f].astype(float), pon_mean[f]
        rc = ref_f - ref_f.mean()
        Xc = Xf - Xf.mean(axis=1, keepdims=True)
        samples[f"profile_r_{f}"] = (Xc @ rc) / (np.sqrt((Xc ** 2).sum(axis=1)) * np.sqrt((rc ** 2).sum()))
        if RAW is not None:
            samples[f"raw_mean_{f}"], samples[f"raw_sd_{f}"] = RAW[f][:, 0], RAW[f][:, 1]
    out = os.path.join(DATA_DIR, "figS10_tss_regression_samples.csv")
    samples.to_csv(out, index=False)
    print(f"wrote {out} ({len(samples)} rows)")

    summary, r2rows, tss_frames = [], [], []
    for task, sc_all in tables.items():
        deep = (sc_all["mean_coverage"] > DEEP_COVERAGE_X).to_numpy()
        assert deep.sum() == (0 if IS_VARIANT else 30) and (sc_all.loc[deep, "y_true"] == 0).all()
        rows_all = np.array([row_of[s] for s in sc_all["sample_id"]])
        for feature in FEATURES:
            Xt = X[feature][rows_all].astype(float)
            cols = unique_columns(Xt)                       # as model_hpc: on the task's full sample set
            if feature == PRIMARY:
                assert len(cols) == N_UNIQUE_PRIMARY, (
                    f"{task} {feature}: {len(cols)} unique columns, the nested runs used {N_UNIQUE_PRIMARY}")
            per_tss_task = {}
            for sample_set, keep in [("all", np.ones(len(sc_all), bool))] + ([] if IS_VARIANT else [("excl_deep", ~deep)]):
                sc = sc_all[keep].reset_index(drop=True)
                s, r, per_tss = analyse(task, feature, sample_set, sc, Xt[keep][:, cols], pon_mean[feature][cols])
                summary += s
                r2rows += r
                if sample_set == "all":
                    per_tss_task.update(per_tss)
                else:
                    per_tss_task.update({f"{k}_excl_deep": v for k, v in per_tss.items()
                                         if k.startswith(("t_class", "q_class"))})
            cls_rows = [x for x in summary if x["task"] == task and x["feature"] == feature
                        and x["term"] == "class" and x["sample_set"] == "all"]
            print(f"  {task:8s} {feature:14s} {len(cols):>6} TSSs | class significant (q<0.05): "
                  + ", ".join(f"{x['model']} {x['n_sig_q05']}" for x in cls_rows))
            if feature == PRIMARY:
                df = pd.DataFrame(per_tss_task)
                df.insert(0, "pon_mean_profile", pon_mean[feature][cols])
                df.insert(0, "gene_name", genes[cols])
                df.insert(0, "tss_window_index", kept[cols])
                df.insert(0, "task", task)
                tss_frames.append(df)

    out = os.path.join(DATA_DIR, "figS10_tss_regression_summary.csv")
    pd.DataFrame(summary).to_csv(out, index=False)
    print(f"wrote {out} ({len(summary)} rows)")
    out = os.path.join(DATA_DIR, "figS10_tss_regression_r2.csv")
    pd.DataFrame(r2rows).to_csv(out, index=False)
    print(f"wrote {out} ({len(r2rows)} rows)")
    tss = pd.concat(tss_frames, ignore_index=True)
    num = tss.select_dtypes("number").columns.difference(["tss_window_index"])
    tss[num] = tss[num].astype(float).apply(lambda c: c.map(lambda v: float(f"{v:.5g}") if np.isfinite(v) else v))
    out = os.path.join(DATA_DIR, "figS10_tss_regression_tss.csv")
    tss.to_csv(out, index=False)
    print(f"wrote {out} ({len(tss)} rows, {tss.shape[1]} columns)")


if __name__ == "__main__":
    main()
