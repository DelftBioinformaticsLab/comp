"""Statistical comparison of the data-driven model with the baselines (ATAC, RNA, ichorCNA), no_deep cohort.

Per task (BRCA/CRC: fig3_predictions.csv; pan-cancer: fig4a_predictions.csv), on the same ten outer folds:
  fold level  per-fold AUROCs and the Nadeau-Bengio (2003) corrected resampled t-test (variance
              (1/k + n_test/n_train) * s^2, k - 1 df; includes variability between training sets)
  pooled      pooled out-of-fold AUROC with a within-class participant bootstrap 95% interval (2,000
              resamples) and the paired DeLong test (Sun and Xu 2014; participant sampling only)
  Holm        over the three baselines of a task, per test
For the Jiang et al. cohort (figS16_jiang_scores.csv, one deployed model per method) only the pooled
part applies, against ATAC and RNA.

Writes data/no_deep/stats_method_comparisons.csv (one row per task and baseline), read by fig3.ipynb.
Run from the repository root:  COHORT_VARIANT=no_deep python src/figures/export/export_method_comparisons.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from _common import COHORT_VARIANT, DATA_DIR  # noqa: E402

TASKS = {"brca_crc": ("BRCA/CRC vs healthy", "fig3_predictions.csv"),
         "pancan": ("Pan-cancer vs healthy", "fig4a_predictions.csv")}
BASELINES = {"atac": "ATAC", "rna": "RNA", "ichorcna": "ichorCNA"}
N_BOOT, SEED = 2000, 20261120


def _midrank(x):
    order = np.argsort(x, kind="mergesort")
    z = x[order]
    ranks = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i
        while j < len(x) and z[j] == z[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(len(x))
    out[order] = ranks
    return out


def delong(y, a, b):
    """Paired two-sided DeLong test (Sun and Xu 2014): AUROCs, difference a - b, its 95% CI, p."""
    y = np.asarray(y).astype(int)
    s = np.vstack([a, b]).astype(float)[:, np.argsort(-y, kind="mergesort")]
    m = int(y.sum())
    n = len(y) - m
    tx = np.array([_midrank(r[:m]) for r in s])
    ty = np.array([_midrank(r[m:]) for r in s])
    tz = np.array([_midrank(r) for r in s])
    auc = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    cov = np.cov((tz[:, :m] - tx) / n) / m + np.cov(1.0 - (tz[:, m:] - ty) / m) / n
    d = auc[0] - auc[1]
    se = np.sqrt(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    return auc[0], auc[1], d, d - 1.96 * se, d + 1.96 * se, 2 * stats.norm.sf(abs(d) / se)


def boot_auc(y, s, rng):
    """Participant-level bootstrap 95% interval of the AUROC, resampling within each class."""
    y, s = np.asarray(y).astype(int), np.asarray(s, float)
    pos, neg = s[y == 1], s[y == 0]
    vals = np.empty(N_BOOT)
    for i in range(N_BOOT):
        p, q = rng.choice(pos, len(pos)), rng.choice(neg, len(neg))
        vals[i] = roc_auc_score(np.r_[np.ones(len(p)), np.zeros(len(q))], np.r_[p, q])
    return np.percentile(vals, [2.5, 97.5])


def corrected_t(diff, ratio):
    """Nadeau-Bengio corrected resampled t-test on k paired fold differences; ratio = n_test / n_train."""
    k = len(diff)
    se = np.sqrt((1.0 / k + ratio) * np.var(diff, ddof=1))
    t = diff.mean() / se
    half = stats.t.ppf(0.975, k - 1) * se
    return t, diff.mean() - half, diff.mean() + half, 2 * stats.t.sf(abs(t), k - 1)


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty(len(p))
    running = 0.0
    for r, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - r) * p[i]))
        adj[i] = running
    return adj


def main() -> None:
    assert COHORT_VARIANT == "no_deep", "run with COHORT_VARIANT=no_deep"
    rng = np.random.default_rng(SEED)
    rows = []
    for task, (label, csv) in TASKS.items():
        d = pd.read_csv(os.path.join(DATA_DIR, csv))
        dd = d[(d["method"] == "data_driven") & d["fold"].notna()].drop_duplicates("sample_id").set_index("sample_id")
        y, fold = dd["y_true"].astype(int).to_numpy(), dd["fold"].astype(int).to_numpy()
        folds = np.unique(fold)
        assert len(folds) == 10
        n_test = np.array([(fold == f).sum() for f in folds])
        ratio = float(np.mean(n_test / (len(y) - n_test)))
        a = dd["score"].to_numpy()
        ci_dd = boot_auc(y, a, rng)
        fa = np.array([roc_auc_score(y[fold == f], a[fold == f]) for f in folds])
        task_rows = []
        for m, mlabel in BASELINES.items():
            o = d[d["method"] == m].drop_duplicates("sample_id").set_index("sample_id").reindex(dd.index)
            assert o["score"].notna().all(), (task, m)
            assert (o["y_true"].astype(int).to_numpy() == y).all(), (task, m)
            if "fold" in o and o["fold"].notna().all():
                assert (o["fold"].astype(int).to_numpy() == fold).all(), f"{task} {m}: different outer folds"
            b = o["score"].to_numpy()
            fb = np.array([roc_auc_score(y[fold == f], b[fold == f]) for f in folds])
            diff = fa - fb
            t, t_lo, t_hi, t_p = corrected_t(diff, ratio)
            auc_a, auc_b, dl, dl_lo, dl_hi, dl_p = delong(y, a, b)
            ci_b = boot_auc(y, b, rng)
            task_rows.append(dict(
                task=label, comparison=f"Data-driven vs {mlabel}", baseline=mlabel, n_cancer=int(y.sum()),
                n_noncancer=int((y == 0).sum()), n_folds=len(folds), test_train_ratio=ratio,
                fold_mean_dd=fa.mean(), fold_sd_dd=fa.std(ddof=1), fold_mean_base=fb.mean(), fold_sd_base=fb.std(ddof=1),
                fold_diff_mean=diff.mean(), fold_diff_sd=diff.std(ddof=1), folds_dd_higher=int((diff > 0).sum()),
                corrected_t=t, corrected_t_ci_lo=t_lo, corrected_t_ci_hi=t_hi, corrected_t_p=t_p,
                pooled_auc_dd=auc_a, pooled_auc_dd_lo=ci_dd[0], pooled_auc_dd_hi=ci_dd[1],
                pooled_auc_base=auc_b, pooled_auc_base_lo=ci_b[0], pooled_auc_base_hi=ci_b[1],
                delong_diff=dl, delong_ci_lo=dl_lo, delong_ci_hi=dl_hi, delong_p=dl_p))
        for test in ["corrected_t", "delong"]:
            adj = holm([r[f"{test}_p"] for r in task_rows])
            for r, p in zip(task_rows, adj):
                r[f"{test}_p_holm"] = p
        rows += task_rows

    j = pd.read_csv(os.path.join(DATA_DIR, "figS16_jiang_scores.csv"))
    y = j["y_true"].astype(int).to_numpy()
    ci_dd = boot_auc(y, j["tss"], rng)
    jrows = []
    for m, mlabel in [("atac", "ATAC"), ("rna", "RNA")]:
        auc_a, auc_b, dl, dl_lo, dl_hi, dl_p = delong(y, j["tss"].to_numpy(), j[m].to_numpy())
        ci_b = boot_auc(y, j[m], rng)
        jrows.append(dict(task="Jiang, HCC vs non-HCC (deployed models)", comparison=f"Data-driven vs {mlabel}", baseline=mlabel,
                          n_cancer=int(y.sum()), n_noncancer=int((y == 0).sum()),
                          pooled_auc_dd=auc_a, pooled_auc_dd_lo=ci_dd[0], pooled_auc_dd_hi=ci_dd[1],
                          pooled_auc_base=auc_b, pooled_auc_base_lo=ci_b[0], pooled_auc_base_hi=ci_b[1],
                          delong_diff=dl, delong_ci_lo=dl_lo, delong_ci_hi=dl_hi, delong_p=dl_p))
    for r, p in zip(jrows, holm([r["delong_p"] for r in jrows])):
        r["delong_p_holm"] = p
    out = pd.DataFrame(rows + jrows)
    out.to_csv(os.path.join(DATA_DIR, "stats_method_comparisons.csv"), index=False, lineterminator="\n")
    show = ["task", "baseline", "fold_mean_dd", "fold_mean_base", "fold_diff_mean", "folds_dd_higher", "corrected_t_ci_lo",
            "corrected_t_ci_hi", "corrected_t_p_holm", "pooled_auc_dd", "pooled_auc_base", "delong_diff", "delong_p_holm"]
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(out[show].round(4).to_string(index=False))
    print(f"\nwrote {os.path.join(DATA_DIR, 'stats_method_comparisons.csv')}")


if __name__ == "__main__":
    main()
