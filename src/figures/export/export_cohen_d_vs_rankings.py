"""Agreement between the interpretation ranking of the TSSs (|Cohen's d| on GD) and the four
ranking functions the nested models select TSSs with (model_hpc.compute_feature_weights:
|mean difference|, -log10 Mann-Whitney p, |log2 ratio of the means|, weighted product), per task
(Spearman over all TSSs, overlap of the top k), plus the function and k every outer fold selected.

Outputs (data/no_deep/):
  cohen_d_vs_rankings.csv          task, ranking, spearman_all, jaccard / overlap of the top k, chance
  cohen_d_vs_rankings_folds.csv    task, fold, selected ranking function and k, overlap of that fold's selected
                                   TSSs with the top k by |d| computed on the same training samples

Run from the repository root (about a minute):
    COHORT_VARIANT=no_deep python src/figures/export/export_cohen_d_vs_rankings.py
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

sys.path.insert(0, os.path.dirname(__file__))
from _common import COHORT_VARIANT, DATA_DIR, rp, variant_dir  # noqa: E402
from export_figS17 import TASKS  # noqa: E402
from export_figS17_cohen_d import cohen_d  # noqa: E402
from export_figS_loco import load  # noqa: E402

KS = [100, 1000, 5000, 10000]


def weights(X, y):
    """model_hpc.compute_feature_weights, all four functions at once."""
    mc, mh = X[y == 1].mean(axis=0), X[y == 0].mean(axis=0)
    out = {"mean_diff": np.nan_to_num(np.abs(mc - mh))}
    with np.errstate(divide="ignore", invalid="ignore"):
        fc = np.divide(mc, mh, out=np.full_like(mc, np.nan), where=mh != 0)
        lfc = np.log2(fc, out=np.full_like(fc, np.nan), where=fc > 1e-10)
    lfc = np.nan_to_num(lfc)
    p = mannwhitneyu(X[y == 1], X[y == 0], alternative="two-sided", axis=0).pvalue
    p = np.where(p > 0, p, 1e-10)
    out["p_value"] = -np.log10(p)
    out["fold_change"] = np.abs(lfc)
    out["weighted"] = np.nan_to_num(np.abs(lfc) * -np.log10(p))
    return out


def top(v, k):
    return set(np.argsort(v)[-k:].tolist())


def main() -> None:
    assert COHORT_VARIANT == "no_deep", "run with COHORT_VARIANT=no_deep"
    sc, X_all = load(cores=1)
    pos = {s: i for i, s in enumerate(sc["sample_id"])}
    disease = dict(zip(sc["sample_id"], sc["disease"]))
    rows, frows = [], []
    for task in TASKS:
        with open(rp(variant_dir(TASKS[task]), "ksweep_tuned", "nested_full_per_fold.pkl"), "rb") as fh:
            pf = pickle.load(fh).sort_values("fold")
        ids = sorted(set().union(*[set(r["train_sample_ids"]) | set(r["test_sample_ids"]) for _, r in pf.iterrows()]))
        X = X_all[[pos[s] for s in ids]]
        y = np.array([0 if disease[s] == "Healthy" else 1 for s in ids])
        n = X.shape[1]
        d = np.abs(cohen_d(X, y))
        W = weights(X, y)
        print(f"\n=== {task}: {len(ids)} samples, {n} TSSs")
        for name, w in W.items():
            rho = spearmanr(d, w).statistic
            r = dict(task=task, ranking=name, spearman_all=rho)
            for k in KS:
                a, b = top(d, k), top(w, k)
                r[f"overlap_top{k}"] = len(a & b) / k
                r[f"jaccard_top{k}"] = len(a & b) / len(a | b)
                r[f"chance_jaccard_top{k}"] = k / (2 * n - k)
            rows.append(r)
            print(f"  |d| vs {name:<12} Spearman {rho:.3f}   share of top 1,000 shared {r['overlap_top1000']:.2f}"
                  f"   top 10,000 {r['overlap_top10000']:.3f}")
        # what the folds selected, and how much of it the top k by |d| (same training samples) would also select
        idx_of = {s: i for i, s in enumerate(ids)}
        sel_col = next(c for c in ("selection_type", "ranking", "selection") if c in pf.columns)
        k_col = next(c for c in ("n_top_genes", "k", "n_features") if c in pf.columns)
        for _, fr in pf.iterrows():
            tr = [idx_of[s] for s in fr["train_sample_ids"]]
            Xt, yt = X[tr], y[tr]
            k = int(min(int(fr[k_col]), n))
            sel = str(fr[sel_col])
            dt = np.abs(cohen_d(Xt, yt))
            if sel in ("mean_diff", "p_value", "fold_change", "weighted"):
                chosen = set(int(i) for i in fr["selected_gene_indices"])      # the fold's own selection
                assert len(chosen) == k, (len(chosen), k)
                ov = len(chosen & top(dt, k)) / k
            else:
                ov = np.nan
            frows.append(dict(task=task, fold=int(fr["fold"]), ranking=sel, k=k, overlap_with_top_k_by_d=ov))
            print(f"  fold {int(fr['fold'])}: {sel:<12} k = {k:>6}   share also in the top k by |d| {ov:.3f}")
    pd.DataFrame(rows).to_csv(os.path.join(DATA_DIR, "cohen_d_vs_rankings.csv"), index=False)
    pd.DataFrame(frows).to_csv(os.path.join(DATA_DIR, "cohen_d_vs_rankings_folds.csv"), index=False)
    print(f"\nwritten to {DATA_DIR}")


if __name__ == "__main__":
    main()
