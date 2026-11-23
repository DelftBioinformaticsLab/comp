"""Stability of the ranked TSS list (Fig S17 comparisons) on Cohen's d, the statistic of the
gene-set analyses (Fig S18). Same folds, half splits and permutations as export_figS17.py;
the signed difference in means is recomputed alongside and asserted to reproduce figS17_summary.json.

Reads the cached GD matrix of export_figS_loco.py (data/no_deep/_cache_loco_gd.npz) and the
fold definitions of the nested runs. Writes to data/no_deep/:
  figS17_stability_cohen_d.csv   as figS17_stability.csv, plus 'statistic' (cohen_d | mean_diff)
  figS17_summary_cohen_d.json    medians per task and statistic
  figS17_tss_cohen_d.csv         per task and TSS (bed_row): Cohen's d, rank by |d|, folds in the top 1,000

Run from the repository root (about a minute):
    COHORT_VARIANT=no_deep python src/figures/export/export_figS17_cohen_d.py
"""
from __future__ import annotations

import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, os.path.dirname(__file__))
from _common import COHORT_VARIANT, DATA_DIR, rp, variant_dir  # noqa: E402
from export_figS17 import N_HALF_SPLITS, N_PERMUTATIONS, SEED, TASKS, TOP_K, agreement, signed_mean_diff, top_set  # noqa: E402
from export_figS_loco import load  # noqa: E402


def cohen_d(X, y):
    """Difference in means (cancer minus healthy) over the pooled standard deviation."""
    a, b = X[y == 1], X[y == 0]
    sd = np.sqrt(((len(a) - 1) * a.var(axis=0, ddof=1) + (len(b) - 1) * b.var(axis=0, ddof=1)) / (len(a) + len(b) - 2))
    return np.nan_to_num((a.mean(axis=0) - b.mean(axis=0)) / np.where(sd > 0, sd, np.nan))


STATS = {"cohen_d": cohen_d, "mean_diff": signed_mean_diff}


def main() -> None:
    assert COHORT_VARIANT == "no_deep", "run with COHORT_VARIANT=no_deep"
    sc, X_all = load(cores=1)
    pos = {s: i for i, s in enumerate(sc["sample_id"])}
    disease = dict(zip(sc["sample_id"], sc["disease"]))
    rng = {name: np.random.default_rng(SEED) for name in STATS}     # the same permutations for both statistics
    rows, summary, per_tss = [], {}, []
    bed_rows = np.sort(pd.read_csv(os.path.join(DATA_DIR, "figS17_tss_effects_brca_crc.csv"), usecols=["bed_row"])["bed_row"].to_numpy())
    assert len(bed_rows) == X_all.shape[1]      # the column order of the cache (export_figS_loco.load)
    stored =json.load(open(os.path.join(DATA_DIR, "figS17_summary.json")))

    for task in TASKS:
        with open(rp(variant_dir(TASKS[task]), "ksweep_tuned", "nested_full_per_fold.pkl"), "rb") as fh:
            pf = pickle.load(fh).sort_values("fold")
        ids = sorted(set().union(*[set(r["train_sample_ids"]) | set(r["test_sample_ids"]) for _, r in pf.iterrows()]))
        X = X_all[[pos[s] for s in ids]]
        y = np.array([0 if disease[s] == "Healthy" else 1 for s in ids])
        strata = np.array([disease[s] for s in ids])
        idx_of = {s: i for i, s in enumerate(ids)}
        n = X.shape[1]
        print(f"\n=== {task}: {len(ids)} samples ({int((y == 0).sum())} healthy, {int(y.sum())} cancer), {n} TSSs")
        summary[task] = {}
        for name, stat in STATS.items():
            stab = []
            full = stat(X, y)
            fold = {int(r["fold"]): stat(X[[idx_of[s] for s in r["train_sample_ids"]]], y[[idx_of[s] for s in r["train_sample_ids"]]])
                    for _, r in pf.iterrows()}
            folds = sorted(fold)
            for f in folds:
                agreement(fold[f], full, "fold_vs_full", task, "fold_vs_full", f"fold{f}", stab)
            for i, a in enumerate(folds):
                for b in folds[i + 1:]:
                    agreement(fold[a], fold[b], "fold_vs_fold", task, "fold_vs_fold", f"fold{a}-fold{b}", stab)
            sss = StratifiedShuffleSplit(n_splits=N_HALF_SPLITS, test_size=0.5, random_state=SEED)
            for i, (h1, h2) in enumerate(sss.split(X, strata)):
                agreement(stat(X[h1], y[h1]), stat(X[h2], y[h2]), "half", task, "half_vs_half", f"split{i}", stab)
            for i in range(N_PERMUTATIONS):
                yp = rng[name].permutation(y)
                agreement(stat(X, yp), full, "perm", task, "permuted_vs_full", f"perm{i}", stab)
            for k in TOP_K:
                stab.append(dict(task=task, comparison="chance_analytic", metric=f"jaccard_top{k}", pair="k/(2n-k)", value=k / (2 * n - k)))
            n_top1000 = np.zeros(n, int)
            for f in folds:
                n_top1000[list(top_set(fold[f], 1000))] += 1
            t = pd.DataFrame(stab)
            t["statistic"] = name
            rows.append(t)
            if name == "cohen_d":
                per_tss.append(pd.DataFrame({"task": task, "bed_row": bed_rows, "cohen_d": full,
                                             "abs_rank_cohen_d": pd.Series(-np.abs(full)).rank(method="first").astype(int).to_numpy(),
                                             "n_folds_top1000_cohen_d": n_top1000}))

            def med(comp, metric):
                return float(t[(t.comparison == comp) & (t.metric == metric)]["value"].median())

            s = {"spearman_signed_fold_vs_full_median": med("fold_vs_full", "spearman_signed"),
                 "spearman_signed_fold_vs_fold_median": med("fold_vs_fold", "spearman_signed"),
                 "spearman_signed_half_vs_half_median": med("half_vs_half", "spearman_signed"),
                 "spearman_signed_permuted_median": med("permuted_vs_full", "spearman_signed"),
                 "jaccard_top1000_fold_vs_fold_median": med("fold_vs_fold", "jaccard_top1000"),
                 "jaccard_top1000_half_vs_half_median": med("half_vs_half", "jaccard_top1000"),
                 "jaccard_top1000_permuted_median": med("permuted_vs_full", "jaccard_top1000"),
                 "jaccard_top100_half_vs_half_median": med("half_vs_half", "jaccard_top100"),
                 "jaccard_top1000_chance": 1000 / (2 * n - 1000),
                 "n_tss_top1000_in_all_folds": int((n_top1000 == len(folds)).sum()),
                 "median_abs": float(np.median(np.abs(full))), "max_abs": float(np.abs(full).max())}
            summary[task][name] = s
            print(f"  {name:9s} " + ", ".join(f"{k.replace('_median', '')} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in s.items()))
            if name == "mean_diff":      # must be the stored analysis: same folds and splits (permutations are redrawn per statistic)
                for k in ["spearman_signed_fold_vs_full_median", "spearman_signed_fold_vs_fold_median", "spearman_signed_half_vs_half_median",
                          "jaccard_top1000_fold_vs_fold_median", "jaccard_top1000_half_vs_half_median", "n_tss_top1000_in_all_folds"]:
                    assert abs(s[k] - stored[task][k]) < 2e-3, (task, k, s[k], stored[task][k])
                print("    mean_diff reproduces figS17_summary.json (folds and half splits)")

    pd.concat(rows).to_csv(os.path.join(DATA_DIR, "figS17_stability_cohen_d.csv"), index=False, lineterminator="\n")
    pd.concat(per_tss).to_csv(os.path.join(DATA_DIR, "figS17_tss_cohen_d.csv"), index=False, lineterminator="\n")
    with open(os.path.join(DATA_DIR, "figS17_summary_cohen_d.json"), "w", newline="\n") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nwrote figS17_stability_cohen_d.csv and figS17_summary_cohen_d.json to {DATA_DIR}")


if __name__ == "__main__":
    main()
