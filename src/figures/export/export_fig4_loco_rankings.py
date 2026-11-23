#!/usr/bin/env python
"""Leave-one-cancer-type-out with two TSS rankings: does a cross-type consensus ranking transfer better?

Same splits as the leave-one-cancer-type-out analysis (slurm/submit_loco.sh, export_fig4_loco.py):
training = 130 healthy donors + six cancer types, test = the left-out type + 34 held-out healthy.
One fixed model (Griffin difference, StandardScaler + elastic-net LR, C = 10, l1_ratio = 0.1)
is fitted on the top k TSSs (k in KS, or all) of a ranking computed on the training samples:
    pooled     |mean cancer - mean healthy| over all training cancers (model_hpc mean_diff)
    consensus  the same difference per training cancer type, ranks averaged over the six types

Writes data/no_deep/fig4_loco_rankings.csv (AUROC with participant bootstrap interval and
top-k overlap per type, ranking and k) and fig4_loco_rankings_summary.json.
Run from the repository root:  python src/figures/export/export_fig4_loco_rankings.py [--cores 6]
"""
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from _common import DATA_DIR  # noqa: E402
from export_figS_loco import CLF, load, per_type_ranks  # noqa: E402

warnings.filterwarnings("ignore")
KS = [100, 250, 500, 1000, 2500, 5000, 10000]
N_BOOT, SEED = 2000, 42


def auc(pos, neg):
    r = rankdata(np.concatenate([pos, neg]))
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def boot_ci(pos, neg):
    rng = np.random.default_rng(SEED)
    b = [auc(pos[rng.integers(0, len(pos), len(pos))], neg[rng.integers(0, len(neg), len(neg))]) for _ in range(N_BOOT)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def select(Xtr, ytr, dtr, ranking, k):
    if ranking == "all":
        return np.arange(Xtr.shape[1])
    if ranking == "pooled":
        w = np.abs(Xtr[ytr == 1].mean(0) - Xtr[ytr == 0].mean(0))
        return np.sort(np.argsort(w)[-k:])
    r = per_type_ranks(Xtr, ytr, dtr)                    # one row of ranks per training cancer type
    return np.sort(np.argsort(r.mean(0))[:k])


def fit_one(X, y, dis, tr, te, ranking, k):
    Xtr, ytr = X[tr], y[tr]
    sel = select(Xtr, ytr, dis[tr], ranking, k)
    other = set(select(Xtr, ytr, dis[tr], "consensus" if ranking == "pooled" else "pooled", k)) if ranking != "all" else set()
    sc = StandardScaler().fit(Xtr[:, sel])
    m = LogisticRegression(**CLF).fit(sc.transform(Xtr[:, sel]), ytr)
    s = m.decision_function(sc.transform(X[te][:, sel]))
    pos, neg = s[y[te] == 1], s[y[te] == 0]
    lo, hi = boot_ci(pos, neg)
    return {"auc": float(auc(pos, neg)), "auc_lo": lo, "auc_hi": hi, "n_tss": int(len(sel)),
            "shared_with_other_ranking": (len(set(sel) & other) / len(sel)) if ranking != "all" else np.nan}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cores", type=int, default=6)
    args = ap.parse_args()
    t0 = time.time()
    sc, X = load(args.cores)
    ids = sc["sample_id"].astype(str).to_numpy()
    y = sc["y_true"].to_numpy().astype(int)
    dis = sc["disease"].to_numpy()
    assert (y == 0).sum() == 164 and (y == 1).sum() == 203, "expected the no_deep pan-cancer cohort"
    per_type = pd.read_csv(os.path.join(DATA_DIR, "fig4_loco_per_type.csv"))
    pred = pd.read_csv(os.path.join(DATA_DIR, "fig4_loco_predictions.csv"))
    pos_of = {s: i for i, s in enumerate(ids)}

    jobs = []
    for _, row in per_type.iterrows():
        test_ids = pred.loc[pred["type"] == row["type"], "sample_id"].astype(str).tolist()
        te = np.array([pos_of[s] for s in test_ids])
        tr = np.setdiff1d(np.arange(len(ids)), te)
        assert (y[te] == 0).sum() == 34 and set(dis[te][y[te] == 1]) == {row["disease"]} and (y[te] == 1).sum() == row["n_cancer"]
        assert (y[tr] == 0).sum() == 130 and row["disease"] not in set(dis[tr]), row["type"]
        jobs.append((row["type"], "all", X.shape[1], tr, te))
        for k in KS:
            for ranking in ("pooled", "consensus"):
                jobs.append((row["type"], ranking, k, tr, te))
    print(f"{len(jobs)} fits on {args.cores} cores ...", flush=True)
    out = Parallel(n_jobs=args.cores, verbose=5)(delayed(fit_one)(X, y, dis, tr, te, ranking, k) for _, ranking, k, tr, te in jobs)
    res = pd.DataFrame([{"type": t, "ranking": ranking, "k": k, **o} for (t, ranking, k, _, _), o in zip(jobs, out)])
    res = res.merge(per_type[["type", "label", "n_cancer", "auc"]].rename(columns={"auc": "auc_deployed_model"}), on="type")
    res.to_csv(os.path.join(DATA_DIR, "fig4_loco_rankings.csv"), index=False, lineterminator="\n")

    chk = res[res["ranking"] == "all"]
    print("\nall TSSs, this refit against the deployed model of Figure 5d:")
    print(chk[["label", "auc", "auc_deployed_model"]].round(3).to_string(index=False))
    mean = res.groupby(["ranking", "k"])["auc"].mean().unstack(0)
    print("\nmean AUROC over the seven left-out types:")
    print(mean.round(3).to_string())
    summary = {"config": {**CLF, "feature": "griffin_diff"}, "ks": KS,
               "mean_auc_over_types": {r: {int(k): float(v) for k, v in mean[r].dropna().items()} for r in mean.columns},
               "max_abs_diff_all_vs_deployed": float((chk["auc"] - chk["auc_deployed_model"]).abs().max()),
               "mean_shared_fraction": {int(k): float(v) for k, v in res[res["ranking"] == "pooled"].groupby("k")["shared_with_other_ranking"].mean().items()},
               "minutes": (time.time() - t0) / 60}
    with open(os.path.join(DATA_DIR, "fig4_loco_rankings_summary.json"), "w", newline="\n") as fh:
        json.dump(summary, fh, indent=1)
    print(f"\nwrote {DATA_DIR}/fig4_loco_rankings.csv and fig4_loco_rankings_summary.json after {summary['minutes']:.1f} min")


if __name__ == "__main__":
    main()
