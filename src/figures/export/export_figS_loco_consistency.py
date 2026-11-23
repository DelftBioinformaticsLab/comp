"""Agreement of per-cancer-type TSS rankings against a split-half null (no_deep cohort).

Per split (N_SPLITS, seeded) the healthy donors and every type's cancers are split into
disjoint halves, and a per-TSS statistic is computed for each half against its own healthy half:
  within  rho_T      = Spearman(s(T1 vs H1), s(T2 vs H2))           (split-half ceiling)
  between rho_AB     = mean of Spearman(s(A1 vs H1), s(B2 vs H2)) and Spearman(s(A2 vs H2), s(B1 vs H1))
  vs rest rho_A,rest = the same with the six other types pooled as B
Disattenuated agreement = mean(rho_AB) / sqrt(mean(rho_A) * mean(rho_B)). Statistics: cohen_d
(Cohen's d of the row-z-scored griffin_diff, all splits) and weight (coefficients of the fixed
LOCO configuration, first N_SPLITS_WEIGHT splits only; slow).

Writes data/no_deep/figS_loco_consistency_{splits,matrix}.csv and figS_loco_consistency_summary.json.
Run from the repository root, after export_figS_loco.py has built its feature cache:
    COHORT_VARIANT=no_deep python src/figures/export/export_figS_loco_consistency.py [--cores 8] [--skip_weights]
"""

import argparse
import itertools
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
from export_figS_loco import CLF, load  # noqa: E402

warnings.filterwarnings("ignore")
N_SPLITS = 200
N_SPLITS_WEIGHT = 20
SEED = 20261103
REST = "other types pooled"


def cohen_d(X, y):
    a, b = X[y == 1], X[y == 0]
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(0, ddof=1) + (nb - 1) * b.var(0, ddof=1)) / (na + nb - 2))
    return np.nan_to_num((a.mean(0) - b.mean(0)) / np.where(sp > 0, sp, np.nan))


def weights(X, y):
    sc = StandardScaler().fit(X)
    return LogisticRegression(**CLF).fit(sc.transform(X), y).coef_.ravel()


def spearman_rows(R):
    """Spearman correlation between all rows of a matrix of ranks."""
    Z = (R - R.mean(1, keepdims=True)) / R.std(1, keepdims=True)
    return Z @ Z.T / R.shape[1]


def halves(rng, idx):
    p = rng.permutation(idx)
    return p[: len(p) // 2], p[len(p) // 2:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cores", type=int, default=8)
    ap.add_argument("--skip_weights", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    sc, X = load(args.cores)
    y = (sc["y_true"].to_numpy() == 1)
    dis = sc["disease"].to_numpy()
    assert (~y).sum() == 164 and y.sum() == 203, "expected the no_deep pan-cancer cohort"
    types = sorted(set(dis[y]))
    groups = types + [REST]
    H = np.flatnonzero(~y)
    rng = np.random.default_rng(SEED)
    splits = []
    for _ in range(N_SPLITS):
        h1, h2 = halves(rng, H)
        c = {t: halves(rng, np.flatnonzero(dis == t)) for t in types}
        splits.append((h1, h2, c))

    rows = []
    jobs_per_stat = {"cohen_d": N_SPLITS} if args.skip_weights else {"cohen_d": N_SPLITS, "weight": N_SPLITS_WEIGHT}
    for stat, ns in jobs_per_stat.items():
        for si in range(ns):
            s = splits[si]
            # designs: each type's halves, and for every type the six-other-types pooled halves
            des = {}
            for g in types:
                for half in (0, 1):
                    des[(g, half)] = np.concatenate([s[2][g][half], (s[0] if half == 0 else s[1])])
                    rest = np.concatenate([s[2][t][half] for t in types if t != g])
                    des[(REST + "|" + g, half)] = np.concatenate([rest, (s[0] if half == 0 else s[1])])
            keys = list(des)
            if stat == "cohen_d":
                vecs = [cohen_d(X[des[k]], y[des[k]].astype(int)) for k in keys]
            else:
                vecs = Parallel(n_jobs=args.cores)(delayed(weights)(X[des[k]], y[des[k]].astype(int)) for k in keys)
            R = np.vstack([rankdata(v) for v in vecs])
            C = spearman_rows(R)
            pos = {k: i for i, k in enumerate(keys)}
            for g in types:
                rows.append(dict(statistic=stat, split=si, a=g, b=g, rho=C[pos[(g, 0)], pos[(g, 1)]]))
                r = C[pos[(REST + "|" + g, 0)], pos[(REST + "|" + g, 1)]]
                rows.append(dict(statistic=stat, split=si, a=REST + "|" + g, b=REST + "|" + g, rho=r))
                rows.append(dict(statistic=stat, split=si, a=g, b=REST,
                                 rho=0.5 * (C[pos[(g, 0)], pos[(REST + "|" + g, 1)]] + C[pos[(g, 1)], pos[(REST + "|" + g, 0)]])))
            for a, b in itertools.combinations(types, 2):
                rows.append(dict(statistic=stat, split=si, a=a, b=b,
                                 rho=0.5 * (C[pos[(a, 0)], pos[(b, 1)]] + C[pos[(a, 1)], pos[(b, 0)]])))
            if si % 20 == 0 or stat == "weight":
                print(f"  {stat}: split {si + 1}/{ns} [{time.time() - t0:.0f} s]", flush=True)

    sp = pd.DataFrame(rows)
    sp.to_csv(os.path.join(DATA_DIR, "figS_loco_consistency_splits.csv"), index=False)

    mat, summary = [], {"n_splits": jobs_per_stat, "seed": SEED, "n_per_type": {t: int((dis == t).sum()) for t in types}, "n_healthy": int(len(H))}
    for stat, g in sp.groupby("statistic"):
        m = g.groupby(["a", "b"])["rho"].agg(["mean", "std", lambda v: np.percentile(v, 2.5), lambda v: np.percentile(v, 97.5)])
        m.columns = ["rho_mean", "rho_sd", "rho_lo", "rho_hi"]
        within = {t: m.loc[(t, t), "rho_mean"] for t in types}
        within_rest = {t: m.loc[(REST + "|" + t, REST + "|" + t), "rho_mean"] for t in types}
        for (a, b), r in m.iterrows():
            if a.startswith(REST):
                continue
            if b == REST:
                den = np.sqrt(within[a] * within_rest[a])
            else:
                den = np.sqrt(within[a] * within[b])
            mat.append(dict(statistic=stat, a=a, b=b, **r.to_dict(),
                            disattenuated=r["rho_mean"] / den if den > 0 else np.nan,
                            within_a=within[a], within_b=within_rest[a] if b == REST else within[b]))
        off = [x for x in mat if x["statistic"] == stat and x["a"] != x["b"] and x["b"] != REST]
        summary[stat] = {"within": within, "within_rest": within_rest,
                         "between_median": float(np.median([x["rho_mean"] for x in off])),
                         "disattenuated_median": float(np.nanmedian([x["disattenuated"] for x in off])),
                         "vs_rest_disattenuated": {x["a"]: x["disattenuated"] for x in mat if x["statistic"] == stat and x["b"] == REST}}
        print(f"\n{stat}: within-type split-half rho " + ", ".join(f"{t.split()[0]} {v:.3f}" for t, v in within.items()))
        print(f"  between types median rho {summary[stat]['between_median']:.3f}, disattenuated median {summary[stat]['disattenuated_median']:.2f}")
        print("  vs the six other types pooled, disattenuated: " + ", ".join(f"{t.split()[0]} {v:.2f}" for t, v in summary[stat]["vs_rest_disattenuated"].items()))
    pd.DataFrame(mat).to_csv(os.path.join(DATA_DIR, "figS_loco_consistency_matrix.csv"), index=False)
    summary["minutes"] = (time.time() - t0) / 60
    with open(os.path.join(DATA_DIR, "figS_loco_consistency_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=float)
    print(f"\nwrote figS_loco_consistency_*.csv / .json to {os.path.relpath(DATA_DIR)} in {summary['minutes']:.1f} min")


if __name__ == "__main__":
    main()
