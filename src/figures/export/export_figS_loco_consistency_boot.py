"""Bootstrap interval for the disattenuated agreement of each cancer type with the six other types
pooled (Figure 6c), no_deep cohort. The point estimate is read from export_figS_loco_consistency.py
(Cohen's d). Each of N_BOOT replicates resamples patients within type and healthy donors with
replacement and repeats the estimate over N_SPLITS splits (copies of an individual stay in one half);
the interval is the 2.5th to 97.5th percentile.

Writes data/no_deep/figS_loco_consistency_bootstrap.csv (one row per type) and _bootstrap_replicates.csv;
--from_replicates redoes only the summary from the saved replicates.
Run from the repository root, after export_figS_loco.py has built its feature cache:
    COHORT_VARIANT=no_deep python src/figures/export/export_figS_loco_consistency_boot.py [--cores 8]
"""

import argparse
import json
import os
import sys
import tempfile
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(__file__))
from _common import DATA_DIR  # noqa: E402
from export_figS_loco import load  # noqa: E402
from export_figS_loco_consistency import cohen_d, spearman_rows  # noqa: E402

N_BOOT = 400
N_SPLITS = 20
SEED = 20261121


def one_replicate(b, X, y, dis, types, seed):
    rng = np.random.default_rng([seed, b])
    groups = {"healthy": np.flatnonzero(~y), **{t: np.flatnonzero(dis == t) for t in types}}
    # bootstrap weights per individual, within its group
    w = {g: np.bincount(rng.integers(0, len(ix), len(ix)), minlength=len(ix)) for g, ix in groups.items()}
    acc = {k: [] for k in ("within", "within_rest", "vs_rest")}
    for _ in range(N_SPLITS):
        half = {}
        for g, ix in groups.items():
            perm = rng.permutation(len(ix))
            parts = (perm[: len(ix) // 2], perm[len(ix) // 2:])
            half[g] = [np.repeat(ix[p], w[g][p]) for p in parts]   # copies of an individual stay in its half
        des = {}
        for g in types:
            for h in (0, 1):
                des[(g, h)] = np.concatenate([half[g][h], half["healthy"][h]])
                rest = np.concatenate([half[t][h] for t in types if t != g])
                des[("rest|" + g, h)] = np.concatenate([rest, half["healthy"][h]])
        keys = list(des)
        R = np.vstack([rankdata(cohen_d(X[des[k]], y[des[k]].astype(int))) for k in keys])
        C = spearman_rows(R)
        pos = {k: i for i, k in enumerate(keys)}
        acc["within"].append([C[pos[(g, 0)], pos[(g, 1)]] for g in types])
        acc["within_rest"].append([C[pos[("rest|" + g, 0)], pos[("rest|" + g, 1)]] for g in types])
        acc["vs_rest"].append([0.5 * (C[pos[(g, 0)], pos[("rest|" + g, 1)]] + C[pos[(g, 1)], pos[("rest|" + g, 0)]]) for g in types])
    m = {k: np.mean(v, axis=0) for k, v in acc.items()}
    den = np.sqrt(np.clip(m["within"], 0, None) * np.clip(m["within_rest"], 0, None))
    return [dict(replicate=b, type=t, rho=m["vs_rest"][i], within=m["within"][i], within_rest=m["within_rest"][i],
                 disattenuated=m["vs_rest"][i] / den[i] if den[i] > 0 else np.nan) for i, t in enumerate(types)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cores", type=int, default=8)
    ap.add_argument("--from_replicates", action="store_true", help="only redo the summary from the saved replicates")
    args = ap.parse_args()
    t0 = time.time()
    if args.from_replicates:
        summarize(pd.read_csv(os.path.join(DATA_DIR, "figS_loco_consistency_bootstrap_replicates.csv")))
        return
    sc, X = load(args.cores)
    y = (sc["y_true"].to_numpy() == 1)
    dis = sc["disease"].to_numpy()
    assert (~y).sum() == 164 and y.sum() == 203, "expected the no_deep pan-cancer cohort"
    types = sorted(set(dis[y]))
    reps = Parallel(n_jobs=args.cores, verbose=5)(delayed(one_replicate)(b, X, y, dis, types, SEED) for b in range(N_BOOT))
    rep = pd.DataFrame([r for rr in reps for r in rr])
    # local copy first, in case the write to a network drive fails
    rep.to_csv(os.path.join(tempfile.gettempdir(), "figS_loco_consistency_bootstrap_replicates.csv"), index=False)
    for attempt in range(5):
        try:
            os.makedirs(DATA_DIR, exist_ok=True)
            rep.to_csv(os.path.join(DATA_DIR, "figS_loco_consistency_bootstrap_replicates.csv"), index=False)
            break
        except OSError as e:
            print(f"write failed ({e}); retrying in 30 s", flush=True)
            time.sleep(30)

    summarize(rep)
    print(f"\nwrote figS_loco_consistency_bootstrap*.csv to {os.path.relpath(DATA_DIR)} in {(time.time() - t0) / 60:.1f} min")


def summarize(rep):
    """Percentile interval per type. A replicate with mean split-half correlation <= 0 counts as +infinity
    (not dropped), so an upper bound of infinity means the data do not bound it."""
    with open(os.path.join(DATA_DIR, "figS_loco_consistency_summary.json")) as fh:
        point = json.load(fh)["cohen_d"]["vs_rest_disattenuated"]
    rows = []
    for t, g in rep.groupby("type"):
        v = g["disattenuated"].to_numpy(dtype=float)
        n_undef = int(np.isnan(v).sum())
        v = np.where(np.isnan(v), np.inf, v)
        rows.append(dict(type=t, disattenuated=point[t], lo=np.percentile(v, 2.5, method="lower"),
                         hi=np.percentile(v, 97.5, method="higher"), boot_median=np.median(v), n_undefined=n_undef,
                         n_boot=len(v), n_splits_per_replicate=N_SPLITS, seed=SEED))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(DATA_DIR, "figS_loco_consistency_bootstrap.csv"), index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
