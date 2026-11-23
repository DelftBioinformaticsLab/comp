"""Leave-one-cancer-type-out analysis of the pan-cancer task with one fixed configuration
(no_deep cohort, frozen folds of data/no_deep/fig4_baseline_scalars.csv).

Model: row-z-scored griffin_diff, StandardScaler, elastic-net LogisticRegression (C=10,
l1_ratio=0.1); score = decision_function. Arms: seen (ordinary CV) and loco (type T left out
of training, tested on all of T plus the healthy of the fold). Cancers are scored as the
fraction of held-out healthy they outscore; AUROC per type with a paired participant
bootstrap. TSS variants: all, top<k> (pooled mean difference) and cons<k> (best mean rank over
the training types), k = 100 / 1,000 / 5,000.

Writes data/no_deep/figS_loco_{per_type,overall,ksweep,participants,core_sizes}.csv and
figS_loco_summary.json; the GD matrix is cached in data/no_deep/_cache_loco_gd.npz.
Run (repository root):  python src/figures/export/export_figS_loco.py [--cores 8]
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
from _common import DATA_DIR, rp  # noqa: E402

warnings.filterwarnings("ignore")
SCALARS = os.path.join(DATA_DIR, "fig4_baseline_scalars.csv")          # no_deep: 367 rows, frozen folds
TSS_TABLE = os.path.join(DATA_DIR, "figS17_tss_effects_brca_crc.csv")   # the 14,221 model TSSs (bed_row)
CACHE = os.path.join(DATA_DIR, "_cache_loco_gd.npz")
BLACKLIST = rp("accessory_files/final_tss_indices_to_remove_hg38.npy")
FEATURES_DIR = rp("extracted_features/biomart_10kb")
CLF = dict(penalty="elasticnet", solver="saga", C=10.0, l1_ratio=0.1, random_state=42, max_iter=10000)
KS = [100, 1000, 5000]
CORE_TOP = 1000      # "consistent in every type" = in the top 1,000 of each training type (reported, too small to fit)
N_BOOT = 2000
SEED = 20261102


# ---------------------------------------------------------------------------- data
def _read_gd(sid, remove, cols):
    d = pd.read_csv(os.path.join(FEATURES_DIR, f"{sid}.hg38.frag.tsv_BIN.csv"), usecols=["griffin_diff"])
    x = d["griffin_diff"].to_numpy(dtype=float)
    kept = np.delete(np.arange(len(x)), remove)
    z = x[kept]
    mu, sd = np.nanmean(z), np.nanstd(z)
    z = np.nan_to_num((z - mu) / (sd if sd > 0 else 1.0))
    pos = {b: j for j, b in enumerate(kept)}
    return z[[pos[b] for b in cols]]


def load(cores):
    sc = pd.read_csv(SCALARS)
    cols = np.sort(pd.read_csv(TSS_TABLE, usecols=["bed_row"])["bed_row"].to_numpy())
    assert len(cols) == 14221
    ids = sc["sample_id"].tolist()
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        if list(z["ids"]) == ids and np.array_equal(z["cols"], cols):
            print(f"features: cache {os.path.relpath(CACHE)}")
            return sc, z["X"]
    print(f"reading griffin_diff of {len(ids)} samples ({cores} cores) ...", flush=True)
    remove = np.load(BLACKLIST)
    X = np.vstack(Parallel(n_jobs=cores, verbose=5)(delayed(_read_gd)(s, remove, cols) for s in ids))
    np.savez(CACHE, ids=np.array(ids), cols=cols, X=X)
    return sc, X


# ---------------------------------------------------------------------------- models
def variant_name(k):
    if k is None:
        return "all"
    return f"cons{k[1]}" if isinstance(k, tuple) else f"top{k}"


def per_type_ranks(Xtr, ytr, dtr):
    """Rank (1 = largest |mean type - mean healthy|) of every TSS within each training cancer type."""
    h = Xtr[ytr == 0].mean(0)
    return np.vstack([rankdata(-np.abs(Xtr[dtr == t].mean(0) - h)) for t in sorted(set(dtr[ytr == 1]))])


def fit_score(X, y, dis, tr, te, k):
    Xtr, ytr = X[tr], y[tr]
    r = per_type_ranks(Xtr, ytr, dis[tr])
    n_core = int((r <= CORE_TOP).all(0).sum())             # TSSs in the top CORE_TOP of every training type
    if k is None:
        sel = np.arange(X.shape[1])
    elif isinstance(k, tuple):                              # consensus: mean rank over the training types
        sel = np.sort(np.argsort(r.mean(0))[:k[1]])
    else:
        w = np.abs(Xtr[ytr == 1].mean(0) - Xtr[ytr == 0].mean(0))   # model_hpc mean_diff ranking, pooled
        sel = np.sort(np.argsort(w)[-k:])
    sc = StandardScaler().fit(Xtr[:, sel])
    m = LogisticRegression(**CLF).fit(sc.transform(Xtr[:, sel]), ytr)
    return te, m.decision_function(sc.transform(X[te][:, sel])), n_core


def auc(pos, neg):
    r = rankdata(np.concatenate([pos, neg]))
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def frac_above(s, sh):
    return np.array([((sh < v).sum() + 0.5 * (sh == v).sum()) / len(sh) for v in s])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cores", type=int, default=8)
    args = ap.parse_args()
    t0 = time.time()
    sc, X = load(args.cores)
    y = sc["y_true"].to_numpy().astype(int)
    dis = sc["disease"].to_numpy()
    fold = sc["fold"].to_numpy().astype(int)
    assert (y == 0).sum() == 164 and (y == 1).sum() == 203, "expected the no_deep pan-cancer cohort"
    types = sorted(set(dis[y == 1]))
    folds = sorted(set(fold))
    nt = {t: int((dis == t).sum()) for t in types}
    variants = [None] + KS + [("cons", k) for k in KS]

    jobs = []
    for f in folds:
        for k in variants:
            jobs.append(("seen", None, f, k, np.where(fold != f)[0], np.where(fold == f)[0]))
        for t in types:
            tr = np.where(((y == 0) & (fold != f)) | ((y == 1) & (dis != t)))[0]
            te = np.where(((y == 0) & (fold == f)) | (dis == t))[0]
            assert not (dis[tr] == t).any()
            for k in variants:
                jobs.append(("loco", t, f, k, tr, te))
    print(f"{len(jobs)} fits on {args.cores} cores ...", flush=True)
    out = Parallel(n_jobs=args.cores, verbose=5)(delayed(fit_score)(X, y, dis, tr, te, k) for *_, k, tr, te in jobs)
    res = {(a, t, f, k): o[:2] for (a, t, f, k, _, _), o in zip(jobs, out)}
    core = pd.DataFrame([{"arm": a, "left_out": t or "", "fold": f, "n_core": o[2]}
                         for (a, t, f, k, _, _), o in zip(jobs, out) if k is None])
    print(f"fits done after {(time.time() - t0) / 60:.1f} min", flush=True)

    # reference: per-fold AUROC of the seen arm, as the nested run reports it
    ref = [auc(s[y[te] == 1], s[y[te] == 0]) for te, s in (res[("seen", None, f, None)] for f in folds)]

    # per-participant healthy-referenced values
    H = np.where(y == 0)[0]
    rows = []

    def collect(arm, t, k):
        cv = {i: [] for i in np.where(dis == t)[0]}
        hv = {}
        for f in folds:
            te, s = res[(arm, t if arm == "loco" else None, f, k)]
            hm, cm = y[te] == 0, dis[te] == t
            sh = s[hm]
            for i, v in zip(te[hm], (rankdata(sh) - 0.5) / len(sh)):
                hv[i] = v
            for i, v in zip(te[cm], frac_above(s[cm], sh)):
                cv[i].append(v)
        assert all(len(v) == (10 if arm == "loco" else 1) for v in cv.values()) and sorted(hv) == sorted(H)
        ci_ = np.array(sorted(cv))
        c = np.array([np.mean(cv[i]) for i in ci_])
        h = np.array([hv[i] for i in H])
        for i, v in zip(ci_, c):
            rows.append((arm, variant_name(k), t, sc["sample_id"].iat[i], 1, v))
        return c, h

    rng = np.random.default_rng(SEED)
    bh = rng.integers(0, len(H), size=(N_BOOT, len(H)))
    bc = {t: rng.integers(0, nt[t], size=(N_BOOT, nt[t])) for t in types}

    def boot(c, h, t):
        return np.array([auc(c[bc[t][b]], h[bh[b]]) for b in range(N_BOOT)])

    def ci(b):
        return tuple(float(x) for x in np.percentile(b, [2.5, 97.5]))

    per_type, ksweep, B = [], [], {}
    for k in variants:
        for t in types:
            c_s, h_s = collect("seen", t, k)
            c_l, h_l = collect("loco", t, k)
            b_s, b_l = boot(c_s, h_s, t), boot(c_l, h_l, t)
            B[(k, t)] = (b_s, b_l)
            r = {"tss": variant_name(k), "type": t, "n": nt[t],
                 "auc_seen": auc(c_s, h_s), "seen_lo": ci(b_s)[0], "seen_hi": ci(b_s)[1],
                 "auc_loco": auc(c_l, h_l), "loco_lo": ci(b_l)[0], "loco_hi": ci(b_l)[1],
                 "diff": auc(c_l, h_l) - auc(c_s, h_s), "diff_lo": ci(b_l - b_s)[0], "diff_hi": ci(b_l - b_s)[1]}
            (per_type if k is None else ksweep).append(r)
        if k is None:
            # healthy rows once (same in every type)
            for i, v in zip(H, h_s):
                rows.append(("seen", "all", "", sc["sample_id"].iat[i], 0, v))
    w = np.array([nt[t] for t in types]) / sum(nt.values())
    overall = []
    for k in variants:
        bs = sum(w[i] * B[(k, t)][0] for i, t in enumerate(types))
        bl = sum(w[i] * B[(k, t)][1] for i, t in enumerate(types))
        tab = pd.DataFrame(per_type if k is None else [r for r in ksweep if r["tss"] == variant_name(k)])
        a_s, a_l = float((w * tab["auc_seen"]).sum()), float((w * tab["auc_loco"]).sum())
        overall.append({"tss": variant_name(k), "auc_seen": a_s, "seen_lo": ci(bs)[0], "seen_hi": ci(bs)[1],
                        "auc_loco": a_l, "loco_lo": ci(bl)[0], "loco_hi": ci(bl)[1],
                        "diff": a_l - a_s, "diff_lo": ci(bl - bs)[0], "diff_hi": ci(bl - bs)[1]})

    pd.DataFrame(per_type).to_csv(os.path.join(DATA_DIR, "figS_loco_per_type.csv"), index=False)
    pd.DataFrame(ksweep).to_csv(os.path.join(DATA_DIR, "figS_loco_ksweep.csv"), index=False)
    pd.DataFrame(overall).to_csv(os.path.join(DATA_DIR, "figS_loco_overall.csv"), index=False)
    core.to_csv(os.path.join(DATA_DIR, "figS_loco_core_sizes.csv"), index=False)
    pd.DataFrame(rows, columns=["arm", "tss", "type", "sample_id", "y_true", "value"]).to_csv(
        os.path.join(DATA_DIR, "figS_loco_participants.csv"), index=False)
    summary = {"config": {k: v for k, v in CLF.items()} | {"feature": "griffin_diff", "n_tss": int(X.shape[1]),
                                                           "n_boot": N_BOOT, "seed": SEED},
               "reference_seen_fold_auc": {"mean": float(np.mean(ref)), "sd": float(np.std(ref)), "per_fold": ref},
               "nested_pancan_no_deep_fold_mean": 0.969,
               "core_top": CORE_TOP,
               "n_core_loco": {t: [int(v) for v in g["n_core"]] for t, g in core[core["arm"] == "loco"].groupby("left_out")},
               "n_core_seen": [int(v) for v in core[core["arm"] == "seen"]["n_core"]], "minutes": (time.time() - t0) / 60}
    with open(os.path.join(DATA_DIR, "figS_loco_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(f"\nreference (seen, all TSSs): fold-mean AUROC {np.mean(ref):.3f} +/- {np.std(ref):.3f} "
              f"(fully nested pan-cancer, no_deep: 0.969)")
        print("\nper type, all TSSs:\n" + pd.DataFrame(per_type).drop(columns="tss").to_string(index=False))
        print("\noverall (weighted by n):\n" + pd.DataFrame(overall).to_string(index=False))
    print(f"\nwrote figS_loco_*.csv / .json to {os.path.relpath(DATA_DIR)} in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
