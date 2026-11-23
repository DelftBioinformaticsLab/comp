"""Source data for Fig S17: one ranked list of TSSs per task and its stability.

The list is the signed mean difference (cancer minus healthy) of the row-z-scored
Griffin coverage-difference feature (GD) per TSS on the whole task cohort; its absolute
value is the pipeline's 'mean_diff' ranking. Cohen's d, per-TSS AUROC and BH q-values
(Welch t-test) are reported alongside. Stability compares the list rebuilt per outer
training fold (fold_vs_full, fold_vs_fold), the loci each fold selected, disjoint
stratified halves, permuted labels, and the refitted fold winners' weights (the refit
must reproduce the stored held-out AUROC).

Inputs: the folds in <task run>/ksweep_tuned, the GD feature of every sample (cache .npz
with ids and gd, built from extracted_features/biomart_10kb/<ID>.hg38.frag.tsv_BIN.csv
when --gd_cache does not exist), accessory_files/cris_manifest.csv, beds/biomart.bed,
accessory_files/final_tss_indices_to_remove_hg38.npy.
Writes to data[/no_deep]/: figS17_tss_effects_<task>.csv, figS17_stability.csv,
figS17_folds.csv, figS17_summary.json.

Run from the repository root (about 30 min, the refits dominate):
    COHORT_VARIANT=no_deep python src/figures/export/export_figS17.py --gd_cache <npz> [--cores 8]
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr, ttest_ind
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, os.path.dirname(__file__))
from _common import COHORT_VARIANT, DATA_DIR, rp, variant_dir  # noqa: E402

TASKS = {"brca_crc": "nested_cv_results_cris/nested_full_brca_crc_nested_full_ksweep",
         "pancan": "nested_cv_results_cris/nested_full_pancan_nested_full_ksweep"}
FEATURE = "griffin_diff"
TOP_K = [100, 250, 500, 1000, 2500, 5000, 10000]
N_HALF_SPLITS = 50
N_PERMUTATIONS = 20
SEED = 0


# ----------------------------------------------------------------------------- data
def load_gd(ids: list[str], cache: str | None) -> np.ndarray:
    """GD feature (n x 19,376 bed rows) for `ids`, from the cache when it holds them all,
    else read from the per-sample BIN files (slow on the share) and cached."""
    if cache and os.path.isfile(cache):
        z = np.load(cache, allow_pickle=True)
        pos = {str(s): i for i, s in enumerate(z["ids"])}
        if all(s in pos for s in ids):
            return z["gd"][[pos[s] for s in ids]].astype(float)
        print(f"  cache {cache} lacks {sum(s not in pos for s in ids)} of the {len(ids)} samples - reading BIN files")
    rows = []
    for i, s in enumerate(ids):
        f = rp(f"extracted_features/biomart_10kb/{s}.hg38.frag.tsv_BIN.csv")
        rows.append(pd.read_csv(f, usecols=[FEATURE])[FEATURE].to_numpy(float))
        if (i + 1) % 50 == 0:
            print(f"    {i + 1}/{len(ids)} files read")
    G = np.vstack(rows)
    if cache:
        np.savez_compressed(cache, ids=np.array(ids), gd=G.astype(np.float32), n_cris=len(ids))
        print(f"  cached {G.shape} to {cache}")
    return G


KEPT_COLUMNS_SOURCE = "nested_cv_results_cris/best_model_best_model_search_dd_rev/preprocessing_metadata.pkl"


def preprocess(G: np.ndarray):
    """model_hpc's TSS preprocessing: blacklist, per-sample z-score, duplicate-column
    removal. Returns X (n x 14,221) and the bed row of every column. The kept columns
    come from the run's metadata (KEPT_COLUMNS_SOURCE), because a dedupe on a float32
    cache would drop columns the float64 run kept."""
    remove = np.load(rp("accessory_files/final_tss_indices_to_remove_hg38.npy"))
    keep = np.setdiff1d(np.arange(G.shape[1]), remove)
    X = G[:, keep]
    mu = np.nanmean(X, axis=1, keepdims=True)
    sd = np.nanstd(X, axis=1, keepdims=True)
    sd[sd == 0] = 1
    X = np.nan_to_num((X - mu) / sd)
    meta = rp(variant_dir(KEPT_COLUMNS_SOURCE))
    if os.path.isfile(meta):
        with open(meta, "rb") as fh:
            kept = np.asarray(pickle.load(fh)["kept_columns_per_feature"][FEATURE])
        pos = np.searchsorted(keep, kept)
        assert np.array_equal(keep[pos], kept), "kept columns of the run are not a subset of the blacklist-filtered columns"
        print(f"  column set ({len(kept)} columns) taken from {meta}")
        return X[:, pos], kept
    print(f"  WARNING: {meta} not found - duplicate columns removed here (float precision may differ from the run)")
    _, uniq = np.unique(X, axis=1, return_index=True)
    uniq = np.sort(uniq)
    return X[:, uniq], keep[uniq]


# ------------------------------------------------------------------------ statistics
def signed_mean_diff(X, y):
    return X[y == 1].mean(axis=0) - X[y == 0].mean(axis=0)


def top_set(v, k):
    """The pipeline's selection: the k columns with the largest |v| (select_top_genes)."""
    k = min(k, len(v))
    return set(np.argsort(np.abs(v))[-k:].tolist())


def ranking_weights(X, y, selection_type):
    """model_hpc.compute_feature_weights for 'mean_diff', 'fold_change' and 'p_value'
    (the nested selection keeps the k columns with the largest weight)."""
    mc, mh = X[y == 1].mean(axis=0), X[y == 0].mean(axis=0)
    if selection_type == "mean_diff":
        return np.nan_to_num(np.abs(mc - mh))
    if selection_type == "fold_change":
        with np.errstate(divide="ignore", invalid="ignore"):
            fc = np.divide(mc, mh, out=np.full_like(mc, np.nan), where=mh != 0)
            lfc = np.log2(fc, out=np.full_like(fc, np.nan), where=fc > 1e-10)
        return np.abs(np.nan_to_num(lfc))
    if selection_type == "p_value":
        p = mannwhitneyu(X[y == 1], X[y == 0], alternative="two-sided", axis=0).pvalue
        p = np.where(p > 0, p, 1e-10)
        return -np.log10(p)
    raise ValueError(f"ranking {selection_type!r} not replicated here")


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else np.nan


def agreement(v1, v2, label, task, comparison, pair, rows, ks=TOP_K):
    rows.append(dict(task=task, comparison=comparison, metric="spearman_signed", pair=pair, value=float(spearmanr(v1, v2)[0])))
    rows.append(dict(task=task, comparison=comparison, metric="spearman_abs", pair=pair, value=float(spearmanr(np.abs(v1), np.abs(v2))[0])))
    for k in ks:
        rows.append(dict(task=task, comparison=comparison, metric=f"jaccard_top{k}", pair=pair, value=jaccard(top_set(v1, k), top_set(v2, k))))


def refit_fold(task, r, X, y, idx_of, sel):
    """Refit the fold winner on its training rows with the stored configuration on the
    stored selected columns `sel`; return (weights over all columns, refit test AUROC)."""
    tr = np.array([idx_of[s] for s in r["train_sample_ids"]])
    te = np.array([idx_of[s] for s in r["test_sample_ids"]])
    sel = np.array(sorted(sel))
    hp = {kk.replace("clf__", ""): vv for kk, vv in dict(r["clf_hyperparams"]).items()}
    if r["classifier"] == "Logistic Regression":
        clf = LogisticRegression(random_state=42, max_iter=10000, **hp)
    elif r["classifier"] == "Support Vector Machine":
        clf = SVC(probability=True, random_state=42, **hp)
        if hp.get("kernel", "rbf") != "linear":
            raise ValueError(f"{task} fold {r['fold']}: {hp.get('kernel')} kernel has no weight vector")
    else:
        raise ValueError(f"{task} fold {r['fold']}: no weight definition for {r['classifier']}")
    sc = StandardScaler().fit(X[tr][:, sel])
    clf.fit(sc.transform(X[tr][:, sel]), y[tr])
    auc = roc_auc_score(y[te], clf.decision_function(sc.transform(X[te][:, sel])))
    w = np.zeros(X.shape[1])
    w[sel] = np.asarray(clf.coef_).ravel()
    return w, float(auc)


# ------------------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gd_cache", default=None, help=".npz with ids and gd (n x 19,376); built from the BIN files if absent")
    ap.add_argument("--cores", type=int, default=4)
    ap.add_argument("--tasks", nargs="+", default=list(TASKS))
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    manifest = pd.read_csv(rp("accessory_files/cris_manifest.csv")).set_index("ID")
    bed = pd.read_csv(rp("beds/biomart.bed"), sep="\t", header=None, names=["chrom", "start", "end", "gene", "strand", "gc"])
    stab, fold_rows, summary = [], [], {}
    os.makedirs(DATA_DIR, exist_ok=True)

    for task in args.tasks:
        t0 = time.time()
        d = variant_dir(TASKS[task])
        with open(rp(d, "ksweep_tuned", "nested_full_per_fold.pkl"), "rb") as fh:
            pf = pickle.load(fh).sort_values("fold")
        assert (pf["feature"] == FEATURE).all(), f"{task}: a fold selected another feature than {FEATURE}"
        # stability of the list uses mean_diff; selected loci use each fold's own ranking
        ids = sorted(set().union(*[set(r["train_sample_ids"]) | set(r["test_sample_ids"]) for _, r in pf.iterrows()]))
        y = np.array([0 if manifest.loc[s, "disease"] == "Healthy" else 1 for s in ids])
        idx_of = {s: i for i, s in enumerate(ids)}
        print(f"\n=== {task} ({COHORT_VARIANT}): {len(ids)} samples ({int((y == 0).sum())} healthy, {int(y.sum())} cancer), folds from {d}")
        X, bed_row = preprocess(load_gd(ids, args.gd_cache))
        n = X.shape[1]
        assert n == int(pf.iloc[0]["n_features_available"][FEATURE]), f"{task}: {n} columns, the run had {pf.iloc[0]['n_features_available'][FEATURE]}"
        print(f"  GD matrix {X.shape} (blacklist, row z-score, duplicate columns removed) - matches the run")

        # ---- the list, on the whole cohort ----
        m_full = signed_mean_diff(X, y)
        abs_rank = pd.Series(-np.abs(m_full)).rank(method="first").astype(int).to_numpy()   # 1 = largest |mean diff|
        sd_pooled = np.sqrt(((y == 1).sum() - 1) * X[y == 1].var(axis=0, ddof=1) + ((y == 0).sum() - 1) * X[y == 0].var(axis=0, ddof=1)) / np.sqrt(len(y) - 2)
        cohen_d = m_full / np.where(sd_pooled > 0, sd_pooled, np.nan)
        p_welch = ttest_ind(X[y == 1], X[y == 0], equal_var=False).pvalue
        u = mannwhitneyu(X[y == 1], X[y == 0], axis=0).statistic
        auroc = u / ((y == 1).sum() * (y == 0).sum())
        order = np.argsort(p_welch)
        q = np.empty(n); q[order] = np.minimum.accumulate((p_welch[order] * n / np.arange(1, n + 1))[::-1])[::-1]
        q = np.clip(q, 0, 1)
        print(f"  full list: {int((q < 0.05).sum())} TSSs at q < 0.05; |mean diff| median {np.median(np.abs(m_full)):.4f}, "
              f"max {np.abs(m_full).max():.4f}; {int((m_full > 0).sum())} positive (GD higher in cancer), {int((m_full < 0).sum())} negative")

        # ---- per fold: lists, selected loci, refit weights ----
        m_fold, sel_fold, sel_md, w_fold = {}, {}, {}, {}
        for _, r in pf.iterrows():
            f = int(r["fold"])
            tr = np.array([idx_of[s] for s in r["train_sample_ids"]])
            m_fold[f] = signed_mean_diff(X[tr], y[tr])
            k = int(r["n_top_genes"])
            st = r["selection_type"]
            own = top_set(ranking_weights(X[tr], y[tr], st), k)           # the fold's own ranking, replicated
            stored = set(np.asarray(r["selected_gene_indices"]).tolist())  # what the run selected
            match = own == stored
            sel_fold[f] = stored                                           # the selected loci = the stored set
            sel_md[f] = top_set(m_fold[f], k)                              # the same k under THE list's ranking
            w, auc_refit = refit_fold(task, r, X, y, idx_of, stored)
            w_fold[f] = w
            agreement(m_fold[f], m_full, "fold_vs_full", task, "fold_vs_full", f"fold{f}", stab)
            fold_rows.append(dict(task=task, fold=f, k=k, selection_type=st, n_selected=len(stored), classifier=r["classifier"],
                                  hyperparameters=json.dumps({kk: (float(v) if isinstance(v, (np.floating, float)) else v) for kk, v in dict(r["clf_hyperparams"]).items()}),
                                  stored_test_auc=float(r["test_auc"]), refit_test_auc=auc_refit,
                                  own_ranking_matches_stored=bool(match), own_ranking_overlap_with_stored=len(own & stored) / len(stored),
                                  jaccard_meandiff_vs_stored_same_k=jaccard(sel_md[f], stored),
                                  n_weights_nonzero=int((w != 0).sum()),
                                  spearman_signed_vs_full=float(spearmanr(m_fold[f], m_full)[0]),
                                  jaccard_selected_vs_full_same_k=jaccard(stored, top_set(m_full, k)),
                                  spearman_weight_vs_fold_effect=float(spearmanr(np.abs(w[w != 0]), np.abs(m_fold[f][w != 0]))[0]),
                                  spearman_weight_vs_full_effect_signed=float(spearmanr(w[w != 0], m_full[w != 0])[0])))
            print(f"  fold {f:2d}: k={k:5d} {st:11s} {r['classifier']}: stored AUROC {float(r['test_auc']):.3f} refit {auc_refit:.3f} | "
                  f"own-ranking set {'matches stored' if match else f'differs from stored ({len(own & stored) / len(stored):.3f} overlap)'} | "
                  f"mean_diff set vs stored J={jaccard(sel_md[f], stored):.3f} | Spearman vs full {spearmanr(m_fold[f], m_full)[0]:.3f} | nonzero weights {int((w != 0).sum())}")
        folds = sorted(m_fold)
        for i, a in enumerate(folds):
            for b in folds[i + 1:]:
                agreement(m_fold[a], m_fold[b], "fold_vs_fold", task, "fold_vs_fold", f"fold{a}-fold{b}", stab)
                stab.append(dict(task=task, comparison="fold_vs_fold", metric="jaccard_selected", pair=f"fold{a}-fold{b}", value=jaccard(sel_fold[a], sel_fold[b])))
                stab.append(dict(task=task, comparison="fold_vs_fold", metric="jaccard_selected_meandiff", pair=f"fold{a}-fold{b}", value=jaccard(sel_md[a], sel_md[b])))
                wa, wb = w_fold[a], w_fold[b]
                both = (wa != 0) & (wb != 0)
                stab.append(dict(task=task, comparison="weight_fold_vs_fold", metric="spearman_signed", pair=f"fold{a}-fold{b}", value=float(spearmanr(wa, wb)[0])))
                stab.append(dict(task=task, comparison="weight_fold_vs_fold", metric="spearman_signed_both_nonzero", pair=f"fold{a}-fold{b}", value=float(spearmanr(wa[both], wb[both])[0])))
                stab.append(dict(task=task, comparison="weight_fold_vs_fold", metric="spearman_abs", pair=f"fold{a}-fold{b}", value=float(spearmanr(np.abs(wa), np.abs(wb))[0])))
                for k in TOP_K:
                    stab.append(dict(task=task, comparison="weight_fold_vs_fold", metric=f"jaccard_top{k}", pair=f"fold{a}-fold{b}", value=jaccard(top_set(wa, k), top_set(wb, k))))
        n_sel = np.zeros(n, int)
        n_sel_md = np.zeros(n, int)
        for f in folds:
            n_sel[list(sel_fold[f])] += 1
            n_sel_md[list(sel_md[f])] += 1
        n_top1000 = np.zeros(n, int)
        for f in folds:
            n_top1000[list(top_set(m_fold[f], 1000))] += 1
        W = np.vstack([w_fold[f] for f in folds])
        M = np.vstack([m_fold[f] for f in folds])

        # ---- references: disjoint halves (sampling noise) and permuted labels (chance) ----
        sss = StratifiedShuffleSplit(n_splits=N_HALF_SPLITS, test_size=0.5, random_state=SEED)
        strata = np.array([manifest.loc[s, "disease"] for s in ids])
        for i, (h1, h2) in enumerate(sss.split(X, strata)):
            agreement(signed_mean_diff(X[h1], y[h1]), signed_mean_diff(X[h2], y[h2]), "half", task, "half_vs_half", f"split{i}", stab)
        for i in range(N_PERMUTATIONS):
            yp = rng.permutation(y)
            agreement(signed_mean_diff(X, yp), m_full, "perm", task, "permuted_vs_full", f"perm{i}", stab)
        for k in TOP_K:
            stab.append(dict(task=task, comparison="chance_analytic", metric=f"jaccard_top{k}", pair="k/(2n-k)", value=k / (2 * n - k)))

        # ---- per-TSS table ----
        tab = bed.iloc[bed_row].reset_index(drop=True)
        tab.insert(0, "bed_row", bed_row)
        tab.insert(0, "column", np.arange(n))
        tab["mean_diff"] = m_full
        tab["mean_healthy"] = X[y == 0].mean(axis=0)   # group means of the row-z-scored GD
        tab["mean_cancer"] = X[y == 1].mean(axis=0)
        tab["abs_rank"] = abs_rank
        tab["cohen_d"] = cohen_d
        tab["auroc"] = auroc
        tab["p_welch"] = p_welch
        tab["q_bh"] = q
        tab["fold_mean_diff_mean"] = M.mean(axis=0)
        tab["fold_mean_diff_sd"] = M.std(axis=0, ddof=1)
        tab["fold_abs_rank_mean"] = np.vstack([pd.Series(-np.abs(M[i])).rank(method="first").to_numpy() for i in range(len(folds))]).mean(axis=0)
        tab["n_folds_selected"] = n_sel                 # by the fold's own ranking (= the stored run)
        tab["n_folds_selected_meandiff"] = n_sel_md     # the same k under the mean_diff ranking
        tab["n_folds_top1000"] = n_top1000
        tab["weight_mean"] = W.mean(axis=0)
        tab["weight_sd"] = W.std(axis=0, ddof=1)
        tab["n_folds_weight_nonzero"] = (W != 0).sum(axis=0)
        tab["weight_abs_rank_mean"] = np.vstack([pd.Series(-np.abs(W[i])).rank(method="first").to_numpy() for i in range(len(folds))]).mean(axis=0)
        out = os.path.join(DATA_DIR, f"figS17_tss_effects_{task}.csv")
        tab.sort_values("abs_rank").to_csv(out, index=False, lineterminator="\n")
        print(f"  wrote {out} ({len(tab)} TSSs)")

        S = pd.DataFrame([s for s in stab if s["task"] == task])
        def med(comp, metric):
            v = S[(S.comparison == comp) & (S.metric == metric)]["value"]
            return float(v.median()) if len(v) else np.nan
        summary[task] = {
            "n_samples": len(ids), "n_healthy": int((y == 0).sum()), "n_cancer": int(y.sum()), "n_tss": n,
            "n_q_below_0.05": int((q < 0.05).sum()), "n_q_below_0.10": int((q < 0.10).sum()),
            "share_positive_mean_diff": float((m_full > 0).mean()),
            "share_opposite_sign_group_means": float((X[y == 0].mean(axis=0) * X[y == 1].mean(axis=0) < 0).mean()),
            "median_abs_mean_diff": float(np.median(np.abs(m_full))), "max_abs_mean_diff": float(np.abs(m_full).max()),
            "median_abs_cohen_d": float(np.nanmedian(np.abs(cohen_d))), "max_abs_cohen_d": float(np.nanmax(np.abs(cohen_d))),
            "k_per_fold": [int(r["n_top_genes"]) for _, r in pf.iterrows()],
            "selection_type_per_fold": [r["selection_type"] for _, r in pf.iterrows()],
            "selected_sets_match_stored": all(fr["own_ranking_matches_stored"] for fr in fold_rows if fr["task"] == task),
            "jaccard_meandiff_vs_stored_same_k_per_fold": [round(fr["jaccard_meandiff_vs_stored_same_k"], 3) for fr in fold_rows if fr["task"] == task],
            "jaccard_selected_meandiff_fold_vs_fold_median": float(np.median([s["value"] for s in stab if s["task"] == task and s["metric"] == "jaccard_selected_meandiff"])),
            "refit_auc_max_abs_diff": float(max(abs(fr["refit_test_auc"] - fr["stored_test_auc"]) for fr in fold_rows if fr["task"] == task)),
            "spearman_signed_fold_vs_full_median": med("fold_vs_full", "spearman_signed"),
            "spearman_signed_fold_vs_fold_median": med("fold_vs_fold", "spearman_signed"),
            "spearman_signed_half_vs_half_median": med("half_vs_half", "spearman_signed"),
            "spearman_signed_permuted_median": med("permuted_vs_full", "spearman_signed"),
            "jaccard_top1000_fold_vs_full_median": med("fold_vs_full", "jaccard_top1000"),
            "jaccard_top1000_fold_vs_fold_median": med("fold_vs_fold", "jaccard_top1000"),
            "jaccard_top1000_half_vs_half_median": med("half_vs_half", "jaccard_top1000"),
            "jaccard_top1000_permuted_median": med("permuted_vs_full", "jaccard_top1000"),
            "jaccard_top1000_chance": 1000 / (2 * n - 1000),
            "jaccard_top100_fold_vs_fold_median": med("fold_vs_fold", "jaccard_top100"),
            "jaccard_top100_half_vs_half_median": med("half_vs_half", "jaccard_top100"),
            "jaccard_selected_fold_vs_fold_median": med("fold_vs_fold", "jaccard_selected"),
            "n_tss_selected_in_all_folds": int((n_sel == len(folds)).sum()),
            "n_tss_top1000_in_all_folds": int((n_top1000 == len(folds)).sum()),
            "weight_spearman_signed_fold_vs_fold_median": med("weight_fold_vs_fold", "spearman_signed"),
            "weight_spearman_both_nonzero_fold_vs_fold_median": med("weight_fold_vs_fold", "spearman_signed_both_nonzero"),
            "weight_jaccard_top1000_fold_vs_fold_median": med("weight_fold_vs_fold", "jaccard_top1000"),
            "weight_jaccard_top100_fold_vs_fold_median": med("weight_fold_vs_fold", "jaccard_top100"),
            "weight_vs_fold_effect_spearman_abs_median": float(np.median([fr["spearman_weight_vs_fold_effect"] for fr in fold_rows if fr["task"] == task])),
            "weight_vs_full_effect_spearman_signed_median": float(np.median([fr["spearman_weight_vs_full_effect_signed"] for fr in fold_rows if fr["task"] == task])),
            "n_weights_nonzero_per_fold": [fr["n_weights_nonzero"] for fr in fold_rows if fr["task"] == task],
            "seconds": round(time.time() - t0),
        }
        print("  " + json.dumps({k: v for k, v in summary[task].items() if not isinstance(v, list)}, indent=None)[:1500])

    pd.DataFrame(stab).to_csv(os.path.join(DATA_DIR, "figS17_stability.csv"), index=False, lineterminator="\n")
    pd.DataFrame(fold_rows).to_csv(os.path.join(DATA_DIR, "figS17_folds.csv"), index=False, lineterminator="\n")
    with open(os.path.join(DATA_DIR, "figS17_summary.json"), "w", newline="\n") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nwrote figS17_stability.csv ({len(stab)} rows), figS17_folds.csv, figS17_summary.json to {DATA_DIR}")


if __name__ == "__main__":
    main()
