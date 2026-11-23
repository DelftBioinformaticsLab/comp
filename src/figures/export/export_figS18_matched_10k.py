"""Fig S18 look-alike comparison with 10,000 matched random sets (instead of 1,000):
rank-AUC of the signed d per Hallmark, C8 and Reactome set and the Zhu promoter set,
matched on GC x coverage x healthy dip depth (4 x 4 x 4 strata), BH within a collection.

Inputs: data[/no_deep]/figS18_tss_sets.csv and figS18_gsea_matched.csv
(export_figS18_gene_sets.py, whose functions are reused) and the MSigDB GMT files.
Writes data[/no_deep]/figS18_gsea_matched_10k.csv and figS18_zhu_lookalike_null.csv.
Run from the repository root:  COHORT_VARIANT=no_deep python src/figures/export/export_figS18_matched_10k.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from _common import DATA_DIR  # noqa: E402
from export_figS18_gene_sets import (GMT, MAX_SET_SIZE, MIN_SET_SIZE, N_STRATA_3D, TASKS,  # noqa: E402
                                     MatchedNull, ranks, read_gmt)

N_NULL = 10_000
CHUNK = 1_000
SEED = 2026
# Reactome (MSigDB 2024.1 c2.cp.reactome) in addition to the exporter's two collections
COLLECTIONS = {**GMT, "reactome": "accessory_files/annotations/msigdb/c2.cp.reactome.v2024.1.Hs.symbols.gmt"}


def bh(p: np.ndarray) -> np.ndarray:
    m = len(p); order = np.argsort(p)
    q = np.empty(m); q[order] = np.minimum.accumulate((p[order] * m / np.arange(1, m + 1))[::-1])[::-1]
    return np.clip(q, 0, 1)


def main() -> None:
    t0 = time.time()
    tss = pd.read_csv(os.path.join(DATA_DIR, "figS18_tss_sets.csv"))
    n = len(tss)
    gene_pos = {g: i for i, g in enumerate(tss["gene"].to_numpy())}
    usable = {}
    for coll, path in COLLECTIONS.items():
        idx = {t: np.array(sorted(gene_pos[g] for g in s if g in gene_pos)) for t, s in read_gmt(path).items()}
        usable[coll] = {t: i for t, i in idx.items() if MIN_SET_SIZE <= len(i) <= MAX_SET_SIZE}
    # the one pre-specified set tested in Fig S18a, b: the TSSs of the Zhu et al. promoter genes (the RNA region set)
    usable["prespecified"] = {"zhu_promoter": np.flatnonzero(tss["in_zhu_promoter"].astype(bool).to_numpy())}
    # the exporter's GC x coverage x dip-depth strata (quarters of each)
    q4 = lambda v: pd.qcut(pd.Series(v).rank(method="first"), N_STRATA_3D, labels=False).to_numpy()
    log_cov = np.log10(tss["pon_mean_coverage"].to_numpy() + 1e-3)
    strata3 = q4(tss["gc"].to_numpy(float)) * 100 + q4(log_cov) * 10 + q4(tss["pon_mean_gd_profile_figS10"].to_numpy(float))
    R = {t: ranks(np.nan_to_num(tss[f"d_{t}"].to_numpy(float))) for t in TASKS}
    null_sums = {(t, c, term): [] for t in TASKS for c in usable for term in usable[c]}
    # the Zhu set also on |d|: same look-alike draws, ranks of |d|
    R_abs = {t: ranks(np.abs(np.nan_to_num(tss[f"d_{t}"].to_numpy(float)))) for t in TASKS}
    zhu_idx = usable["prespecified"]["zhu_promoter"]
    null_abs = {t: [] for t in TASKS}

    rng = np.random.default_rng(SEED)
    for k in range(N_NULL // CHUNK):
        null = MatchedNull(strata3, CHUNK, rng)
        for t in TASKS:
            cs = null.cumsums(R[t])
            for c in usable:
                for term, idx in usable[c].items():
                    null_sums[(t, c, term)].append(null.null_sums(cs, idx))
            null_abs[t].append(null.null_sums(null.cumsums(R_abs[t]), zhu_idx))
        del null
        print(f"  chunk {k + 1}/{N_NULL // CHUNK} [{time.time() - t0:.0f} s]", flush=True)

    rows, zhu_null = [], []
    for (t, c, term), parts in null_sums.items():
        idx = usable[c][term]
        n1 = len(idx); n2 = n - n1
        base = n1 * (n1 + 1) / 2
        obs = (R[t][idx].sum() - base) / (n1 * n2)
        nv = (np.concatenate(parts) - base) / (n1 * n2)
        pg, pl = (1 + (nv >= obs).sum()) / (N_NULL + 1), (1 + (nv <= obs).sum()) / (N_NULL + 1)
        rows.append({"statistic": "cohen_d", "matching": "gc_coverage_dip", "contrast": t, "collection": c, "term": term,
                     "n_genes_matched": n1, "rank_auc_signed_d": obs, "rank_auc_signed_d_null_mean": nv.mean(),
                     "rank_auc_signed_d_null_lo": np.percentile(nv, 2.5), "rank_auc_signed_d_null_hi": np.percentile(nv, 97.5),
                     "rank_auc_signed_d_z": (obs - nv.mean()) / nv.std(ddof=1), "rank_auc_signed_d_p": min(1.0, 2 * min(pg, pl)),
                     "n_null": N_NULL})
        if c == "prespecified":      # the whole null distributions (signed d and |d|), drawn in Fig S18a, b
            obs_a = (R_abs[t][idx].sum() - base) / (n1 * n2)
            nv_a = (np.concatenate(null_abs[t]) - base) / (n1 * n2)
            pg_a, pl_a = (1 + (nv_a >= obs_a).sum()) / (N_NULL + 1), (1 + (nv_a <= obs_a).sum()) / (N_NULL + 1)
            rows[-1].update({"rank_auc_abs_d": obs_a, "rank_auc_abs_d_null_mean": nv_a.mean(),
                             "rank_auc_abs_d_null_lo": np.percentile(nv_a, 2.5), "rank_auc_abs_d_null_hi": np.percentile(nv_a, 97.5),
                             "rank_auc_abs_d_z": (obs_a - nv_a.mean()) / nv_a.std(ddof=1), "rank_auc_abs_d_p": min(1.0, 2 * min(pg_a, pl_a))})
            zhu_null.append(pd.DataFrame({"contrast": t, "term": term, "rank_auc_signed_d_lookalike": nv, "rank_auc_abs_d_lookalike": nv_a}))
    out = pd.DataFrame(rows)
    out["rank_auc_signed_d_q"] = np.nan
    for (t, c), g in out.groupby(["contrast", "collection"]):
        out.loc[g.index, "rank_auc_signed_d_q"] = bh(g["rank_auc_signed_d_p"].to_numpy())

    # consistency with the 1,000-set run: identical observed statistics, close null means
    old = pd.read_csv(os.path.join(DATA_DIR, "figS18_gsea_matched.csv"))
    old = old[(old["statistic"] == "cohen_d") & (old["matching"] == "gc_coverage_dip") & old["contrast"].isin(TASKS)]
    chk = out.merge(old[["contrast", "collection", "term", "rank_auc_signed_d", "rank_auc_signed_d_null_mean", "rank_auc_signed_d_q"]],
                    on=["contrast", "collection", "term"], suffixes=("", "_1k"), validate="1:1")
    assert len(chk) == out["collection"].isin(["hallmark", "c8_cell_type"]).sum(), \
        "set lists differ from the 1,000-set run (Reactome and the Zhu set have no 1,000-set run)"
    assert np.allclose(chk["rank_auc_signed_d"], chk["rank_auc_signed_d_1k"], atol=1e-12), "observed statistics differ"
    print(f"  observed rank-AUCs identical to the 1,000-set run; null means differ by at most "
          f"{(chk['rank_auc_signed_d_null_mean'] - chk['rank_auc_signed_d_null_mean_1k']).abs().max():.4f}")
    for (t, c), g in chk.groupby(["contrast", "collection"]):
        print(f"  {t:8s} {c:13s} {len(g):4d} sets: q < 0.05 with 1,000 random sets {int((g['rank_auc_signed_d_q_1k'] < 0.05).sum()):4d}, "
              f"with {N_NULL:,} {int((g['rank_auc_signed_d_q'] < 0.05).sum()):4d}")
    out.to_csv(os.path.join(DATA_DIR, "figS18_gsea_matched_10k.csv"), index=False)
    pd.concat(zhu_null, ignore_index=True).to_csv(os.path.join(DATA_DIR, "figS18_zhu_lookalike_null.csv"), index=False)
    for _, r in out[out["collection"] == "prespecified"].iterrows():
        print(f"  Zhu promoter set, {r['contrast']}: {r['n_genes_matched']} TSSs, rank-AUC {r['rank_auc_signed_d']:.3f} vs look-alikes "
              f"{r['rank_auc_signed_d_null_mean']:.3f} [{r['rank_auc_signed_d_null_lo']:.3f}-{r['rank_auc_signed_d_null_hi']:.3f}], "
              f"z {r['rank_auc_signed_d_z']:+.2f}, p {r['rank_auc_signed_d_p']:.4f} | |d|: {r['rank_auc_abs_d']:.3f} vs "
              f"{r['rank_auc_abs_d_null_mean']:.3f} [{r['rank_auc_abs_d_null_lo']:.3f}-{r['rank_auc_abs_d_null_hi']:.3f}], "
              f"z {r['rank_auc_abs_d_z']:+.2f}, p {r['rank_auc_abs_d_p']:.4f}")
    print(f"wrote {os.path.join(DATA_DIR, 'figS18_gsea_matched_10k.csv')} ({len(out)} rows) [{time.time() - t0:.0f} s]")


if __name__ == "__main__":
    main()
