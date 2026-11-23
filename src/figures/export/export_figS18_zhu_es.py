"""Fig S18a, b: the GSEA running enrichment score of the Zhu promoter TSSs against look-alike TSSs.

Pre-ranked GSEA running sum (weight 1, as gseapy.prerank) along the TSSs ranked by d, with a null
of 10,000 look-alike sets matched on quarters of GC, coverage and dip depth (the strata of
export_figS18_matched_10k.py); two-sided empirical p. The band is 2.5-97.5% of the running sums
of the first 1,000 look-alike sets.

Inputs: data[/no_deep]/figS18_tss_sets.csv, figS18_gsea.csv (gseapy ES of the Zhu set, to check the curve).
Outputs: data[/no_deep]/figS18_zhu_es_curves.csv (rank, observed curve and look-alike band per task),
figS18_zhu_es_summary.csv (ES, look-alike ES mean and 95% range, p, per task).
Run from the repository root:  COHORT_VARIANT=no_deep python src/figures/export/export_figS18_zhu_es.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from _common import DATA_DIR  # noqa: E402
from export_figS18_gene_sets import N_STRATA_3D, TASKS, MatchedNull  # noqa: E402

N_NULL = 10_000
CHUNK = 1_000
N_BAND = 1_000
SEED = 2026


def running_sums(order_pos: np.ndarray, members: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Running enrichment sums for many sets at once. order_pos[i] = position of TSS i in the ranked
    list; members: (n_sets, n) TSS indices; w: |d| per TSS. Returns (n_sets, N) float32."""
    n_sets, n = members.shape
    N = len(order_pos)
    pos = order_pos[members]                                     # (n_sets, n) list positions of the hits
    hw = w[members]; hw = hw / hw.sum(axis=1, keepdims=True)     # hit weights
    step = np.full((n_sets, N), -1.0 / (N - n), dtype=np.float32)
    rows = np.repeat(np.arange(n_sets), n)
    step[rows, pos.ravel()] = hw.ravel().astype(np.float32)
    return np.cumsum(step, axis=1)


def es_of(curves: np.ndarray) -> np.ndarray:
    """GSEA enrichment score: the running sum's largest deviation from zero, signed."""
    mx, mn = curves.max(axis=1), curves.min(axis=1)
    return np.where(mx >= -mn, mx, mn)


def main() -> None:
    tss = pd.read_csv(os.path.join(DATA_DIR, "figS18_tss_sets.csv"))
    gsea = pd.read_csv(os.path.join(DATA_DIR, "figS18_gsea.csv"))
    zhu = np.flatnonzero(tss["in_zhu_promoter"].astype(bool).to_numpy())
    q4 = lambda v: pd.qcut(pd.Series(v).rank(method="first"), N_STRATA_3D, labels=False).to_numpy()
    log_cov = np.log10(tss["pon_mean_coverage"].to_numpy() + 1e-3)
    strata3 = q4(tss["gc"].to_numpy(float)) * 100 + q4(log_cov) * 10 + q4(tss["pon_mean_gd_profile_figS10"].to_numpy(float))
    comp = list(zip(*np.unique(strata3[zhu], return_counts=True)))
    N = len(tss)
    curves_out = {"rank": np.arange(1, N + 1)}
    summary = []
    rng = np.random.default_rng(SEED)
    nulls = [MatchedNull(strata3, CHUNK, rng) for _ in range(N_NULL // CHUNK)]    # the same draws for both tasks
    for t in TASKS:
        d = np.nan_to_num(tss[f"d_{t}"].to_numpy(float))
        order = np.argsort(-d, kind="stable")
        order_pos = np.empty(N, int); order_pos[order] = np.arange(N)
        w = np.abs(d)
        obs_curve = running_sums(order_pos, zhu[None, :], w)[0]
        es_obs = float(es_of(obs_curve[None, :])[0])
        es_ref = float(gsea[(gsea["contrast"] == t) & (gsea["statistic"] == "cohen_d") & (gsea["collection"] == "prespecified")
                            & (gsea["term"] == "zhu_promoter")]["es"].iloc[0])
        assert abs(es_obs - es_ref) < 0.005, f"{t}: ES {es_obs:.4f} does not reproduce gseapy {es_ref:.4f}"
        es_null, band = [], None
        for k, null in enumerate(nulls):
            members = np.concatenate([null.perm[s][:, :c] for s, c in comp], axis=1)
            cur = running_sums(order_pos, members, w)
            es_null.append(es_of(cur))
            if k == 0:
                band = np.percentile(cur[:N_BAND], [2.5, 50, 97.5], axis=0)
            del cur
        es_null = np.concatenate(es_null)
        pg, pl = (1 + (es_null >= es_obs).sum()) / (N_NULL + 1), (1 + (es_null <= es_obs).sum()) / (N_NULL + 1)
        curves_out.update({f"{t}_zhu": obs_curve, f"{t}_lookalike_lo": band[0], f"{t}_lookalike_median": band[1], f"{t}_lookalike_hi": band[2]})
        summary.append({"contrast": t, "term": "zhu_promoter", "n_tss": len(zhu), "es": es_obs, "es_gseapy": es_ref,
                        "es_lookalike_mean": es_null.mean(), "es_lookalike_lo": np.percentile(es_null, 2.5),
                        "es_lookalike_hi": np.percentile(es_null, 97.5), "p_lookalike": min(1.0, 2 * min(pg, pl)), "n_null": N_NULL})
        print(f"  {t}: Zhu ES {es_obs:+.3f} (gseapy {es_ref:+.3f}); look-alike ES {es_null.mean():+.3f} "
              f"[{np.percentile(es_null, 2.5):+.3f}, {np.percentile(es_null, 97.5):+.3f}]; p {summary[-1]['p_lookalike']:.4f}")
    pd.DataFrame(curves_out).to_csv(os.path.join(DATA_DIR, "figS18_zhu_es_curves.csv"), index=False, float_format="%.5f")
    pd.DataFrame(summary).to_csv(os.path.join(DATA_DIR, "figS18_zhu_es_summary.csv"), index=False)
    print(f"wrote figS18_zhu_es_curves.csv and figS18_zhu_es_summary.csv to {DATA_DIR}")


if __name__ == "__main__":
    main()
