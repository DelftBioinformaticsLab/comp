"""Reactome GSEA (cancer-vs-healthy d and the healthy dip ranking) and the healthy-vs-cancer NES
comparison for Hallmark, C8 and Reactome.

Same GSEA settings as export_figS18_gene_sets.py. Inputs: data[/no_deep]/figS18_tss_sets.csv,
figS18_gsea.csv, figS18_c8_classes.csv and
accessory_files/annotations/msigdb/c2.cp.reactome.v2024.1.Hs.symbols.gmt (MSigDB 2024.1).
Outputs: data[/no_deep]/figS18_gsea_reactome.csv and output[/no_deep]/figS18_healthy_vs_cancer.png
(NES on the healthy ranking vs NES on cancer-vs-healthy d, per collection and task).
Run from the repository root (several minutes; Reactome has ~1,700 sets):
    COHORT_VARIANT=no_deep python src/figures/export/export_figS18_reactome_healthy.py --cores 4
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from _common import COHORT_VARIANT, DATA_DIR, IS_VARIANT, rp  # noqa: E402
from export_figS18_gene_sets import MAX_SET_SIZE, MIN_SET_SIZE, N_GSEA_PERMUTATIONS, SEED, TASKS  # noqa: E402

REACTOME = "accessory_files/annotations/msigdb/c2.cp.reactome.v2024.1.Hs.symbols.gmt"
LISTS = {**{t: (f"d_{t}", "cohen_d") for t in TASKS}, "healthy": ("healthy_mean_gd", "healthy_mean_gd")}


def run_reactome(cores: int) -> pd.DataFrame:
    import gseapy
    tss = pd.read_csv(os.path.join(DATA_DIR, "figS18_tss_sets.csv"))
    genes = tss["gene"].to_numpy()
    out = []
    for contrast, (col, stat) in LISTS.items():
        rnk = pd.Series(np.nan_to_num(tss[col].to_numpy(float)), index=genes).sort_values(ascending=False)
        res = gseapy.prerank(rnk=rnk, gene_sets=rp(REACTOME), permutation_num=N_GSEA_PERMUTATIONS, min_size=MIN_SET_SIZE,
                             max_size=MAX_SET_SIZE, seed=SEED, threads=cores, outdir=None, verbose=False).res2d
        for _, r in res.iterrows():
            out.append(dict(contrast=contrast, collection="reactome", statistic=stat, term=r["Term"], es=float(r["ES"]),
                            nes=float(r["NES"]), p_nominal=float(r["NOM p-val"]), fdr=float(r["FDR q-val"])))
        print(f"  reactome / {contrast}: {len(res)} sets, {int((res['FDR q-val'].astype(float) < 0.05).sum())} at FDR < 0.05", flush=True)
    df = pd.DataFrame(out)
    df.to_csv(os.path.join(DATA_DIR, "figS18_gsea_reactome.csv"), index=False)
    return df


def plot(reactome: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr
    g = pd.concat([pd.read_csv(os.path.join(DATA_DIR, "figS18_gsea.csv")), reactome], ignore_index=True)
    cls = pd.read_csv(os.path.join(DATA_DIR, "figS18_c8_classes.csv"))[["term", "cell_class"]]
    H = g[g["statistic"] == "healthy_mean_gd"][["collection", "term", "nes", "fdr"]].rename(columns={"nes": "nes_h", "fdr": "fdr_h"})
    colors = {"blood / immune": "#C0392B", "epithelial / parenchymal": "#2E86C1", "endothelial": "#17A589",
              "stromal / mesenchymal": "#D68910", "neural / glial": "#7D3C98", "other": "0.6"}
    task_label = {"brca_crc": "Healthy vs BRCA + CRC", "pancan": "Healthy vs PANCAN"}
    fig, axes = plt.subplots(3, 2, figsize=(12, 16))
    for i, coll in enumerate(["hallmark", "c8_cell_type", "reactome"]):
        for j, t in enumerate(TASKS):
            C = g[(g["statistic"] == "cohen_d") & (g["contrast"] == t) & (g["collection"] == coll)][["term", "nes", "fdr"]]
            x = H[H["collection"] == coll].merge(C, on="term")
            ax = axes[i, j]
            sig = x["fdr"] < 0.05
            if coll == "c8_cell_type":
                x = x.merge(cls, on="term", how="left")
                for k, col in colors.items():
                    s = x[x["cell_class"] == k]
                    ax.scatter(s["nes_h"], s["nes"], s=10, color=col, alpha=0.6, label=k)
            else:
                ax.scatter(x.loc[~sig, "nes_h"], x.loc[~sig, "nes"], s=12, color="0.75")
                ax.scatter(x.loc[sig, "nes_h"], x.loc[sig, "nes"], s=16, color="0.2", label="FDR < 0.05 in cancer")
            # label the strongest significant same-direction sets
            same = x[sig & (np.sign(x["nes_h"]) == np.sign(x["nes"]))]
            for _, r in same.reindex(same["nes"].abs().sort_values(ascending=False).index[:8]).iterrows():
                lab = r["term"].replace("HALLMARK_", "").replace("REACTOME_", "").lower()[:38] if coll != "c8_cell_type" else ""
                if lab:
                    ax.annotate(lab, (r["nes_h"], r["nes"]), fontsize=6, xytext=(3, 2), textcoords="offset points")
            rho = spearmanr(x["nes_h"], x["nes"])[0]
            n_sig = int(sig.sum()); n_same = int((sig & (np.sign(x["nes_h"]) == np.sign(x["nes"]))).sum())
            ax.axhline(0, color="0.5", lw=0.7); ax.axvline(0, color="0.5", lw=0.7)
            ax.set_title(f"{coll} | {task_label[t]}\nSpearman {rho:+.2f}; {n_sig} sets at FDR < 0.05 in cancer, "
                         f"{n_sig - n_same} mirror the healthy pattern, {n_same} same direction", fontsize=8.5)
            ax.set_xlabel("NES on the healthy ranking (> 0: deep dip in healthy plasma)", fontsize=8.5)
            ax.set_ylabel("NES on cancer - healthy d (> 0: deeper dip in cancer)", fontsize=8.5)
            if j == 0:
                ax.legend(fontsize=7, frameon=False)
            print(f"  {coll:13s} {t:8s} rho {rho:+.2f}; FDR<0.05 in cancer {n_sig}, mirror {n_sig - n_same}, same direction {n_same}")
    out_dir = rp("src/figures/output", COHORT_VARIANT) if IS_VARIANT else rp("src/figures/output")
    plt.tight_layout(); fig.savefig(os.path.join(out_dir, "figS18_healthy_vs_cancer.png"), dpi=200)
    print("wrote", os.path.join(out_dir, "figS18_healthy_vs_cancer.png"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cores", type=int, default=4)
    ap.add_argument("--plot_only", action="store_true", help="reuse figS18_gsea_reactome.csv")
    args = ap.parse_args()
    p = os.path.join(DATA_DIR, "figS18_gsea_reactome.csv")
    reactome = pd.read_csv(p) if args.plot_only and os.path.exists(p) else run_reactome(args.cores)
    plot(reactome)


if __name__ == "__main__":
    main()
