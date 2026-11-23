"""Supplementary table: class of cell type of every MSigDB C8 signature tested, with its enrichment.

One row per C8 signature tested in Figure 6d (15 to 500 genes among the 14,221 TSSs): name parts,
class of cell type (`c8_cell_class` in export_figS18_gene_sets.py, keyword lists on the name only),
number of TSSs, and per task the z and BH q of the matched test (figS18_gsea_matched_10k.csv).

Reads data[/no_deep]/figS18_c8_classes.csv and figS18_gsea_matched_10k.csv; writes
data[/no_deep]/tableS_c8_classes.csv (Supplementary Table S6; workbook by build_supplementary_tables.py).

Run: COHORT_VARIANT=no_deep python src/figures/export/export_tableS_c8_classes.py
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from _common import DATA_DIR  # noqa: E402  (data/no_deep for the default cohort variant)

OUT_DIR = DATA_DIR
CLASS_ORDER = ["blood / immune", "epithelial / parenchymal", "neural / glial", "stromal / mesenchymal", "endothelial", "other"]
TASKS = {"brca_crc": "BRCA/CRC versus healthy", "pancan": "pan-cancer versus healthy"}


def main():
    classes = pd.read_csv(os.path.join(DATA_DIR, "figS18_c8_classes.csv"))
    la = pd.read_csv(os.path.join(DATA_DIR, "figS18_gsea_matched_10k.csv"))
    la = la[(la["matching"] == "gc_coverage_dip") & (la["collection"] == "c8_cell_type")]
    assert set(la["contrast"]) == set(TASKS), sorted(la["contrast"].unique())

    wide = None
    for task in TASKS:
        t = la[la["contrast"] == task][["term", "n_genes_matched", "rank_auc_signed_d_z", "rank_auc_signed_d_q"]]
        t = t.rename(columns={"rank_auc_signed_d_z": f"z_{task}", "rank_auc_signed_d_q": f"q_{task}",
                              "n_genes_matched": f"n_tss_{task}"})
        wide = t if wide is None else wide.merge(t, on="term", how="outer")
    assert (wide["n_tss_brca_crc"] == wide["n_tss_pancan"]).all()
    wide = wide.drop(columns="n_tss_pancan").rename(columns={"n_tss_brca_crc": "n_tss"})

    tab = wide.merge(classes, on="term", how="left", validate="one_to_one")
    assert tab["cell_class"].notna().all(), "a tested signature has no class"
    for task in TASKS:
        sig = tab[f"q_{task}"] < 0.05
        tab[f"direction_{task}"] = ""
        tab.loc[sig & (tab[f"z_{task}"] < 0), f"direction_{task}"] = "less active in cancer"
        tab.loc[sig & (tab[f"z_{task}"] > 0), f"direction_{task}"] = "more active in cancer"
    tab["cell_class"] = pd.Categorical(tab["cell_class"], CLASS_ORDER, ordered=True)
    tab = tab.sort_values(["cell_class", "term"]).rename(columns={
        "term": "signature", "study_tissue": "study_tissue_prefix", "cell_type": "cell_type_part",
        "cell_class": "class_of_cell_type"})
    cols = ["signature", "study_tissue_prefix", "cell_type_part", "class_of_cell_type", "n_tss",
            "z_brca_crc", "q_brca_crc", "direction_brca_crc", "z_pancan", "q_pancan", "direction_pancan"]
    tab = tab[cols]

    # count checks
    counts = tab["class_of_cell_type"].value_counts().reindex(CLASS_ORDER).astype(int).to_dict()
    print("signatures per class:", counts, "total", len(tab))
    assert len(tab) == 738, len(tab)
    for task in TASKS:
        n = int((tab[f"q_{task}"] < 0.05).sum())
        less = tab[tab[f"direction_{task}"] == "less active in cancer"]
        more = tab[tab[f"direction_{task}"] == "more active in cancer"]
        print(f"{task}: {n} enriched; less active {len(less)} (blood / immune {int((less['class_of_cell_type'] == 'blood / immune').sum())}), "
              f"more active {len(more)} (other than blood / immune {int((more['class_of_cell_type'] != 'blood / immune').sum())})")
    assert int((tab["q_brca_crc"] < 0.05).sum()) == 148 and int((tab["q_pancan"] < 0.05).sum()) == 158

    os.makedirs(OUT_DIR, exist_ok=True)
    csv = os.path.join(OUT_DIR, "tableS_c8_classes.csv")
    tab.to_csv(csv, index=False, lineterminator="\n", float_format="%.4g")
    print("wrote", csv)


if __name__ == "__main__":
    main()
