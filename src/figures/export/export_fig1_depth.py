"""Source data for the Figure 1 depth companion (supplementary figure sup_cov_esfahani):
  a) the Fig 1c correlation (per-sample Pearson r of Esfahani et al. PBMC gene-group
     expression with each feature) for the 50 panel-of-normals samples at native depth and
     downsampled to 1x and 0.1x (extracted_features/downsampled/, seed 42);
  b) per-sample depth of the cohort, the panel of normals and the 30 excluded deep-sequenced donors.

Writes src/figures/data/fig1_depth_correlations.csv and fig1_depth_cohort_coverage.csv.
The native-depth rows are asserted to equal fig1c_expression_correlations.csv when it exists.

Run from the repository root (the panel of normals does not depend on the cohort variant):
    COHORT_VARIANT=all python src/figures/export/export_fig1_depth.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from _common import DATA_DIR, rp  # noqa: E402
from export_fig1 import (BIOMART_BED, MANIFEST, PON_N, TSS_FEATURES,  # noqa: E402
                         esfahani_groups, expression_correlations, tss_filter_genes)

COVERAGE_SUMMARY = "results/cris_coverage_summary.csv"
DEEP_LIST = "accessory_files/cris_exclude_deep_healthy.txt"
DEPTH_DIRS = {
    "native": "extracted_features/biomart_10kb",
    "1x": "extracted_features/downsampled/cov1x_seed42/biomart_10kb",
    "0.1x": "extracted_features/downsampled/cov0.1x_seed42/biomart_10kb",
}


def read_deep_ids():
    ids = []
    for line in open(rp(DEEP_LIST)):
        tok = line.split("#")[0].strip()
        if tok:
            ids.append(tok.split()[0])
    return ids


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    manifest = pd.read_csv(rp(MANIFEST))
    cov = pd.read_csv(rp(COVERAGE_SUMMARY)).set_index("sample_name")["mean_coverage"]
    deep = set(read_deep_ids())
    assert len(deep) == 30, f"expected 30 deep-sequenced donors, got {len(deep)}"

    pon_ids = manifest.loc[manifest["role"] == "panel_of_normals", "ID"].tolist()
    assert len(pon_ids) == PON_N
    assert not deep & set(pon_ids)
    pon_cov = cov.reindex(pon_ids)
    assert pon_cov.notna().all()
    print(f"panel of normals: {len(pon_ids)} samples, native depth median {pon_cov.median():.2f}x "
          f"(range {pon_cov.min():.2f}-{pon_cov.max():.2f}); "
          f"{(pon_cov < 1).sum()} below the 1x target")

    biomart = pd.read_csv(rp(BIOMART_BED), sep="\t", header=None,
                          names=["chrom", "start", "end", "gene_name", "strand", "GC"])
    groups = esfahani_groups(biomart, tss_filter_genes(biomart))
    print(f"{len(groups)} Esfahani gene groups mapped")

    frames = []
    for depth, d in DEPTH_DIRS.items():
        mats = {f: [] for f in TSS_FEATURES}
        for sid in pon_ids:
            b = pd.read_csv(rp(f"{d}/{sid}.hg38.frag.tsv_BIN.csv"), usecols=["gene_name"] + TSS_FEATURES)
            assert b["gene_name"].reset_index(drop=True).equals(biomart["gene_name"].reset_index(drop=True)), \
                f"{depth} {sid}: window order differs from {BIOMART_BED}"
            for f in TSS_FEATURES:
                mats[f].append(b[f].to_numpy(float))
        fa = {f: np.vstack(v) for f, v in mats.items()}
        c = expression_correlations(fa, pon_ids, groups)
        c.insert(0, "depth", depth)
        c["native_coverage"] = c["sample_id"].map(pon_cov)
        frames.append(c)
        zero = {f: float(np.mean(fa[f] == 0)) for f in TSS_FEATURES}
        print(f"  {depth:>6}: " + ", ".join(
            f"{f} r={c[(c.feature == f) & (c.tss_set == 'qc')].pearson_r.mean():+.3f} (zeros {zero[f]:.0%})"
            for f in TSS_FEATURES))
    out = pd.concat(frames, ignore_index=True)

    ref_p = os.path.join(DATA_DIR, "fig1c_expression_correlations.csv")
    if os.path.exists(ref_p):
        ref = pd.read_csv(ref_p)
        nat = out[out.depth == "native"].merge(ref, on=["sample_id", "feature", "tss_set"], suffixes=("", "_fig1c"))
        assert len(nat) == len(ref) and np.allclose(nat.pearson_r, nat.pearson_r_fig1c, equal_nan=True), \
            "native-depth correlations differ from Fig 1c"
        print("native depth matches fig1c_expression_correlations.csv")
    out.to_csv(os.path.join(DATA_DIR, "fig1_depth_correlations.csv"), index=False)
    print(f"wrote fig1_depth_correlations.csv ({len(out)} rows)")

    # panel b: depth of the revised cohort, the panel of normals and the excluded deep donors
    rows = []
    for _, r in manifest.iterrows():
        sid = r["ID"]
        if sid in deep:
            group = "deep_excluded"
        elif r["role"] == "panel_of_normals":
            group = "panel_of_normals"
        elif r["role"] == "model" and bool(r["in_task_pancan"]):
            group = "cohort"
        else:
            continue
        rows.append({"sample_id": sid, "group": group, "disease": r["disease"],
                     "mean_coverage": cov.get(sid, np.nan)})
    cc = pd.DataFrame(rows)
    assert cc["mean_coverage"].notna().all(), cc[cc.mean_coverage.isna()]
    assert (cc.group == "deep_excluded").sum() == 30
    cc.to_csv(os.path.join(DATA_DIR, "fig1_depth_cohort_coverage.csv"), index=False)
    print(f"wrote fig1_depth_cohort_coverage.csv ({len(cc)} rows)")
    print(cc.groupby("group")["mean_coverage"].describe().round(2).to_string())


if __name__ == "__main__":
    main()
