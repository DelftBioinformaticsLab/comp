"""Derive the TSS exclusion filter from the panel of normals.

This replaces the lost derivation step for
accessory_files/final_tss_indices_to_remove_hg38.npy (and its hg19 twin,
consumed by model_hpc.py). Composition, as verified against the submitted
filter and figure S2c:

  blacklist    biomart windows overlapping the centromere exclusion list
               (precomputed 'overlap' column in
               beds/biomart_blacklist_overlaps.bed)                  [geometric]
  chrX         all chrX windows                                      [geometric]
  overlapping  another TSS (window center) within < 5,000 bp         [geometric]
  high-var     per-TSS std of mean_coverage across the panel of
               normals >= its 99.5th percentile, minus the
               geometric sets                                  [panel-dependent]

Provenance notes from the audit:
  * The submitted final filter (5,152) = geometric union (5,080) + 72
    high-variability windows. The exact derivation of the 72 was recorded
    nowhere, but a check showed they behave
    like SD >= 99.5th percentile on the historical panel (48 records,
    iloc[2:50], EE87934 phantom-matched to EE87930's file): 75 candidates,
    70/72 recovered, matching the submitted S2c legend
    (">=99.5th percentile"). A 95th-percentile rule selects ~10x more
    windows and does not describe the submitted set. This script applies the 99.5th-percentile rule cleanly
    on the clean 50-participant panel.
  * blacklist_and_std_outlier_indices.npy historically contained ONLY the
    blacklist set (the notebook unioned the blacklist with an
    *intersection* of the std outliers and the blacklist); it is
    regenerated with identical semantics for figure-S2 compatibility.
  * The hg19 files are exact gene-name mappings of their hg38 twins onto
    beds/biomart_10kb_hg19.bed (verified equal for the submitted files).
  * Panel = manifest role == 'panel_of_normals': 50 unique healthy
    participants, base sequencing runs only (no replicates), loaded by exact ID; a missing feature file is an error.

The submitted originals are backed up once to
accessory_files/submitted_backup/ before overwriting, and a delta report
is printed.  Run:  python src/comp/derive_tss_filter.py
"""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
ACC = REPO / "accessory_files"
BACKUP_DIR = ACC / "submitted_backup"

MANIFEST = ACC / "cris_manifest.csv"
BIOMART_BED = REPO / "beds" / "biomart_10kb.bed"
BIOMART_HG19_BED = REPO / "beds" / "biomart_10kb_hg19.bed"
BLACKLIST_OVERLAPS_BED = REPO / "beds" / "biomart_blacklist_overlaps.bed"
FEATURES_DIR = REPO / "extracted_features" / "biomart_10kb"

OUTPUTS = {
    "base": ACC / "blacklist_and_std_outlier_indices.npy",
    "chrx": ACC / "blacklist_and_std_outlier_and_chrX_indices.npy",
    "chrx_hg19": ACC / "blacklist_and_std_outlier_and_chrX_indices_hg19.npy",
    "final": ACC / "final_tss_indices_to_remove_hg38.npy",
    "final_hg19": ACC / "final_tss_indices_to_remove_hg19.npy",
}

STD_PERCENTILE = 0.995
# 0.995 per that check:
# the submitted 72-window set behaves like SD >= 99.5th percentile
# (historical panel: 75 candidates, 70/72 recovered; the submitted S2c
# legend also said ">=99.5th percentile"), NOT a 95th percentile, which
# selects ~10x more windows. On the clean
# 50-participant panel this yields ~70 extra windows and a final removal
# set of ~5,150 (submitted: 5,152) - submission-faithful in scale.
TSS_DISTANCE_BP = 5000     # S2c "(<5kb)": another TSS center within 5,000 bp


def load_panel_matrix(panel_ids):
    """Exact-ID loading of the panel's mean_coverage columns; raises on a
    missing file (no fuzzy matching — that is how phantoms happened)."""
    missing = [i for i in panel_ids
               if not (FEATURES_DIR / f"{i}.hg38.frag.tsv_BIN.csv").exists()]
    if missing:
        raise FileNotFoundError(f"panel feature files missing: {missing}")
    cols = [pd.read_csv(FEATURES_DIR / f"{i}.hg38.frag.tsv_BIN.csv",
                        usecols=["mean_coverage"])["mean_coverage"].to_numpy()
            for i in panel_ids]
    return np.vstack(cols)


def main() -> None:
    biomart = pd.read_csv(BIOMART_BED, sep="\t", header=None,
                          names=["chrom", "start", "end", "name", "strand", "GC"])
    assert len(biomart) == 19376, f"expected 19,376 windows, got {len(biomart)}"

    # --- geometric sets ----------------------------------------------------
    bl_bed = pd.read_csv(BLACKLIST_OVERLAPS_BED, skiprows=1, sep="\t",
                         names=["chrom", "start", "end", "name",
                                "overlap", "overlap_encode"])
    assert len(bl_bed) == len(biomart)
    blacklist_idx = set(np.where(bl_bed["overlap"] > 0)[0].tolist())

    chrx_idx = set(biomart.index[biomart["chrom"] == "chrX"].tolist())

    overlap_idx = set()
    for _, g in biomart.groupby("chrom"):
        g = g.sort_values("start")
        idx = g.index.to_numpy()
        tss = (g["start"] + (g["end"] - g["start"]) // 2).to_numpy()
        d = np.diff(tss)
        for i in np.where(d < TSS_DISTANCE_BP)[0]:
            overlap_idx.add(int(idx[i]))
            overlap_idx.add(int(idx[i + 1]))

    geometric = blacklist_idx | chrx_idx | overlap_idx
    print(f"geometric sets: blacklist {len(blacklist_idx)}, chrX {len(chrx_idx)}, "
          f"overlapping(<{TSS_DISTANCE_BP/1000:g}kb) {len(overlap_idx)}, "
          f"union {len(geometric)}")

    # --- panel-dependent high-variability set ------------------------------
    manifest = pd.read_csv(MANIFEST)
    panel = manifest[manifest["role"] == "panel_of_normals"]
    panel_ids = panel["ID"].tolist()
    assert len(panel_ids) == 50 and panel["participant_id"].nunique() == 50, \
        "panel must be 50 unique participants — rerun build_cris_manifest.py"
    assert not panel["replicate_second_run"].any()

    mat = load_panel_matrix(panel_ids)
    print(f"panel matrix: {mat.shape} (samples x TSS windows), "
          f"from manifest role == 'panel_of_normals'")

    std = np.nanstd(mat, axis=0)
    thr = pd.Series(std).quantile(STD_PERCENTILE)
    outlier_std = set(np.where(std >= thr)[0].tolist())
    high_var = outlier_std - geometric
    print(f"std >= {STD_PERCENTILE:.1%} percentile ({thr:.4f}): "
          f"{len(outlier_std)} windows; {len(high_var)} outside the geometric sets")

    # --- assemble ----------------------------------------------------------
    new = {
        "base": blacklist_idx,                       # historical content: blacklist only
        "chrx": blacklist_idx | chrx_idx | high_var,  # blacklist + std + chrX
        "final": geometric | high_var,
    }

    hg19 = pd.read_csv(BIOMART_HG19_BED, sep="\t", header=None,
                       names=["chrom", "start", "end", "name", "strand", "gc"])
    for src, dst in (("chrx", "chrx_hg19"), ("final", "final_hg19")):
        names = set(biomart.loc[sorted(new[src]), "name"])
        new[dst] = set(hg19.index[hg19["name"].isin(names)].tolist())

    # --- backup + write + delta report -------------------------------------
    BACKUP_DIR.mkdir(exist_ok=True)
    print()
    for key, path in OUTPUTS.items():
        old = set(np.load(path).tolist()) if path.exists() else None
        if old is not None:
            bak = BACKUP_DIR / path.name
            if not bak.exists():
                shutil.copy2(path, bak)
        arr = np.array(sorted(new[key]), dtype=np.int64)
        np.save(path, arr)
        if old is None:
            print(f"{path.name}: NEW, {len(arr)} indices")
        elif old == new[key]:
            print(f"{path.name}: unchanged ({len(arr)} indices)")
        else:
            print(f"{path.name}: {len(old)} -> {len(arr)} "
                  f"(+{len(new[key] - old)} added, -{len(old - new[key])} removed)")

    kept = 19376 - len(new["final"])
    print(f"\nretained TSS windows (hg38): {kept} "
          f"(submitted filter retained {19376 - 5152})")
    print(f"originals preserved in {BACKUP_DIR}")


if __name__ == "__main__":
    main()
