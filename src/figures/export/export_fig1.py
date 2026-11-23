"""Source data for Figure 1: a) normalized coverage profiles at housekeeping (HK) and
unexpressed (PAU) TSSs across the panel of normals; b) per-locus mean coverage in the two
gene sets; c) per-sample Pearson r of each feature's gene-group means with Esfahani et al.
expression (TPM[log2]), for all TSSs and QC-filtered TSSs.

Panel of normals: the 50 participants with role panel_of_normals in
accessory_files/cris_manifest.csv. Panel a needs the per-locus coverage arrays
(features.py --coverage); samples without one are left out of panel a only.
Writes to src/figures/data/: fig1a_coverage_profiles.csv, fig1b_locus_coverage.csv,
fig1c_expression_correlations.csv.

Run from the repository root:
    python src/figures/export/export_fig1.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(__file__))
from _common import DATA_DIR, rp  # noqa: E402

FEATURES_DIR = "extracted_features/biomart_10kb"
MANIFEST = "accessory_files/cris_manifest.csv"
BIOMART_BED = "beds/biomart_10kb.bed"
PAU_BED = "beds/pau.bed"
HK_BED = "beds/hk_genes.bed"
FINAL_NPY = "accessory_files/final_tss_indices_to_remove_hg38.npy"
ESFAHANI_XLSX = "accessory_files/esfahani_supplement.xlsx"
PON_N = 50
WINDOW = 10000
TSS_FEATURES = ["fslr", "fslr_coverage", "griffin_diff", "rcov", "mwh"]


def load_bed_genes(path):
    bed = pd.read_csv(rp(path), sep="\t", header=None)
    return bed[3].unique()


def tss_filter_genes(biomart):
    """Gene names of the windows removed by the TSS quality filter."""
    removed = np.load(rp(FINAL_NPY))
    return set(biomart.iloc[removed]["gene_name"].unique())


def esfahani_groups(biomart, blacklisted_genes):
    """Esfahani et al. PBMC gene groups mapped to biomart window indices."""
    esf = pd.read_excel(rp(ESFAHANI_XLSX), sheet_name="Supplementary Table 2",
                        header=6, usecols=["Gene group", "TPM [log2]"])
    groups = []
    for _, row in esf.iterrows():
        genes = [g.strip() for g in row["Gene group"].split("|")]
        all_idx, unfiltered_idx, n_filtered = [], [], 0
        for gene in genes:
            idx = biomart.index[biomart["gene_name"] == gene].tolist()
            if idx:
                all_idx.extend(idx)
                if gene in blacklisted_genes:
                    n_filtered += 1
                else:
                    unfiltered_idx.extend(idx)
        if all_idx:
            groups.append({"tpm": row["TPM [log2]"], "n_filtered": n_filtered,
                           "all_idx": all_idx, "unfiltered_idx": unfiltered_idx})
    return groups


def expression_correlations(feature_arrays, sample_ids, groups):
    """Per-sample Pearson r of gene-group feature means vs TPM[log2] (Fig 1c).
    "all": every mapped group, all loci; "qc": groups with <=1 filtered gene,
    non-removed loci only."""
    rows = []
    sel = [g for g in groups if g["n_filtered"] <= 1 and g["unfiltered_idx"]]
    for feat, fa in feature_arrays.items():
        for label, gs, key in [("all", groups, "all_idx"), ("qc", sel, "unfiltered_idx")]:
            tpm_v = np.array([g["tpm"] for g in gs])
            means = np.vstack([np.nanmean(fa[:, g[key]], axis=1) for g in gs])
            for si in range(means.shape[1]):
                fv = means[:, si]
                mask = ~(np.isnan(tpm_v) | np.isnan(fv))
                r = pearsonr(tpm_v[mask], fv[mask])[0] if mask.sum() >= 3 else np.nan
                rows.append({"sample_index": si, "sample_id": sample_ids[si],
                             "feature": feat, "tss_set": label, "pearson_r": r,
                             "n_groups": int(mask.sum())})
    return pd.DataFrame(rows)


def coverage_path(sid):
    p = Path(rp(f"{FEATURES_DIR}/{sid}.hg38.frag.tsv_coverage.npy"))
    if p.exists():
        return p
    alt = Path(rp(f"{FEATURES_DIR}/{sid}_coverage.npy"))
    return alt if alt.exists() else None


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

    manifest = pd.read_csv(rp(MANIFEST))
    pon_ids = manifest.loc[manifest["role"] == "panel_of_normals", "ID"].tolist()
    assert len(pon_ids) == PON_N, f"manifest panel has {len(pon_ids)} records, expected {PON_N}"
    assert manifest.loc[manifest["role"] == "panel_of_normals", "participant_id"].is_unique
    for sid in pon_ids:
        assert Path(rp(f"{FEATURES_DIR}/{sid}.hg38.frag.tsv_BIN.csv")).exists(), f"no BIN.csv for {sid}"
    cov_ids = [sid for sid in pon_ids if coverage_path(sid) is not None]
    no_cov = sorted(set(pon_ids) - set(cov_ids))
    print(f"PoN: {len(pon_ids)} participants (manifest); coverage arrays for {len(cov_ids)}"
          + (f"; panel a leaves out {no_cov} (run features.py --coverage)" if no_cov else ""))

    biomart = pd.read_csv(rp(BIOMART_BED), sep="\t", header=None,
                          names=["chrom", "start", "end", "gene_name", "strand", "GC"])
    pau_genes = load_bed_genes(PAU_BED)
    hk_genes = load_bed_genes(HK_BED)
    removed = np.load(rp(FINAL_NPY))
    removed_set = set(removed.tolist())
    blacklisted_genes = tss_filter_genes(biomart)

    def gene_indices(genes):
        idx = biomart.index[biomart["gene_name"].isin(set(genes))].to_numpy()
        return np.array([i for i in idx if i not in removed_set])

    pau_idx = gene_indices(pau_genes)
    hk_idx = gene_indices(hk_genes)
    print(f"loci after QC filter: PAU {len(pau_idx)}, HK {len(hk_idx)} "
          f"(panel legend: 529 / 1423)")

    # ---------------------------------------------------- panel a: profiles
    pau_profiles, hk_profiles = [], []
    locus_rows = []
    hk_set, pau_set = set(hk_genes), set(pau_genes)
    bins = {}
    for n, sid in enumerate(pon_ids, 1):
        if sid in cov_ids:
            cov = np.load(coverage_path(sid), mmap_mode="r")
        else:
            cov = None
        for indices, sink in ([] if cov is None else [(pau_idx, pau_profiles), (hk_idx, hk_profiles)]):
            agg = np.asarray(cov[np.sort(indices), :]).sum(axis=0)
            if agg.shape[0] > WINDOW:  # center-trim (11001 -> 10000)
                start = (agg.shape[0] - WINDOW) // 2
                agg = agg[start:start + WINDOW]
            mean_cov = agg.mean()
            sink.append(agg / mean_cov if mean_cov > 0 else agg)
        del cov

        # panel b: per-locus mean coverage; drop loci below the 1st percentile of
        # positive coverage among HK+PAU loci
        bin_df = pd.read_csv(rp(f"{FEATURES_DIR}/{sid}.hg38.frag.tsv_BIN.csv"),
                             usecols=["gene_name", "mean_coverage"] + TSS_FEATURES)
        bins[sid] = bin_df[TSS_FEATURES].to_numpy()
        bl_mask = bin_df["gene_name"].isin(blacklisted_genes)
        hk_mask = bin_df["gene_name"].isin(hk_set) & ~bl_mask
        pau_mask = bin_df["gene_name"].isin(pau_set) & ~bl_mask
        all_cov = bin_df.loc[hk_mask | pau_mask, "mean_coverage"].dropna()
        min_thr = all_cov[all_cov > 0].quantile(0.01)
        cov_mask = bin_df["mean_coverage"] > min_thr
        for gs, mask in [("HK", hk_mask), ("PAU", pau_mask)]:
            for v in bin_df.loc[mask & cov_mask, "mean_coverage"].dropna():
                locus_rows.append({"sample_id": sid, "gene_set": gs, "mean_coverage": v})
        print(f"  [{n}/{len(pon_ids)}] {sid} done")

    x = np.arange(-WINDOW // 2, WINDOW // 2)
    rows = []
    for gs, profiles in [("HK", hk_profiles), ("PAU", pau_profiles)]:
        stack = np.array(profiles)
        rows.append(pd.DataFrame({"position": x, "gene_set": gs,
                                  "mean_normalized_coverage": stack.mean(axis=0),
                                  "sd_normalized_coverage": stack.std(axis=0),
                                  "n_samples": len(stack)}))
    prof = pd.concat(rows, ignore_index=True)
    prof.to_csv(os.path.join(DATA_DIR, "fig1a_coverage_profiles.csv"), index=False)
    print(f"wrote fig1a_coverage_profiles.csv ({len(prof)} rows, n_samples={len(cov_ids)}, "
          f"n_hk_tss={len(hk_idx)}, n_pau_tss={len(pau_idx)})")

    locus = pd.DataFrame(locus_rows)
    locus.to_csv(os.path.join(DATA_DIR, "fig1b_locus_coverage.csv"), index=False)
    print(f"wrote fig1b_locus_coverage.csv ({len(locus)} rows)")

    # ------------------------------------------------- panel c: correlations
    print("panel c: loading feature matrices for the PoN...")
    feature_arrays = {}
    for j, feat in enumerate(TSS_FEATURES):   # columns read once per sample in the loop above
        feature_arrays[feat] = np.vstack([bins[sid][:, j] for sid in pon_ids])
        print(f"  {feat}: {feature_arrays[feat].shape}")

    groups = esfahani_groups(biomart, blacklisted_genes)
    print(f"panel c: {len(groups)} Esfahani gene groups mapped")
    corr = expression_correlations(feature_arrays, pon_ids, groups)
    corr.to_csv(os.path.join(DATA_DIR, "fig1c_expression_correlations.csv"), index=False)
    print("wrote fig1c_expression_correlations.csv")
    print("\npanel c summary (manuscript Fig 1c values for comparison):")
    for feat in TSS_FEATURES:
        for label in ["all", "qc"]:
            v = corr[(corr.feature == feat) & (corr.tss_set == label)]["pearson_r"].dropna()
            print(f"  {feat:<14} {label:<4} r = {v.mean():+.3f} ± {v.std(ddof=0):.3f}")


if __name__ == "__main__":
    main()
