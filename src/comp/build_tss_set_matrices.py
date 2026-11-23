#!/usr/bin/env python
"""TSS features aggregated over Hallmark pathways and chromosome arms (rebuttal R2.3 / R3.4).

Reviewer 2 (comment 3) suggests grouping TSSs by functional elements; Reviewer 3
(comment 4) asks whether the generic, locus-level TSS set is what matters. This
script turns the 14,221 model TSSs into set-level features, in two ways, for the
five fragmentomic features of Fig 4c (griffin_diff, rcov, mwh, fslr, fslr_coverage):

  averaged  the per-TSS feature values, per-sample row z-scored exactly as the
            classifier sees them (TSS blacklist applied, then z-score over the
            remaining TSSs), averaged over the TSSs of each set. No extraction.
  pooled    the feature computed once on the set as a whole, as in Fig 4c and the
            ATAC comparator: griffin_diff / rcov / mwh on the summed coverage profile
            of the set's TSS windows (needs the aggregate extraction, see `beds`),
            fslr / fslr_coverage as log2(summed short / summed long) over the set's
            TSSs, from the per-TSS _BIN.csv counts.

Sets (fixed in advance, label-free):
  hallmark  the 50 MSigDB Hallmark gene sets (same file as Fig S18), TSSs matched
            on gene symbol; a TSS can belong to several sets, most belong to none.
  arm       autosomal chromosome arms (p / q split at the ichorCNA hg38
            centromere table); a partition of the TSSs.
  genome    one set of all 14,221 TSSs (pooled only): the genome-wide aggregate of
            Figure 4c (sbatch slurm/extract_tss_sets.sh genome, no array needed).
Sets with fewer than MIN_SET_SIZE model TSSs are dropped.

Each (set type, mode, feature) becomes one predefined matrix per task, in the
row order of the task's baseline scalar table (full cohort, manifest order); the
no_deep exclusion is applied by model_hpc (--exclude_samples), as for the
covariate panels, so the outer folds are the frozen ones.

Steps (repository root):
  python src/comp/build_tss_set_matrices.py beds       # set definitions, bed files, BAM lists
  python src/comp/build_tss_set_matrices.py averaged   # averaged matrices + pooled fslr/fslr_coverage
  sbatch slurm/extract_tss_sets.sh hallmark; sbatch slurm/extract_tss_sets.sh arm   # pooled profiles
  python src/comp/build_tss_set_matrices.py pooled     # pooled griffin_diff / rcov / mwh
  ./slurm/submit_tss_set_panels.sh                     # the nested runs
Outputs: accessory_files/tss_sets_membership.csv, beds/tss_sets/<type>/<set>.bed,
input/tss_sets_<type>_beds.txt, input/cris_model_{bam,gc}.txt,
extracted_features/tss_sets/<type>_<mode>_<feature>[_pancan].{npy,metadata.csv,tf_names.csv}.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

TSS_TABLE = REPO / "src/figures/data/no_deep/figS17_tss_effects_brca_crc.csv"   # the 14,221 model TSSs (bed_row)
BED = REPO / "beds/biomart_10kb.bed"
BLACKLIST = REPO / "accessory_files/final_tss_indices_to_remove_hg38.npy"
GMT = REPO / "accessory_files/annotations/msigdb/h.all.v2024.1.Hs.symbols.gmt"
CENTROMERES = REPO / "accessory_files/annotations/GRCh38_centromere_acen.txt"
FEATURES_DIR = REPO / "extracted_features/biomart_10kb"
OUT = REPO / "extracted_features/tss_sets"
MEMBERSHIP = REPO / "accessory_files/tss_sets_membership.csv"
FEATURES = ["griffin_diff", "rcov", "mwh", "fslr", "fslr_coverage"]
PROFILE_FEATURES = ["griffin_diff", "rcov", "mwh"]          # pooled: need the summed coverage profile
SET_TYPES = ["hallmark", "arm", "genome"]
# genome (Figure 4c): one set holding all 14,221 model TSSs, i.e. the genome-wide
# aggregate of the submitted Fig 4c recomputed on the current TSS filter. Pooled mode only (the
# averaged z-score over all TSSs is zero by construction); one value per sample, evaluated as a
# training-oriented scalar by export_fig5.py, not through the nested panels.
POOLED_ONLY = {"genome"}
MIN_SET_SIZE = 10
TASKS = {"brca_crc": ("src/figures/data/fig3_baseline_scalars.csv", ""),
         "pancan": ("src/figures/data/fig4_baseline_scalars.csv", "_pancan")}
# BAM / GCfix lists the biomart features were extracted from (the recovery list holds seven
# samples that are not in cris_all.txt)
BAM_LISTS = [(REPO / "input/cris_all.txt", REPO / "input/cris_all_gc.txt"),
             (REPO / "input/cris_bam_recovery.txt", REPO / "input/cris_gc_recovery.txt")]


# ---------------------------------------------------------------------------- set definitions
def model_tss() -> pd.DataFrame:
    t = pd.read_csv(TSS_TABLE, usecols=["bed_row", "chrom", "start", "end", "gene"])
    assert len(t) == 14221 and t["bed_row"].is_unique
    return t.sort_values("bed_row").reset_index(drop=True)


def define_sets() -> pd.DataFrame:
    tss = model_tss()
    rows = []
    cen = pd.read_csv(CENTROMERES, sep="\t").set_index("Chr")
    mid = (tss["start"] + tss["end"]) // 2
    for chrom, idx in tss.groupby("chrom").groups.items():
        c = cen.loc[chrom]
        for i in idx:
            arm = "p" if mid[i] < c["Start"] else ("q" if mid[i] > c["End"] else None)
            if arm:
                rows.append(("arm", f"{chrom}{arm}", tss.at[i, "bed_row"], tss.at[i, "gene"]))
    by_gene = tss.assign(g=tss["gene"].astype(str).str.upper()).groupby("g")["bed_row"].apply(list).to_dict()
    with open(GMT) as fh:
        for line in fh:
            name, _, *genes = line.rstrip("\n").split("\t")
            for g in sorted({g.upper() for g in genes if g}):
                for br in by_gene.get(g, []):
                    rows.append(("hallmark", name, br, g))
    for br, g in zip(tss["bed_row"], tss["gene"]):
        rows.append(("genome", "all_tss", br, g))
    m = pd.DataFrame(rows, columns=["set_type", "set_name", "bed_row", "gene"]).drop_duplicates()
    size = m.groupby(["set_type", "set_name"])["bed_row"].transform("size")
    dropped = m[size < MIN_SET_SIZE].groupby(["set_type", "set_name"]).size()
    if len(dropped):
        print(f"  sets below {MIN_SET_SIZE} TSSs dropped: " + ", ".join(f"{n} ({k})" for (_, n), k in dropped.items()))
    return m[size >= MIN_SET_SIZE].reset_index(drop=True)


def load_membership() -> pd.DataFrame:
    assert MEMBERSHIP.exists(), f"{MEMBERSHIP} missing - run the 'beds' step first"
    return pd.read_csv(MEMBERSHIP)


def set_names(m: pd.DataFrame, set_type: str) -> list[str]:
    return sorted(m.loc[m["set_type"] == set_type, "set_name"].unique(), key=_arm_order if set_type == "arm" else str)


def _arm_order(name: str):
    return (int(name[3:-1]), name[-1])


def model_samples() -> list[str]:
    ids = pd.read_csv(REPO / TASKS["pancan"][0], usecols=["sample_id"])["sample_id"].tolist()
    brca = pd.read_csv(REPO / TASKS["brca_crc"][0], usecols=["sample_id"])["sample_id"].tolist()
    assert set(brca) <= set(ids), "BRCA/CRC samples must be a subset of the pan-cancer table"
    return ids


def step_beds() -> None:
    m = define_sets()
    MEMBERSHIP.parent.mkdir(parents=True, exist_ok=True)
    m.to_csv(MEMBERSHIP, index=False, lineterminator="\n")
    bed = pd.read_csv(BED, sep="\t", header=None)
    for st in SET_TYPES:
        d = REPO / "beds/tss_sets" / st
        d.mkdir(parents=True, exist_ok=True)
        names = set_names(m, st)
        paths = []
        for name in names:
            rows = sorted(m.loc[(m["set_type"] == st) & (m["set_name"] == name), "bed_row"])
            p = d / f"{name}.bed"
            bed.iloc[rows].to_csv(p, sep="\t", header=False, index=False, lineterminator="\n")
            paths.append(f"beds/tss_sets/{st}/{name}.bed")
        with open(REPO / f"input/tss_sets_{st}_beds.txt", "w", newline="\n") as fh:
            fh.write("\n".join(paths) + "\n")
        sizes = m[m["set_type"] == st].groupby("set_name").size()
        print(f"  {st}: {len(names)} sets, {sizes.min()}-{sizes.max()} TSSs (median {int(sizes.median())}), "
              f"{m.loc[m['set_type'] == st, 'bed_row'].nunique()} of 14,221 TSSs covered")
    # BAM / GC lists of the model samples only (the extraction does not need the other 130 records)
    ids = set(model_samples())
    pairs = {}
    for bam_list, gc_list in BAM_LISTS:
        bams = [l.strip() for l in open(bam_list) if l.strip()]
        gcs = [l.strip() for l in open(gc_list) if l.strip()]
        assert len(bams) == len(gcs), bam_list
        for b, g in zip(bams, gcs):
            pairs.setdefault(Path(b).name.split(".")[0], (b, g))
    missing = sorted(ids - set(pairs))
    assert not missing, f"model samples in no BAM list: {missing}"
    keep = [pairs[s] for s in sorted(ids)]
    for b, g in keep:
        assert Path(b).name.split(".")[0] == Path(g).name.split(".")[0]
    for path, col in [(REPO / "input/cris_model_bam.txt", 0), (REPO / "input/cris_model_gc.txt", 1)]:
        with open(path, "w", newline="\n") as fh:
            fh.write("\n".join(k[col] for k in keep) + "\n")
    print(f"wrote {MEMBERSHIP.relative_to(REPO)}, beds/tss_sets/, input/tss_sets_*_beds.txt, "
          f"input/cris_model_{{bam,gc}}.txt ({len(keep)} samples)")


# ---------------------------------------------------------------------------- per-sample values
def _sample_values(sid: str, m: pd.DataFrame, kept: np.ndarray, bed_genes: np.ndarray) -> dict:
    """Averaged values of the five features and pooled fslr / fslr_coverage, per set, for one sample."""
    d = pd.read_csv(FEATURES_DIR / f"{sid}.hg38.frag.tsv_BIN.csv",
                    usecols=["gene_name"] + FEATURES + ["absolute_short_fragments", "absolute_long_fragments",
                                                        "mean_coverage_short", "mean_coverage_long"])
    assert len(d) == len(bed_genes), f"{sid}: {len(d)} BIN rows, bed has {len(bed_genes)}"
    same = (d["gene_name"].to_numpy() == bed_genes) | (d["gene_name"].isna().to_numpy() & pd.isna(bed_genes))
    assert same.all(), f"{sid}: BIN rows do not follow the bed file"
    col_of = {br: j for j, br in enumerate(kept)}
    out = {}
    for f in FEATURES:                                   # row z-score over the blacklist-filtered TSSs, as model_hpc
        x = d[f].to_numpy(dtype=float)[kept]
        mu, sd = np.nanmean(x), np.nanstd(x)
        z = np.nan_to_num((x - mu) / (sd if sd > 0 else 1.0))
        for (st, name), br in m.groupby(["set_type", "set_name"])["bed_row"]:
            out[("averaged", st, f, name)] = float(z[[col_of[b] for b in br]].mean())
    for f, s_col, l_col in [("fslr", "absolute_short_fragments", "absolute_long_fragments"),
                            ("fslr_coverage", "mean_coverage_short", "mean_coverage_long")]:
        s, l = d[s_col].to_numpy(dtype=float), d[l_col].to_numpy(dtype=float)
        for (st, name), br in m.groupby(["set_type", "set_name"])["bed_row"]:
            tot_s, tot_l = np.nansum(s[br.to_numpy()]), np.nansum(l[br.to_numpy()])
            # Fig 4c / baseline_aggregate.py: log2(max(short, 1) / max(long, 1)); the floor never binds for these sets
            assert tot_s > 1 and tot_l > 1, f"{sid} {name} {f}: summed {tot_s:.2f} / {tot_l:.2f}"
            out[("pooled", st, f, name)] = float(np.log2(tot_s / tot_l))
    return out


def _sample_profile_values(sid: str, set_type: str, names: list[str]) -> dict:
    from extract_aggregate_features import (extract_single_sample_griffin_diff, extract_single_sample_mwh,
                                            extract_single_sample_rcov)
    fn = {"griffin_diff": extract_single_sample_griffin_diff, "rcov": extract_single_sample_rcov,
          "mwh": extract_single_sample_mwh}
    out = {}
    for name in names:
        p = REPO / f"extracted_features/tss_sets_{set_type}" / name / f"{sid}.hg38.frag.tsv_coverage_aggregate.npy"
        assert p.exists(), f"{p} missing - run slurm/extract_tss_sets.sh {set_type}"
        prof = np.load(p)
        with open(os.devnull, "w") as dn:                 # the Fig 4c extractors print every value
            old, sys.stdout = sys.stdout, dn
            try:
                for f in PROFILE_FEATURES:
                    out[("pooled", set_type, f, name)] = float(fn[f](prof))
            finally:
                sys.stdout = old
    return out


# ---------------------------------------------------------------------------- matrices
def write_matrices(values: pd.DataFrame, m: pd.DataFrame, modes_features: list[tuple[str, str]]) -> None:
    """values: index sample_id, columns MultiIndex (mode, set_type, feature, set_name)."""
    OUT.mkdir(parents=True, exist_ok=True)
    for task, (csv, suffix) in TASKS.items():
        sc = pd.read_csv(REPO / csv, usecols=["sample_id", "disease"])
        for st in SET_TYPES:
            names = set_names(m, st)
            for mode, f in modes_features:
                if st in POOLED_ONLY and mode != "pooled":
                    continue
                X = values.loc[sc["sample_id"], [(mode, st, f, n) for n in names]].to_numpy(dtype=float)
                assert np.isfinite(X).all(), f"{task} {st} {mode} {f}: non-finite values"
                stem = OUT / f"{st}_{mode}_{f}{suffix}"
                np.save(f"{stem}.npy", X)
                sc.to_csv(f"{stem}.metadata.csv", index=False, lineterminator="\n")
                pd.DataFrame({"feature": names}).to_csv(f"{stem}.tf_names.csv", index=False, lineterminator="\n")
        print(f"  {task}: {len(sc)} samples, {len(SET_TYPES) * len(modes_features)} matrices")


def step_averaged(cores: int) -> None:
    m = load_membership()
    ids = model_samples()
    remove = np.load(BLACKLIST)
    bed_genes = pd.read_csv(BED, sep="\t", header=None)[3].to_numpy()
    kept = np.delete(np.arange(len(bed_genes)), remove)
    assert set(m["bed_row"]) <= set(kept), "a model TSS is on the blacklist"
    print(f"reading {len(ids)} _BIN.csv files ({cores} cores) ...", flush=True)
    res = Parallel(n_jobs=cores, verbose=5)(delayed(_sample_values)(s, m, kept, bed_genes) for s in ids)
    values = pd.DataFrame(res, index=ids)
    values.columns = pd.MultiIndex.from_tuples(values.columns)
    OUT.mkdir(parents=True, exist_ok=True)
    values.to_pickle(OUT / "_cache_averaged.pkl")
    write_matrices(values, m, [("averaged", f) for f in FEATURES] + [("pooled", "fslr"), ("pooled", "fslr_coverage")])


def step_pooled(cores: int) -> None:
    m = load_membership()
    ids = model_samples()
    parts = []
    for st in SET_TYPES:
        names = set_names(m, st)
        print(f"{st}: {len(names)} pooled profiles x {len(ids)} samples ...", flush=True)
        parts.append(pd.DataFrame(Parallel(n_jobs=cores, verbose=5)(
            delayed(_sample_profile_values)(s, st, names) for s in ids), index=ids))
    values = pd.concat(parts, axis=1)
    values.columns = pd.MultiIndex.from_tuples(values.columns)
    write_matrices(values, m, [("pooled", f) for f in PROFILE_FEATURES])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["beds", "averaged", "pooled"])
    ap.add_argument("--cores", type=int, default=8)
    args = ap.parse_args()
    os.chdir(REPO)
    {"beds": step_beds, "averaged": lambda: step_averaged(args.cores),
     "pooled": lambda: step_pooled(args.cores)}[args.step]()


if __name__ == "__main__":
    main()
