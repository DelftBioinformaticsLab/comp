"""Source data for the cohort comparison (Supplementary Figure S2) and the cohort characteristics
table (Supplementary Table S2): Cristiano vs Jiang technical differences.

Writes to src/figures/data/<COHORT_VARIANT>/:
  figS12_cohort_scalars.csv        per sample: coverage, fragment count, fragment-length
                                   statistics, mean GC, sex from the reads, ichorCNA TF, metadata
  figS12_fraglen_distributions.csv per-sample fragment-length density, 100-220 bp
  tableS_cohorts.csv / .tex        cohort table (accessory_files/cohort_protocols.csv + computed
                                   median [min-max])

Inputs: accessory_files/{cris,jiang}_manifest.csv, results/cris_coverage_summary.csv,
results/jiang_coverage/jiang_mean_coverage.csv, results/{cris,jiang}_sex/*_sex.csv,
extracted_features/biomart_10kb/<ID>.hg38.frag.tsv_gc_fragment_length_distribution.npy and
_BIN.csv, ichorCNA/ichorCNA_summary.tsv and ichorCNA_jiang/ichorCNA_summary.tsv.

Run from the repository root:  python src/figures/export/export_figS12_cohorts.py
  --skip_gc       leave mean_gc empty for samples without a cached value (the
                  Jiang _BIN.csv reads are the one slow step: 225 x ~45 MB)
  --recompute_gc  ignore cached values
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from _common import DATA_DIR, excluded_sample_ids, resolve_sex_call, rp  # noqa: E402

BIOMART_DIR = "extracted_features/biomart_10kb"
COVERAGE = {"cris": "results/cris_coverage_summary.csv",
            "jiang": "results/jiang_coverage/jiang_mean_coverage.csv"}
SEX = {"cris": "results/cris_sex/cris_sex.csv", "jiang": "results/jiang_sex/jiang_sex.csv"}
# ichorCNA with the Cristiano 50-sample panel of normals for both cohorts
ICHOR_TSV = {"Cristiano": "ichorCNA/ichorCNA_summary.tsv", "Jiang": "ichorCNA_jiang/ichorCNA_summary.tsv"}
GC_CACHE = ["src/figures/data/figS12_cohort_scalars.csv",
            "src/figures/data/fig4_baseline_scalars.csv",
            "src/figures/data/fig3_baseline_scalars.csv"]
LENGTHS = np.arange(100, 221)
SHORT = LENGTHS <= 150          # short/long cut as in the pipeline's FSLR (100-150 vs 151-220)

GROUP = {
    "cris": lambda d: "Healthy" if d == "Healthy" else (
        "BRCA+CRC" if d in ("Breast cancer", "Colorectal cancer") else "Other cancer"),
    "jiang": lambda d: {"Healthy": "Healthy", "Hepatitis B": "Chronic hepatitis B",
                        "Cirrhosis": "HBV cirrhosis", "Liver cancer": "HCC"}[d],
}


def load_samples() -> pd.DataFrame:
    frames = []
    for cohort in ["cris", "jiang"]:
        m = pd.read_csv(rp(f"accessory_files/{cohort}_manifest.csv"))
        m = m[m["role"] == "model"].copy()
        meta_cols = ["readlen", "instrument", "seq_layout"]
        if not set(meta_cols) <= set(m.columns):   # cris manifest lacks them: take the FinaleDB export
            raw = pd.read_csv(rp(f"accessory_files/{cohort}_metadata.csv")).drop_duplicates("ID")
            m = m.merge(raw[["ID"] + meta_cols], on="ID", how="left")
        t = pd.DataFrame({
            "cohort": "Cristiano" if cohort == "cris" else "Jiang",
            "sample_id": m["ID"].to_numpy(), "sample_name": m["sample_name"].to_numpy(),
            "disease": m["disease"].to_numpy(),
            "group": m["disease"].map(GROUP[cohort]).to_numpy(),
            "is_cancer": (~m["disease"].isin(["Healthy", "Hepatitis B", "Cirrhosis"])).astype(int).to_numpy(),
            "readlen": m["readlen"].to_numpy(), "instrument": m["instrument"].to_numpy(),
            "seq_layout": m["seq_layout"].to_numpy(),
            "total_fragments": m["fragNum"].astype(float).to_numpy(),
        })
        cov = pd.read_csv(rp(COVERAGE[cohort]))
        t["mean_coverage"] = t["sample_id"].map(dict(zip(cov["sample_name"], cov["mean_coverage"])))
        sex = resolve_sex_call(pd.read_csv(rp(SEX[cohort])))
        call = dict(zip(sex["sample_name"], sex["sex_call_resolved"]))
        t["sex_female_bam"] = t["sample_id"].map(call).map({"female": 1.0, "male": 0.0})
        frames.append(t)
    t = pd.concat(frames, ignore_index=True)
    excluded = set(excluded_sample_ids())
    if excluded:
        n0 = len(t)
        t = t[~t["sample_id"].isin(excluded)].reset_index(drop=True)
        print(f"  cohort variant: {n0 - len(t)} excluded sample(s) removed")
    assert t["sample_id"].is_unique
    assert t["mean_coverage"].notna().all(), "coverage missing for some samples"
    assert t["sex_female_bam"].notna().all(), "sex call missing or unassigned for some samples"
    return t


def add_fragment_lengths(t: pd.DataFrame):
    """Length-marginal statistics and per-sample density (100-220 bp). Only the length
    marginal of the GC x length histogram is used."""
    stats, dens_rows = [], []
    for sid in t["sample_id"]:
        a = np.load(rp(BIOMART_DIR, f"{sid}.hg38.frag.tsv_gc_fragment_length_distribution.npy"))
        assert a.shape == (101, 121), f"{sid}: unexpected histogram shape {a.shape}"
        h = a.sum(axis=0).astype(float)
        assert h.sum() > 0
        p = h / h.sum()
        cdf = np.cumsum(p)
        stats.append({
            "fragment_length_mean": float((p * LENGTHS).sum()),
            "fragment_length_median": float(LENGTHS[np.searchsorted(cdf, 0.5)]),
            "fragment_length_mode": float(LENGTHS[np.argmax(p)]),
            "fragment_length_iqr": float(LENGTHS[np.searchsorted(cdf, 0.75)] - LENGTHS[np.searchsorted(cdf, 0.25)]),
            "short_long_ratio": float(p[SHORT].sum() / p[~SHORT].sum()),
        })
        dens_rows.append(p)
    for k in stats[0]:
        t[k] = [s[k] for s in stats]
    dens = pd.DataFrame(np.vstack(dens_rows), columns=[str(l) for l in LENGTHS])
    dens.insert(0, "sample_id", t["sample_id"].to_numpy())
    dens.insert(0, "group", t["group"].to_numpy())
    dens.insert(0, "cohort", t["cohort"].to_numpy())
    return dens


def add_mean_gc(t: pd.DataFrame, skip: bool, recompute: bool) -> None:
    cached = {}
    if not recompute:
        for c in GC_CACHE:
            if os.path.exists(rp(c)):
                old = pd.read_csv(rp(c))
                if "mean_gc" in old.columns:
                    cached.update({k: v for k, v in zip(old["sample_id"], old["mean_gc"]) if pd.notna(v)})
    vals = []
    todo = [s for s in t["sample_id"] if s not in cached]
    print(f"  mean_gc: {len(t) - len(todo)} cached, {len(todo)} to read"
          + (" (skipped: --skip_gc)" if skip and todo else ""))
    for i, sid in enumerate(t["sample_id"], 1):
        if sid in cached or skip:
            vals.append(cached.get(sid, np.nan))
            continue
        d = pd.read_csv(rp(BIOMART_DIR, f"{sid}.hg38.frag.tsv_BIN.csv"),
                        usecols=["mean_fragment_gc_content", "absolute_fragment_counts"])
        w = d["absolute_fragment_counts"]
        vals.append(float((d["mean_fragment_gc_content"] * w).sum() / w.sum()))
        if i % 25 == 0:
            print(f"    {i}/{len(t)}")
    t["mean_gc"] = vals
    assert t["mean_gc"].dropna().between(20, 70).all()


def add_ichor(t: pd.DataFrame) -> None:
    tf = {}
    for path in ICHOR_TSV.values():
        ich = pd.read_csv(rp(path), sep="\t")
        tf.update(zip(ich["sample"], ich["tumor_fraction"]))
    t["ichorcna_tf"] = t["sample_id"].map(tf)
    for cohort in ICHOR_TSV:
        assert t.loc[t["cohort"] == cohort, "ichorcna_tf"].notna().mean() > 0.95, f"ichorCNA TF missing for {cohort}"


# printed labels of the analysis groups (GROUP keeps the short names the S12 notebook uses)
TABLE_GROUP = {"Healthy": "Healthy", "BRCA+CRC": "BRCA/CRC", "Other cancer": "other cancer types",
               "Chronic hepatitis B": "chronic hepatitis B", "HBV cirrhosis": "HBV cirrhosis", "HCC": "HCC"}
TABLE_ORDER = {"Cristiano": ["Healthy", "BRCA+CRC", "Other cancer"],
               "Jiang": ["Healthy", "Chronic hepatitis B", "HBV cirrhosis", "HCC"]}
CRISTIANO_S1 = "accessory_files/cristiano_metadata.xlsx"   # Cristiano et al. Supplementary Table 1 (age at diagnosis)
TABLE_NOTE = ("Protocol rows follow the Methods of both studies. Computed rows are the median "
              "[minimum\u2013maximum] over the samples analysed, from our pipeline: coverage from the BAM files, "
              "fragment statistics in the 100\u2013220 bp window, sex from the reads on chromosomes X and Y. "
              "Jiang et al. report a median tumour fraction of 2.1% in HCC with their own method.")


def summary_table(t: pd.DataFrame) -> pd.DataFrame:
    """Supplementary Table S2. Literature rows from accessory_files/cohort_protocols.csv, computed rows from
    the per-sample scalars; `source` is provenance for the repository and is not printed."""
    proto = pd.read_csv(rp("accessory_files/cohort_protocols.csv")).set_index("characteristic")

    def fmt(s, nd=2):
        s = s.dropna()
        return "" if s.empty else f"{s.median():.{nd}f} [{s.min():.{nd}f}\u2013{s.max():.{nd}f}]"

    def per_group(cohort, f):
        return ", ".join(f"{TABLE_GROUP[g]} {f(t[(t['cohort'] == cohort) & (t['group'] == g)])}"
                         for g in TABLE_ORDER[cohort])

    def lit(name):
        r = proto.loc[name]
        return {"characteristic": name, "cristiano": r["cristiano"], "jiang": r["jiang"], "source": r["source"]}

    rows = [lit("Study"), lit("Data source")]
    rows.append({"characteristic": "Samples analysed", "cristiano": per_group("Cristiano", len),
                 "jiang": per_group("Jiang", len), "source": "manifests"})
    rows += [lit(n) for n in ["Comparison group", "Blood collection and plasma processing",
                              "cfDNA extraction", "Library preparation", "Size selection"]]
    seq = {}
    for cohort in ["Cristiano", "Jiang"]:
        sub = t[t["cohort"] == cohort]
        seq[cohort] = (f"{' / '.join(sub['instrument'].unique())}, {' / '.join(map(str, sub['readlen'].unique()))} bp "
                       f"{' / '.join(sub['seq_layout'].str.lower().unique())}-end")
    rows.append({"characteristic": "Sequencing", "cristiano": seq["Cristiano"], "jiang": seq["Jiang"],
                 "source": "FinaleDB export metadata"})

    # age at diagnosis: Cristiano only (not reported per sample for Jiang)
    s1 = pd.read_excel(rp(CRISTIANO_S1), sheet_name="1", header=1).drop_duplicates("Patient")
    age = dict(zip(s1["Patient"], pd.to_numeric(s1["Age at Diagnosis"], errors="coerce")))
    t["age"] = np.where(t["cohort"] == "Cristiano", t["sample_name"].map(age), np.nan)
    cris = t[t["cohort"] == "Cristiano"]
    missing_by = cris.loc[cris["age"].isna(), "disease"].value_counts()
    print(f"  age: {cris['age'].notna().sum()}/{len(cris)} Cristiano samples; missing: {missing_by.to_dict()}")
    age_txt = per_group("Cristiano", lambda d: fmt(d["age"], 0))
    if len(missing_by):
        age_txt += f" (not reported for {missing_by.sum()} cancer samples)"
        assert "Healthy" not in missing_by.index
    rows.append({"characteristic": "Age (years)", "cristiano": age_txt, "jiang": "Not reported",
                 "source": "Cristiano et al. Supplementary Table 1 (age at diagnosis)"})
    rows.append({"characteristic": "Female, inferred from the reads (%)",
                 **{k: per_group(c, lambda d: f"{100 * d['sex_female_bam'].mean():.0f}")
                    for k, c in [("cristiano", "Cristiano"), ("jiang", "Jiang")]},
                 "source": "src/comp/compute_sex.py (R_Y + R_X), resolved as _common.resolve_sex_call"})

    for name, col, nd in [("Genome-wide coverage (\u00d7)", "mean_coverage", 2),
                          ("Fragments per sample (millions)", "total_fragments", 1),
                          ("Mean fragment length (bp)", "fragment_length_mean", 1),
                          ("Modal fragment length (bp)", "fragment_length_mode", 0),
                          ("Short/long fragment ratio (100\u2013150 / 151\u2013220 bp)", "short_long_ratio", 3),
                          ("Mean fragment GC (%)", "mean_gc", 1),
                          ("ichorCNA tumour fraction, cancer samples", "ichorcna_tf", 3)]:
        vals = {}
        for cohort in ["Cristiano", "Jiang"]:
            s = t.loc[t["cohort"] == cohort, col]
            if col == "total_fragments":
                s = s / 1e6
            if col == "ichorcna_tf":
                s = s[t.loc[t["cohort"] == cohort, "is_cancer"] == 1]
            vals[cohort] = fmt(s, nd)
        rows.append({"characteristic": name, "cristiano": vals["Cristiano"], "jiang": vals["Jiang"],
                     "source": "computed on our pipeline (median [min-max])"})
    return pd.DataFrame(rows)


def write_table_tex(table: pd.DataFrame, path: str) -> None:
    """The printed Supplementary Table S2 as a LaTeX tabular (without the source column)."""
    def tex(x):
        x = str(x)
        for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"),
                     ("\u2013", "--"), ("\u2212", "$-$"), (" \u00b0C", r"~$^\circ$C"), ("\u00d7", r"$\times$")]:
            x = x.replace(a, b)
        return x
    lines = ["% Supplementary Table S2, written by src/figures/export/export_figS12_cohorts.py",
             r"\begin{tabular}{p{0.22\linewidth}p{0.37\linewidth}p{0.37\linewidth}}", r"\hline",
             r" & Cristiano et al. 2019 & Jiang et al. 2015 \\", r"\hline"]
    lines += [f"{tex(r.characteristic)} & {tex(r.cristiano)} & {tex(r.jiang)} \\\\" for r in table.itertuples()]
    lines += [r"\hline", r"\end{tabular}", "", f"% Note: {tex(TABLE_NOTE)}", ""]
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip_gc", action="store_true")
    ap.add_argument("--recompute_gc", action="store_true")
    args = ap.parse_args()
    os.makedirs(DATA_DIR, exist_ok=True)

    print("=== per-sample scalars (E4) ===")
    t = load_samples()
    print(f"  {len(t)} samples: " + ", ".join(f"{c} {n}" for c, n in t["cohort"].value_counts().items()))
    dens = add_fragment_lengths(t)
    add_mean_gc(t, args.skip_gc, args.recompute_gc)
    add_ichor(t)
    t.to_csv(os.path.join(DATA_DIR, "figS12_cohort_scalars.csv"), index=False)
    dens.to_csv(os.path.join(DATA_DIR, "figS12_fraglen_distributions.csv"), index=False)
    print(f"wrote figS12_cohort_scalars.csv ({len(t)} rows) + figS12_fraglen_distributions.csv")

    table = summary_table(t)
    table.to_csv(os.path.join(DATA_DIR, "tableS_cohorts.csv"), index=False, lineterminator="\n")
    write_table_tex(table, os.path.join(DATA_DIR, "tableS_cohorts.tex"))
    print("wrote tableS_cohorts.csv + tableS_cohorts.tex")
    print(table.to_string(index=False, max_colwidth=60))



if __name__ == "__main__":
    main()
