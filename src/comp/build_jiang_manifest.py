"""Build the canonical Jiang sample manifest: accessory_files/jiang_manifest.csv.

The Jiang et al. 2015 (PNAS, doi 10.1073/pnas.1500076112) cohort as served by
FinaleDB: 90 hepatocellular carcinoma (HCC) patients and 135 non-HCC
comparators — 32 healthy subjects, 67 chronic hepatitis B (CHB) patients and
36 patients with HBV-associated cirrhosis. The 135 are a heterogeneous
non-HCC comparison group, NOT 135 healthy people (reviewer 4, major
comment 4); the manifest therefore carries an explicit `control_group`
column next to the binary task label.

Sample accounting (reviewer 4, major comment 1, applied to this cohort):

  accessory_files/jiang_metadata.csv is the raw FinaleDB export and has 249
  rows for 225 unique sample IDs: 20 IDs are listed two or three times with
  byte-identical rows (a FinaleDB export artefact, no conflicting labels).
  model_hpc.py always dropped these at load time (drop_duplicates on ID), so
  every reported number was computed on the 225 unique samples; this file
  makes that explicit and reproducible.

  input/jiang_bam.txt and input/jiang_gc.txt list 249 IDs: the 225 Jiang
  samples plus 24 FinaleDB accessions from the same EE857xx-EE860xx numeric
  range that are not part of the Jiang export, have no fragment file and
  zero coverage in preprocessing/jiang/coverage.csv (they were never
  retrieved). They are recorded in jiang_sample_flow.csv as
  `not_a_jiang_record` and never enter any analysis.

Every row gets `role == 'model'`: there is no panel of normals for this
cohort (the ichorCNA panel and the TSS filter are Cristiano-derived and
frozen), and no exclusions were needed. model_hpc.py detects the `role`
column and switches to manifest-driven selection with exact-ID feature
loading, replacing the legacy longest-substring file matching.

Regenerate:  python src/comp/build_jiang_manifest.py
Writes:      accessory_files/jiang_manifest.csv
             accessory_files/jiang_sample_flow.csv  (sample-flow table)
"""

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "accessory_files" / "jiang_metadata.csv"
INPUT_LIST = REPO / "input" / "jiang_bam.txt"
OUT_MANIFEST = REPO / "accessory_files" / "jiang_manifest.csv"
OUT_FLOW = REPO / "accessory_files" / "jiang_sample_flow.csv"
FEATURES_DIR = REPO / "extracted_features" / "biomart_10kb"

# Published composition (Jiang et al. 2015, Table 1) — the invariants below
# assert that the deduplicated FinaleDB export reproduces it exactly.
PUBLISHED = {"Healthy": 32, "Cirrhosis": 36, "Hepatitis B": 67, "Liver cancer": 90}

# Human-readable comparison-group labels (the `disease` strings themselves are
# frozen: model_hpc.py's Jiang branch selects on them).
CONTROL_GROUP = {"Healthy": "Healthy",
                 "Hepatitis B": "Chronic hepatitis B",
                 "Cirrhosis": "HBV-associated cirrhosis",
                 "Liver cancer": ""}


def main() -> None:
    raw = pd.read_csv(SOURCE)
    n_raw = len(raw)

    # --- duplicate export rows ------------------------------------------------
    dup_ids = raw.loc[raw["ID"].duplicated(), "ID"].unique().tolist()
    for sid in dup_ids:
        rows = raw[raw["ID"] == sid]
        assert len(rows.drop_duplicates()) == 1, \
            f"{sid}: duplicate rows are not identical — resolve by hand"
    m = raw.drop_duplicates(subset="ID").reset_index(drop=True)
    n_dup_rows = n_raw - len(m)

    # --- roles / labels -------------------------------------------------------
    m["participant_id"] = m["sample_name"]  # one plasma sample per participant
    m["control_group"] = m["disease"].map(CONTROL_GROUP)
    m["is_hcc"] = (m["disease"] == "Liver cancer").astype(int)
    m["role"] = "model"
    m["exclusion_reason"] = ""
    m["finaledb_export_rows"] = m["ID"].map(raw["ID"].value_counts())
    m["features_available"] = m["ID"].map(
        lambda i: (FEATURES_DIR / f"{i}.hg38.frag.tsv_BIN.csv").exists())
    m["gcfix_pending"] = ~m["features_available"]
    for col in ["fold_hcc_vs_nonhcc", "oof_probability_hcc_vs_nonhcc"]:
        m[col] = ""

    cols = ["ID", "sample_name", "participant_id", "disease", "control_group",
            "is_hcc", "role", "exclusion_reason", "finaledb_export_rows",
            "features_available", "gcfix_pending",
            "readlen", "instrument", "seq_layout", "fragNum",
            "fold_hcc_vs_nonhcc", "oof_probability_hcc_vs_nonhcc"]
    out = m[cols]
    out.to_csv(OUT_MANIFEST, index=False, lineterminator="\n")

    # --- accessions in the input lists that are not Jiang records -------------
    listed = [Path(l.strip()).name.split(".")[0] for l in INPUT_LIST.read_text().splitlines()
              if l.strip()]
    not_jiang = sorted(set(listed) - set(m["ID"]))
    missing_from_list = sorted(set(m["ID"]) - set(listed))

    # --- sample-flow table ----------------------------------------------------
    counts = m["disease"].value_counts()
    flow = [
        ("finaledb_export_rows", n_raw, "rows in accessory_files/jiang_metadata.csv (FinaleDB export)"),
        ("duplicate_export_rows", n_dup_rows,
         f"byte-identical repeat rows for {len(dup_ids)} IDs, dropped (drop_duplicates on ID)"),
        ("unique_samples", len(m), "one plasma sample per participant"),
        ("healthy", int(counts.get("Healthy", 0)), "control_group = Healthy"),
        ("chronic_hepatitis_b", int(counts.get("Hepatitis B", 0)), "control_group = Chronic hepatitis B"),
        ("hbv_cirrhosis", int(counts.get("Cirrhosis", 0)), "control_group = HBV-associated cirrhosis"),
        ("non_hcc_total", int((m["is_hcc"] == 0).sum()), "class 0 of the HCC-vs-non-HCC task"),
        ("hcc", int(counts.get("Liver cancer", 0)), "class 1"),
        ("features_available", int(m["features_available"].sum()),
         "extracted_features/biomart_10kb/<ID>.hg38.frag.tsv_BIN.csv present on this machine"),
        ("role_model", int((m["role"] == "model").sum()), "enter the modelling / external-application analyses"),
        ("excluded", int((m["role"] == "excluded").sum()), "none"),
        ("not_a_jiang_record", len(not_jiang),
         "accessions in input/jiang_bam.txt from the same EE-number range that are not in the "
         "FinaleDB Jiang export, have no fragment file and zero coverage; never analysed: "
         + " ".join(not_jiang)),
    ]
    pd.DataFrame(flow, columns=["step", "n", "note"]).to_csv(OUT_FLOW, index=False, lineterminator="\n")

    # --- summary + invariants --------------------------------------------------
    print(f"wrote {OUT_MANIFEST}")
    print(f"wrote {OUT_FLOW}")
    print(f"\nFinaleDB export: {n_raw} rows -> {len(m)} unique samples "
          f"({n_dup_rows} duplicate rows over {len(dup_ids)} IDs)")
    print(m.groupby(["disease", "control_group"]).size().to_string())
    print(f"\ninput list: {len(listed)} accessions, {len(not_jiang)} not Jiang records, "
          f"{len(missing_from_list)} Jiang samples missing from the list")
    print(f"features on disk: {int(m['features_available'].sum())}/{len(m)}")

    for disease, n in PUBLISHED.items():
        assert int(counts.get(disease, 0)) == n, \
            f"{disease}: {counts.get(disease, 0)} != published {n}"
    assert len(m) == sum(PUBLISHED.values()) == 225
    assert m["ID"].is_unique and m["sample_name"].is_unique
    assert not missing_from_list, f"Jiang samples absent from {INPUT_LIST}: {missing_from_list}"
    assert (m["readlen"] == 75).all() and (m["seq_layout"] == "PAIRED").all()
    print("\nall invariants hold: 225 unique samples = 32 healthy / 36 cirrhosis / "
          "67 CHB / 90 HCC (Jiang et al. 2015), no replicates, one sample per participant")


if __name__ == "__main__":
    main()
