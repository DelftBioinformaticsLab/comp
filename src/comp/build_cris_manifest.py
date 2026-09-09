"""Build the canonical Cristiano sample manifest: accessory_files/cris_manifest.csv.

One row per FinaleDB sequencing run, with an explicit `role` for every record:

  panel_of_normals  the 50 plasma-cfDNA healthy samples held out for TSS QC
                    and the ichorCNA panel: the first 50 unique healthy
                    PARTICIPANTS in retrieval order, one (base) sequencing
                    run each — replicate runs are never used anywhere
                    (decision Sep 2026)
  model             enters the classification tasks (one sequencing run per
                    participant)
  excluded          never used, with `exclusion_reason` saying why

This file is the single source of truth: it is the manuscript sample table
(the fold_* / oof_probability_* columns are filled by the rerun), and
model_hpc.py selects samples from it when passed as --metadata_file (it
detects the `role` column and switches to manifest-driven selection with
exact-ID feature loading).

Regenerate after the GCfix recovery run so features_available/gcfix_pending
reflect the disk:  python src/comp/build_cris_manifest.py

Derivation sources: accessory_files/cris_metadata.csv (our FinaleDB retrieval
manifest) cross-checked against the published supplement
(accessory_files/cristiano_metadata.xlsx, Table S4) during the Sep 2026
sample-accounting audit.
"""

import os
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "accessory_files" / "cris_metadata.csv"
OUT_MANIFEST = REPO / "accessory_files" / "cris_manifest.csv"
OUT_PON_WIGS = REPO / "input" / "cris_pon_wigs.txt"

FEATURES_DIR = REPO / "extracted_features" / "biomart_10kb"
# preprocessing store as mounted on this machine / on the cluster
WIG_DIR_LOCAL = Path("u:/KWFcfDNA/preprocessing/cris/wigs")
WIG_DIR_CLUSTER = "/tudelft.net/staff-umbrella/KWFcfDNA/preprocessing/cris/wigs"

# CGH14/CGH15: "Human adult elutriated lymphocytes" in Cristiano Table S4
# (~9.5x germline WGS) — not plasma cfDNA. FinaleDB's tissue tag is wrong.
LYMPHOCYTE_SAMPLES = {"CGH14", "CGH15"}
N_PANEL = 50

# The seven task cancer types (matches CANCER_MAP in model_hpc.py). Plain
# "Lung cancer" records are serial monitoring samples (not treatment-naïve).
TASK_CANCERS = {"Breast cancer", "Bile duct cancer", "Colorectal cancer",
                "Gastric cancer", "Lung cancer naive nsclc", "Ovarian cancer",
                "Pancreatic cancer"}


def main() -> None:
    m = pd.read_csv(SOURCE)
    m["participant_id"] = m["sample_name"].str.replace(r"_Rep$", "", regex=True)
    m["material"] = m["sample_name"].map(
        lambda s: "lymphocyte_gDNA" if s in LYMPHOCYTE_SAMPLES else "plasma_cfDNA")
    m["replicate_second_run"] = False
    m["replicate_of"] = ""
    m["role"] = ""
    m["exclusion_reason"] = ""

    # --- second sequencing runs -------------------------------------------
    # healthy pairs are distinguished by the _Rep suffix; the pancreatic pair
    # (CGPLPA95) shares one name, so the later retrieval (higher EE id) is
    # taken as the second run.
    for pid, grp in m.groupby("participant_id"):
        if len(grp) < 2:
            continue
        grp = grp.sort_values("ID")
        is_rep = grp["sample_name"].str.endswith("_Rep")
        second = grp[is_rep] if is_rep.any() else grp.iloc[1:]
        first_id = grp.loc[~grp["ID"].isin(second["ID"]), "ID"].iloc[0]
        m.loc[m["ID"].isin(second["ID"]), "replicate_second_run"] = True
        m.loc[m["ID"].isin(second["ID"]), "replicate_of"] = first_id

    # --- roles -------------------------------------------------------------
    # lymphocyte gDNA
    lymph = m["material"] == "lymphocyte_gDNA"
    m.loc[lymph, ["role", "exclusion_reason"]] = ["excluded", "lymphocyte_gDNA"]

    # panel of normals: first N_PANEL unique cfDNA healthy PARTICIPANTS in
    # file order, each represented by the base (non-second-run) record only.
    healthy_cfdna = m[(m["disease"] == "Healthy") & ~lymph]
    panel_base = healthy_cfdna[~healthy_cfdna["replicate_second_run"]]
    panel_ids = panel_base["ID"].iloc[:N_PANEL].tolist()
    m.loc[m["ID"].isin(panel_ids), "role"] = "panel_of_normals"

    # remaining healthy: every second run is excluded (including second runs
    # of panel participants — replicates are not used anywhere), rest model
    rest_healthy = (m["disease"] == "Healthy") & (m["role"] == "")
    m.loc[rest_healthy & m["replicate_second_run"],
          ["role", "exclusion_reason"]] = ["excluded", "replicate_second_run"]
    m.loc[(m["disease"] == "Healthy") & (m["role"] == ""), "role"] = "model"

    # cancer records
    cancer = m["disease"] != "Healthy"
    m.loc[cancer & (m["disease"] == "Lung cancer"),
          ["role", "exclusion_reason"]] = ["excluded", "not_treatment_naive_serial"]
    m.loc[cancer & (m["disease"] == "Duodenal cancer"),
          ["role", "exclusion_reason"]] = ["excluded", "single_case"]
    m.loc[cancer & (m["role"] == "") & m["replicate_second_run"],
          ["role", "exclusion_reason"]] = ["excluded", "replicate_second_run"]
    m.loc[cancer & (m["role"] == ""), "role"] = "model"

    # --- disk status -------------------------------------------------------
    m["features_available"] = m["ID"].map(
        lambda i: (FEATURES_DIR / f"{i}.hg38.frag.tsv_BIN.csv").exists())
    m["gcfix_pending"] = (m["role"].isin(["model", "panel_of_normals"])
                          & ~m["features_available"])

    # --- task membership + result placeholders -----------------------------
    m["in_task_brca_crc"] = (m["role"] == "model") & m["disease"].isin(
        {"Healthy", "Breast cancer", "Colorectal cancer"})
    m["in_task_pancan"] = (m["role"] == "model") & (
        (m["disease"] == "Healthy") | m["disease"].isin(TASK_CANCERS))
    for col in ["fold_brca_crc", "oof_probability_brca_crc",
                "fold_pancan", "oof_probability_pancan"]:
        m[col] = ""

    cols = ["ID", "sample_name", "participant_id", "disease", "material",
            "role", "exclusion_reason", "replicate_second_run", "replicate_of",
            "in_task_brca_crc", "in_task_pancan",
            "features_available", "gcfix_pending", "fragNum",
            "fold_brca_crc", "oof_probability_brca_crc",
            "fold_pancan", "oof_probability_pancan"]
    out = m[cols]
    out.to_csv(OUT_MANIFEST, index=False)

    # --- panel wig list for createPanelOfNormals.R -------------------------
    missing_wigs = [i for i in panel_ids
                    if not (WIG_DIR_LOCAL / f"{i}.hg38.frag.tsv.wig").exists()]
    with open(OUT_PON_WIGS, "w", newline="\n") as fh:
        for i in panel_ids:
            fh.write(f"{WIG_DIR_CLUSTER}/{i}.hg38.frag.tsv.wig\n")

    # --- summary + invariants ---------------------------------------------
    print(f"wrote {OUT_MANIFEST}")
    print(f"wrote {OUT_PON_WIGS} ({len(panel_ids)} wigs; missing on disk: {missing_wigs or 'none'})")
    print("\nrole counts:")
    print(m.groupby(["role", "exclusion_reason"], dropna=False).size().to_string())

    panel = m[m["role"] == "panel_of_normals"]
    model = m[m["role"] == "model"]
    model_h = model[model["disease"] == "Healthy"]
    model_c = model[model["disease"] != "Healthy"]
    print(f"\npanel_of_normals: {len(panel)} records / {panel['participant_id'].nunique()} participants "
          f"({int(panel['replicate_second_run'].sum())} second runs — must be 0)")
    print(f"model healthy:    {len(model_h)} ({int(model_h['gcfix_pending'].sum())} pending GC recovery)")
    print(f"model cancer:     {len(model_c)} ({int(model_c['gcfix_pending'].sum())} pending GC recovery)")
    print(f"BRCA/CRC task:    {int(m['in_task_brca_crc'].sum())}   PANCAN task: {int(m['in_task_pancan'].sum())}")

    assert len(panel) == N_PANEL, f"panel must be {N_PANEL}, got {len(panel)}"
    assert panel["participant_id"].nunique() == N_PANEL, "panel must be unique participants"
    assert not panel["replicate_second_run"].any(), "no replicate runs in the panel"
    assert len(model_h) == 194 and model_h["participant_id"].nunique() == 194
    assert len(model_c) == 203 and model_c["participant_id"].nunique() == 203
    assert not m.loc[m["role"] == "model", "replicate_second_run"].any()
    assert not set(panel["participant_id"]) & set(model["participant_id"]), \
        "participant crosses the panel/model boundary"
    assert (m["role"] != "").all()
    assert len(m) == 534
    print("\nall invariants hold: 50-participant panel / 194 healthy / 203 cancer, "
          "no replicates anywhere, boundary clean")


if __name__ == "__main__":
    main()
