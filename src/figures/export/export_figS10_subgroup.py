"""Source data for the deep-sequenced healthy subgroup of Figure S10: the depth columns of
Cristiano et al. 2019 Table S4 next to our own measurements, for the 30 healthy samples
sequenced to 6-14x and the rest of the pan-cancer task.

Writes src/figures/data/figS10_depth_subgroup.csv, one row per pan-cancer task sample (397):
sample_id, sample_name, y_true, disease, deep_subgroup (mean_coverage > DEEP_COVERAGE_X),
mean_coverage, total_fragments, s4_total_bases_sequenced / s4_coverage / s4_read_length
(Table S4, WGS rows).

Only under COHORT_VARIANT=all. Run from the repository root (after
export_fig3_baselines.py --task pancan):
    COHORT_VARIANT=all python src/figures/export/export_figS10_subgroup.py
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from _common import DATA_DIR, rp  # noqa: E402
from _common import IS_VARIANT, COHORT_VARIANT  # noqa: E402

MANIFEST_CSV = "accessory_files/cris_manifest.csv"
COVERAGE_CSV = "results/cris_coverage_summary.csv"
CRISTIANO_XLSX = "accessory_files/cristiano_metadata.xlsx"
BIOMART_DIR = "extracted_features/biomart_10kb"
SCALARS_CSV = "fig4_baseline_scalars.csv"
OUT_CSV = "figS10_depth_subgroup.csv"

DEEP_COVERAGE_X = 6.0   # the cohort has no sample between 5.3x and 6.3x
N_WINDOWS = 19376



def main() -> None:
    if IS_VARIANT:
        # the cohort variant has no deep-sequenced subgroup
        print(f"export_figS10_subgroup.py: not applicable under COHORT_VARIANT={COHORT_VARIANT} - nothing written")
        return
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()

    sc = pd.read_csv(os.path.join(DATA_DIR, SCALARS_CSV))
    assert len(sc) == 397 and sc["y_true"].sum() == 203, "expected the 397-sample PANCAN scalar table"
    t = sc[["sample_id", "sample_name", "y_true", "disease", "mean_coverage", "total_fragments"]].copy()

    # our own measurements must be the ones the Fig S10 covariates were built from
    mf = pd.read_csv(rp(MANIFEST_CSV)).set_index("ID")
    cov = pd.read_csv(rp(COVERAGE_CSV)).set_index("sample_name")["mean_coverage"]
    assert np.allclose(t["sample_id"].map(cov), t["mean_coverage"]), "coverage differs from the scalar table"
    assert np.allclose(t["sample_id"].map(mf["fragNum"]).astype(float), t["total_fragments"]), "fragNum differs"
    assert (t["sample_id"].map(mf["role"]) == "model").all()

    s4 = pd.read_excel(rp(CRISTIANO_XLSX), sheet_name="4", header=1)
    s4 = s4[s4["Analysis type"] == "WGS"].groupby("Patient").first()
    for col, src in [("s4_total_bases_sequenced", "Total Bases Sequenced"),
                     ("s4_coverage", "Coverage"), ("s4_read_length", "Read Length")]:
        t[col] = t["sample_name"].map(s4[src]).astype(float)
        assert t[col].notna().all(), f"{src}: task samples unmatched in Cristiano S4"

    t["deep_subgroup"] = t["mean_coverage"] > DEEP_COVERAGE_X

    # the subgroup is the 30 healthy samples of the exclusion list
    deep = t[t["deep_subgroup"]]
    assert len(deep) == 30 and (deep["y_true"] == 0).all(), (len(deep), deep["y_true"].sum())
    excluded = {l.split("#")[0].strip() for l in open(rp("accessory_files/cris_exclude_deep_healthy.txt"))} - {""}
    assert set(deep["sample_id"]) == excluded, "coverage-defined subgroup differs from the exclusion list"
    gap = (t.loc[~t["deep_subgroup"], "mean_coverage"].max(), deep["mean_coverage"].min())
    assert gap[0] < DEEP_COVERAGE_X < gap[1], gap
    pon = mf[mf["role"] == "panel_of_normals"].index
    n_pon_deep = int((cov.reindex(pon) > DEEP_COVERAGE_X).sum())
    assert n_pon_deep == 0, "deep-sequenced samples in the panel of normals"

    cols = ["sample_id", "sample_name", "y_true", "disease", "deep_subgroup", "mean_coverage",
            "total_fragments", "s4_total_bases_sequenced", "s4_coverage", "s4_read_length"]
    out = os.path.join(DATA_DIR, OUT_CSV)
    t[cols].to_csv(out, index=False)
    print(f"wrote {out} ({len(t)} rows)")

    grp = np.where(t["y_true"] == 1, "cancer", np.where(t["deep_subgroup"], "healthy, deep", "healthy, other"))
    with pd.option_context("display.width", 200):
        print(t.groupby(grp)[["mean_coverage", "s4_coverage"]].median().round(4).T.to_string())
    print(f"coverage gap between the groups: {gap[0]:.2f}x -> {gap[1]:.2f}x; "
          f"deep-sequenced samples in the panel of normals: {n_pon_deep}")


if __name__ == "__main__":
    main()
