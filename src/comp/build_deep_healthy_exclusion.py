#!/usr/bin/env python
"""Write the exclusion list of the deep-sequenced healthy samples.

30 of the 194 healthy model samples of the Cristiano cohort were sequenced to a
median of 9.6x where every other sample sits at 2.3-2.5x (Fig S10 Section 2c;
no sample lies between 5.3x and 6.3x). They are a batch-like structure that is
confounded with class (all healthy), so the revision repeats every model
without them (model_hpc.py --exclude_samples).

Definition: role == 'model', disease == 'Healthy' in accessory_files/
cris_manifest.csv, and mean_coverage > DEEP_COVERAGE_X in
results/cris_coverage_summary.csv (mosdepth chain, slurm/cris_coverage.sh).
The same threshold is used by figS10.ipynb and the Fig S10 exporters.

Output: accessory_files/cris_exclude_deep_healthy.txt (one ID per line, with a
header comment; read by model_hpc.py --exclude_samples and by the figure
exporters when COHORT_VARIANT=no_deep).

Run from the repository root:
    python src/comp/build_deep_healthy_exclusion.py
"""

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DEEP_COVERAGE_X = 6.0
OUT = REPO / "accessory_files" / "cris_exclude_deep_healthy.txt"


def main() -> None:
    mf = pd.read_csv(REPO / "accessory_files" / "cris_manifest.csv")
    cov = pd.read_csv(REPO / "results" / "cris_coverage_summary.csv").set_index("sample_name")["mean_coverage"]
    model_healthy = mf[(mf["role"] == "model") & (mf["disease"] == "Healthy")].copy()
    model_healthy["mean_coverage"] = model_healthy["ID"].map(cov)
    assert model_healthy["mean_coverage"].notna().all(), "healthy model samples without a coverage value"
    deep = model_healthy[model_healthy["mean_coverage"] > DEEP_COVERAGE_X].sort_values("ID")
    below = model_healthy.loc[model_healthy["mean_coverage"] <= DEEP_COVERAGE_X, "mean_coverage"].max()
    assert len(deep) == 30, f"expected 30 deep-sequenced healthy samples, found {len(deep)}"
    print(f"{len(deep)} healthy model samples above {DEEP_COVERAGE_X:g}x "
          f"(min {deep['mean_coverage'].min():.2f}x; highest other healthy sample {below:.2f}x)")
    lines = [
        "# Deep-sequenced healthy samples of the Cristiano cohort (Fig S10 Section 2c):",
        f"# role == 'model', disease == 'Healthy', mean_coverage > {DEEP_COVERAGE_X:g}x",
        "# (results/cris_coverage_summary.csv). Written by src/comp/build_deep_healthy_exclusion.py.",
        "# Used by: model_hpc.py --exclude_samples, slurm wrappers with EXCLUDE_SAMPLES set,",
        "# and the figure exporters with COHORT_VARIANT=no_deep.",
        "# ID  # sample_name  mean_coverage",
    ]
    lines += [f"{r.ID}  # {r.sample_name}  {r.mean_coverage:.2f}x" for r in deep.itertuples()]
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
