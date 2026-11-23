"""Inputs of the chr21 test, derived from the files the manuscript runs use.

    python tests/chr21/prepare_inputs.py <work_dir>

Writes into <work_dir>:
  biomart_10kb_chr21.bed       the chr21 rows of beds/biomart_10kb.bed, in their original order (216 TSS windows)
  tss_filter_chr21.npy         accessory_files/final_tss_indices_to_remove_hg38.npy restricted to those rows and
                               renumbered (56 windows removed, 160 kept)
  cris_chr21_manifest.csv      the 20 records of tests/chr21/chr21_test.csv with role 'model' (model_hpc.py selects
                               samples by role; the file name must contain 'cris' for the Cristiano label mapping)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def main(work: Path) -> None:
    work.mkdir(parents=True, exist_ok=True)

    bed = pd.read_csv(ROOT / "beds" / "biomart_10kb.bed", sep="\t", header=None)
    rows = np.flatnonzero(bed[0].to_numpy() == "chr21")
    assert len(rows) == 216 and np.all(np.diff(rows) == 1), "unexpected chr21 rows in beds/biomart_10kb.bed"
    bed.iloc[rows].to_csv(work / "biomart_10kb_chr21.bed", sep="\t", header=False, index=False, lineterminator="\n")

    removed = np.load(ROOT / "accessory_files" / "final_tss_indices_to_remove_hg38.npy")
    new_index = {old: new for new, old in enumerate(rows)}
    filt = np.array(sorted(new_index[i] for i in removed if i in new_index), dtype=np.int64)
    np.save(work / "tss_filter_chr21.npy", filt)

    ids = pd.read_csv(Path(__file__).with_name("chr21_test.csv"), comment="#")
    assert len(ids) == 20 and ids["ID"].is_unique
    manifest = ids[["ID", "sample_name", "disease"]].assign(role="model")
    full = pd.read_csv(ROOT / "accessory_files" / "cris_manifest.csv").set_index("ID")
    assert (full.loc[manifest["ID"], "disease"].to_numpy() == manifest["disease"].to_numpy()).all()
    manifest.to_csv(work / "cris_chr21_manifest.csv", index=False, lineterminator="\n")

    print(f"{work}: {len(rows)} chr21 TSS windows, {len(filt)} removed by the TSS filter, "
          f"{len(rows) - len(filt)} kept; manifest with {len(manifest)} records "
          f"({manifest['disease'].value_counts().to_dict()})")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(Path(sys.argv[1]))
