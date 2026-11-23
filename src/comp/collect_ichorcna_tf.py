#!/usr/bin/env python3
"""Refresh the ichorcna_tf column of accessory_files/cris_stage_vaf_coverage_ichor.csv
from the (re)run ichorCNA outputs in results/ichorCNA/.

For every row's ID, reads results/ichorCNA/<ID>.hg38.frag.tsv/<ID>.hg38.frag.tsv.params.txt
and takes the Tumor Fraction of the selected solution (line 2 of the header table —
runIchorCNA reports the highest-log-likelihood solution there, matching Methods 2.5).

Also deduplicates the historical byte-identical duplicate rows (the 16 doubled healthy
samples): with the manifest-driven rerun each sample must appear once.

Usage (repo root, after the ichorCNA rerun with the pon_cris50 panel):
    python src/comp/collect_ichorcna_tf.py            # in-place update
    python src/comp/collect_ichorcna_tf.py --out /tmp/preview.csv

Afterwards rerun extract_aggregate_features (fills the aggregate_* columns in the same
CSV), then copy the base CSV over the _aggregate variant that the figure exporters read:
    cp accessory_files/cris_stage_vaf_coverage_ichor.csv \\
       accessory_files/cris_stage_vaf_coverage_ichor_aggregate.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ICHOR_DIR = Path("results/ichorCNA")
CSV = Path("accessory_files/cris_stage_vaf_coverage_ichor.csv")


def read_tumor_fraction(params_path: Path) -> float:
    """Selected-solution Tumor Fraction from an ichorCNA params.txt (line 2, col 2)."""
    with open(params_path) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        values = fh.readline().rstrip("\n").split("\t")
    if header[:3] != ["Sample", "Tumor Fraction", "Ploidy"]:
        raise ValueError(f"{params_path}: unexpected header {header[:3]}")
    return float(values[1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=str(CSV))
    ap.add_argument("--ichor_dir", default=str(ICHOR_DIR))
    ap.add_argument("--out", default=None, help="Output path (default: overwrite --csv in place)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, index_col=0)
    print(f"Loaded {len(df)} rows from {args.csv}")

    # Deduplicate byte-identical rows (assert they really are identical first)
    dup_ids = df["ID"][df["ID"].duplicated()].unique().tolist()
    for sid in dup_ids:
        grp = df[df["ID"] == sid]
        if (grp.nunique(dropna=False) > 1).any():
            sys.exit(f"ERROR: duplicate rows for {sid} differ — refusing to dedup automatically")
    if dup_ids:
        df = df.drop_duplicates(subset="ID").reset_index(drop=True)
        print(f"Deduplicated {len(dup_ids)} doubled sample(s): now {len(df)} rows")

    missing, changed = [], []
    new_tf = np.full(len(df), np.nan)
    for i, sid in enumerate(df["ID"]):
        stem = f"{sid}.hg38.frag.tsv"
        params = Path(args.ichor_dir) / stem / f"{stem}.params.txt"
        if not params.is_file():
            missing.append(sid)
            continue
        new_tf[i] = read_tumor_fraction(params)
        old = df.iloc[i]["ichorcna_tf"]
        if not np.isclose(old, new_tf[i], equal_nan=True):
            changed.append((sid, old, new_tf[i]))

    if missing:
        sys.exit(f"ERROR: no params.txt for {len(missing)} sample(s), e.g. {missing[:10]} — "
                 f"has the ichorCNA rerun finished for every sample?")

    df["ichorcna_tf"] = new_tf
    print(f"Tumor fractions refreshed: {len(changed)} of {len(df)} values changed")
    if changed:
        preview = ", ".join(f"{s}: {o:.4g}->{n:.4g}" for s, o, n in changed[:5])
        print(f"  e.g. {preview}")

    out = args.out or args.csv
    df.to_csv(out)  # keeps the unnamed index column the downstream readers expect
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
