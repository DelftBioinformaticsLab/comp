#!/usr/bin/env python3
"""Fill the fold_<task> / oof_probability_<task> columns of the Cristiano manifest
from a fully nested CV run's out-of-fold predictions.

Usage (repo root):
    python src/comp/fill_manifest_oof.py nested_cv_results_cris/nested_full_brca_crc_nested_full --task brca_crc
    python src/comp/fill_manifest_oof.py <dir-or-oof-csv> --task pancan

The positional argument is either a nested_full output directory (containing
nested_full_oof_predictions.csv) or a path to the OOF CSV itself.

Validations are strict and loud: every predicted sample must be a manifest
record with role == 'model' and in_task_<task> == True, every in-task sample
must have exactly one prediction, and the labels must agree with the manifest's
disease column. Any violation aborts without writing.

NOTE: build_cris_manifest.py recreates the manifest with EMPTY fold/oof columns;
rerun this script after every manifest rebuild.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

HEALTHY = "Healthy"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("oof", help="nested_full output dir or path to nested_full_oof_predictions.csv")
    ap.add_argument("--task", required=True, choices=["brca_crc", "pancan"])
    ap.add_argument("--manifest", default="accessory_files/cris_manifest.csv")
    ap.add_argument("--out", default=None, help="Output path (default: overwrite --manifest in place)")
    args = ap.parse_args()

    oof_path = Path(args.oof)
    if oof_path.is_dir():
        oof_path = oof_path / "nested_full_oof_predictions.csv"
    if not oof_path.is_file():
        sys.exit(f"ERROR: OOF predictions not found: {oof_path}")

    oof = pd.read_csv(oof_path)
    required = {"sample_id", "y_true", "y_pred_proba", "outer_fold"}
    if not required.issubset(oof.columns):
        sys.exit(f"ERROR: {oof_path} lacks columns {sorted(required - set(oof.columns))}")

    manifest = pd.read_csv(args.manifest)
    fold_col = f"fold_{args.task}"
    prob_col = f"oof_probability_{args.task}"
    task_col = f"in_task_{args.task}"
    for col in (fold_col, prob_col, task_col):
        if col not in manifest.columns:
            sys.exit(f"ERROR: manifest lacks column '{col}' — regenerate with build_cris_manifest.py")

    # --- strict cross-checks -------------------------------------------------
    dupes = oof["sample_id"][oof["sample_id"].duplicated()].tolist()
    if dupes:
        sys.exit(f"ERROR: duplicated sample_ids in OOF predictions: {dupes[:10]}")

    m = manifest.set_index("ID")
    unknown = [s for s in oof["sample_id"] if s not in m.index]
    if unknown:
        sys.exit(f"ERROR: {len(unknown)} OOF sample(s) not in manifest, e.g. {unknown[:10]}")

    sel = m.loc[oof["sample_id"]]
    bad_role = sel[sel["role"] != "model"]
    if len(bad_role):
        sys.exit(f"ERROR: {len(bad_role)} predicted sample(s) do not have role='model': "
                 f"{bad_role.index.tolist()[:10]}")
    not_in_task = sel[~sel[task_col].astype(bool)]
    if len(not_in_task):
        sys.exit(f"ERROR: {len(not_in_task)} predicted sample(s) have {task_col}=False: "
                 f"{not_in_task.index.tolist()[:10]}")

    expected = set(manifest.loc[manifest[task_col].astype(bool) & (manifest["role"] == "model"), "ID"])
    missing = sorted(expected - set(oof["sample_id"]))
    if missing:
        sys.exit(f"ERROR: {len(missing)} in-task sample(s) have no OOF prediction, e.g. {missing[:10]}")

    label_from_disease = (sel["disease"] != HEALTHY).astype(int).to_numpy()
    mismatch = oof.loc[label_from_disease != oof["y_true"].to_numpy(), "sample_id"].tolist()
    if mismatch:
        sys.exit(f"ERROR: y_true disagrees with manifest disease for: {mismatch[:10]}")

    if not oof["outer_fold"].between(1, 10).all():
        sys.exit("ERROR: outer_fold values outside 1..10")
    if not oof["y_pred_proba"].between(0.0, 1.0).all():
        sys.exit("ERROR: y_pred_proba values outside [0, 1]")

    # --- fill ---------------------------------------------------------------
    fold_map = dict(zip(oof["sample_id"], oof["outer_fold"].astype(int)))
    prob_map = dict(zip(oof["sample_id"], oof["y_pred_proba"].astype(float)))
    manifest[fold_col] = manifest["ID"].map(fold_map).astype("Int64")
    manifest[prob_col] = manifest["ID"].map(prob_map)

    out = args.out or args.manifest
    manifest.to_csv(out, index=False)

    n = int(manifest[fold_col].notna().sum())
    print(f"Filled {fold_col} / {prob_col} for {n} samples "
          f"({int(oof['y_true'].sum())} cancer, {int((oof['y_true'] == 0).sum())} healthy) "
          f"from {oof_path}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
