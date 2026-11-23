"""Check that a chr21 test run produced complete, well-formed outputs (a smoke test).

    python tests/chr21/check_outputs.py <work_dir> [--grid quick|full]

No numbers are compared with stored values: with 20 records and one chromosome the model results
carry no information, and classifier fits can differ in the last digits between CPU types and library
builds. The checks are that every step ran and wrote what the next step and the figure code read:

  features      one BIN file per record, all 216 chr21 TSS windows, the five model feature types and
                the fragment count present, finite in most windows, fragments counted in every record
  model         10 outer folds, each with a selected feature type, ranking function, k and classifier;
                one held-out probability in [0, 1] per record, each record in exactly one outer fold;
                summary with the pooled AUROC

Exits 1 on the first failed check and prints which one.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

N_WINDOWS = 216
FEATURES = ["absolute_fragment_counts", "fslr_coverage", "fslr", "griffin_diff", "rcov", "mwh"]
SELECTION = ["fold", "feature", "selection_type", "n_top_genes", "classifier", "clf_hyperparams"]


def fail(msg: str) -> None:
    print(f"FAILED: {msg}")
    print("chr21 test FAILED")
    sys.exit(1)


def check_features(work: Path, ids: pd.Series) -> None:
    feat_dir = work / "features" / "biomart_10kb_chr21"
    for sid in ids:
        f = feat_dir / f"{sid}.hg38.frag.tsv_BIN.csv"
        if not f.exists():
            fail(f"feature file missing: {f}")
        df = pd.read_csv(f)
        missing = [c for c in ["chrom", "start", "end", "gene_name"] + FEATURES if c not in df.columns]
        if missing:
            fail(f"{f.name}: columns missing: {missing}")
        if len(df) != N_WINDOWS or set(df["chrom"].astype(str)) != {"chr21"}:
            fail(f"{f.name}: {len(df)} windows on {sorted(set(df['chrom'].astype(str)))}, expected {N_WINDOWS} on chr21")
        if df["absolute_fragment_counts"].sum() <= 0:
            fail(f"{f.name}: no fragments counted")
        for c in FEATURES[1:]:
            finite = np.isfinite(df[c].to_numpy(float)).mean()
            if finite < 0.5:
                fail(f"{f.name}: {c} finite in only {finite:.0%} of windows")
    print(f"features     ok ({len(ids)} records x {N_WINDOWS} windows)")


def check_model(work: Path, grid: str, ids: pd.Series) -> None:
    nested = work / "nested" / f"nested_full_chr21_test_{grid}"
    for name in ["nested_full_per_fold.csv", "nested_full_oof_predictions.csv", "nested_full_summary.json"]:
        if not (nested / name).exists():
            fail(f"model output missing: {nested / name}")
    sel = pd.read_csv(nested / "nested_full_per_fold.csv")
    missing = [c for c in SELECTION if c not in sel.columns]
    if missing:
        fail(f"nested_full_per_fold.csv: columns missing: {missing}")
    if len(sel) != 10 or sel[SELECTION].isna().any().any():
        fail(f"nested_full_per_fold.csv: {len(sel)} outer folds or empty selections, expected 10 complete rows")
    oof = pd.read_csv(nested / "nested_full_oof_predictions.csv")
    if sorted(oof["sample_id"]) != sorted(ids) or oof["sample_id"].duplicated().any():
        fail("nested_full_oof_predictions.csv: not exactly one held-out prediction per record")
    p = oof["y_pred_proba"].to_numpy(float)
    if not (np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()):
        fail("nested_full_oof_predictions.csv: probabilities outside [0, 1]")
    if oof["outer_fold"].nunique() != 10:
        fail(f"nested_full_oof_predictions.csv: records spread over {oof['outer_fold'].nunique()} outer folds, expected 10")
    summary = json.loads((nested / "nested_full_summary.json").read_text())
    if "oof_auc_pooled" not in summary:
        fail("nested_full_summary.json: no pooled AUROC")
    print(f"model        ok (10 outer folds, {len(oof)} held-out predictions; pooled AUROC "
          f"{summary['oof_auc_pooled']:.2f}, meaningless at this size)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("work_dir", type=Path)
    ap.add_argument("--grid", choices=["quick", "full"], default="quick", help="which model run to read")
    args = ap.parse_args()
    ids = pd.read_csv(args.work_dir / "cris_chr21_manifest.csv")["ID"]
    check_features(args.work_dir, ids)
    check_model(args.work_dir, args.grid, ids)
    print("chr21 test PASSED")


if __name__ == "__main__":
    main()
