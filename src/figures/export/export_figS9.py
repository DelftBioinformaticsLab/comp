"""Source data for Figure S9: training-cohort size sweep (fully nested CV on nested
stratified subsets of 90% .. 10% of the cohort, both tasks where results exist).

Reads SIZESWEEP_DIRS, SIZESWEEP_SEED_REPLICATE_GLOBS and the standalone full-cohort runs
(used as the 100% point when the sweep has none) from _common.py. Writes to
src/figures/data/: figS9_predictions.csv (OOF scores per task, fraction, sample),
figS9_sweep.csv (AUROC summaries per fraction), figS9_fold_configs.csv (winning
configuration per fold), figS9_seed_replicates.csv (other subsample seeds; may be empty).

Run from the repository root:
    python src/figures/export/export_figS9.py [--no-full-ref]
--arm task=dir overrides a sweep directory (testing only; do not commit its CSVs).
"""

import argparse
import os
import pickle
import sys
from glob import glob

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from _common import (  # noqa: E402
    DATA_DIR,
    NESTED_FULL_DIR,
    PANCAN_NESTED_FULL_DIR,
    SIZESWEEP_DIRS,
    SIZESWEEP_SEED_REPLICATE_GLOBS,
    assert_predictions_match_stored,
    rp,
)

# Standalone full-cohort references per task, used only when the sweep has no 1.0 arm.
FULL_REF_DIRS = {"brca_crc": NESTED_FULL_DIR, "pancan": PANCAN_NESTED_FULL_DIR}


def load_fraction_dir(frac_dir: str):
    """Aggregated per-fold table of one fraction subdirectory (schema-aligned
    with assert_predictions_match_stored) + that fraction's meta."""
    with open(rp(frac_dir, "nested_full_per_fold.pkl"), "rb") as fh:
        per_fold = pickle.load(fh)
    per_fold = per_fold.rename(columns={"test_auc": "auc_full_model",
                                        "n_top_genes": "top_genes"})
    first_partial = sorted(os.listdir(rp(frac_dir)))
    first_partial = [f for f in first_partial if f.startswith("nested_full_fold_")
                     and f.endswith(".pkl")][0]
    with open(rp(frac_dir, first_partial), "rb") as fh:
        meta = pickle.load(fh)["meta"]
    oof = pd.read_csv(rp(frac_dir, "nested_full_oof_predictions.csv"))
    return per_fold, meta, oof


def collect_task(task: str, sweep_dir: str, full_ref_dir: str | None):
    """All (fraction, per_fold, meta) triples of one task, largest first; fractions
    without nested_full_per_fold.pkl (incomplete) are skipped."""
    sweep_has_full = os.path.exists(
        rp(sweep_dir, "sizesweep_f100", "nested_full_per_fold.pkl"))
    if full_ref_dir is not None and sweep_has_full:
        print(f"NOTE: {task}: sweep carries its own 1.0 arm - standalone reference "
              f"{full_ref_dir} not added.")
        full_ref_dir = None
    entries = []
    if full_ref_dir is not None:
        with open(rp(full_ref_dir, "nested_full_per_fold.pkl"), "rb") as fh:
            per_fold = pickle.load(fh)
        per_fold = per_fold.rename(columns={"test_auc": "auc_full_model",
                                            "n_top_genes": "top_genes"})
        n = sum(len(r) for r in per_fold["test_sample_ids"])
        meta = {"sample_fraction": 1.0, "n_samples_subset": n,
                "n_outer_folds": len(per_fold), "subsample_seed": None}
        entries.append((1.0, per_fold, meta,
                        pd.read_csv(rp(full_ref_dir, "nested_full_oof_predictions.csv")),
                        full_ref_dir))
    for name in sorted(os.listdir(rp(sweep_dir)), reverse=True):
        if not name.startswith("sizesweep_f"):
            continue
        if not os.path.exists(rp(sweep_dir, name, "nested_full_per_fold.pkl")):
            print(f"NOTE: {task}: {name} has no aggregated results (incomplete "
                  f"folds or aggregate not rerun) - fraction skipped.")
            continue
        per_fold, meta, oof = load_fraction_dir(os.path.join(sweep_dir, name))
        entries.append((float(meta["sample_fraction"]), per_fold, meta, oof,
                        os.path.join(sweep_dir, name)))
    entries.sort(key=lambda e: -e[0])
    return entries


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", action="append", default=[], metavar="TASK=DIR",
                    help="override a sweep directory (testing only; repeatable)")
    ap.add_argument("--no-full-ref", action="store_true",
                    help="do not add the standalone full-cohort runs as 100%% points")
    args = ap.parse_args()

    dirs = dict(SIZESWEEP_DIRS)
    overridden = False
    for spec in args.arm:
        task, _, d = spec.partition("=")
        if task not in dirs or not d:
            raise SystemExit(f"--arm expects one of {list(dirs)}=<dir>, got {spec!r}")
        dirs[task] = d
        overridden = True
    if overridden:
        print("!! source-directory overrides active - do NOT commit the resulting CSVs\n")

    available = {t: d for t, d in dirs.items()
                 if os.path.isdir(rp(d)) and os.path.exists(rp(d, "nested_full_sizesweep.csv"))}
    for t in dirs:
        if t not in available:
            print(f"NOTE: no aggregated sweep for task '{t}' at {dirs[t]} - skipped.")
    if not available:
        print("\nNo sample-size sweep results found. Submit them first, e.g.:")
        print("  ./slurm/submit_nested_full.sh sizesweep                        # brca_crc")
        print("  CANCER_TYPES=PANCAN EXPERIMENT_NAME=pancan_nested_full \\")
        print("  SIZESWEEP_FRACTIONS=\"1.0 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1\" \\")
        print("  ./slurm/submit_nested_full.sh sizesweep                        # pancan")
        raise SystemExit(1)

    os.makedirs(DATA_DIR, exist_ok=True)

    pred_rows, sweep_rows, cfg_rows = [], [], []
    for task, sweep_dir in available.items():
        full_ref = None if args.no_full_ref else FULL_REF_DIRS.get(task)
        if full_ref is not None and not os.path.exists(rp(full_ref, "nested_full_per_fold.pkl")):
            print(f"NOTE: {task}: standalone reference {full_ref} not found - sweep only.")
            full_ref = None
        entries = collect_task(task, sweep_dir, full_ref)
        print(f"\n=== task {task}: {len(entries)} cohort sizes ===")
        for frac, per_fold, meta, oof, src in entries:
            label = f"{task} @ {frac:g}"
            assert_predictions_match_stored(per_fold, label)

            # cross-check per-fold scores against the run's own OOF CSV
            oof_map = dict(zip(oof["sample_id"], oof["y_pred_proba"]))
            strat_map = dict(zip(oof["sample_id"], oof["stratification_label"])) \
                if "stratification_label" in oof.columns else {}
            n_sub = int(meta["n_samples_subset"])
            assert len(oof) == n_sub, f"{label}: OOF rows {len(oof)} != n {n_sub}"
            for _, row in per_fold.iterrows():
                for sid, s in zip(row["test_sample_ids"], row["y_pred_proba"]):
                    assert np.isclose(oof_map[sid], s), f"{label}: OOF mismatch for {sid}"

            aucs = per_fold["auc_full_model"].to_numpy(dtype=float)
            from sklearn.metrics import roc_auc_score
            pooled = float(roc_auc_score(oof["y_true"], oof["y_pred_proba"]))
            print(f"  {frac:>4g}: n={n_sub:>3} folds={meta['n_outer_folds']:>2} "
                  f"AUROC {aucs.mean():.4f} ± {aucs.std(ddof=0):.4f} (pooled {pooled:.4f}) "
                  f"[{src}]")

            sweep_rows.append({
                "task": task, "sample_fraction": frac, "n_samples": n_sub,
                "n_outer_folds": int(meta["n_outer_folds"]),
                "subsample_seed": meta.get("subsample_seed"),
                "mean_test_auc": float(aucs.mean()),
                "std_test_auc": float(aucs.std(ddof=0)),
                "oof_auc_pooled": pooled,
                "modal_feature": per_fold["feature"].mode().iloc[0],
                "modal_classifier": per_fold["classifier"].mode().iloc[0],
                "n_distinct_configs": len(per_fold[["feature", "selection_type",
                                                    "top_genes", "classifier"]]
                                          .drop_duplicates()),
                "source_dir": src,
            })
            for _, r in per_fold.iterrows():
                thr = float(r["full_model_thresholds"]["95_percent_specificity"])
                for sid, y, s in zip(r["test_sample_ids"], r["y_test"], r["y_pred_proba"]):
                    pred_rows.append({
                        "task": task, "sample_fraction": frac, "n_samples": n_sub,
                        "n_outer_folds": int(meta["n_outer_folds"]),
                        "fold": int(r["fold"]), "sample_id": sid, "y_true": int(y),
                        "score": float(s), "threshold_95spec": thr,
                        "stratification_label": strat_map.get(sid),
                    })
                cfg_rows.append({
                    "task": task, "sample_fraction": frac, "fold": int(r["fold"]),
                    "feature": r["feature"], "classifier": r["classifier"],
                    "selection_type": r["selection_type"], "top_genes": int(r["top_genes"]),
                    "inner_cv_auc": float(r["inner_best_score"]),
                    "test_auc": float(r["auc_full_model"]),
                })

    # ---- subset-selection variance replicates (other subsample seeds) ----
    # AUROC summaries only, from each replicate's aggregated sweep CSV.
    REP_COLS = ["task", "subsample_seed", "sample_fraction", "n_samples",
                "n_outer_folds", "mean_test_auc", "std_test_auc",
                "oof_auc_pooled", "source_dir"]
    rep_rows = []
    for task in available:
        for d in sorted(glob(rp(SIZESWEEP_SEED_REPLICATE_GLOBS[task]))):
            rel = os.path.relpath(d, rp())
            csv_path = os.path.join(d, "nested_full_sizesweep.csv")
            if not os.path.exists(csv_path):
                print(f"NOTE: {task}: seed replicate {rel} not aggregated - skipped.")
                continue
            rep = pd.read_csv(csv_path)
            for _, r in rep.iterrows():
                if int(r["subsample_seed"]) == 42:
                    print(f"WARNING: {task}: {rel} contains the MAIN seed 42 - "
                          f"row skipped (it would duplicate the main curve).")
                    continue
                rep_rows.append({
                    "task": task, "subsample_seed": int(r["subsample_seed"]),
                    "sample_fraction": float(r["sample_fraction"]),
                    "n_samples": int(r["n_samples"]),
                    "n_outer_folds": int(r["n_outer_folds"]),
                    "mean_test_auc": float(r["mean_test_auc"]),
                    "std_test_auc": float(r["std_test_auc"]),
                    "oof_auc_pooled": float(r["oof_auc_pooled"]),
                    "source_dir": rel,
                })
        n_seeds = len({r["subsample_seed"] for r in rep_rows if r["task"] == task})
        print(f"{task}: {n_seeds} replicate seed(s) found.")

    for name, rows in [("figS9_predictions.csv", pred_rows),
                       ("figS9_sweep.csv", sweep_rows),
                       ("figS9_fold_configs.csv", cfg_rows)]:
        out = os.path.join(DATA_DIR, name)
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"wrote {out} ({len(rows)} rows)")
    out = os.path.join(DATA_DIR, "figS9_seed_replicates.csv")
    pd.DataFrame(rep_rows, columns=REP_COLS).to_csv(out, index=False)
    print(f"wrote {out} ({len(rep_rows)} rows)")


if __name__ == "__main__":
    main()
