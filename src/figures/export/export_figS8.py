"""Export source data for Figure S8 (depth-downsampling robustness).

The fully nested data-driven model at full depth and on features downsampled to 1x and 0.1x,
for brca_crc and pancan; optional cross-depth rows (xd_*: full-depth models on downsampled test
folds, slurm/cross_depth_eval.sh). Sources: DEPTH_ARM_DIRS and CROSS_DEPTH_DIRS in _common.py.
Fold partitions must be identical across depths (asserted); tasks with missing arms are skipped.

Writes to src/figures/data/<COHORT_VARIANT>/: figS8_predictions.csv (OOF score, fold, label,
threshold, stage, disease per task, depth and sample), figS8_config.csv (modal configuration
and mean AUROC per task and depth) and figS8_fold_configs.csv (per-fold configurations).

Run from the repository root:
    python src/figures/export/export_figS8.py [--arm task:depth=dir ...]
--arm overrides a source directory for testing; never commit CSVs made with it.
"""

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from _common import (  # noqa: E402
    CROSS_DEPTH_DIRS,
    DATA_DIR,
    DEPTH_ARM_DIRS,
    ICHOR_AGG_CSV,
    assert_predictions_match_stored,
    rp,
)

DEPTH_ORDER = ["full", "cov1x", "cov0.1x"]


def load_nested_full_dir(result_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-fold table (columns renamed for assert_predictions_match_stored) and the OOF CSV."""
    with open(rp(result_dir, "nested_full_per_fold.pkl"), "rb") as fh:
        per_fold = pickle.load(fh)
    oof = pd.read_csv(rp(result_dir, "nested_full_oof_predictions.csv"))
    per_fold = per_fold.rename(columns={"test_auc": "auc_full_model",
                                        "n_top_genes": "top_genes"})
    return per_fold, oof


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", action="append", default=[], metavar="TASK:DEPTH=DIR",
                    help="override a source directory (testing only; repeatable)")
    args = ap.parse_args()

    dirs = {t: dict(d) for t, d in DEPTH_ARM_DIRS.items()}
    overridden = False
    for spec in args.arm:
        key, _, d = spec.partition("=")
        task, _, depth = key.partition(":")
        if task not in dirs or depth not in dirs[task] or not d:
            raise SystemExit(
                f"--arm expects <task>:<depth>=<dir> with task in {list(dirs)} and "
                f"depth in {DEPTH_ORDER}, got {spec!r}")
        dirs[task][depth] = d
        overridden = True
    if overridden:
        print("!! source-directory overrides active - do NOT commit the resulting CSVs\n")

    tasks = {}
    for task, task_dirs in dirs.items():
        missing = {k: d for k, d in task_dirs.items()
                   if not os.path.exists(rp(d, "nested_full_per_fold.pkl"))}
        if missing:
            print(f"NOTE: task '{task}' skipped - arm(s) not aggregated yet:")
            for k, d in missing.items():
                print(f"    {k:>8}: {d}/nested_full_per_fold.pkl missing")
            continue
        tasks[task] = task_dirs
    if not tasks:
        print("\nNo task has all three depth arms. Submit the missing runs, e.g.:")
        print("  FEATURES_DIR=extracted_features/downsampled/cov1x_seed42/biomart_10kb \\")
        print("  CANCER_TYPES=PANCAN EXPERIMENT_NAME=pancan_nested_full_cov1x \\")
        print("  ./slurm/submit_nested_full.sh full")
        print("(and the cov0.1x twin; drop CANCER_TYPES/use brca_crc_nested_full_cov<T>x")
        print(" for the BRCA+CRC task), then re-run this exporter.")
        raise SystemExit(1)

    os.makedirs(DATA_DIR, exist_ok=True)

    # stage + disease annotations (Cristiano et al. publication metadata)
    ann = pd.read_csv(rp(ICHOR_AGG_CSV), index_col=0).drop_duplicates(subset="ID").set_index("ID")

    pred_frames, cfg_rows, fold_rows = [], [], []
    for task, task_dirs in tasks.items():
        print(f"\n=== task {task} ===")
        arms: dict[str, pd.DataFrame] = {}
        for depth in DEPTH_ORDER:
            per_fold, oof = load_nested_full_dir(task_dirs[depth])
            assert len(per_fold) == 10, (
                f"{task}/{depth}: expected 10 folds, got {len(per_fold)}")
            oof_map = dict(zip(oof["sample_id"], oof["y_pred_proba"]))
            for _, row in per_fold.iterrows():
                for sid, s in zip(row["test_sample_ids"], row["y_pred_proba"]):
                    assert np.isclose(oof_map[sid], s), (
                        f"{task}/{depth}: OOF mismatch for {sid}")
            arms[depth] = per_fold
            print(f"loaded {depth:>8}: {task_dirs[depth]} - mean outer AUC "
                  f"{per_fold['auc_full_model'].mean():.4f} ± "
                  f"{per_fold['auc_full_model'].std(ddof=0):.4f}")

        # ------------------------------------------------ fold-partition check
        def fold_map(sel: pd.DataFrame) -> dict[int, frozenset]:
            return {int(r["fold"]): frozenset(r["test_sample_ids"])
                    for _, r in sel.iterrows()}

        ref = fold_map(arms["full"])
        for depth in DEPTH_ORDER[1:]:
            assert fold_map(arms[depth]) == ref, (
                f"{task}: outer-fold partition of {depth} differs from the full-depth "
                f"run; per-fold AUCs would not be paired")
        print("fold partitions identical across depths (per-fold AUCs are paired)")

        for depth, sel in arms.items():
            assert_predictions_match_stored(sel, f"{task}/{depth}")
        print("stored AUCs + 95%-spec confusion matrices reproduced for all depths")

        # --------------------------------------------------------- assemble
        records = []
        for depth, sel in arms.items():
            for _, row in sel.iterrows():
                thr = float(row["full_model_thresholds"]["95_percent_specificity"])
                for sid, y, s in zip(row["test_sample_ids"], row["y_test"],
                                     row["y_pred_proba"]):
                    records.append({"task": task, "sample_id": sid, "depth": depth,
                                    "fold": int(row["fold"]), "y_true": int(y),
                                    "score": float(s), "threshold_95spec": thr})
        task_pred = pd.DataFrame(records)

        # ---------------------------- optional cross-depth rows (xd_<depth>)
        # full-depth models on downsampled test folds; its 'full' rows must match the full arm
        xd_dir = CROSS_DEPTH_DIRS.get(task)
        xd_csv = rp(xd_dir, "cross_depth_predictions.csv") if xd_dir else None
        if xd_csv and os.path.exists(xd_csv):
            xd = pd.read_csv(xd_csv)
            full_arm = task_pred[task_pred["depth"] == "full"].set_index(["fold", "sample_id"])
            xd_full = xd[xd["depth"] == "full"].set_index(["fold", "sample_id"])
            joined = full_arm.join(xd_full, rsuffix="_xd", how="inner")
            assert len(joined) == len(full_arm), (
                f"{task}: cross-depth file covers {len(joined)} of {len(full_arm)} "
                f"full-arm predictions")
            assert np.allclose(joined["score"], joined["score_xd"]), (
                f"{task}: cross-depth 'full' predictions deviate from the full arm - "
                f"stale cross_depth run? re-run slurm/cross_depth_eval.sh")
            xd_depths = [d for d in xd["depth"].unique() if d != "full"]
            for d in xd_depths:
                sub = xd[xd["depth"] == d]
                task_pred = pd.concat([task_pred, pd.DataFrame({
                    "task": task, "sample_id": sub["sample_id"], "depth": f"xd_{d}",
                    "fold": sub["fold"].astype(int), "y_true": sub["y_true"].astype(int),
                    "score": sub["score"], "threshold_95spec": sub["threshold_95spec"],
                })], ignore_index=True)
            xd_fold = pd.read_csv(rp(xd_dir, "cross_depth_per_fold.csv"))
            for d in xd_depths:
                sub = xd_fold[xd_fold["depth"] == d]
                cfg_rows.append({
                    "task": task, "depth": f"xd_{d}",
                    "modal_feature": arms["full"]["feature"].mode().iloc[0],
                    "modal_classifier": arms["full"]["classifier"].mode().iloc[0],
                    "modal_weighting_function": arms["full"]["selection_type"].mode().iloc[0],
                    "modal_top_genes": int(arms["full"]["top_genes"].mode().iloc[0]),
                    "n_distinct_configs": len(arms["full"][["feature", "selection_type",
                                                            "top_genes", "classifier"]]
                                              .drop_duplicates()),
                    "mean_auc": float(sub["auc"].mean()),
                    "std_auc": float(sub["auc"].std(ddof=0)),
                    "source_dir": xd_dir,
                    "note": "full-depth models re-evaluated on downsampled test folds",
                })
            print(f"cross-depth rows appended: {['xd_' + d for d in xd_depths]} "
                  f"('full' rows verified against the full arm)")
        elif xd_dir:
            print(f"NOTE: no cross-depth results at {xd_dir} - xd rows omitted "
                  f"(sbatch slurm/cross_depth_eval.sh to add them).")

        task_pred["stage"] = task_pred["sample_id"].map(ann["Stage"])
        task_pred["disease"] = task_pred["sample_id"].map(ann["disease"])
        pred_frames.append(task_pred)
        n_task = task_pred[task_pred["depth"] == "full"]["sample_id"].nunique()
        print(f"task samples: {n_task}; rows: {len(task_pred)} "
              f"({task_pred['depth'].nunique()} depths)")

        for depth, sel in arms.items():
            n_distinct = len(sel[["feature", "selection_type", "top_genes",
                                  "classifier"]].drop_duplicates())
            cfg_rows.append({
                "task": task, "depth": depth,
                "modal_feature": sel["feature"].mode().iloc[0],
                "modal_classifier": sel["classifier"].mode().iloc[0],
                "modal_weighting_function": sel["selection_type"].mode().iloc[0],
                "modal_top_genes": int(sel["top_genes"].mode().iloc[0]),
                "n_distinct_configs": n_distinct,
                "mean_auc": float(sel["auc_full_model"].mean()),
                "std_auc": float(sel["auc_full_model"].std(ddof=0)),
                "source_dir": task_dirs[depth],
                "note": "fully nested: config chosen per fold; modal shown",
            })
            for _, r in sel.iterrows():
                fold_rows.append({"task": task, "depth": depth, "fold": int(r["fold"]),
                                  "feature": r["feature"], "classifier": r["classifier"],
                                  "selection_type": r["selection_type"],
                                  "top_genes": int(r["top_genes"]),
                                  "inner_cv_auc": float(r["inner_best_score"]),
                                  "test_auc": float(r["auc_full_model"])})

    pred = pd.concat(pred_frames, ignore_index=True)
    out = os.path.join(DATA_DIR, "figS8_predictions.csv")
    pred.to_csv(out, index=False)
    print(f"\nwrote {out} ({len(pred)} rows, tasks: {sorted(tasks)})")

    cfg = pd.DataFrame(cfg_rows)
    out_cfg = os.path.join(DATA_DIR, "figS8_config.csv")
    cfg.to_csv(out_cfg, index=False)
    print(f"wrote {out_cfg}")
    print(cfg[["task", "depth", "modal_feature", "modal_classifier",
               "n_distinct_configs", "mean_auc", "std_auc"]].to_string(index=False))

    out_folds = os.path.join(DATA_DIR, "figS8_fold_configs.csv")
    pd.DataFrame(fold_rows).to_csv(out_folds, index=False)
    print(f"wrote {out_folds} ({len(fold_rows)} rows)")


if __name__ == "__main__":
    main()
