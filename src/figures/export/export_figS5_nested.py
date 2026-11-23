"""Source data for the label-permutation null of the fully nested procedure (Supplementary
Figure S5), both tasks. Labels are shuffled once per seed before the outer folds are drawn and
the whole selection is repeated inside every outer fold (slurm/nested_full.sh with
LABEL_PERMUTATION_SEED, experiment <task>_nested_full_perm<seed>_nodeep). The pan-cancer
permutations search K_GRID="10000 15000" (the values the true-label folds chose); the
true-label reference is the task's main nested run.

Writes data/no_deep/figS5_nested_runs.csv (one row per run), figS5_nested_folds.csv (selected
configuration per outer fold) and figS5_nested_oof.csv (out-of-fold scores), each with a `task`
column. Unaggregated runs are skipped, or read from their fold partials with --draft.

Run from the repository root:
    COHORT_VARIANT=no_deep python src/figures/export/export_figS5_nested.py [--draft]
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
from glob import glob

import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from _common import DATA_DIR, NESTED_FULL_DIR, PANCAN_NESTED_FULL_DIR, rp  # noqa: E402

TASKS = {   # task: (true-label run, permutation runs)
    "pancan": (PANCAN_NESTED_FULL_DIR, "nested_cv_results_cris/nested_full_pancan_nested_full_perm*_nodeep"),
    "brca_crc": (NESTED_FULL_DIR, "nested_cv_results_cris/nested_full_brca_crc_nested_full_perm*_nodeep"),
}
N_TSS = 14221   # TSSs after filtering; k = 15000 in the grid means all of them


def k_label(ks) -> str:
    ks = sorted(int(k) for k in ks)
    return " ".join("all" if k >= N_TSS else str(k) for k in ks)


def read_run(task: str, run: str, seed, d: str) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    with open(rp(d, "nested_full_per_fold.pkl"), "rb") as fh:
        pf = pickle.load(fh)
    with open(rp(d, "nested_full_summary.json")) as fh:
        summary = json.load(fh)
    assert summary.get("label_permutation_seed") == seed, f"{d}: summary says permutation seed {summary.get('label_permutation_seed')}"
    oof = pd.read_csv(rp(d, "nested_full_oof_predictions.csv"))
    row = {"task": task, "run": run, "permutation_seed": seed, "n_samples": len(oof), "n_folds": len(pf),
           "auc_fold_mean": float(pf["test_auc"].mean()), "auc_fold_sd": float(pf["test_auc"].std(ddof=0)),
           "auc_pooled_oof": float(roc_auc_score(oof["y_true"], oof["y_pred_proba"])),
           "k_grid": k_label(summary["top_genes_searched"]), "complete": True, "source": d}
    folds = pf[["fold", "feature", "selection_type", "n_top_genes", "classifier", "inner_best_score", "test_auc"]].copy()
    folds.insert(0, "run", run)
    folds.insert(0, "task", task)
    oof = oof[["sample_id", "y_true", "outer_fold", "y_pred_proba"]].rename(columns={"y_pred_proba": "score"})
    oof.insert(0, "run", run)
    oof.insert(0, "task", task)
    return row, folds, oof


def read_partial_run(task: str, run: str, seed, d: str) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """An unaggregated run from its finished fold partials (draft figures only)."""
    recs, oofs = [], []
    for f in sorted(glob(rp(d, "nested_full_fold_*.pkl"))):
        with open(f, "rb") as fh:
            p = pickle.load(fh)
        assert p["meta"].get("label_permutation_seed") == seed, f"{f}: permutation seed {p['meta']}"
        recs.append(p["fold_record"])
        oofs.append(pd.DataFrame(p["oof_rows"]))
    pf, oof = pd.DataFrame(recs).sort_values("fold"), pd.concat(oofs, ignore_index=True)
    ks = sorted(pf["n_top_genes"].astype(int).unique())
    row = {"task": task, "run": run, "permutation_seed": seed, "n_samples": len(oof), "n_folds": len(pf),
           "auc_fold_mean": float(pf["test_auc"].mean()), "auc_fold_sd": float(pf["test_auc"].std(ddof=0)),
           "auc_pooled_oof": float(roc_auc_score(oof["y_true"], oof["y_pred_proba"])),
           "k_grid": k_label(ks), "complete": False, "source": d}   # grid as selected so far (the summary holds the searched grid)
    folds = pf[["fold", "feature", "selection_type", "n_top_genes", "classifier", "inner_best_score", "test_auc"]].copy()
    folds.insert(0, "run", run)
    folds.insert(0, "task", task)
    oof = oof[["sample_id", "y_true", "outer_fold", "y_pred_proba"]].rename(columns={"y_pred_proba": "score"})
    oof.insert(0, "run", run)
    oof.insert(0, "task", task)
    return row, folds, oof


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", action="store_true", help="include unaggregated runs from their fold partials")
    args = ap.parse_args()
    runs, pending = [], []
    for task, (true_dir, perm_glob) in TASKS.items():
        task_runs = [read_run(task, "true_labels", None, true_dir)]
        for d in sorted(glob(rp(perm_glob)), key=lambda p: int(re.search(r"_perm(\d+)_", p).group(1))):
            rel = os.path.relpath(d, rp(""))
            seed = int(re.search(r"_perm(\d+)_", os.path.basename(d)).group(1))
            if not os.path.exists(os.path.join(d, "nested_full_per_fold.pkl")):
                n_folds = len(glob(os.path.join(d, "nested_full_fold_*.pkl")))
                pending.append(f"{rel} ({n_folds}/10 folds)" + (" - included as a draft" if args.draft and n_folds else ""))
                if args.draft and n_folds:
                    task_runs.append(read_partial_run(task, f"perm{seed}", seed, rel))
                continue
            task_runs.append(read_run(task, f"perm{seed}", seed, rel))
        # the true-label reference is valid for a restricted grid only if its own selections lie inside it
        true_k = set(task_runs[0][1]["n_top_genes"].astype(int).map(lambda k: min(k, N_TSS)))
        for row, _, _ in task_runs[1:]:
            grid = {N_TSS if k == "all" else int(k) for k in row["k_grid"].split()}
            row["true_selection_in_grid"] = true_k <= grid
            if not row["true_selection_in_grid"]:
                print(f"WARNING {task} {row['run']}: true-label folds chose k {sorted(true_k)}, outside the grid {row['k_grid']}")
        runs += task_runs
    table = pd.DataFrame([r[0] for r in runs])
    table.to_csv(os.path.join(DATA_DIR, "figS5_nested_runs.csv"), index=False)
    pd.concat([r[1] for r in runs]).to_csv(os.path.join(DATA_DIR, "figS5_nested_folds.csv"), index=False)
    pd.concat([r[2] for r in runs]).to_csv(os.path.join(DATA_DIR, "figS5_nested_oof.csv"), index=False)
    print(table.drop(columns="source").round(3).to_string(index=False))
    if pending:
        print("\nnot aggregated yet:\n  " + "\n  ".join(pending))
    print(f"\nwrote figS5_nested_*.csv to {os.path.relpath(DATA_DIR)}: "
          + ", ".join(f"{t} {int((table['task'] == t).sum()) - 1} permutations" for t in TASKS))


if __name__ == "__main__":
    main()
