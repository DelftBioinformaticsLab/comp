"""TSS features aggregated over Hallmark pathways and chromosome arms (no_deep cohort).

Reads the runs of slurm/nested_cv_tss_sets.sh on the set-level matrices of
src/comp/build_tss_set_matrices.py (2 set types x 2 modes x 5 features per task, frozen outer
folds). The classifier is chosen per outer fold on its inner-CV AUROC (fully nested); the post-hoc
best classifier is reported beside it as a diagnostic only. Reference arms (data-driven, ATAC,
RNA) come from fig3_predictions.csv / fig4a_predictions.csv.

Writes data/no_deep/figS_tss_sets_{runs,folds,predictions,reference}.csv; unfinished runs are
listed and skipped.
Run from the repository root:  python src/figures/export/export_figS_tss_sets.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from _common import DATA_DIR, best_mean_auc_config, load_results, rp  # noqa: E402

TASKS = {
    "brca_crc": dict(scalars="fig3_baseline_scalars.csv", reference="fig3_predictions.csv", matrix_suffix=""),
    "pancan": dict(scalars="fig4_baseline_scalars.csv", reference="fig4a_predictions.csv", matrix_suffix="_pancan"),
}
SET_TYPES = ["hallmark", "arm"]
MODES = ["averaged", "pooled"]
FEATURES = ["griffin_diff", "rcov", "mwh", "fslr", "fslr_coverage"]
REFERENCE_METHODS = ["data_driven", "atac", "rna"]
KEEP = ["feature", "classifier", "fold", "auc_full_model", "inner_cv_best_score", "best_params",
        "test_sample_ids", "y_test", "y_pred_proba", "full_model_thresholds"]


def fold_mean_auc(df: pd.DataFrame) -> tuple[float, float]:
    a = [roc_auc_score(g["y_true"], g["score"]) for _, g in df.groupby("fold")]
    return float(np.mean(a)), float(np.std(a))


def export_run(task: str, st: str, mode: str, feat: str, task_fold: dict) -> tuple[dict, list, list] | None:
    run_dir = f"nested_cv_results_cris/{task}_tss_sets_{st}_{mode}_{feat}_nodeep"
    if not os.path.isdir(rp(run_dir)):
        return None
    res = load_results(f"{run_dir}/results*.pkl", KEEP)
    if res["classifier"].nunique() < 6:
        print(f"  {run_dir}: only {res['classifier'].nunique()} of 6 classifiers - skipped (incomplete)")
        return None
    assert (res["feature"] == f"{st}_{mode}_{feat}").all(), f"{run_dir}: unexpected feature {res['feature'].unique()}"
    assert res["inner_cv_best_score"].notna().all(), f"{run_dir}: no inner_cv_best_score"
    sel = res.loc[res.groupby("fold")["inner_cv_best_score"].idxmax()].sort_values("fold")
    assert sorted(sel["fold"]) == list(range(1, 11)), f"{run_dir}: expected 10 folds"
    for _, r in sel.iterrows():                       # frozen-fold identity with the headline run
        seen = {task_fold[s] for s in r["test_sample_ids"]}
        assert seen == {int(r["fold"])}, f"{run_dir} fold {r['fold']}: test samples span folds {seen}"

    pred = [{"task": task, "set_type": st, "mode": mode, "feature": feat, "fold": int(r["fold"]),
             "sample_id": sid, "y_true": int(y), "score": float(s)}
            for _, r in sel.iterrows() for sid, y, s in zip(r["test_sample_ids"], r["y_test"], r["y_pred_proba"])]
    p = pd.DataFrame(pred)
    m, sd = fold_mean_auc(p)
    assert np.isclose(m, sel["auc_full_model"].mean(), atol=1e-6), f"{run_dir}: fold AUCs do not reproduce"
    posthoc = best_mean_auc_config(res, ["classifier"])
    names = pd.read_csv(rp(f"extracted_features/tss_sets/{st}_{mode}_{feat}{TASKS[task]['matrix_suffix']}.tf_names.csv"))
    chosen = sel["classifier"].value_counts()
    row = {"task": task, "set_type": st, "mode": mode, "feature": feat, "n_sets": len(names),
           "auc_fold_mean": m, "auc_fold_sd": sd, "auc_pooled_oof": float(roc_auc_score(p["y_true"], p["score"])),
           "classifiers": ", ".join(f"{k} x{v}" for k, v in chosen.items()),
           "posthoc_best_classifier": posthoc["classifier"], "posthoc_auc_fold_mean": float(posthoc["mean_auc"])}
    folds = [{"task": task, "set_type": st, "mode": mode, "feature": feat, "fold": int(r["fold"]),
              "classifier": r["classifier"], "inner_cv_auc": float(r["inner_cv_best_score"]),
              "test_auc": float(r["auc_full_model"]), "best_params": str(r["best_params"])} for _, r in sel.iterrows()]
    return row, folds, pred


def main() -> None:
    runs, folds, preds, refs, missing = [], [], [], [], []
    for task, cfg in TASKS.items():
        sc = pd.read_csv(os.path.join(DATA_DIR, cfg["scalars"]), usecols=["sample_id", "fold"])
        task_fold = dict(zip(sc["sample_id"], sc["fold"].astype(int)))
        ref = pd.read_csv(os.path.join(DATA_DIR, cfg["reference"]))
        for meth in REFERENCE_METHODS:
            g = ref[ref["method"] == meth]
            m, sd = fold_mean_auc(g)
            refs.append({"task": task, "method": meth, "auc_fold_mean": m, "auc_fold_sd": sd,
                         "auc_pooled_oof": float(roc_auc_score(g["y_true"], g["score"]))})
        for st in SET_TYPES:
            for mode in MODES:
                for feat in FEATURES:
                    out = export_run(task, st, mode, feat, task_fold)
                    if out is None:
                        missing.append(f"{task}_tss_sets_{st}_{mode}_{feat}_nodeep")
                        continue
                    runs.append(out[0]); folds += out[1]; preds += out[2]

    runs, refs = pd.DataFrame(runs), pd.DataFrame(refs)
    runs.to_csv(os.path.join(DATA_DIR, "figS_tss_sets_runs.csv"), index=False)
    pd.DataFrame(folds).to_csv(os.path.join(DATA_DIR, "figS_tss_sets_folds.csv"), index=False)
    pd.DataFrame(preds).to_csv(os.path.join(DATA_DIR, "figS_tss_sets_predictions.csv"), index=False)
    refs.to_csv(os.path.join(DATA_DIR, "figS_tss_sets_reference.csv"), index=False)

    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print("\nreference arms (fold-mean AUROC):\n" + refs.to_string(index=False))
        print("\nTSS-set runs (fully nested, classifier per fold):")
        print(runs.drop(columns=["posthoc_best_classifier"]).sort_values(
            ["task", "auc_fold_mean"], ascending=[True, False]).to_string(index=False))
    if missing:
        print(f"\n{len(missing)} run(s) not (yet) available: " + ", ".join(missing))
    print(f"\nwrote figS_tss_sets_*.csv to {os.path.relpath(DATA_DIR)} ({len(runs)} runs)")


if __name__ == "__main__":
    main()
