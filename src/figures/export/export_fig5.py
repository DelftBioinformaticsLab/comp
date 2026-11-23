"""Source data for Figure 5 (pan-cancer) and Supplementary Figure S9 (BRCA/CRC):
  a, b) AUROC vs number of selected TSSs (k), per feature and per classifier;
  c)    AUROC by aggregation level (genome-wide, chromosome arms, Hallmark sets, individual TSSs);
  and AUROC vs k per ranking function (Supplementary Figure S9).

Inputs: the per-configuration sweep (DD_SWEEP_DIRS, slurm/nested_cv_dd.sh; configurations are
not chosen inside the folds, so the best curve is descriptive),
extracted_features/tss_sets/genome_pooled_<feature>[_pancan].npy (build_tss_set_matrices.py;
scalars oriented on the training folds) and figS_tss_sets_{runs,reference,predictions}.csv
(export_figS_tss_sets.py). Incomplete sweeps or missing matrices are reported and left out.

Writes (DATA_DIR):
  fig5_auc_sweep.csv          task, feature, classifier, weighting_function, top_genes, fold, auc
  fig5c_resolution.csv        task, level, n_units, feature, mode, auc_fold_mean, auc_fold_sd,
                              auc_pooled_oof, protocol, source
  fig5c_genome_predictions.csv  the oriented genome-wide scalars per sample and fold
  fig5_status.csv             what exists per task and source
Run (repository root; COHORT_VARIANT defaults to no_deep):
    python src/figures/export/export_fig5.py
"""

import os
import sys
from glob import glob

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from _common import DATA_DIR, DD_SWEEP_DIRS, DD_SWEEP_RANKINGS, load_results, rp  # noqa: E402
from export_fig3_baselines import evaluate_scalar  # noqa: E402

TASKS = {"brca_crc": dict(fold_col="fold_brca_crc", in_col="in_task_brca_crc", suffix=""),
         "pancan": dict(fold_col="fold_pancan", in_col="in_task_pancan", suffix="_pancan")}
FEATURES = ["griffin_diff", "rcov", "mwh", "fslr", "fslr_coverage"]
N_CLASSIFIERS, N_K, N_FOLDS = 6, 12, 10
N_TSS = 14221                      # k = 15000 means "all TSSs left after the filter"
SET_LEVELS = {"arm": "chromosome arms", "hallmark": "Hallmark gene sets"}
KEEP = ["feature", "classifier", "top_genes", "fold", "auc_full_model"]


def export_sweep(task, status):
    d = DD_SWEEP_DIRS[task]
    frames = []
    for wf in DD_SWEEP_RANKINGS:
        if glob(rp(f"{d}/results*{wf}.pkl")):
            frames.append(load_results(f"{d}/results*{wf}.pkl", KEEP, extra={"weighting_function": wf}))
    expected = len(DD_SWEEP_RANKINGS) * N_CLASSIFIERS * len(FEATURES) * N_K * N_FOLDS
    n = sum(len(f) for f in frames)
    state = "complete" if n == expected else ("pending" if n == 0 else "partial")
    status.append(dict(task=task, source="sweep", path=d, rows=n, expected=expected, state=state))
    if state != "complete":
        print(f"  {task} sweep: {n}/{expected} fold rows in {d} ({state}) - left out")
        return None
    dd = pd.concat(frames, ignore_index=True)
    dd["top_genes"] = dd["top_genes"].replace(15000, N_TSS)
    out = dd.rename(columns={"auc_full_model": "auc"})
    out.insert(0, "task", task)
    best = out.groupby(["feature", "classifier", "weighting_function", "top_genes"])["auc"].mean()
    print(f"  {task} sweep: {len(out)} fold rows; best configuration mean AUROC {best.max():.3f} "
          f"({' / '.join(map(str, best.idxmax()))}; chosen on the test folds, descriptive)")
    return out[["task", "feature", "classifier", "weighting_function", "top_genes", "fold", "auc"]]


def fold_stats(df):
    aucs = [roc_auc_score(g["y_true"], g["score"]) for _, g in df.groupby("fold")]
    return float(np.mean(aucs)), float(np.std(aucs)), float(roc_auc_score(df["y_true"], df["score"]))


def export_resolution(task, cfg, part, status):
    rows, preds = [], []
    t = part[part[cfg["in_col"]] == True].copy()  # noqa: E712
    t["fold"] = t[cfg["fold_col"]].astype(int)
    t["y_true"] = (t["disease"] != "Healthy").astype(int)
    t = t.rename(columns={"ID": "sample_id"})

    # the folds of the nested set runs must be these frozen folds
    sp = pd.read_csv(os.path.join(DATA_DIR, "figS_tss_sets_predictions.csv"))
    sp = sp[sp["task"] == task].drop_duplicates("sample_id").set_index("sample_id")["fold"]
    assert (sp.reindex(t["sample_id"]).to_numpy() == t["fold"].to_numpy()).all(), f"{task}: set-run folds differ"

    # genome-wide: one pooled value per sample, oriented on the training folds
    for f in FEATURES:
        stem = rp(f"extracted_features/tss_sets/genome_pooled_{f}{cfg['suffix']}")
        if not os.path.exists(stem + ".npy"):
            status.append(dict(task=task, source=f"genome_{f}", path=os.path.relpath(stem, rp(".")) + ".npy", rows=0, expected=1, state="pending"))
            print(f"  {task} genome-wide {f}: matrix missing (sbatch slurm/extract_tss_sets.sh genome, then "
                  f"python src/comp/build_tss_set_matrices.py pooled) - left out")
            continue
        X = np.load(stem + ".npy")
        meta = pd.read_csv(stem + ".metadata.csv")
        assert X.shape == (len(meta), 1)
        col = f"genome_{f}"
        t[col] = t["sample_id"].map(pd.Series(X[:, 0], index=meta["sample_id"]))
        assert t[col].notna().all(), f"{task} {col}: task samples missing from the matrix"
        p = evaluate_scalar(t, col)
        p.insert(0, "task", task)
        preds.append(p)
        m, s, pooled = fold_stats(p)
        rows.append(dict(task=task, level="genome-wide", n_units=1, feature=f, mode="pooled", auc_fold_mean=m,
                         auc_fold_sd=s, auc_pooled_oof=pooled, protocol="scalar oriented on the training folds",
                         source=os.path.relpath(stem, rp(".")) + ".npy"))
        status.append(dict(task=task, source=col, path=os.path.relpath(stem, rp(".")) + ".npy", rows=len(p), expected=len(t), state="complete"))

    # chromosome arms and Hallmark sets: fully nested set runs
    runs = pd.read_csv(os.path.join(DATA_DIR, "figS_tss_sets_runs.csv"))
    for _, r in runs[runs["task"] == task].iterrows():
        rows.append(dict(task=task, level=SET_LEVELS[r["set_type"]], n_units=int(r["n_sets"]), feature=r["feature"],
                         mode=r["mode"], auc_fold_mean=float(r["auc_fold_mean"]), auc_fold_sd=float(r["auc_fold_sd"]),
                         auc_pooled_oof=float(r["auc_pooled_oof"]), protocol="fully nested (classifier per fold)",
                         source="figS_tss_sets_runs.csv"))
    # individual TSSs: the fully nested data-driven model (feature, k and classifier chosen per fold)
    ref = pd.read_csv(os.path.join(DATA_DIR, "figS_tss_sets_reference.csv"))
    r = ref[(ref["task"] == task) & (ref["method"] == "data_driven")].iloc[0]
    rows.append(dict(task=task, level="individual TSSs", n_units=N_TSS, feature="selected per fold", mode="",
                     auc_fold_mean=float(r["auc_fold_mean"]), auc_fold_sd=float(r["auc_fold_sd"]),
                     auc_pooled_oof=float(r["auc_pooled_oof"]), protocol="fully nested data-driven model",
                     source="figS_tss_sets_reference.csv"))
    return rows, preds


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    part = pd.read_csv(os.path.join(DATA_DIR, "tableS_participants.csv"))
    status, sweeps, res_rows, res_preds = [], [], [], []
    for task, cfg in TASKS.items():
        print(f"{task}:")
        s = export_sweep(task, status)
        if s is not None:
            sweeps.append(s)
        r, p = export_resolution(task, cfg, part, status)
        res_rows += r
        res_preds += p

    cols = ["task", "feature", "classifier", "weighting_function", "top_genes", "fold", "auc"]
    (pd.concat(sweeps, ignore_index=True) if sweeps else pd.DataFrame(columns=cols)).to_csv(
        os.path.join(DATA_DIR, "fig5_auc_sweep.csv"), index=False, lineterminator="\n")
    res = pd.DataFrame(res_rows)
    res.to_csv(os.path.join(DATA_DIR, "fig5c_resolution.csv"), index=False, lineterminator="\n")
    (pd.concat(res_preds, ignore_index=True) if res_preds else pd.DataFrame()).to_csv(
        os.path.join(DATA_DIR, "fig5c_genome_predictions.csv"), index=False, lineterminator="\n")
    pd.DataFrame(status).to_csv(os.path.join(DATA_DIR, "fig5_status.csv"), index=False, lineterminator="\n")

    print("\npanel c, fold-mean AUROC (best feature/mode per level):")
    for (task, level), g in res.groupby(["task", "level"], sort=False):
        b = g.loc[g["auc_fold_mean"].idxmax()]
        print(f"  {task:8s} {level:20s} n = {int(b['n_units']):>6,}: {b['auc_fold_mean']:.3f} ({b['feature']} {b['mode']})")
    print(f"wrote fig5_auc_sweep.csv ({sum(len(s) for s in sweeps)} rows), fig5c_resolution.csv ({len(res)} rows), "
          f"fig5c_genome_predictions.csv, fig5_status.csv to {DATA_DIR}")


if __name__ == "__main__":
    main()
