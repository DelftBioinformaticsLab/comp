"""Supplementary Tables S7 (configuration and counts per outer fold) and S1 (participants), no_deep cohort.

  Part A  per task, method and outer fold: the selected configuration, inner-CV and test AUROC, the
          training-partition 95%-specificity threshold, and TP/FN/TN/FP on the test fold.
  Part B  the counts pooled over the ten folds with Clopper-Pearson 95% intervals and the per-fold
          specificity range, plus the Jiang cohort at the threshold fixed on Cristiano (HCC against
          all non-HCC participants and against each comparison group).

Inputs (data/no_deep/): figS13_fold_configs.csv, figS13_predictions.csv, fig3_baseline_scalars.csv,
fig4_baseline_scalars.csv, fig4bc_fixed_threshold.csv; ichorCNA/ichorCNA_summary.tsv;
accessory_files/cris_manifest.csv (read only).

Writes data/no_deep/tableS_folds_counts_{A_per_fold,B_pooled}.csv (S7) and
data/no_deep/tableS_participants.csv (S1: the manifest with the 30 high-coverage healthy controls
marked excluded, task membership, and per task fold, out-of-fold scores, threshold and call).
The workbooks are built by build_supplementary_tables.py.
Run (repository root):  COHORT_VARIANT=no_deep python src/figures/export/export_tableS_folds_counts.py
"""
import glob
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, os.path.dirname(__file__))
from _common import DATA_DIR, rp  # noqa: E402

TASKS = {"brca_crc": ("BRCA/CRC vs healthy", "fig3_baseline_scalars.csv"), "pancan": ("Pan-cancer vs healthy", "fig4_baseline_scalars.csv")}
ARMS = [("data_driven", "tuned", "Data-driven"), ("atac", "fixed", "ATAC"), ("rna", "fixed", "RNA")]
FEATURE = {"griffin_diff": "Griffin difference", "rcov": "RCOV", "mwh": "MWH", "fslr": "FSLR", "fslr_coverage": "FSLR (coverage)", "ulz_atac_tfbs_features": "TFBS features"}
RANK = {"mean_diff": "mean difference", "fold_change": "fold change", "p_value": "Mann-Whitney p", "weighted": "weighted"}


def cp(k, n):
    return (0.0 if k == 0 else float(beta.ppf(.025, k, n - k + 1)), 1.0 if k == n else float(beta.ppf(.975, k + 1, n - k)))


def counts(y, call):
    y, call = np.asarray(y).astype(int), np.asarray(call).astype(bool)
    return dict(TP=int((call & (y == 1)).sum()), FN=int((~call & (y == 1)).sum()), TN=int((~call & (y == 0)).sum()), FP=int((call & (y == 0)).sum()))


def pooled(task, method, c, per_fold_spec=None):
    p, n = c["TP"] + c["FN"], c["TN"] + c["FP"]
    se, sp = cp(c["TP"], p), cp(c["TN"], n)
    return dict(task=task, method=method, **c, sensitivity=c["TP"] / p, sensitivity_lo=se[0], sensitivity_hi=se[1], specificity=c["TN"] / n,
                specificity_lo=sp[0], specificity_hi=sp[1], specificity_fold_min=None if per_fold_spec is None else min(per_fold_spec),
                specificity_fold_max=None if per_fold_spec is None else max(per_fold_spec))


def hyper(h):
    try:
        d = json.loads(h)
    except (TypeError, ValueError):
        return str(h)
    return ", ".join(f"{k} = {v}" for k, v in sorted(d.items()))


def comparator_hyperparameters(t, m, clf, feature, fold, inner):
    """best_params of a fixed-region comparator fold, from its per-classifier result file (checked on the inner score)."""
    f = glob.glob(rp(f"nested_cv_results_cris/{t}_{m}_rev_nodeep/results_{clf.lower().replace(' ', '_')}_{feature}_all_features.pkl"))
    assert len(f) == 1, (t, m, clf, f)
    with open(f[0], "rb") as fh:
        d = pickle.load(fh)
    r = d[d["fold"] == fold]
    assert len(r) == 1, (t, m, clf, fold)
    r = r.iloc[0]
    assert abs(float(r["inner_cv_best_score"]) - float(inner)) < 1e-3, (t, m, clf, fold, r["inner_cv_best_score"], inner)
    return ", ".join(f"{k.split('__')[-1]} = {v}" for k, v in sorted(r["best_params"].items()))


DEEP_REASON = "high_coverage_substudy"


def participants(pred, ichor):
    """The manifest as the participant-level table of the no_deep cohort, with the data-driven out-of-fold scores."""
    m = pd.read_csv(rp("accessory_files/cris_manifest.csv"))
    deep = set(pd.read_csv(rp("accessory_files/cris_exclude_deep_healthy.txt"), header=None, comment="#")[0].astype(str).str.strip())
    assert deep <= set(m["ID"].astype(str)) or deep <= set(m["sample_name"].astype(str)), "deep-healthy list does not match the manifest"
    key = "ID" if deep <= set(m["ID"].astype(str)) else "sample_name"
    is_deep = m[key].astype(str).isin(deep)
    assert is_deep.sum() == 30 and (m.loc[is_deep, "role"] == "model").all(), (is_deep.sum(), m.loc[is_deep, "role"].unique())
    m.loc[is_deep, "role"] = "excluded"
    m.loc[is_deep, "exclusion_reason"] = DEEP_REASON
    for t, n in [("brca_crc", 244), ("pancan", 367)]:
        x = pred[(pred.task == t) & (pred.method == "data_driven") & (pred.arm == "tuned")].set_index("sample_id")
        assert len(x) == n and x.index.is_unique, (t, len(x))
        ids = m["ID"].astype(str)
        assert x.index.astype(str).isin(ids).all(), t
        m[f"in_task_{t}"] = ids.isin(x.index.astype(str))
        m[f"fold_{t}"] = ids.map(x["fold"]).astype("Int64")
        m[f"oof_probability_{t}"] = ids.map(x["score"])
        m[f"threshold_95spec_{t}"] = ids.map(x["threshold_95spec"])
        m[f"called_cancer_{t}"] = (m[f"oof_probability_{t}"] >= m[f"threshold_95spec_{t}"]).astype("boolean")
        m.loc[~m[f"in_task_{t}"], f"called_cancer_{t}"] = pd.NA   # empty outside the task
        for b in ["atac", "rna"]:   # the comparator arms of the main text and of Part A/B
            xb = pred[(pred.task == t) & (pred.method == b) & (pred.arm == "fixed")].set_index("sample_id")
            assert len(xb) == n and set(xb.index) == set(x.index), (t, b)
            assert (xb["fold"].reindex(x.index) == x["fold"]).all(), f"{t} {b}: other outer folds"
            m[f"oof_score_{b}_{t}"] = ids.map(xb["score"])
    tf = ichor.set_index("sample_id")["tumor_fraction"]
    in_any = m["in_task_pancan"] | m["in_task_brca_crc"]
    m["ichorcna_tumour_fraction"] = m["ID"].astype(str).map(tf).where(in_any)
    assert m.loc[in_any, "ichorcna_tumour_fraction"].notna().all()
    assert (m.loc[m["role"] != "model", ["in_task_brca_crc", "in_task_pancan"]].sum() == 0).all()
    assert (m["role"] == "model").sum() == 367 and (m["role"] == "panel_of_normals").sum() == 50
    assert m["features_available"].all(), "every record has features: the column carries no information"
    m = m.drop(columns=[c for c in ["gcfix_pending", "features_available"] if c in m]).rename(columns={"fragNum": "n_fragments"})
    per_task = [f"{c}_{t}" for t in ["brca_crc", "pancan"]
                for c in ["fold", "oof_probability", "threshold_95spec", "called_cancer", "oof_score_atac", "oof_score_rna"]]
    per_task.append("ichorcna_tumour_fraction")
    m = m[[c for c in m.columns if c not in per_task] + per_task]
    m.to_csv(os.path.join(DATA_DIR, "tableS_participants.csv"), index=False, float_format="%.5f", lineterminator="\n")
    print("participants:", m["role"].value_counts().to_dict(), "| task brca_crc", int(m.in_task_brca_crc.sum()), "pancan", int(m.in_task_pancan.sum()),
          "| participants", m["participant_id"].nunique(), "| PoN participants also in a task:",
          len(set(m.loc[m.role == "panel_of_normals", "participant_id"]) & set(m.loc[m.role == "model", "participant_id"])))
    return m


def main():
    cfg = pd.read_csv(os.path.join(DATA_DIR, "figS13_fold_configs.csv"))
    pred = pd.read_csv(os.path.join(DATA_DIR, "figS13_predictions.csv"))
    ichor = pd.read_csv(rp("ichorCNA/ichorCNA_summary.tsv"), sep="\t").rename(columns={"sample": "sample_id"})[["sample_id", "tumor_fraction"]]
    A, B = [], []
    for t, (tl, scal) in TASKS.items():
        for m, arm, ml in ARMS:
            x = pred[(pred.task == t) & (pred.method == m) & (pred.arm == arm)]
            total, specs = dict(TP=0, FN=0, TN=0, FP=0), []
            fold_cfg = cfg[(cfg.task == t) & (cfg.method == m) & (cfg.arm == arm)].set_index("fold")
            for f, g in x.groupby("fold"):
                c = counts(g.y_true, g.score >= g.threshold_95spec)
                total = {k: total[k] + c[k] for k in total}
                specs.append(c["TN"] / (c["TN"] + c["FP"]))
                r = fold_cfg.loc[f]
                nk = "all" if (m == "data_driven" and r.get("k", 0) >= 14221) or (m != "data_driven" and pd.isna(r.get("k"))) else (f"{int(r['k']):,}" if pd.notna(r.get("k")) else "all")
                A.append(dict(task=tl, method=ml, fold=int(f), feature=FEATURE.get(r["feature"], r["feature"]), ranking=RANK.get(r["selection_type"], r["selection_type"] if pd.notna(r["selection_type"]) else "none"),
                              k=nk, classifier=r["classifier"], hyperparameters=hyper(r["hyperparameters"]) if m == "data_driven" else comparator_hyperparameters(t, m, r["classifier"], r["feature"], f, r["inner_cv_auc"]), inner_cv_auroc=r["inner_cv_auc"], test_auroc=r["test_auc"],
                              threshold_95spec=float(g.threshold_95spec.iloc[0]), **c))
            B.append(pooled(tl, ml, total, specs))
        d = pd.read_csv(os.path.join(DATA_DIR, scal)).merge(ichor, on="sample_id")
        total, specs = dict(TP=0, FN=0, TN=0, FP=0), []
        for f, g in d.groupby("fold"):
            tr = d[d.fold != f]
            fpr, _, th = roc_curve(tr.y_true, tr.tumor_fraction)
            thr = float(th[np.where(1 - fpr >= .95)[0][-1]])
            c = counts(g.y_true, g.tumor_fraction >= thr)
            total = {k: total[k] + c[k] for k in total}
            specs.append(c["TN"] / (c["TN"] + c["FP"]))
            A.append(dict(task=tl, method="ichorCNA", fold=int(f), feature="tumour fraction", ranking="none", k="none", classifier="none", hyperparameters="", inner_cv_auroc=np.nan,
                          test_auroc=roc_auc_score(g.y_true, g.tumor_fraction), threshold_95spec=thr, **c))
        B.append(pooled(tl, "ichorCNA", total, specs))
    j = pd.read_csv(os.path.join(DATA_DIR, "fig4bc_fixed_threshold.csv"))
    j = j[j.operating_point == "95_percent_specificity"]
    # all non-HCC participants first, then each comparison group; TP/FN are the same HCC patients in every block
    JIANG_GROUPS = [("all_non_hcc", "non-HCC"), ("Healthy", "healthy"), ("Hepatitis B", "chronic hepatitis B"), ("Cirrhosis", "cirrhosis")]
    for grp, gl in JIANG_GROUPS:
        for m, ml in [("dd", "Data-driven"), ("rna", "RNA"), ("atac", "ATAC")]:
            r = j[(j.method == m) & (j.comparison_group == grp)]
            assert len(r) == 1, (m, grp, len(r))
            r = r.iloc[0]
            B.append(pooled(f"Jiang, HCC vs {gl} (threshold fixed on Cristiano)", ml, dict(TP=int(r.tp), FN=int(r.fn), TN=int(r.tn), FP=int(r.fp))))
    A, B = pd.DataFrame(A), pd.DataFrame(B)
    participants(pred, ichor)
    A.to_csv(os.path.join(DATA_DIR, "tableS_folds_counts_A_per_fold.csv"), index=False, float_format="%.4f", lineterminator="\n")
    B.to_csv(os.path.join(DATA_DIR, "tableS_folds_counts_B_pooled.csv"), index=False, float_format="%.4f", lineterminator="\n")
    print("wrote tableS_participants.csv and tableS_folds_counts_{A_per_fold,B_pooled}.csv")
    print(len(A), "fold rows;", len(B), "pooled rows")
    print(B[["task", "method", "TP", "FN", "TN", "FP"]].to_string(index=False))
    print(A[A.method == "Data-driven"].drop(columns=["method", "hyperparameters"]).round(3).to_string(index=False))
    print(A[A.method.isin(["ATAC", "RNA"])][["task", "method", "fold", "k", "classifier", "hyperparameters"]].iloc[[0, 10, 20, 30]].to_string(index=False))


if __name__ == "__main__":
    main()
