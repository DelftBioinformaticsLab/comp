#!/usr/bin/env python
"""Source data for the leave-one-cancer-type-out (LOCO) analysis of the pan-cancer task.

Per cancer type, slurm/submit_loco.sh (splits from src/comp/build_loco_splits.py) runs a
single-CV best-model search on the training healthy donors and the other six types, refits it
with a frozen 95%-specificity threshold and deploys it once on the left-out type plus 34
held-out healthy donors. This script only reads the stored deployments; nothing is refitted.
The reference ("type in training") is the fully nested pan-cancer CV (fig4a_predictions.csv).

Outputs (data/no_deep/):
    fig4_loco_per_type.csv                per left-out type: configuration, AUROC with bootstrap
                                          interval, frozen-threshold sensitivity / specificity
    fig4_loco_predictions.csv             per sample and left-out type: score, threshold, call
    fig4_loco_ksweep.csv                  per type and number of TSSs: AUROC on the left-out set
    fig4_loco_summary.json                totals over the seven types
    fig4_loco_comparators.csv             the same per-type rows for the RNA and ATAC arms
                                          (ARM=rna|atac slurm/submit_loco.sh); missing arms are
                                          reported as pending
    fig4_loco_comparator_predictions.csv  per sample, arm and left-out type
Run (repository root, no_deep only):
    python src/figures/export/export_fig4_loco.py
"""
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from _common import COHORT_VARIANT, DATA_DIR, rp  # noqa: E402

TYPES = {   # key of slurm/submit_loco.sh -> (disease label of the manifest, label in the figure)
    "breast": ("Breast cancer", "Breast"),
    "pancreatic": ("Pancreatic cancer", "Pancreatic"),
    "ovarian": ("Ovarian cancer", "Ovarian"),
    "gastric": ("Gastric cancer", "Gastric"),
    "colorectal": ("Colorectal cancer", "Colorectal"),
    "bileduct": ("Bile duct cancer", "Bile duct"),
    "lung": ("Lung cancer naive nsclc", "Lung"),
}
N_BOOT, SEED = 2000, 42


COMPARATORS = {"rna": 1, "atac": 1}   # arm -> configurations in its search (ARM=<arm> slurm/submit_loco.sh)


def deploy_dir(key, arm="dd"):
    return f"nested_cv_results_cris/external_validation_comprehensive_external_cris_on_cris_loco_{key}_{arm}_rev_nodeep"


def search_dir(key, arm="dd"):
    return f"nested_cv_results_cris/best_model_best_model_search_{arm}_loco_{key}_rev_nodeep"


def comparator_rows(task, held_ref):
    """Per-type rows for the RNA / ATAC arms; arms/types not on disk are returned as pending.
    Reference = the fully nested pan-cancer CV of the same arm (fig4a_predictions.csv)."""
    fig4a = pd.read_csv(os.path.join(DATA_DIR, "fig4a_predictions.csv"))
    rows, preds, pending = [], [], []
    for arm, n_want in COMPARATORS.items():
        ref = fig4a[fig4a["method"] == arm].set_index("sample_id")["score"]
        assert len(ref) == 367 and ref.index.is_unique, arm
        for key, (disease, label) in TYPES.items():
            d = deploy_dir(key, arm)
            summ_path = rp(d, "comprehensive_validation_summary.pkl")
            sdir = rp(search_dir(key, arm))
            n_cfg = len([f for f in os.listdir(sdir) if f.startswith("summary_") and f.endswith(".pkl")]) if os.path.isdir(sdir) else 0
            if not os.path.isfile(summ_path) or n_cfg < n_want:
                pending.append(f"{arm}/{key}")
                continue
            with open(summ_path, "rb") as fh:
                summ = pickle.load(fh)
            fr = {k: v for k, v in summ["feature_results"].items() if v is not None}
            best = max(fr, key=lambda k: fr[k].get("training_cv_auc", 0))
            with open(rp(d, f"external_validation_{best}_results.pkl"), "rb") as fh:
                r = pickle.load(fh)
            info = r["best_model_info"]
            assert r["training_dataset"] == search_dir(key, arm), (arm, key, r["training_dataset"])
            ids = [str(x) for x in r["sample_ids"]]
            y = np.asarray(r["y_true"], dtype=int)
            s = np.asarray(r["y_pred_proba"], dtype=float)
            assert np.isclose(roc_auc_score(y, s), r["overall_auc"], atol=1e-9), (arm, key)
            healthy_ids = sorted(i for i, yi in zip(ids, y) if yi == 0)
            cancer_ids = [i for i, yi in zip(ids, y) if yi == 1]
            assert healthy_ids == held_ref, f"{arm}/{key}: test healthy donors differ from the TSS model's"
            assert set(task.loc[cancer_ids, "disease"]) == {disease} and len(cancer_ids) == int((task["disease"] == disease).sum()), (arm, key)
            thr = float(r["deployment_thresholds"]["95_percent_specificity"])
            tp, fn = int(((s >= thr) & (y == 1)).sum()), int(((s < thr) & (y == 1)).sum())
            tn, fp = int(((s < thr) & (y == 0)).sum()), int(((s >= thr) & (y == 0)).sum())
            auc, lo, hi = boot_auc(y, s)
            ref_ids = healthy_ids + cancer_ids
            ref_y = np.r_[np.zeros(len(healthy_ids), int), np.ones(len(cancer_ids), int)]
            ref_auc, ref_lo, ref_hi = boot_auc(ref_y, ref.loc[ref_ids].to_numpy())
            rows.append({"arm": arm, "type": key, "disease": disease, "label": label,
                         "n_cancer": len(cancer_ids), "n_healthy": len(healthy_ids),
                         "feature": info["feature"], "classifier": info["classifier"], "n_features": int(info["n_genes"]),
                         "training_cv_auc": float(info["cv_auc"]), "auc": auc, "auc_lo": lo, "auc_hi": hi,
                         "threshold_95spec_train": thr, "tp": tp, "fn": fn, "tn": tn, "fp": fp,
                         "sensitivity": tp / (tp + fn), "specificity": tn / (tn + fp),
                         "auc_in_training": ref_auc, "auc_in_training_lo": ref_lo, "auc_in_training_hi": ref_hi,
                         "source_dir": d})
            for i, yi, si in zip(ids, y, s):
                preds.append({"arm": arm, "type": key, "sample_id": i, "y_true": int(yi), "score": float(si),
                              "threshold_95spec_train": thr, "called_cancer": bool(si >= thr),
                              "score_in_training": float(ref.loc[i])})
            print(f"  {arm:4s} {label:10s}: {info['classifier']} (training CV {info['cv_auc']:.3f}) | left out {auc:.3f} "
                  f"[{lo:.2f}-{hi:.2f}] | in training {ref_auc:.3f} | frozen threshold: sens {tp}/{tp + fn}, spec {tn}/{tn + fp}")
    return rows, preds, pending


def boot_auc(y, s, n_boot=N_BOOT, seed=SEED):
    """Participant bootstrap of the AUROC, resampling within each class."""
    y, s = np.asarray(y, dtype=int), np.asarray(s, dtype=float)
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        boots[b] = roc_auc_score(y[idx], s[idx])
    return float(roc_auc_score(y, s)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def exact_ci(k, n):
    """Clopper-Pearson 95% interval."""
    lo = 0.0 if k == 0 else float(beta_dist.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta_dist.ppf(0.975, k + 1, n - k))
    return lo, hi


def main():
    assert COHORT_VARIANT == "no_deep", "the leave-one-cancer-type-out runs exist for the no_deep cohort only"
    part = pd.read_csv(os.path.join(DATA_DIR, "tableS_participants.csv"))
    task = part[part["in_task_pancan"] == True].set_index("ID")  # noqa: E712
    assert len(task) == 367
    nested = pd.read_csv(os.path.join(DATA_DIR, "fig4a_predictions.csv"))
    nested = nested[nested["method"] == "data_driven"].set_index("sample_id")["score"]
    assert len(nested) == 367 and nested.index.is_unique

    rows, pred_rows, k_rows, held_ref = [], [], [], None
    for key, (disease, label) in TYPES.items():
        d = deploy_dir(key)
        summ_path = rp(d, "comprehensive_validation_summary.pkl")
        assert os.path.isfile(summ_path), f"{d} not on disk (slurm/submit_loco.sh deploy)"
        n_cfg = len([f for f in os.listdir(rp(search_dir(key))) if f.startswith("summary_") and f.endswith(".pkl")])
        assert n_cfg == 60, f"{key}: search holds {n_cfg}/60 configurations"
        with open(summ_path, "rb") as fh:
            summ = pickle.load(fh)
        fr = {k: v for k, v in summ["feature_results"].items() if v is not None}
        best = max(fr, key=lambda k: fr[k].get("training_cv_auc", 0))
        with open(rp(d, f"external_validation_{best}_results.pkl"), "rb") as fh:
            r = pickle.load(fh)
        info = r["best_model_info"]
        assert info["feature"] == best and r["training_dataset"] == search_dir(key), (key, info["feature"], r["training_dataset"])
        ids = [str(x) for x in r["sample_ids"]]
        y = np.asarray(r["y_true"], dtype=int)
        s = np.asarray(r["y_pred_proba"], dtype=float)
        assert np.isclose(roc_auc_score(y, s), r["overall_auc"], atol=1e-9), key

        # the test set is what the design says: all patients of the type + the same 34 healthy donors
        healthy_ids = sorted(i for i, yi in zip(ids, y) if yi == 0)
        cancer_ids = [i for i, yi in zip(ids, y) if yi == 1]
        assert set(task.loc[healthy_ids, "disease"]) == {"Healthy"} and len(healthy_ids) == 34, key
        assert set(task.loc[cancer_ids, "disease"]) == {disease} and len(cancer_ids) == int((task["disease"] == disease).sum()), key
        held_ref = held_ref or healthy_ids
        assert healthy_ids == held_ref, f"{key}: held-out healthy donors differ between types"

        # the threshold frozen on the training samples, and the calls it gives on the left-out set
        dep = r["deployment_thresholds"]
        thr = float(dep["95_percent_specificity"])
        ov = r["fixed_threshold_evaluation"]["95_percent_specificity"]["overall"]
        tp, fn = int(((s >= thr) & (y == 1)).sum()), int(((s < thr) & (y == 1)).sum())
        tn, fp = int(((s < thr) & (y == 0)).sum()), int(((s >= thr) & (y == 0)).sum())
        assert (tp, fn, tn, fp) == (int(ov["tp"]), int(ov["fn"]), int(ov["tn"]), int(ov["fp"])), (key, (tp, fn, tn, fp), ov)
        auc, lo, hi = boot_auc(y, s)
        sens_lo, sens_hi = exact_ci(tp, tp + fn)
        spec_lo, spec_hi = exact_ci(tn, tn + fp)

        # reference: the type in training (fully nested pan-cancer CV, pooled out-of-fold), same 34 healthy donors
        ref_ids = healthy_ids + cancer_ids
        ref_y = np.r_[np.zeros(len(healthy_ids), int), np.ones(len(cancer_ids), int)]
        ref_auc, ref_lo, ref_hi = boot_auc(ref_y, nested.loc[ref_ids].to_numpy())

        rows.append({
            "type": key, "disease": disease, "label": label, "n_cancer": len(cancer_ids), "n_healthy": len(healthy_ids),
            "feature": info["feature"], "classifier": info["classifier"], "n_tss": int(info["n_genes"]),
            "selection_type": info.get("selection_type", ""), "best_params": json.dumps({k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in (info.get("best_params") or {}).items()}),
            "training_cv_auc": float(info["cv_auc"]), "auc": auc, "auc_lo": lo, "auc_hi": hi,
            "threshold_95spec_train": thr, "threshold_source": dep.get("threshold_source", ""),
            "tp": tp, "fn": fn, "tn": tn, "fp": fp,
            "sensitivity": tp / (tp + fn), "sensitivity_lo": sens_lo, "sensitivity_hi": sens_hi,
            "specificity": tn / (tn + fp), "specificity_lo": spec_lo, "specificity_hi": spec_hi,
            "auc_in_training": ref_auc, "auc_in_training_lo": ref_lo, "auc_in_training_hi": ref_hi,
            "source_dir": d,
        })
        for i, yi, si in zip(ids, y, s):
            pred_rows.append({"type": key, "sample_id": i, "y_true": int(yi), "disease": task.loc[i, "disease"],
                              "score": float(si), "threshold_95spec_train": thr, "called_cancer": bool(si >= thr),
                              "score_in_training": float(nested.loc[i])})

        # the search's winner at every number of TSSs, applied to the left-out set
        for k, g in sorted(summ.get("gene_count_results", {}).items()):
            if not g or g.get("y_true") is None:
                continue
            gy = np.asarray(g["y_true"], dtype=int)
            gs = np.asarray(g.get("y_scores_for_roc", g["y_pred_proba"]), dtype=float)
            ka, klo, khi = boot_auc(gy, gs)
            assert np.isclose(ka, g["auc"], atol=1e-9), (key, k)
            k_rows.append({"type": key, "label": label, "k": int(k), "feature": g.get("feature", ""), "classifier": g.get("classifier", ""),
                           "training_cv_auc": float(g.get("training_cv_auc", np.nan)), "auc": ka, "auc_lo": klo, "auc_hi": khi})
        print(f"  {label:10s} n = {len(cancer_ids):2d}: {info['feature']} k={info['n_genes']} {info['classifier']} (training CV {info['cv_auc']:.3f}) | "
              f"left out {auc:.3f} [{lo:.2f}-{hi:.2f}] | in training {ref_auc:.3f} | frozen threshold {thr:.3f}: sens {tp}/{tp + fn}, spec {tn}/{tn + fp}")

    per_type = pd.DataFrame(rows)
    ks = pd.DataFrame(k_rows)
    per_type.to_csv(os.path.join(DATA_DIR, "fig4_loco_per_type.csv"), index=False, lineterminator="\n")
    pd.DataFrame(pred_rows).to_csv(os.path.join(DATA_DIR, "fig4_loco_predictions.csv"), index=False, lineterminator="\n")
    ks.to_csv(os.path.join(DATA_DIR, "fig4_loco_ksweep.csv"), index=False, lineterminator="\n")

    tp, n_c = int(per_type["tp"].sum()), int(per_type["n_cancer"].sum())
    tn, n_h = int(per_type["tn"].sum()), int(per_type["n_healthy"].sum())
    summary = {
        "n_types": len(per_type), "n_cancer": n_c, "held_out_healthy": held_ref,
        "auc_mean_over_types": float(per_type["auc"].mean()),
        "auc_mean_weighted_by_n": float(np.average(per_type["auc"], weights=per_type["n_cancer"])),
        "auc_in_training_mean_over_types": float(per_type["auc_in_training"].mean()),
        "auc_in_training_mean_weighted_by_n": float(np.average(per_type["auc_in_training"], weights=per_type["n_cancer"])),
        "sensitivity_at_frozen_threshold": {"tp": tp, "n": n_c, "value": tp / n_c, "ci95": exact_ci(tp, n_c)},
        "specificity_at_frozen_threshold_over_7_models": {"tn": tn, "n": n_h, "value": tn / n_h,
                                                          "note": "the same 34 donors scored by seven models; not 238 independent donors"},
        "configurations": per_type.groupby(["feature", "classifier", "n_tss"]).size().reset_index(name="n_types").to_dict("records"),
        "ksweep_mean_auc_over_types": {int(k): float(v) for k, v in ks.groupby("k")["auc"].mean().items()},
        "ksweep_mean_auc_weighted_by_n": {int(k): float(np.average(g["auc"], weights=per_type.set_index("type").loc[g["type"], "n_cancer"]))
                                          for k, g in ks.groupby("k")},
    }
    with open(os.path.join(DATA_DIR, "fig4_loco_summary.json"), "w", newline="\n") as fh:
        json.dump(summary, fh, indent=1)
    print(f"\nmean AUROC over the seven types: left out {summary['auc_mean_over_types']:.3f} (weighted {summary['auc_mean_weighted_by_n']:.3f}), "
          f"in training {summary['auc_in_training_mean_over_types']:.3f} (weighted {summary['auc_in_training_mean_weighted_by_n']:.3f})")
    print(f"frozen threshold: sensitivity {tp}/{n_c} = {tp / n_c:.3f}; specificity over the seven models {tn}/{n_h} = {tn / n_h:.3f}")
    print("mean AUROC by number of TSSs:", {k: round(v, 3) for k, v in summary["ksweep_mean_auc_over_types"].items()})
    print(f"wrote {DATA_DIR}/fig4_loco_{{per_type,predictions,ksweep}}.csv and fig4_loco_summary.json")

    print("\ncomparator arms (RNA, ATAC):")
    c_rows, c_preds, pending = comparator_rows(task, held_ref)
    comp_cols = ["arm", "type", "disease", "label", "n_cancer", "n_healthy", "feature", "classifier", "n_features",
                 "training_cv_auc", "auc", "auc_lo", "auc_hi", "threshold_95spec_train", "tp", "fn", "tn", "fp",
                 "sensitivity", "specificity", "auc_in_training", "auc_in_training_lo", "auc_in_training_hi", "source_dir"]
    pd.DataFrame(c_rows, columns=comp_cols).to_csv(os.path.join(DATA_DIR, "fig4_loco_comparators.csv"), index=False, lineterminator="\n")
    pd.DataFrame(c_preds, columns=["arm", "type", "sample_id", "y_true", "score", "threshold_95spec_train", "called_cancer",
                                   "score_in_training"]).to_csv(os.path.join(DATA_DIR, "fig4_loco_comparator_predictions.csv"),
                                                                index=False, lineterminator="\n")
    if c_rows:
        c = pd.DataFrame(c_rows)
        for arm, g in c.groupby("arm"):
            print(f"  {arm}: mean AUROC over {len(g)} types: left out {g['auc'].mean():.3f}, in training {g['auc_in_training'].mean():.3f}")
    if pending:
        print(f"  pending ({len(pending)}): {' '.join(pending)}  (ARM=rna|atac ./slurm/submit_loco.sh all)")
    print(f"wrote fig4_loco_comparators.csv ({len(c_rows)} rows) and fig4_loco_comparator_predictions.csv")


if __name__ == "__main__":
    main()
