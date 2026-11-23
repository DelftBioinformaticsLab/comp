#!/usr/bin/env python
"""Write the splits of the leave-one-cancer-type-out analysis (slurm/submit_loco.sh).

Same procedure as the Jiang transfer: one single-CV best-model search on the training
samples, refit, applied once to samples the search never saw. For each of the seven
cancer types of the pan-cancer task:

    training  all healthy donors outside the held-out set + the six other cancer types
    test      all cancers of the left-out type + the held-out healthy donors

The held-out healthy donors are the same for every type: the healthy donors of two
outer folds of the frozen pan-cancer fold assignment (FOLDS, chosen by rule as the
first two, without looking at any per-fold result). The cancers of those folds stay
in training unless they are of the left-out type.

Outputs, one pair per type (<key> = the CANCER_MAP key in model_hpc.py):
    accessory_files/cris_loco_<key>_exclude.txt    training: model_hpc.py --exclude_samples
                                                   (held-out healthy + left-out type + the
                                                   30 deep healthy controls)
    accessory_files/cris_loco_<key>_manifest.csv   test: --external_metadata_file (the rows
                                                   of cris_manifest.csv; the name makes
                                                   slurm/external_validation.sh find it)
    extracted_features/ulz_atac_tfbs_features_cris_loco_<key>.{npy,metadata.csv,tf_names.csv}
                                                   test rows of the pan-cancer ATAC matrix, for
                                                   the ATAC comparator arm (ARM=atac)
Written with LF line endings: the files are read on the cluster.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ACC = os.path.join(ROOT, "accessory_files")
FOLDS = (1, 2)
TYPES = {
    "breast": "Breast cancer",
    "bileduct": "Bile duct cancer",
    "colorectal": "Colorectal cancer",
    "gastric": "Gastric cancer",
    "lung": "Lung cancer naive nsclc",
    "ovarian": "Ovarian cancer",
    "pancreatic": "Pancreatic cancer",
}


def read_ids(path):
    with open(path) as fh:
        return {x for x in (ln.split("#", 1)[0].strip() for ln in fh) if x}


def main():
    m = pd.read_csv(os.path.join(ACC, "cris_manifest.csv"))
    deep = read_ids(os.path.join(ACC, "cris_exclude_deep_healthy.txt"))
    assert len(deep) == 30 and deep <= set(m["ID"]), "deep-healthy list does not match the manifest"

    # the frozen pan-cancer folds of the no_deep cohort, from the participant table
    p = pd.read_csv(os.path.join(ROOT, "src", "figures", "data", "no_deep", "tableS_participants.csv"))
    task = p[p["in_task_pancan"] == True]  # noqa: E712 (the column reads as object)
    assert len(task) == 367 and task["participant_id"].is_unique, len(task)
    assert not set(task["ID"]) & deep, "a deep healthy control is in the no_deep task"
    held = task[(task["disease"] == "Healthy") & task["fold_pancan"].astype(int).isin(FOLDS)]["ID"].tolist()
    assert len(held) == 34, len(held)

    in_task = m[m["ID"].isin(task["ID"])]
    assert set(in_task["disease"]) == set(TYPES.values()) | {"Healthy"}, set(in_task["disease"])
    assert (in_task["role"] == "model").all()

    rows = []
    for key, disease in TYPES.items():
        cancers = in_task.loc[in_task["disease"] == disease, "ID"].tolist()
        excl = sorted(deep | set(held) | set(cancers))
        with open(os.path.join(ACC, f"cris_loco_{key}_exclude.txt"), "w", newline="\n") as fh:
            fh.write(f"# Leave-one-type-out training exclusions, {disease} (src/comp/build_loco_splits.py):\n"
                     f"# {len(cancers)} {disease} + {len(held)} held-out healthy (pan-cancer folds {FOLDS}) + {len(deep)} deep healthy\n")
            fh.write("\n".join(excl) + "\n")
        test = in_task[in_task["ID"].isin(set(held) | set(cancers))]
        assert len(test) == len(cancers) + len(held)
        test.to_csv(os.path.join(ACC, f"cris_loco_{key}_manifest.csv"), index=False, lineterminator="\n")

        train = in_task[~in_task["ID"].isin(excl)]
        assert not set(train["participant_id"]) & set(test["participant_id"]), f"{key}: participant in both"
        assert set(train["disease"]) == set(TYPES.values()) - {disease} | {"Healthy"}
        rows.append(dict(type=key, train_healthy=int((train["disease"] == "Healthy").sum()),
                         train_cancer=int((train["disease"] != "Healthy").sum()),
                         test_healthy=len(held), test_cancer=len(cancers)))
    print(pd.DataFrame(rows).to_string(index=False))

    # ATAC comparator arm: the ATAC deployment takes its test samples from the
    # rows of a predefined matrix (slurm/external_validation.sh atac <dataset> reads
    # extracted_features/ulz_atac_tfbs_features_<dataset>.npy), not from the manifest, so each
    # type gets the rows of the pan-cancer matrix that the search was trained on, cut to its
    # test samples. Same columns (TFs) as the training matrix, same values.
    atac = os.path.join(ROOT, "extracted_features", "ulz_atac_tfbs_features_pancan")
    if os.path.exists(atac + ".npy"):
        mat = np.load(atac + ".npy")
        meta = pd.read_csv(atac + ".metadata.csv")
        tf = pd.read_csv(atac + ".tf_names.csv")
        assert len(meta) == mat.shape[0] and len(tf) == mat.shape[1]
        for key in TYPES:
            test_ids = pd.read_csv(os.path.join(ACC, f"cris_loco_{key}_manifest.csv"))["ID"]
            rows_idx = meta.index[meta["ID"].isin(set(test_ids))].to_numpy()
            assert len(rows_idx) == len(test_ids), f"{key}: {len(test_ids) - len(rows_idx)} test samples missing from the ATAC matrix"
            stem = os.path.join(ROOT, "extracted_features", f"ulz_atac_tfbs_features_cris_loco_{key}")
            np.save(stem + ".npy", mat[rows_idx])
            sub = meta.iloc[rows_idx].reset_index(drop=True)
            sub["row_index"] = np.arange(len(sub))
            sub.to_csv(stem + ".metadata.csv", index=False, lineterminator="\n")
            tf.to_csv(stem + ".tf_names.csv", index=False, lineterminator="\n")
        print(f"ATAC test matrices: extracted_features/ulz_atac_tfbs_features_cris_loco_<type>.npy ({mat.shape[1]} TFs)")
    else:
        print(f"{atac}.npy not found: no ATAC test matrices written")

    # balance of the held-out healthy donors against the training healthy donors (report only)
    s = pd.read_csv(os.path.join(ROOT, "src", "figures", "data", "no_deep", "fig4_baseline_scalars.csv"))
    h = s[s["disease"] == "Healthy"].copy()
    assert len(h) == 164, len(h)
    h["held_out"] = h["sample_id"].isin(held)
    cols = ["mean_coverage", "total_fragments", "fragment_length_mean", "mean_gc", "age", "sex_female_bam"]
    t = h.groupby("held_out")[cols].median().T.rename(columns={False: "training", True: "held_out"})
    t["held_out_mean"] = h[h["held_out"]][cols].mean()
    t["training_mean"] = h[~h["held_out"]][cols].mean()
    print(f"\nHealthy donors: {int(h['held_out'].sum())} held out, {int((~h['held_out']).sum())} in training")
    print(t.round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
