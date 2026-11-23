"""Source data for Fig S18: where pre-specified gene sets sit in the per-TSS effect list of
Fig S17 (export_figS17.py), and gene-set enrichment of that list.

The list holds, per TSS, Cohen's d and BH q of the row-z-scored GD feature (cancer vs healthy), per
task and per cancer type of the pan-cancer task; the task-level values are recomputed from the GD
matrix and asserted equal to the Fig S17 tables.
  pre-specified sets  Zhu promoter genes, housekeeping genes, genes unexpressed in blood (SETS),
                      against N_NULL random TSS sets matched on GC x coverage strata
  enrichment          gseapy.prerank on MSigDB Hallmark and C8 (and HPA sets) per task and per
                      cancer type; controls: matched-null screens (GC x coverage [x dip depth]),
                      a circular-shift null, genomic clustering, and residual lists after removing
                      the GC / coverage / dip-depth axis

Outputs (data[/no_deep]/): figS18_tss_sets.csv (per TSS), figS18_sets.csv, figS18_set_contrasts.csv,
figS18_gsea.csv, figS18_gsea_matched.csv, figS18_c8_classes.csv, figS18_hpa_sets.csv,
figS18_prespecified_curves.csv, figS18_prespecified_hits.csv, figS18_summary.json.

Inputs: data[/no_deep]/figS17_tss_effects_<task>.csv, the GD cache of export_figS17.py
(--gd_cache), a panel-of-normals coverage cache (--pon_cache; built from the 50
panel_of_normals BIN files when absent), beds/, accessory_files/annotations/msigdb/*.gmt
and hpa/ (see accessory_files/annotations/README.md), data/figS10_tss_regression_tss.csv (full-cohort companion, matched by gene).
Run from the repository root (a few minutes; the C8 enrichment dominates):
    COHORT_VARIANT=no_deep python src/figures/export/export_figS18_gene_sets.py \
        --gd_cache <npz> --pon_cache <npz> [--cores 8]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr, ttest_ind

sys.path.insert(0, os.path.dirname(__file__))
from _common import COHORT_VARIANT, DATA_DIR, rp  # noqa: E402
from export_figS17 import load_gd, preprocess, signed_mean_diff  # noqa: E402

TASKS = ["brca_crc", "pancan"]
SAMPLE_TABLES = {"brca_crc": "fig3_baseline_scalars.csv", "pancan": "fig4_baseline_scalars.csv"}
SETS = {
    "zhu_promoter": ("beds/hypothesis/zhu_all_after_lift.bed", "promoter"),
    "hk": ("beds/hk_genes.bed", None),
    "pau": ("beds/pau.bed", None),
}
SET_LABELS = {"zhu_promoter": "Zhu promoter genes (RNA region set)", "hk": "housekeeping genes (Fig 1)",
              "pau": "genes unexpressed in blood (Fig 1)"}
GMT = {"hallmark": "accessory_files/annotations/msigdb/h.all.v2024.1.Hs.symbols.gmt",
       "c8_cell_type": "accessory_files/annotations/msigdb/c8.all.v2024.1.Hs.symbols.gmt"}
HPA_TSV = "accessory_files/annotations/hpa/proteinatlas.tsv"
HPA_ELEVATED_TISSUE = ["Tissue enriched", "Group enriched", "Tissue enhanced"]
HPA_ELEVATED_LINEAGE = ["Lineage enriched", "Group enriched", "Lineage enhanced"]
HPA_HAEMATOPOIETIC = {"bone marrow", "lymphoid tissue"}
HPA_LINEAGES = {"granulocytes": "granulocytes", "monocytes": "monocytes", "T-cells": "t_cells", "B-cells": "b_cells",
                "NK-cells": "nk_cells", "dendritic cells": "dendritic_cells"}
HPA_TISSUES = {"breast": "breast", "intestine": "intestine", "stomach 1": "stomach", "pancreas": "pancreas",
               "lung": "lung", "ovary": "ovary", "liver": "liver"}
N_NULL = 1000
N_STRATA_GC = 5
N_STRATA_COV = 5
N_GSEA_PERMUTATIONS = 1000
MIN_SET_SIZE = 15
MAX_SET_SIZE = 500
N_STRATA_3D = 4             # bins per axis of the GC x coverage x dip-depth matching
RESIDUAL_BINS = (5, 10, 10)  # GC x coverage x dip-depth cells of the non-parametric residual
CLUSTER_WINDOW = 500_000    # bp; genomic clustering of a set
NEIGHBOUR_LAGS = [1, 2, 5, 10, 50]
SEED = 0
COMPANION_FIGS10 = "src/figures/data/figS10_tss_regression_tss.csv"


# ------------------------------------------------------------------------ inputs
def load_pon_coverage(cache: str | None) -> pd.DataFrame:
    """Panel-of-normals mean of mean_coverage and absolute_fragment_counts per bed row."""
    if cache and os.path.isfile(cache):
        z = np.load(cache, allow_pickle=True)
        return pd.DataFrame({"gene": z["genes"], "pon_mean_coverage": z["mean_coverage"],
                             "pon_fragment_counts": z["fragment_counts"]})
    m = pd.read_csv(rp("accessory_files/cris_manifest.csv"))
    pon = m.loc[m["role"] == "panel_of_normals", "ID"].tolist()
    assert len(pon) == 50, f"{len(pon)} panel-of-normals samples, expected 50"
    cov, cnt = [], []
    for i, s in enumerate(pon, 1):
        d = pd.read_csv(rp(f"extracted_features/biomart_10kb/{s}.hg38.frag.tsv_BIN.csv"),
                        usecols=["gene_name", "mean_coverage", "absolute_fragment_counts"])
        cov.append(d["mean_coverage"].to_numpy(float)); cnt.append(d["absolute_fragment_counts"].to_numpy(float))
        if i % 10 == 0:
            print(f"    panel of normals: {i}/{len(pon)} files read")
    out = pd.DataFrame({"gene": d["gene_name"].to_numpy(), "pon_mean_coverage": np.vstack(cov).mean(axis=0),
                        "pon_fragment_counts": np.vstack(cnt).mean(axis=0)})
    if cache:
        np.savez_compressed(cache, ids=np.array(pon), genes=out["gene"].to_numpy(),
                            mean_coverage=out["pon_mean_coverage"].to_numpy(), fragment_counts=out["pon_fragment_counts"].to_numpy())
    return out


def load_set_genes(bed: str, kind: str | None) -> set[str]:
    b = pd.read_csv(rp(bed), sep="\t", header=None)
    if kind is not None:
        b = b[b[4] == kind]
    return set(b[3].astype(str))


def hpa_sets() -> dict[str, set[str]]:
    """Blood-cell-restricted, blood-lineage and tissue gene sets from the HPA consensus RNA data.

    blood_restricted   tissue-elevated (enriched / group enriched / enhanced) only in bone
                       marrow and / or lymphoid tissue, i.e. not elevated in any other tissue
    blood_<lineage>    blood_restricted and elevated in that blood lineage (HPA blood atlas)
    not_detected_immune  not detected in any of the 18 immune cell types (reference, cf. pau)
    tissue_<tissue>    tissue-elevated in that tissue and not in bone marrow / lymphoid tissue
    """
    h = pd.read_csv(rp(HPA_TSV), sep="\t", low_memory=False,
                    usecols=["Gene", "RNA tissue specificity", "RNA tissue specific nTPM", "RNA blood lineage specificity",
                             "RNA blood lineage specific nTPM", "RNA blood cell specificity"])

    def names(v):
        return {t.split(":")[0].strip() for t in v.split(";")} if isinstance(v, str) else set()
    tis = h["RNA tissue specific nTPM"].map(names).where(h["RNA tissue specificity"].isin(HPA_ELEVATED_TISSUE), other=set())
    lin = h["RNA blood lineage specific nTPM"].map(names).where(h["RNA blood lineage specificity"].isin(HPA_ELEVATED_LINEAGE), other=set())
    restricted = tis.map(lambda s: bool(s) and s <= HPA_HAEMATOPOIETIC)
    out = {"blood_restricted": set(h.loc[restricted, "Gene"])}
    for name, key in HPA_LINEAGES.items():
        out[f"blood_{key}"] = set(h.loc[restricted & lin.map(lambda s: name in s), "Gene"])
    out["not_detected_immune"] = set(h.loc[h["RNA blood cell specificity"] == "Not detected in immune cells", "Gene"])
    for name, key in HPA_TISSUES.items():
        out[f"tissue_{key}"] = set(h.loc[tis.map(lambda s: name in s and not (s & HPA_HAEMATOPOIETIC)), "Gene"])
    return out


def read_gmt(path: str) -> dict[str, set[str]]:
    sets = {}
    with open(rp(path)) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            sets[parts[0]] = set(parts[2:])
    return sets



# ------------------------------------------------------------------------ C8 cell-type classes
# MSigDB C8 names are STUDY_TISSUE_[CLUSTER_]CELLTYPE. The study/tissue prefix is stripped and the
# cell-type part is matched against keyword lists (padded with "_" so that e.g. "_T_CELL" does not
# match "GIANT_CELLS"). Rule order matters: endothelial > blood/immune > neural > stromal > epithelial.
# Sets with no match are "other" (marker-gene-named clusters such as "AFP_ALB_POSITIVE_CELLS").
_C8_PREFIX = re.compile(
    r"^(?:AIZARANI_LIVER_C\d+_|BUSSLINGER_(?:GASTRIC|DUODENAL|ESOPHAGEAL)_|CUI_DEVELOPING_HEART_(?:C\d+_)?|"
    r"DESCARTES_(?:FETAL_[A-Z]+_|MAIN_FETAL_)|DURANTE_ADULT_OLFACTORY_NEUROEPITHELIUM_|"
    r"FAN_(?:OVARY_CL\d+_|EMBRYONIC_CTX_(?:BIG_GROUPS_)?)|GAO_LARGE_INTESTINE_[A-Z0-9]+_C[A-Z0-9]+_|"
    r"GAUTAM_EYE_(?:CHOROID_SCLERA|IRIS_CILIARY_BODY|CORNEA|RETINA|[A-Z]+)_|HAY_BONE_MARROW_|HE_LIM_SUN_FETAL_LUNG_C\d+_|"
    r"HU_FETAL_RETINA_|JONES_OVARY_|LAKE_ADULT_KIDNEY_C\d+_|MANNO_MIDBRAIN_NEUROTYPES_|MENON_FETAL_KIDNEY_\d+_|"
    r"MURARO_PANCREAS_|RUBENSTEIN_SKELETAL_MUSCLE_|TRAVAGLINI_LUNG_|VANGURP_PANCREATIC_|ZHENG_CORD_BLOOD_C\d+_|ZHONG_PFC_C\d+_)")
_C8_MANNO = {"HMGL": "blood / immune", "HPERIC": "stromal / mesenchymal", "HENDO": "endothelial"}   # the rest are neuron / glia types
_C8_RULES = [
    ("endothelial", ("ENDOTHEL", "MVECS", "_CAPILLARY", "_CAP_CELL", "AEROCYTE", "ARTER", "_VEIN", "VENOUS", "LYMPHATIC", "LSEC", "SINUSOID",
                     "ENDOCARDIAL", "BRONCHIAL_VESSEL", "_AVR", "_DVR")),
    ("blood / immune", ("_T_CELL", "_B_CELL", "_NK_", "_NKT_", "NATURAL_KILLER", "NEUTROPHIL", "MACROPHAGE", "MONOCYTE", "MYELOID", "MYELOCYTE",
                        "MICROGLIA", "_MAST_", "DENDRITIC", "_DC1_", "_DC2_", "_PDC_", "_DC5_", "_ADC_", "ERYTHRO", "RED_BLOOD", "RETICULOCYTE",
                        "MEGAKARYO", "LYMPHOCYTE", "LYMPHOID", "LEUKOCYTE", "PLASMA_CELL", "HEMATOPOIETIC", "_HSC_", "GRANULOCYTE", "_GRAN_",
                        "_GMP_", "BASOPHIL", "EOSINOPHIL", "KUPFFER", "LANGERHANS", "IMMUNE", "TREG", "_CD4_", "_CD8_", "_CD34_", "PLASMACYTOID",
                        "PLATELET", "THYMOCYTE", "_ILC", "IL5RA", "_PRE_B_", "_PRO_B_", "_CLP_", "_LMPP_", "_MEP_", "_MKP_", "MULTILIN",
                        "_PRE_PC_", "_EO_B_MAST", "ANTIGEN_PRESENTING", "MULTILYMPHOID", "MULTIPOTENT_PROGENITOR", "_B_LYMPH", "_T_LYMPH",
                        "_MAST_CELL", "MEMORY_EFFECTOR", "NAIVE_T", "_B_PLASMA")),
    ("neural / glial", ("NEURON", "NEURO", "GLIA", "ASTROCYTE", "OLIGODENDRO", "SCHWANN", "_NPC", "_NSC", "_INP", "PURKINJE", "GRANULE",
                        "PHOTORECEPTOR", "AMACRINE", "BIPOLAR", "HORIZONTAL_CELL", "GANGLION", "CHROMAFFIN", "_ENS_", "RADIAL", "EPENDYMAL",
                        "MULLER", "_CONE", "_ROD", "SYMPATHOBLAST", "INTERNEURON", "_OPC", "_RPC", "UNIPOLAR_BRUSH", "_RGC", "EXCITATORY",
                        "INHIBITORY", "_NE_PRECURSOR", "NEURAL", "RETINAL_PROGENITOR", "ENSHEATHING", "SENSORY")),
    ("stromal / mesenchymal", ("STROMAL", "FIBROBLAST", "MESENCHYM", "MESOTHELIAL", "SMOOTH_MUSCLE", "_SMC", "PERICYTE", "MYOFIBROBLAST",
                               "STELLATE", "MESANGIAL", "CHONDRO", "OSTEO", "ADIPO", "THECA", "INTERSTIT", "MYOCYTE", "MYOBLAST", "SATELLITE",
                               "_MUSCLE", "MYOGENIC", "SKELETAL", "MESODERM", "EPICARDIAL", "VALVAR", "_CARDIAC_", "TENOCYTE", "FIBROCYTE",
                               "SCLERA", "MATRIX", "LIMB_MESENCHYME")),
    ("epithelial / parenchymal", ("EPITHEL", "HEPATO", "CHOLANGIO", "ENTEROCYTE", "GOBLET", "PANETH", "TUFT", "ENTEROENDOCRINE", "ACINAR", "DUCT",
                                  "_ALPHA", "_BETA", "_DELTA", "_GAMMA", "EPSILON", "POLYPEPTIDE", "ISLET", "ENDOCRINE", "TUBUL", "PODOCYTE",
                                  "ASCENDING_LIMB", "DESCENDING_LIMB", "HENLE", "NEPHRON", "ALVEOLAR", "_CLUB", "CILIATED", "_BASAL", "SECRETORY",
                                  "KERATINOCYTE", "UROTHEL", "PARIETAL", "CHIEF", "MUCOUS", "_STEM", "TRANSIT_AMPLIFYING", "GRANULOSA", "SERTOLI",
                                  "LUMINAL", "TROPHOBLAST", "IONOCYTE", "MESONEPH", "METANEPHRIC", "URETERIC", "ADRENOCORTICAL", "THYROID",
                                  "_PIT_", "FOVEOLAR", "ENDOMETRI", "SQUAMOUS", "_AT1", "_AT2", "PNEUMOCYTE", "CRYPT", "COLONOCYTE", "INTERCALATED",
                                  "PRINCIPAL", "PROXIMAL", "_PT_", "DISTAL", "CONNECTING", "GERM", "OOCYTE", "SPERMATO", "OOGONIA", "PRIMORDIAL",
                                  "_LENS", "PIGMENT", "CONJUNCTIVAL", "CILIARY_BODY", "CORNEA", "LIMBAL", "_D_CELLS", "_G_CELLS", "_K_CELLS",
                                  "_EC_CELLS", "_X_CELLS", "_I_CELLS", "_L_CELLS", "_N_CELLS", "_S_CELLS", "ISTHMUS", "_NECK", "ZYMOGENIC",
                                  "LYZ_POSITIVE", "METALLOTHIONEIN", "REG3A", "PPP1R1B", "BCHE", "_TIP_", "_STALK_", "DEUTEROSOMAL", "SEROUS",
                                  "MKI67", "PROGENITOR", "SUBPOPULATION", "SUSTENTACULAR", "GLAND", "MICROVILLAR", "HEPATOCYTE", "_LOOP")),
]


def c8_cell_class(term: str) -> tuple[str, str, str]:
    """(study/tissue prefix, cell-type part, class) of a C8 set name."""
    t = term.upper()
    m = _C8_PREFIX.match(t)
    prefix, part = (m.group(0), t[m.end():]) if m else ("", t)
    if prefix == "MANNO_MIDBRAIN_NEUROTYPES_":
        return prefix, part, next((v for k, v in _C8_MANNO.items() if part.startswith(k)), "neural / glial")
    padded = f"_{part}_"
    for cls, keys in _C8_RULES:
        if any(k in padded for k in keys):
            return prefix, part, cls
    if prefix in ("HAY_BONE_MARROW_",) or prefix.startswith("ZHENG_CORD_BLOOD_"):
        return prefix, part, "blood / immune"
    return prefix, part, "other"


def write_c8_classes(terms, path: str) -> pd.DataFrame:
    rows = [dict(term=t, study_tissue=p.rstrip("_"), cell_type=c, cell_class=k) for t in terms for p, c, k in [c8_cell_class(t)]]
    out = pd.DataFrame(rows)
    out.to_csv(path, index=False)
    return out


# ------------------------------------------------------------------------ statistics
def effects(X: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    """Signed mean difference, Cohen's d (pooled sd), Welch p and BH q, cancer (1) vs healthy (0)."""
    n1, n0 = int(y.sum()), int((y == 0).sum())
    m = signed_mean_diff(X, y)
    sd = np.sqrt(((n1 - 1) * X[y == 1].var(axis=0, ddof=1) + (n0 - 1) * X[y == 0].var(axis=0, ddof=1)) / (n1 + n0 - 2))
    d = m / np.where(sd > 0, sd, np.nan)
    p = ttest_ind(X[y == 1], X[y == 0], equal_var=False).pvalue
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n); q[order] = np.minimum.accumulate((p[order] * n / np.arange(1, n + 1))[::-1])[::-1]
    return {"mean_diff": m, "d": d, "p": p, "q": np.clip(q, 0, 1)}


def rank_auc_from_ranks(ranks: np.ndarray, idx: np.ndarray) -> float:
    """Mann-Whitney AUC of the values at idx against all other values, from precomputed ranks."""
    n1 = len(idx); n2 = len(ranks) - n1
    return float((ranks[idx].sum() - n1 * (n1 + 1) / 2) / (n1 * n2))


def set_statistics(idx: np.ndarray, d: np.ndarray, m: np.ndarray, q: np.ndarray,
                   r_abs_d: np.ndarray, r_d: np.ndarray, r_abs_m: np.ndarray) -> dict[str, float]:
    return {"median_d": float(np.median(d[idx])), "median_abs_d": float(np.median(np.abs(d[idx]))),
            "mean_d": float(np.mean(d[idx])),
            "rank_auc_abs_d": rank_auc_from_ranks(r_abs_d, idx), "rank_auc_signed_d": rank_auc_from_ranks(r_d, idx),
            "rank_auc_abs_mean_diff": rank_auc_from_ranks(r_abs_m, idx), "share_q_below_0.05": float((q[idx] < 0.05).mean())}


def matched_draw(rng: np.random.Generator, strata: np.ndarray, members: np.ndarray, complement_by_stratum: dict) -> np.ndarray:
    """A random set with the same composition over strata as `members`, from the out-of-set TSSs."""
    out = []
    for s, c in zip(*np.unique(strata[members], return_counts=True)):
        pool = complement_by_stratum[s]
        out.append(rng.choice(pool, size=c, replace=len(pool) < c))
    return np.concatenate(out)


def ranks(v: np.ndarray) -> np.ndarray:
    return pd.Series(v).rank(method="average").to_numpy()


class MatchedNull:
    """N_NULL random sets with a given composition over strata, all at once: every stratum's
    pool is pre-permuted N_NULL times, a draw takes the first c_s elements of each stratum's
    permutation, and the sum of any per-TSS value over a draw is a cumulative-sum lookup."""

    def __init__(self, strata: np.ndarray, n_null: int, rng: np.random.Generator):
        self.strata = strata
        self.labels = np.unique(strata)
        self.perm = {s: rng.permuted(np.tile(np.flatnonzero(strata == s), (n_null, 1)), axis=1) for s in self.labels}
        self.n_null = n_null

    def cumsums(self, v: np.ndarray) -> dict:
        return {s: np.cumsum(v[p], axis=1) for s, p in self.perm.items()}

    def null_sums(self, cs: dict, idx: np.ndarray) -> np.ndarray:
        s_lab, counts = np.unique(self.strata[idx], return_counts=True)
        out = np.zeros(self.n_null)
        for s, c in zip(s_lab, counts):
            out += cs[s][:, c - 1]
        return out


def screen_sets(name_to_idx: dict, d: np.ndarray, null: "MatchedNull", n_total: int) -> pd.DataFrame:
    """Matched-null statistics (rank-AUC of signed d and of |d|, mean d) for many sets."""
    vals = {"rank_auc_signed_d": ranks(d), "rank_auc_abs_d": ranks(np.abs(d)), "mean_d": d}
    cs = {k: null.cumsums(v) for k, v in vals.items()}
    rows = []
    for term, idx in name_to_idx.items():
        n1 = len(idx); n2 = n_total - n1
        row = {"term": term, "n_genes_matched": n1}
        for k, v in vals.items():
            obs_sum, null_sum = v[idx].sum(), null.null_sums(cs[k], idx)
            if k.startswith("rank_auc"):
                obs, nv = (obs_sum - n1 * (n1 + 1) / 2) / (n1 * n2), (null_sum - n1 * (n1 + 1) / 2) / (n1 * n2)
            else:
                obs, nv = obs_sum / n1, null_sum / n1
            pg, pl = (1 + (nv >= obs).sum()) / (null.n_null + 1), (1 + (nv <= obs).sum()) / (null.n_null + 1)
            row.update({k: obs, f"{k}_null_mean": nv.mean(), f"{k}_null_lo": np.percentile(nv, 2.5), f"{k}_null_hi": np.percentile(nv, 97.5),
                        f"{k}_z": (obs - nv.mean()) / nv.std(ddof=1) if nv.std(ddof=1) > 0 else np.nan, f"{k}_p": min(1.0, 2 * min(pg, pl))})
        rows.append(row)
    out = pd.DataFrame(rows)
    for k in vals:
        p = out[f"{k}_p"].to_numpy(); m = len(p); order = np.argsort(p)
        q = np.empty(m); q[order] = np.minimum.accumulate((p[order] * m / np.arange(1, m + 1))[::-1])[::-1]
        out[f"{k}_q"] = np.clip(q, 0, 1)
    return out


class GenomeOrder:
    """TSSs in genomic order: circular-shift null of a per-TSS value, neighbour correlation,
    and the genomic clustering of a set."""

    def __init__(self, chrom: np.ndarray, start: np.ndarray, n_null: int, rng: np.random.Generator):
        self.order = np.lexsort((start, chrom))
        self.pos_of = np.empty(len(chrom), int); self.pos_of[self.order] = np.arange(len(chrom))
        self.chrom, self.start = chrom[self.order], start[self.order].astype(float)
        self.n = len(chrom)
        self.shifts = rng.choice(np.arange(1, self.n), size=n_null, replace=False)
        # number of other TSSs within CLUSTER_WINDOW of each TSS (same chromosome), for the clustering expectation
        self.n_within = np.zeros(self.n, int)
        for c in np.unique(self.chrom):
            m = np.flatnonzero(self.chrom == c); s = self.start[m]
            self.n_within[m] = np.searchsorted(s, s + CLUSTER_WINDOW, side="right") - np.searchsorted(s, s - CLUSTER_WINDOW, side="left") - 1

    def neighbour_spearman(self, v: np.ndarray) -> dict:
        w = v[self.order]
        return {str(lag): float(spearmanr(w[:-lag], w[lag:], nan_policy="omit")[0]) for lag in NEIGHBOUR_LAGS}

    def shift_null(self, v: np.ndarray, idx: np.ndarray):
        """Observed sum of v over idx and the sums under the circular shifts."""
        w = v[self.order]; w2 = np.concatenate([w, w])
        p = self.pos_of[idx]
        return w[p].sum(), w2[p[:, None] + self.shifts[None, :]].sum(axis=0)

    def clustering(self, idx: np.ndarray) -> tuple[float, float]:
        """Share of members with another member within CLUSTER_WINDOW (observed, and expected
        for a random set of the same size)."""
        p = np.sort(self.pos_of[idx]); c, s = self.chrom[p], self.start[p]
        has = np.zeros(len(p), bool)
        for i in range(len(p)):
            lo, hi = i - 1, i + 1
            if lo >= 0 and c[lo] == c[i] and s[i] - s[lo] <= CLUSTER_WINDOW:
                has[i] = True
            elif hi < len(p) and c[hi] == c[i] and s[hi] - s[i] <= CLUSTER_WINDOW:
                has[i] = True
        expected = 1 - (1 - self.n_within[p] / (self.n - 1)) ** (len(p) - 1)
        return float(has.mean()), float(expected.mean())


def residualize(d: np.ndarray, covariates: np.ndarray) -> np.ndarray:
    """Residual of d after OLS on the covariates, their squares and an intercept."""
    Z = np.column_stack([np.ones(len(d)), covariates, covariates ** 2])
    ok = np.isfinite(d)
    beta, *_ = np.linalg.lstsq(Z[ok], d[ok], rcond=None)
    out = np.full_like(d, np.nan); out[ok] = d[ok] - Z[ok] @ beta
    return out


def residualize_binned(d: np.ndarray, covariates: np.ndarray, bins=RESIDUAL_BINS) -> np.ndarray:
    """d minus the mean d of the TSS's cell of a quantile grid over the covariates."""
    cell = np.zeros(len(d), int)
    for j, b in enumerate(bins):
        cell = cell * b + pd.qcut(pd.Series(covariates[:, j]).rank(method="first"), b, labels=False).to_numpy()
    return d - pd.Series(d).groupby(cell).transform("mean").to_numpy()


# ------------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gd_cache", required=True, help=".npz with ids and gd (n x 19,376), as for export_figS17.py")
    ap.add_argument("--pon_cache", default=None, help=".npz with the panel-of-normals per-TSS coverage (built when absent)")
    ap.add_argument("--cores", type=int, default=4)
    ap.add_argument("--skip_gsea", action="store_true", help="B2 only (no gseapy needed)")
    ap.add_argument("--c8_classes_only", action="store_true", help="only (re)write figS18_c8_classes.csv (the C8 cell-type class of every set) and exit")
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    os.makedirs(DATA_DIR, exist_ok=True)
    classes = write_c8_classes(sorted(read_gmt(GMT["c8_cell_type"])), os.path.join(DATA_DIR, "figS18_c8_classes.csv"))
    print("C8 cell-type classes (all 840 sets): " + ", ".join(f"{k} {v}" for k, v in classes["cell_class"].value_counts().items()))
    if args.c8_classes_only:
        return

    # ---- the list (Fig S17) and the TSS universe ----
    eff = {t: pd.read_csv(os.path.join(DATA_DIR, f"figS17_tss_effects_{t}.csv")).sort_values("column").reset_index(drop=True) for t in TASKS}
    assert np.array_equal(eff["brca_crc"]["column"], eff["pancan"]["column"]), "the two tasks' TSS columns differ"
    tss = eff["brca_crc"][["column", "bed_row", "chrom", "start", "end", "gene", "strand", "gc"]].copy()
    assert tss["gene"].is_unique, "a gene maps to more than one TSS window"
    n = len(tss)
    universe_all = set(pd.read_csv(rp("beds/biomart_10kb.bed"), sep="\t", header=None)[3].astype(str))
    pon = load_pon_coverage(args.pon_cache)
    assert len(pon) == 19376
    tss = tss.merge(pon, on="gene", how="left", validate="1:1")
    assert tss["pon_mean_coverage"].notna().all()
    print(f"TSS universe: {n} TSSs (cohort variant {COHORT_VARIANT}); panel-of-normals coverage median "
          f"{tss['pon_mean_coverage'].median():.3f}, GC median {tss['gc'].median():.1f}")

    # ---- sets ----
    set_rows, set_idx = [], {}
    for name, (bed, kind) in SETS.items():
        genes = load_set_genes(bed, kind)
        in_universe = genes & universe_all
        mapped = genes & set(tss["gene"])
        tss[f"in_{name}"] = tss["gene"].isin(mapped)
        set_idx[name] = np.flatnonzero(tss[f"in_{name}"].to_numpy())
        set_rows.append(dict(set=name, label=SET_LABELS[name], n_genes_in_bed=len(genes), n_in_tss_universe_19376=len(in_universe),
                             n_removed_by_tss_filter=len(in_universe) - len(mapped), n_mapped_to_14221=len(mapped)))
        print(f"  set {name}: {len(genes)} genes in the bed, {len(in_universe)} in the 19,376 TSS windows, "
              f"{len(in_universe) - len(mapped)} removed by the TSS filter, {len(mapped)} in the list")
    assert not (tss["in_hk"] & tss["in_pau"]).any()
    pd.DataFrame(set_rows).to_csv(os.path.join(DATA_DIR, "figS18_sets.csv"), index=False)

    # ---- contrasts: the two tasks (from Fig S17, checked against the GD matrix) and the cancer types ----
    manifest = pd.read_csv(rp("accessory_files/cris_manifest.csv")).set_index("ID")
    contrasts: dict[str, dict] = {}
    for t in TASKS:
        contrasts[t] = {"mean_diff": eff[t]["mean_diff"].to_numpy(), "d": eff[t]["cohen_d"].to_numpy(), "q": eff[t]["q_bh"].to_numpy(),
                        "label": {"brca_crc": "Healthy vs BRCA + CRC", "pancan": "Healthy vs PANCAN"}[t]}
    samples = pd.read_csv(os.path.join(DATA_DIR, SAMPLE_TABLES["pancan"]))
    ids = samples["sample_id"].tolist()
    X, bed_row = preprocess(load_gd(ids, args.gd_cache))
    assert np.array_equal(bed_row, tss["bed_row"].to_numpy()), "GD columns differ from the Fig S17 table"
    y_all = (samples["disease"] != "Healthy").to_numpy().astype(int)
    assert np.array_equal(y_all, (manifest.loc[ids, "disease"] != "Healthy").to_numpy().astype(int))
    chk = effects(X, y_all)
    assert np.allclose(chk["d"], contrasts["pancan"]["d"], atol=1e-6, equal_nan=True), "PANCAN d recomputed from the GD matrix differs from Fig S17"
    sub = samples["disease"].isin(["Healthy", "Breast cancer", "Colorectal cancer"]).to_numpy()
    chk = effects(X[sub], y_all[sub])
    assert np.allclose(chk["d"], contrasts["brca_crc"]["d"], atol=1e-6, equal_nan=True), "BRCA/CRC d recomputed from the GD matrix differs from Fig S17"
    print(f"  task-level d recomputed from the GD matrix ({X.shape}) matches the Fig S17 tables")
    contrasts["pancan"]["n_cancer"] = int(y_all.sum()); contrasts["brca_crc"]["n_cancer"] = int(y_all[sub].sum())
    types = [d for d in samples["disease"].value_counts().index if d != "Healthy"]
    for ty in types:
        m = samples["disease"].isin(["Healthy", ty]).to_numpy()
        e = effects(X[m], y_all[m])
        key = "type_" + ty.lower().replace(" cancer", "").replace(" naive nsclc", "").replace(" ", "_")
        contrasts[key] = {"mean_diff": e["mean_diff"], "d": e["d"], "q": e["q"], "label": f"Healthy vs {ty}", "n_cancer": int(y_all[m].sum())}
        print(f"  {key}: {int(y_all[m].sum())} cancers vs {int((y_all[m] == 0).sum())} healthy, {int((e['q'] < 0.05).sum())} TSSs at q < 0.05, "
              f"median |d| {np.nanmedian(np.abs(e['d'])):.3f}")
    for key, c in contrasts.items():
        tss[f"d_{key}"] = c["d"]; tss[f"q_{key}"] = c["q"]
    # the healthy ranking: mean row-z-scored GD of the cohort's healthy donors (the depth of the healthy TSS dip)
    tss["healthy_mean_gd"] = X[y_all == 0].mean(axis=0)

    # ---- companion: the confounder-adjusted class t (Fig S10), full cohort ----
    s10 = pd.read_csv(rp(COMPANION_FIGS10), usecols=["task", "gene_name", "pon_mean_profile", "t_class_unadj_excl_deep", "t_class_confounders_excl_deep", "q_class_confounders_excl_deep"])
    prof = {t: s10[s10["task"] == t].set_index("gene_name")["pon_mean_profile"].reindex(tss["gene"]).to_numpy() for t in TASKS}
    assert np.allclose(prof["brca_crc"], prof["pancan"]), "the panel-of-normals profile differs between the tasks' Fig S10 tables"
    tss["pon_mean_gd_profile_figS10"] = prof["pancan"]          # the depth of the healthy TSS dip (mean z-scored GD of the panel of normals)
    print(f"  healthy ranking: mean GD of the {int((y_all == 0).sum())} healthy donors, Spearman with the panel-of-normals profile "
          f"{spearmanr(tss['healthy_mean_gd'], tss['pon_mean_gd_profile_figS10'], nan_policy='omit')[0]:+.3f}")
    for t in TASKS:
        s = s10[s10["task"] == t].rename(columns={"gene_name": "gene", "t_class_unadj_excl_deep": f"t_class_unadj_excl_deep_{t}_figS10",
                                                  "t_class_confounders_excl_deep": f"t_class_confounders_excl_deep_{t}_figS10",
                                                  "q_class_confounders_excl_deep": f"q_class_confounders_excl_deep_{t}_figS10"}).drop(columns="task")
        tss = tss.merge(s, on="gene", how="left", validate="1:1")
    print(f"  companion matched: adjusted t for {tss['t_class_confounders_excl_deep_pancan_figS10'].notna().sum()}")

    # ---- the technical / promoter-class axis of the list, and the residual list ----
    log_cov = np.log10(tss["pon_mean_coverage"].to_numpy() + 1e-3)
    axis = np.column_stack([tss["gc"].to_numpy(float), log_cov, tss["pon_mean_gd_profile_figS10"].to_numpy(float)])
    axis_corr = {}
    for key, c in contrasts.items():
        axis_corr[key] = {nm: float(spearmanr(c["d"], axis[:, j], nan_policy="omit")[0]) for j, nm in enumerate(["gc", "log_coverage", "dip_depth"])}
    for t in TASKS:
        contrasts[t]["d_residual"] = residualize(contrasts[t]["d"], axis)
        contrasts[t]["d_residual_binned"] = residualize_binned(np.nan_to_num(contrasts[t]["d"]), axis)
        tss[f"d_residual_{t}"] = contrasts[t]["d_residual"]
        tss[f"d_residual_binned_{t}"] = contrasts[t]["d_residual_binned"]
        print(f"  {t}: Spearman of d with GC {axis_corr[t]['gc']:+.3f}, log coverage {axis_corr[t]['log_coverage']:+.3f}, dip depth {axis_corr[t]['dip_depth']:+.3f}; "
              f"the quadratic regression on the three explains {1 - np.nanvar(contrasts[t]['d_residual']) / np.nanvar(contrasts[t]['d']):.1%} of the variance of d, "
              f"the {'x'.join(map(str, RESIDUAL_BINS))} cell means {1 - np.nanvar(contrasts[t]['d_residual_binned']) / np.nanvar(contrasts[t]['d']):.1%}")
    companions = {t: tss[f"t_class_confounders_excl_deep_{t}_figS10"].to_numpy(float) for t in TASKS}
    tss.to_csv(os.path.join(DATA_DIR, "figS18_tss_sets.csv"), index=False)

    # ---- B2: set statistics against the GC x coverage matched null ----
    gc_bin = pd.qcut(tss["gc"], N_STRATA_GC, labels=False, duplicates="drop")
    cov_bin = pd.qcut(np.log10(tss["pon_mean_coverage"] + 1e-3), N_STRATA_COV, labels=False, duplicates="drop")
    strata = (gc_bin * 10 + cov_bin).to_numpy()
    rows = []
    for key, c in contrasts.items():
        d, m, q = np.nan_to_num(c["d"]), c["mean_diff"], c["q"]
        r_abs_d, r_d, r_abs_m = ranks(np.abs(d)), ranks(d), ranks(np.abs(m))
        for name, idx in set_idx.items():
            members = np.zeros(n, bool); members[idx] = True
            comp = {s: np.flatnonzero((strata == s) & ~members) for s in np.unique(strata[idx])}
            short = [s for s, pool in comp.items() if len(pool) < (strata[idx] == s).sum()]
            if short:
                print(f"    WARNING {key}/{name}: {len(short)} strata have fewer out-of-set TSSs than the set has (drawn with replacement)")
            obs = set_statistics(idx, d, m, q, r_abs_d, r_d, r_abs_m)
            null = pd.DataFrame([set_statistics(matched_draw(rng, strata, idx, comp), d, m, q, r_abs_d, r_d, r_abs_m) for _ in range(N_NULL)])
            p_mw = {"abs_d": mannwhitneyu(np.abs(d[idx]), np.abs(d[~members])).pvalue,
                    "signed_d": mannwhitneyu(d[idx], d[~members]).pvalue,
                    "abs_m": mannwhitneyu(np.abs(m[idx]), np.abs(m[~members])).pvalue}
            unmatched = {"median_d": "signed_d", "mean_d": "signed_d", "rank_auc_signed_d": "signed_d", "median_abs_d": "abs_d",
                         "rank_auc_abs_d": "abs_d", "rank_auc_abs_mean_diff": "abs_m"}
            for stat, v in obs.items():
                nv = null[stat].to_numpy()
                rows.append(dict(contrast=key, contrast_label=c["label"], n_cancer=c["n_cancer"], set=name, n_set=len(idx), statistic=stat,
                                 observed=v, null_mean=nv.mean(), null_sd=nv.std(ddof=1), null_lo=np.percentile(nv, 2.5), null_hi=np.percentile(nv, 97.5),
                                 p_greater=(1 + (nv >= v).sum()) / (N_NULL + 1), p_less=(1 + (nv <= v).sum()) / (N_NULL + 1),
                                 z=(v - nv.mean()) / nv.std(ddof=1) if nv.std(ddof=1) > 0 else np.nan,
                                 p_mannwhitney_unmatched=p_mw.get(unmatched.get(stat), np.nan)))
        if key in TASKS:
            for name in set_idx:
                r = {x["statistic"]: x for x in rows if x["contrast"] == key and x["set"] == name}
                print(f"  {key} / {name} (n={len(set_idx[name])}): median d {r['median_d']['observed']:+.3f} (null {r['median_d']['null_lo']:+.3f}..{r['median_d']['null_hi']:+.3f}), "
                      f"rank-AUC |d| {r['rank_auc_abs_d']['observed']:.3f} (null {r['rank_auc_abs_d']['null_lo']:.3f}..{r['rank_auc_abs_d']['null_hi']:.3f}, "
                      f"p_greater {r['rank_auc_abs_d']['p_greater']:.3f}), share q<0.05 {r['share_q_below_0.05']['observed']:.3f} vs null {r['share_q_below_0.05']['null_mean']:.3f}")
    con = pd.DataFrame(rows)
    con.to_csv(os.path.join(DATA_DIR, "figS18_set_contrasts.csv"), index=False)
    print(f"  wrote figS18_set_contrasts.csv ({len(con)} rows) [{time.time() - t0:.0f} s]")

    # ---- B5: pre-ranked enrichment ----
    summary = {"n_tss": n, "cohort_variant": COHORT_VARIANT, "contrasts": {k: {"label": c["label"], "n_cancer": c["n_cancer"]} for k, c in contrasts.items()},
               "sets": {r["set"]: r for r in set_rows}, "n_null": N_NULL, "strata": f"{N_STRATA_GC} GC x {N_STRATA_COV} coverage quantile bins",
               "axis_spearman": axis_corr}
    if not args.skip_gsea:
        import gseapy
        genes = tss["gene"].to_numpy()
        gene_pos = {g: i for i, g in enumerate(genes)}
        collections = {k: read_gmt(v) for k, v in GMT.items()}
        usable_sets = {}
        for k, sets in collections.items():
            idx = {t: np.array(sorted(gene_pos[g] for g in s if g in gene_pos)) for t, s in sets.items()}
            usable_sets[k] = {t: i for t, i in idx.items() if MIN_SET_SIZE <= len(i) <= MAX_SET_SIZE}
            covered = int(tss["gene"].isin(set().union(*sets.values())).sum())
            summary[f"gsea_{k}"] = {"n_sets": len(sets), "n_sets_usable": len(usable_sets[k]), "n_tss_in_any_set": covered}
            print(f"  {k}: {len(sets)} sets, {len(usable_sets[k])} with {MIN_SET_SIZE}-{MAX_SET_SIZE} genes among the TSS genes; {covered} of {n} TSS genes in at least one set")

        # matched-null screens of the same sets
        q4 = lambda v: pd.qcut(pd.Series(v).rank(method="first"), N_STRATA_3D, labels=False).to_numpy()
        strata3 = q4(tss["gc"].to_numpy(float)) * 100 + q4(log_cov) * 10 + q4(tss["pon_mean_gd_profile_figS10"].to_numpy(float))
        nulls = {"gc_coverage": MatchedNull(strata, N_NULL, rng), "gc_coverage_dip": MatchedNull(strata3, N_NULL, rng)}
        genome = GenomeOrder(tss["chrom"].to_numpy(), tss["start"].to_numpy(), N_NULL, rng)
        summary["neighbour_spearman"] = {t: genome.neighbour_spearman(np.nan_to_num(contrasts[t]["d"])) for t in TASKS}
        for t in TASKS:
            print(f"  {t}: Spearman of d between genomic neighbours at lag 1/2/5/10/50: " + " / ".join(f"{v:+.2f}" for v in summary["neighbour_spearman"][t].values()))
        screens = [(k, "hallmark") for k in contrasts] + [(t, "c8_cell_type") for t in TASKS]
        shift_rows, clust = {}, {}
        for key, coll in screens:                                   # circular-shift null and clustering, once per contrast x set
            d = np.nan_to_num(contrasts[key]["d"]); R = ranks(d)
            for term, idx in usable_sets[coll].items():
                n1 = len(idx); n2 = n - n1
                obs_sum, null_sum = genome.shift_null(R, idx)
                obs, nv = (obs_sum - n1 * (n1 + 1) / 2) / (n1 * n2), (null_sum - n1 * (n1 + 1) / 2) / (n1 * n2)
                pg, pl = (1 + (nv >= obs).sum()) / (N_NULL + 1), (1 + (nv <= obs).sum()) / (N_NULL + 1)
                shift_rows[(key, coll, term)] = min(1.0, 2 * min(pg, pl))
                if (coll, term) not in clust:
                    clust[(coll, term)] = genome.clustering(idx)
        scr = []
        jobs = [("cohen_d", matching, key, coll) for matching in nulls for key, coll in screens] + \
               [("cohen_d_residual_binned", "gc_coverage_dip", t, coll) for t in TASKS for coll in GMT]
        for stat, matching, key, coll in jobs:
            v = np.nan_to_num(contrasts[key][{"cohen_d": "d", "cohen_d_residual_binned": "d_residual_binned"}[stat]])
            s = screen_sets(usable_sets[coll], v, nulls[matching], n)
            s.insert(0, "statistic", stat); s.insert(1, "matching", matching); s.insert(2, "contrast", key); s.insert(3, "contrast_label", contrasts[key]["label"]); s.insert(4, "collection", coll)
            for j, nm in enumerate(["set_mean_gc", "set_mean_log_coverage", "set_mean_dip_depth"]):
                s[nm] = [np.nanmean(axis[idx, j]) for idx in usable_sets[coll].values()]
            s["set_mean_adjusted_class_t_figS10"] = [np.nanmean(companions[key if key in TASKS else "pancan"][idx]) for idx in usable_sets[coll].values()]
            if stat == "cohen_d":
                s["rank_auc_signed_d_shift_p"] = [shift_rows[(key, coll, term)] for term in s["term"]]
                p = s["rank_auc_signed_d_shift_p"].to_numpy(); m = len(p); order = np.argsort(p)
                q = np.empty(m); q[order] = np.minimum.accumulate((p[order] * m / np.arange(1, m + 1))[::-1])[::-1]
                s["rank_auc_signed_d_shift_q"] = np.clip(q, 0, 1)
            s["clustering_observed"] = [clust[(coll, term)][0] for term in s["term"]]
            s["clustering_expected_random"] = [clust[(coll, term)][1] for term in s["term"]]
            scr.append(s)
            print(f"  matched screen ({matching}, {stat}) {key} / {coll}: {len(s)} sets; rank-AUC signed d outside the matched band at q < 0.05: {int((s['rank_auc_signed_d_q'] < 0.05).sum())}, "
                  f"q < 0.25: {int((s['rank_auc_signed_d_q'] < 0.25).sum())}; |d| rank-AUC at q < 0.05: {int((s['rank_auc_abs_d_q'] < 0.05).sum())}"
                  + (f"; circular-shift null q < 0.05: {int((s['rank_auc_signed_d_shift_q'] < 0.05).sum())}" if stat == "cohen_d" else "") + f" [{time.time() - t0:.0f} s]")
        scr = pd.concat(scr, ignore_index=True)
        scr.to_csv(os.path.join(DATA_DIR, "figS18_gsea_matched.csv"), index=False)
        base = scr[scr["statistic"] == "cohen_d"]
        summary["matched_screen_counts"] = {f"{k}|{c}|{mt}": {"n_tested": int(len(g)), "n_signed_q_0.05": int((g["rank_auc_signed_d_q"] < 0.05).sum()),
                                                              "n_signed_q_0.25": int((g["rank_auc_signed_d_q"] < 0.25).sum()), "n_abs_q_0.05": int((g["rank_auc_abs_d_q"] < 0.05).sum()),
                                                              "n_shift_q_0.05": int((g["rank_auc_signed_d_shift_q"] < 0.05).sum()), "n_shift_q_0.25": int((g["rank_auc_signed_d_shift_q"] < 0.25).sum())}
                                            for (mt, k, c), g in base.groupby(["matching", "contrast", "collection"])}
        summary["matched_screen_counts_residual_binned"] = {f"{k}|{c}": {"n_tested": int(len(g)), "n_signed_q_0.05": int((g["rank_auc_signed_d_q"] < 0.05).sum()),
                                                                         "n_signed_q_0.25": int((g["rank_auc_signed_d_q"] < 0.25).sum()), "n_abs_q_0.05": int((g["rank_auc_abs_d_q"] < 0.05).sum())}
                                                            for (k, c), g in scr[scr["statistic"] == "cohen_d_residual_binned"].groupby(["contrast", "collection"])}
        summary["set_level_spearman"] = {}
        for (k, c), g in base[base["matching"] == "gc_coverage"].groupby(["contrast", "collection"]):
            summary["set_level_spearman"][f"{k}|{c}"] = {"mean_d_vs_adjusted_t": float(spearmanr(g["mean_d"], g["set_mean_adjusted_class_t_figS10"], nan_policy="omit")[0]),
                                                         "mean_d_vs_log_coverage": float(spearmanr(g["mean_d"], g["set_mean_log_coverage"])[0])}
            if k in TASKS:
                print(f"  set-level Spearman over the {len(g)} {c} sets, {k}: mean cancer d "
                      f"vs mean adjusted class t {summary['set_level_spearman'][f'{k}|{c}']['mean_d_vs_adjusted_t']:+.2f}, vs mean log coverage {summary['set_level_spearman'][f'{k}|{c}']['mean_d_vs_log_coverage']:+.2f}")

        runs = [(t, "hallmark", "cohen_d") for t in TASKS] + [(t, "c8_cell_type", "cohen_d") for t in TASKS] + \
               [(k, "hallmark", "cohen_d") for k in contrasts if k.startswith("type_")] + [(t, "hallmark", "weight_mean") for t in TASKS] + \
               [(t, coll, "cohen_d_residual") for t in TASKS for coll in GMT] + [(t, coll, "cohen_d_residual_binned") for t in TASKS for coll in GMT]
        out = []
        for key, coll, stat in runs:
            v = {"cohen_d": lambda: np.nan_to_num(contrasts[key]["d"]), "weight_mean": lambda: eff[key]["weight_mean"].to_numpy(),
                 "cohen_d_residual": lambda: np.nan_to_num(contrasts[key]["d_residual"]),
                 "cohen_d_residual_binned": lambda: np.nan_to_num(contrasts[key]["d_residual_binned"])}[stat]()
            rnk = pd.Series(v, index=genes).sort_values(ascending=False)
            res = gseapy.prerank(rnk=rnk, gene_sets=rp(GMT[coll]), permutation_num=N_GSEA_PERMUTATIONS, min_size=MIN_SET_SIZE, max_size=MAX_SET_SIZE,
                                 seed=SEED, threads=args.cores, outdir=None, verbose=False).res2d
            for _, r in res.iterrows():
                tag = str(r["Tag %"]).split("/")
                out.append(dict(contrast=key, contrast_label=contrasts[key]["label"], collection=coll, statistic=stat, term=r["Term"], es=r["ES"], nes=r["NES"],
                                p_nominal=r["NOM p-val"], fdr=r["FDR q-val"], n_genes_matched=int(tag[1]) if len(tag) == 2 else np.nan,
                                n_leading_edge=int(tag[0]) if len(tag) == 2 else np.nan, lead_genes=r["Lead_genes"]))
            g = pd.DataFrame([o for o in out if o["contrast"] == key and o["collection"] == coll and o["statistic"] == stat])
            top = g.sort_values("fdr").iloc[0]
            print(f"  GSEA {key} / {coll} / {stat}: {len(g)} sets tested, {int((g['fdr'] < 0.05).sum())} at FDR < 0.05, {int((g['fdr'] < 0.25).sum())} at FDR < 0.25; "
                  f"top {top['term']} (NES {top['nes']:+.2f}, FDR {top['fdr']:.3f}) [{time.time() - t0:.0f} s]")
        # the pre-specified sets as a GSEA collection ('prespecified'): signed d, |mean difference| and
        # binned residual d; running-sum curves saved for the figure
        pre_sets = {name: [genes[i] for i in idx] for name, idx in set_idx.items()}
        pre_runs = [(k, "cohen_d") for k in contrasts] + [(t, "abs_mean_diff") for t in TASKS] + [(t, "cohen_d_residual_binned") for t in TASKS]
        curves, hits_rows = {}, []
        for key, stat in pre_runs:
            v = {"cohen_d": lambda: np.nan_to_num(contrasts[key]["d"]), "abs_mean_diff": lambda: np.abs(contrasts[key]["mean_diff"]),
                 "cohen_d_residual_binned": lambda: np.nan_to_num(contrasts[key]["d_residual_binned"])}[stat]()
            rnk = pd.Series(v, index=genes).sort_values(ascending=False)
            res = gseapy.prerank(rnk=rnk, gene_sets=pre_sets, permutation_num=N_GSEA_PERMUTATIONS, min_size=MIN_SET_SIZE, max_size=5000,
                                 seed=SEED, threads=args.cores, outdir=None, verbose=False).res2d
            for _, r in res.iterrows():
                tag = str(r["Tag %"]).split("/")
                out.append(dict(contrast=key, contrast_label=contrasts[key]["label"], collection="prespecified", statistic=stat, term=r["Term"], es=r["ES"], nes=r["NES"],
                                p_nominal=r["NOM p-val"], fdr=r["FDR q-val"], n_genes_matched=int(tag[1]) if len(tag) == 2 else np.nan,
                                n_leading_edge=int(tag[0]) if len(tag) == 2 else np.nan, lead_genes=r["Lead_genes"]))
            # running enrichment score (weighted Kolmogorov-Smirnov sum, weight 1, as gseapy) for the figure
            order = np.argsort(-v, kind="stable"); vs = np.abs(v[order])
            for name, idx in set_idx.items():
                hit = np.isin(order, idx)                      # hit[i]: the gene at rank i+1 is in the set
                p_hit = np.cumsum(vs * hit) / max((vs * hit).sum(), 1e-12); p_miss = np.cumsum(~hit) / (n - hit.sum())
                curves[f"{key}|{stat}|{name}"] = (p_hit - p_miss).astype(np.float32)
                hits_rows += [dict(contrast=key, statistic=stat, set=name, rank=int(r0) + 1) for r0 in np.flatnonzero(hit)]
            print(f"  GSEA of the pre-specified sets, {key} / {stat}: " + "; ".join(f"{r['Term']} NES {r['NES']:+.2f} FDR {r['FDR q-val']:.3f}" for _, r in res.iterrows()) + f" [{time.time() - t0:.0f} s]")
        # HPA sets (collection 'hpa'): signed d of every contrast and the binned residual of the two tasks
        hpa = hpa_sets()
        hpa_rows = [dict(set=k, n_genes_hpa=len(v), n_mapped_to_14221=len(v & set(genes))) for k, v in hpa.items()]
        pd.DataFrame(hpa_rows).to_csv(os.path.join(DATA_DIR, "figS18_hpa_sets.csv"), index=False)
        print("  HPA sets (mapped to the list): " + ", ".join(f"{r['set']} {r['n_mapped_to_14221']}" for r in hpa_rows))
        hpa_gs = {k: sorted(v & set(genes)) for k, v in hpa.items()}
        for key, stat in [(k, "cohen_d") for k in contrasts] + [(t, "cohen_d_residual_binned") for t in TASKS]:
            v = np.nan_to_num(contrasts[key]["d" if stat == "cohen_d" else "d_residual_binned"])
            rnk = pd.Series(v, index=genes).sort_values(ascending=False)
            res = gseapy.prerank(rnk=rnk, gene_sets=hpa_gs, permutation_num=N_GSEA_PERMUTATIONS, min_size=MIN_SET_SIZE, max_size=6000,
                                 seed=SEED, threads=args.cores, outdir=None, verbose=False).res2d
            for _, r in res.iterrows():
                tag = str(r["Tag %"]).split("/")
                out.append(dict(contrast=key, contrast_label=contrasts[key]["label"], collection="hpa", statistic=stat, term=r["Term"], es=r["ES"], nes=r["NES"],
                                p_nominal=r["NOM p-val"], fdr=r["FDR q-val"], n_genes_matched=int(tag[1]) if len(tag) == 2 else np.nan,
                                n_leading_edge=int(tag[0]) if len(tag) == 2 else np.nan, lead_genes=r["Lead_genes"]))
            print(f"  GSEA of the HPA sets, {key} / {stat}: " + "; ".join(f"{r['Term']} {r['NES']:+.2f} ({r['FDR q-val']:.3f})"
                                                                         for _, r in res.sort_values("Term").iterrows()) + f" [{time.time() - t0:.0f} s]")
        # the same collections on the healthy ranking (mean GD of the healthy donors; statistic 'healthy_mean_gd')
        rnk = pd.Series(np.nan_to_num(tss["healthy_mean_gd"].to_numpy()), index=genes).sort_values(ascending=False)
        for coll, gs_, mx in [("hallmark", rp(GMT["hallmark"]), MAX_SET_SIZE), ("c8_cell_type", rp(GMT["c8_cell_type"]), MAX_SET_SIZE), ("hpa", hpa_gs, 6000)]:
            res = gseapy.prerank(rnk=rnk, gene_sets=gs_, permutation_num=N_GSEA_PERMUTATIONS, min_size=MIN_SET_SIZE, max_size=mx,
                                 seed=SEED, threads=args.cores, outdir=None, verbose=False).res2d
            for _, r in res.iterrows():
                tag = str(r["Tag %"]).split("/")
                out.append(dict(contrast="healthy", contrast_label="Healthy donors (mean GD)", collection=coll, statistic="healthy_mean_gd", term=r["Term"], es=r["ES"],
                                nes=r["NES"], p_nominal=r["NOM p-val"], fdr=r["FDR q-val"], n_genes_matched=int(tag[1]) if len(tag) == 2 else np.nan,
                                n_leading_edge=int(tag[0]) if len(tag) == 2 else np.nan, lead_genes=r["Lead_genes"]))
            print(f"  GSEA on the healthy ranking, {coll}: {int((res['FDR q-val'].astype(float) < 0.05).sum())} of {len(res)} sets at FDR < 0.05 [{time.time() - t0:.0f} s]")
        pd.DataFrame({"rank": np.arange(1, n + 1), **curves}).to_csv(os.path.join(DATA_DIR, "figS18_prespecified_curves.csv"), index=False, float_format="%.4f")
        pd.DataFrame(hits_rows).to_csv(os.path.join(DATA_DIR, "figS18_prespecified_hits.csv"), index=False)
        gs = pd.DataFrame(out)
        gs.to_csv(os.path.join(DATA_DIR, "figS18_gsea.csv"), index=False)
        summary["gsea_counts"] = {f"{k}|{c}|{s}": {"n_tested": int(len(g)), "n_fdr_0.05": int((g["fdr"] < 0.05).sum()), "n_fdr_0.25": int((g["fdr"] < 0.25).sum())}
                                  for (k, c, s), g in gs.groupby(["contrast", "collection", "statistic"])}
    with open(os.path.join(DATA_DIR, "figS18_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=float)
    print(f"wrote figS18_tss_sets.csv, figS18_sets.csv, figS18_set_contrasts.csv{'' if args.skip_gsea else ', figS18_gsea.csv, figS18_gsea_matched.csv'}, "
          f"figS18_summary.json to {DATA_DIR} [{time.time() - t0:.0f} s]")


if __name__ == "__main__":
    main()
