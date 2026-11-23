#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HPC entry point for every cfDNA classification experiment in this project.

One script, one CLI, many modes. The SLURM wrappers in slurm/ all end up calling
`python src/comp/model_hpc.py <mode flag> ...`. Mode flags map onto functions here:

    CLI mode flag                                 implementation
    -------------------------------------------  ------------------------------------------
    --run_best_model_search                      run_best_model_search
    --run_label_shuffle_experiment               run_label_shuffle_experiment
    --run_nested_cv                              run_single_experiment / _no_selection
    --run_nested_cv_full [...]                   run_nested_experiment_full
      (also the comparator region sweep: the same fully nested procedure, k grid
       included, on the RNA (Zhu) region set via --no_gene_selection --features_dir
       and on the ATAC (Ulz TFBS) matrix via --predefined_feature_matrix; see
       slurm/nested_full.sh FEATURE_SET=rna|atac)
    --run_cross_depth_eval                       run_cross_depth_eval
    --recompute_inner_scores DIR                 run_nested_classifier_selection
    --run_baseline_aggregate                     run_baseline_aggregate_study
    --validate_external                          validate_best_model_on_external_dataset
    --validate_external + --comprehensive_...    run_comprehensive_external_validation
    --run_within_dataset_generalization_train    run_within_dataset_generalization_training
    --run_within_dataset_generalization_eval     run_within_dataset_generalization_evaluation
    --run_within_dataset_generalization_compare  run_within_dataset_generalization_compare

Before dispatching to a mode, main() does the work every mode shares: resolve the
cohort from the metadata file (or manifest), load the feature matrices (per-sample
CSVs or a predefined .npy matrix), apply blacklist/duplicate-column filtering,
normalize, and build the label arrays. The stages are marked with STAGE banners
inside main().

File layout (top to bottom):
    - data loading & preprocessing helpers
    - feature selection (TopGeneSelector) and operating-point threshold helpers
    - shared evaluation helpers (per-fold and per-cancer-type metrics)
    - one section per experiment mode (in the order of the table above)
    - main() and the argparse CLI
"""

import os
import copy
import glob
import json
import pickle
import socket
import threading
import time
import argparse
import itertools
import multiprocessing
from pathlib import Path
from typing import List, Optional
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from difflib import SequenceMatcher
from scipy.stats import mannwhitneyu, spearmanr

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, roc_curve, auc, classification_report, confusion_matrix, precision_recall_curve, average_precision_score

# xgboost >= 2.1.3 is required for scikit-learn >= 1.6 compatibility (__sklearn_tags__).
# Guarded so that a container built before xgboost was added to pyproject.toml can still
# run every other classifier; main() prints a warning when XGBoost is skipped.
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Sibling module (this script is run from the repo root as src/comp/model_hpc.py,
# so src/comp/ is on sys.path)
from baseline_aggregate import aggregate_feature_from_raw_data

# Short CLI names (-t/--cancer_types) -> disease labels in the CRIS metadata
CANCER_MAP = {
    'breast': 'Breast cancer',
    'bileduct': 'Bile duct cancer',
    'colorectal': 'Colorectal cancer',
    'gastric': 'Gastric cancer',
    'lung': 'Lung cancer naive nsclc',
    'ovarian': 'Ovarian cancer',
    'pancreatic': 'Pancreatic cancer'
}

# =================================================================================
# --- Data loading & preprocessing helpers ---
# =================================================================================

def sync_file_lists(list1: List[str], list2: List[Path]) -> List[Optional[Path]]:
    """
    Syncs two lists of file identifiers based on the longest matching substring.
    """
    synced_list: List[Optional[Path]] = []
    if not list2:
        return [None] * len(list1)

    for path1 in list1:
        str_path1 = str(path1)
        best_match_path = None
        max_match_size = -1

        for path2 in list2:
            str_path2 = str(path2)
            matcher = SequenceMatcher(a=str_path1, b=str_path2, autojunk=False)
            match = matcher.find_longest_match()

            if match.size > max_match_size:
                max_match_size = match.size
                best_match_path = path2
        
        synced_list.append(best_match_path)

    return synced_list


def exact_file_lists(ids: List[str], folder_path: str) -> List[Path]:
    """
    Exact-ID feature-file resolution: <folder>/<ID>.hg38.frag.tsv_BIN.csv.

    Replaces the longest-substring matching of sync_file_lists for
    manifest-driven runs. Substring matching silently substituted a
    neighbouring sample's file when an ID had no feature file (the "phantom"
    defect); this resolver
    raises instead, listing every missing sample.
    """
    paths: List[Path] = []
    missing: List[str] = []
    for sid in ids:
        p = Path(folder_path) / f"{sid}.hg38.frag.tsv_BIN.csv"
        if p.exists():
            paths.append(p)
        else:
            missing.append(str(sid))
    if missing:
        raise FileNotFoundError(
            f"exact_file_lists: {len(missing)} sample(s) have no feature file in "
            f"{folder_path}: {', '.join(missing[:10])}"
            + (" ..." if len(missing) > 10 else "")
            + " - run GCfix + feature extraction for them, or regenerate the "
              "manifest (src/comp/build_cris_manifest.py) so they are marked "
              "gcfix_pending and skipped explicitly.")
    return paths


def _worker_load_feature(file_path: Path, feature_name: str):
    """
    Worker function to read one feature column from a single CSV file.
    Returns the file_path and the data, or None if it fails.
    """
    if file_path is None:
        return None
    try:
        df = pd.read_csv(file_path, usecols=[feature_name])
        return (file_path, df[feature_name].values)
    except Exception as e:
        print(f"Error processing file {os.path.basename(str(file_path))} (feature: {feature_name}): {e}")
        return None

def load_feature_in_parallel(folder_path: str, feature_name: str, file_filter: Optional[List[str]] = None, num_cores: int = 4, exact: bool = False) -> Optional[np.ndarray]:
    """
    Loads a single feature from all matching CSV files in a folder in parallel.

    exact=True resolves file_filter entries as exact sample IDs
    (<ID>.hg38.frag.tsv_BIN.csv) and raises on any missing file; exact=False
    keeps the legacy longest-substring matching for old metadata files.
    """
    print(f"--- Loading feature '{feature_name}' in parallel ---")
    if exact and file_filter:
        all_files = exact_file_lists(file_filter, folder_path)
        print(f"Resolved {len(all_files)} files by exact sample ID.")
    else:
        all_files = sorted(glob.glob(os.path.join(folder_path, '*_BIN.csv')))
        if not all_files:
            print(f"Warning: No '*_BIN.csv' files found in {folder_path}")
            return None

        if file_filter:
            all_files = sync_file_lists(file_filter, [Path(f) for f in all_files])
            # Filter out None values that may result from sync_file_lists
            all_files = [f for f in all_files if f is not None]
            print(f"Filtered to {len(all_files)} files after applying the filter.")

    if not all_files:
        print(f"Warning: No files remained after filtering for feature '{feature_name}'.")
        return None

    results = Parallel(n_jobs=num_cores)(
        delayed(_worker_load_feature)(f, feature_name) for f in tqdm(all_files)
    )

    valid_results = [res for res in results if res is not None]
    if not valid_results:
        print(f"Error: Could not successfully load the feature '{feature_name}' from any file.")
        return None

    # Check if all arrays have the same shape
    data_arrays = [data for file_path, data in valid_results]
    final_array = np.stack(data_arrays, axis=0)

    print(f"Successfully created array of shape: {final_array.shape}")
    return final_array

def normalize_feature_arrays(feature_dict: dict) -> dict:
    """
    Performs row-wise Z-score normalization on a dictionary of feature arrays.
    """
    normalized_dict = {}
    for feature, array in feature_dict.items():
        if array is None:
            normalized_dict[feature] = None
            continue
        row_means = np.nanmean(array, axis=1, keepdims=True)
        row_stds = np.nanstd(array, axis=1, keepdims=True)
        row_stds[row_stds == 0] = 1
        normalized_array = (array - row_means) / row_stds
        normalized_dict[feature] = np.nan_to_num(normalized_array)
        print(f"Normalized feature '{feature}' with shape {array.shape} to shape {normalized_dict[feature].shape}")
    return normalized_dict


def standardize_with_training_stats(feature_dict: dict, training_stats: dict) -> dict:
    """
    Column-wise standardization using training means/stds per feature.

    Expects training_stats to contain:
      - feature_means_per_feature
      - feature_stds_per_feature
    """
    if training_stats is None:
        return feature_dict

    means_map = training_stats.get('feature_means_per_feature', {})
    stds_map = training_stats.get('feature_stds_per_feature', {})
    # A model trained with --augment_covariates computed its column statistics on
    # [TSS columns | covariate block]; the external TSS arrays arrive here without the
    # block (it is appended, and standardized with the trailing statistics, in main).
    n_trailing = int((training_stats.get('covariate_block') or {}).get('n') or 0)

    standardized = {}
    for feature, array in feature_dict.items():
        if array is None:
            standardized[feature] = None
            continue
        if feature not in means_map or feature not in stds_map:
            print(f"No training stats for feature '{feature}', leaving unchanged")
            standardized[feature] = array
            continue
        means = np.array(means_map[feature])
        stds = np.array(stds_map[feature])
        if n_trailing and array.shape[1] + n_trailing == len(means):
            means, stds = means[:array.shape[1]], stds[:array.shape[1]]
            print(f"'{feature}': training stats cover {n_trailing} trailing covariate column(s) "
                  f"as well; they are standardized when the covariate block is appended")
        if array.shape[1] != len(means):
            print(f"Stats length mismatch for '{feature}': array has {array.shape[1]} cols, stats have {len(means)}")
            standardized[feature] = array
            continue
        stds[stds == 0] = 1
        standardized[feature] = (array - means) / stds
        print(f"Applied training standardization to '{feature}' with shape {array.shape}")

    return standardized

def format_distribution(labels: np.ndarray) -> str:
    """Formats the class distribution counts into a readable string."""
    unique, counts = np.unique(labels, return_counts=True)
    return ", ".join([f"{label}: {count}" for label, count in zip(unique, counts)])

def compute_feature_weights(X: np.ndarray, y: np.ndarray, selection_type: str = 'mean_diff') -> np.ndarray:
    """
    Computes feature selection weights for all features.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Labels (n_samples,)
        selection_type: Type of selection metric to compute
        
    Returns:
        Array of weights for each feature (n_features,)
    """
    cancer_samples = X[y == 1]
    control_samples = X[y == 0]
    
    if selection_type == 'random':
        # Random selection doesn't have meaningful weights
        return None
    
    mean_cancer = np.mean(cancer_samples, axis=0)
    mean_control = np.mean(control_samples, axis=0)
    
    if selection_type == 'mean_diff':
        mean_diff = np.abs(mean_cancer - mean_control)
        return np.nan_to_num(mean_diff)
    
    # Calculate log2 fold change (used by multiple methods)
    def safe_log2_fold_change(mean_cancer, mean_control, eps=1e-10):
        with np.errstate(divide="ignore", invalid="ignore"):
            fold_change = np.divide(
                mean_cancer,
                mean_control,
                out=np.full_like(mean_cancer, np.nan),
                where=mean_control != 0,
            )
            log2_fc = np.log2(
                fold_change, 
                out=np.full_like(fold_change, np.nan), 
                where=fold_change > eps
            )
        return np.nan_to_num(log2_fc)
    
    log2_fold_change = safe_log2_fold_change(mean_cancer, mean_control)
    
    if selection_type == 'fold_change':
        return np.abs(log2_fold_change)
    
    # Calculate p-values (used by 'p_value' and 'weighted')
    p_values = []
    for gene_idx in range(X.shape[1]):
        cancer_gene_values = cancer_samples[:, gene_idx]
        control_gene_values = control_samples[:, gene_idx]
        if len(cancer_gene_values) == 0 or len(control_gene_values) == 0:
            p_values.append(1.0)
            continue
        try:
            _, p = mannwhitneyu(cancer_gene_values, control_gene_values, alternative="two-sided")
            p_values.append(p if p > 0 else 1e-10)
        except:
            p_values.append(1.0)
    
    p_values = np.array(p_values)
    
    if selection_type == 'p_value':
        # Lower p-value = higher importance, so use -log10(p)
        return -np.log10(p_values)
    
    if selection_type == 'weighted':
        # Combination of fold change magnitude and statistical significance
        weighted_score = np.abs(log2_fold_change) * -np.log10(p_values)
        return np.nan_to_num(weighted_score)
    
    raise ValueError(f"Unknown selection_type: {selection_type}")

# Per-process memo of the slow rankings. The ranking statistic depends on
# (X_train, y_train, selection_type) only - not on n_top_genes and not on the
# classifier - yet GridSearchCV refits the selector for every candidate, so in a
# fully nested fold the per-TSS Mann-Whitney ranking (p_value / weighted, ~5 s on
# 220 x 14k) was recomputed ~11,000 times per feature where 6 distinct rankings
# exist. Keyed on a content hash of the inputs, so the result is identical to
# recomputing; bounded size; each loky worker keeps its own copy. The cheap
# rankings (mean_diff, fold_change, ~ms) are not cached, 'random' never is.
_RANKING_MEMO: dict = {}
_RANKING_MEMO_MAX = 64
_RANKING_MEMO_TYPES = frozenset({'p_value', 'weighted'})


def _ranking_memo_key(X_train: np.ndarray, y_train: np.ndarray, selection_type: str) -> bytes:
    import hashlib
    Xc = np.ascontiguousarray(X_train, dtype=np.float64)
    yc = np.ascontiguousarray(y_train)
    h = hashlib.blake2b(digest_size=16)
    h.update(selection_type.encode())
    h.update(repr(Xc.shape).encode())
    h.update(memoryview(Xc))
    h.update(repr((yc.shape, str(yc.dtype))).encode())
    h.update(memoryview(yc))
    return h.digest()


def select_top_genes(X_train: np.ndarray, y_train: np.ndarray, n_top_genes: int, selection_type='mean_diff') -> np.ndarray:
    """
    Selects the top genes based on a metric calculated from the training data.
    Supported selection types: 'p_value', 'fold_change', 'weighted', 'mean_diff', 'random'.
    """
    if selection_type == 'random':
        num_genes_to_select = min(n_top_genes, X_train.shape[1])
        return np.random.choice(X_train.shape[1], size=num_genes_to_select, replace=False)
    
    # Compute weights using the shared function (memoised for the slow rankings)
    if selection_type in _RANKING_MEMO_TYPES:
        key = _ranking_memo_key(X_train, y_train, selection_type)
        weights = _RANKING_MEMO.get(key)
        if weights is None:
            weights = compute_feature_weights(X_train, y_train, selection_type)
            if len(_RANKING_MEMO) >= _RANKING_MEMO_MAX:
                _RANKING_MEMO.pop(next(iter(_RANKING_MEMO)))
            _RANKING_MEMO[key] = weights
    else:
        weights = compute_feature_weights(X_train, y_train, selection_type)
    
    # Select top N genes based on weights
    num_genes_to_select = min(n_top_genes, len(weights))
    return np.argsort(weights)[-num_genes_to_select:]


def derive_operating_point_thresholds(y_train_true, train_scores):
    """
    Choose operating-point thresholds (95% specificity, 90% specificity, Youden's J,
    and the median) using ONLY training-fold scores, applied unchanged to the test fold.

    This avoids data leakage: in nested CV the threshold must be selected without
    looking at the held-out test fold. Deriving the threshold from the test fold (the
    original behaviour) tuned the operating point to the very samples being scored.

    IMPORTANT: for a fitted model `train_scores` must be the INNER-LOOP cross-validated
    predictions on the training set (inner_cv_train_scores: the selected configuration
    scored on the inner validation folds of the nested procedure), NOT the model's
    resubstitution predictions. A flexible classifier (e.g. RBF-SVM) memorises its
    training set, so its in-sample scores are degenerate (all healthy ~0, all cancer ~1).
    A threshold read off those is meaningless (the 95%-specificity point does not exist
    when training separation is perfect, giving an infinite threshold that classifies
    everything negative on the test fold). Inner-loop scores have a realistic spread and
    transfer to the test fold.

    Specificity thresholds are taken as quantiles of the HEALTHY training scores (a
    threshold at the q-quantile of healthy scores yields ~q specificity, score >= thr =>
    positive). This is robust and always finite, unlike reading the FPR=0.05 point off a
    ROC curve. Youden's threshold comes from the ROC, guarded against non-finite values.

    Returns the chosen thresholds plus the train-fold (fpr, tpr) achieved at each.
    """
    train_scores = np.asarray(train_scores, dtype=float)
    y_tr = np.asarray(y_train_true)
    valid = ~np.isnan(train_scores)
    y_tr = y_tr[valid]
    s_tr = train_scores[valid]
    healthy = s_tr[y_tr == 0]
    median_thr = float(np.median(s_tr)) if s_tr.size > 0 else 0.5

    def _spec_threshold(target_spec):
        # Threshold s.t. ~target_spec of healthy scores fall below it.
        if healthy.size == 0:
            return median_thr
        return float(np.quantile(healthy, target_spec))

    t95 = _spec_threshold(0.95)
    t90 = _spec_threshold(0.90)

    # Youden's J from the ROC; guard against the non-finite ROC endpoint.
    fpr, tpr, thr = roc_curve(y_tr, s_tr)
    youden = tpr - fpr
    iy = int(np.argmax(youden))
    t_youden = float(thr[iy])
    if not np.isfinite(t_youden):
        t_youden = median_thr

    def _achieved(thr_val):
        yhat = (s_tr >= thr_val).astype(int)
        tn = int(np.sum((y_tr == 0) & (yhat == 0)))
        fp = int(np.sum((y_tr == 0) & (yhat == 1)))
        fn = int(np.sum((y_tr == 1) & (yhat == 0)))
        tp = int(np.sum((y_tr == 1) & (yhat == 1)))
        fpr_a = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        tpr_a = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return fpr_a, tpr_a

    fpr95, tpr95 = _achieved(t95)
    fpr90, tpr90 = _achieved(t90)
    fpr_y, tpr_y = _achieved(t_youden)

    return {
        '95_percent_specificity': t95,
        '90_percent_specificity': t90,
        'youden_optimal': t_youden,
        'median': median_thr,
        'train_fpr_at_95spec': fpr95, 'train_tpr_at_95spec': tpr95,
        'train_fpr_at_90spec': fpr90, 'train_tpr_at_90spec': tpr90,
        'train_fpr_at_youden': fpr_y, 'train_tpr_at_youden': tpr_y,
    }


# Provenance marker stored with every threshold set derived by
# inner_cv_operating_thresholds(). Result pickles written before this derivation
# (thresholds from a separate 5-fold pass over the training partition) lack it; the
# figure exporters use it to tell the two apart.
THRESHOLD_SOURCE_INNER_CV = 'inner_cv_selected_config'


def inner_cv_train_scores(fitted_estimator, X_train, y_train, inner_splits,
                          expected_mean_auc=None, n_jobs=1, label=''):
    """
    Held-out predict_proba scores of the SELECTED configuration on the inner validation
    folds of the nested procedure - the same `inner_splits` model selection scored it
    on. Every training sample is scored once, by the inner-fold model that was fitted
    without it. Operating-point thresholds are read off these scores, so they come
    from the nested cross-validated training predictions and from nothing else: there
    is no cross-validation scheme beyond the outer and inner loops.

    GridSearchCV keeps each candidate's inner-fold scores but not its per-sample
    predictions, so the winner's inner-fold models are fitted again here on the same
    splits. Every estimator is seeded, so these are the models the search evaluated;
    `expected_mean_auc` (GridSearchCV.best_score_ of the winner) is compared with the
    mean inner-fold AUROC of the returned scores to confirm it.
    """
    y_train = np.asarray(y_train)
    inner_splits = [(np.asarray(tr), np.asarray(va)) for tr, va in inner_splits]
    proba = cross_val_predict(
        clone(fitted_estimator), X_train, y_train,
        cv=inner_splits, method='predict_proba', n_jobs=n_jobs,
    )
    scores = proba[:, 1]
    if expected_mean_auc is not None and np.isfinite(expected_mean_auc):
        fold_aucs = [roc_auc_score(y_train[va], scores[va]) for _, va in inner_splits
                     if len(np.unique(y_train[va])) == 2]
        if fold_aucs and abs(float(np.mean(fold_aucs)) - float(expected_mean_auc)) > 1e-4:
            print(f"  WARNING{' (' + label + ')' if label else ''}: mean inner-fold AUROC of "
                  f"the threshold scores ({np.mean(fold_aucs):.6f}) differs from the inner "
                  f"score the configuration was selected on ({expected_mean_auc:.6f}) - "
                  f"the inner-fold models were not reproduced exactly.")
    return scores


def inner_cv_operating_thresholds(fitted_estimator, X_train, y_train, inner_splits,
                                  expected_mean_auc=None, n_jobs=1, label=''):
    """Operating-point thresholds of the selected configuration from its inner-loop
    cross-validated training scores. Returns (op_thr, inner_cv_scores)."""
    scores = inner_cv_train_scores(fitted_estimator, X_train, y_train, inner_splits,
                                   expected_mean_auc=expected_mean_auc, n_jobs=n_jobs,
                                   label=label)
    op_thr = derive_operating_point_thresholds(y_train, scores)
    op_thr['threshold_source'] = THRESHOLD_SOURCE_INNER_CV
    return op_thr, scores

class TopGeneSelector(BaseEstimator, TransformerMixin):
    """
    A scikit-learn transformer for selecting top genes based on select_top_genes function.
    
    This is used in a Pipeline to allow GridSearchCV to search over 'n_top_genes'.
    """
    def __init__(self, n_top_genes=100, selection_type='mean_diff', n_passthrough=0):
        self.n_top_genes = n_top_genes
        self.selection_type = selection_type
        # n_passthrough > 0: the LAST n_passthrough columns of X are a covariate block
        # (--augment_covariates) that bypasses the ranking and is always kept, appended
        # after the selected TSS columns.
        self.n_passthrough = n_passthrough
        self.indices_ = None

    def fit(self, X, y=None):
        """
        Selects the top genes and stores their indices.
        """
        if y is None:
            raise ValueError("y (labels) must be provided to TopGeneSelector for gene selection.")
        n_pass = int(self.n_passthrough or 0)
        n_rank = X.shape[1] - n_pass
        selected = select_top_genes(X[:, :n_rank], y, self.n_top_genes, self.selection_type)
        if n_pass:
            selected = np.concatenate([np.asarray(selected, dtype=int),
                                       np.arange(n_rank, X.shape[1], dtype=int)])
        self.indices_ = selected
        return self

    def transform(self, X):
        """
        Transforms the data to include only the selected genes.
        """
        if self.indices_ is None:
            raise RuntimeError("The selector has not been fitted yet.")
        if len(self.indices_) == 0:
            print("Warning: TopGeneSelector selected 0 genes. Returning array with 1 column of zeros.")
            return np.zeros((X.shape[0], 1))
        return X[:, self.indices_]
        
    def _get_tags(self):
        # This tag is important to tell scikit-learn that this transformer needs 'y' in 'fit'
        return {"requires_y": True}


class _StoredIndexSelector(TopGeneSelector):
    """TopGeneSelector with a fixed, stored selection (cross-depth refit of a source model)."""
    def __init__(self, stored_indices=None):
        super().__init__()
        self.stored_indices = stored_indices

    def fit(self, X, y=None):
        self.indices_ = np.asarray(self.stored_indices, dtype=int)
        return self


# =================================================================================
# --- Covariate block: TSS features + sample-level covariates in ONE nested model ---
# =================================================================================
# Does the data-driven model need the sample-level covariates
# of Fig S10, and does its TSS signal survive adjusting for the confounders among
# them? Two modes, both set once in main() and read by _nested_full_one_fold:
#   augment     the covariate columns are appended to every TSS matrix (after the
#               per-sample row z-score and the duplicate-column removal), BYPASS the
#               TSS ranking (TopGeneSelector n_passthrough) and enter the classifier
#               next to the selected TSSs, column-scaled like them.
#   residualize the covariate columns travel with the matrix but a transformer
#               fitted on the training partition regresses them out of every TSS
#               column (OLS on [1, covariates]); the classifier sees residual TSS
#               features only.
# Fold records then carry 'covariate_block' (names, mode, the fitted weights or
# importances of the covariates ranked among the TSS weights, block permutation
# importance on the test fold, residual R2 summaries).
COVARIATE_BLOCK = {'mode': None, 'n': 0, 'names': [], 'stem': None}
COVARIATE_PERM_REPEATS = 20


def load_covariate_block(stem, sample_ids):
    """Covariate matrix of a panel stem (build_scalar_panel_matrix.py: <stem>.npy,
    .metadata.csv, .tf_names.csv) aligned to sample_ids. Raises on missing samples
    or NaN (the builder imputes cohort medians, label-free)."""
    stem = str(stem)
    X = np.load(stem + '.npy').astype(float)
    meta = pd.read_csv(stem + '.metadata.csv')
    names = pd.read_csv(stem + '.tf_names.csv').iloc[:, 0].astype(str).tolist()
    id_col = 'sample_id' if 'sample_id' in meta.columns else 'ID'
    pos = {sid: i for i, sid in enumerate(meta[id_col].astype(str))}
    missing = [sid for sid in sample_ids if str(sid) not in pos]
    if missing:
        raise ValueError(f"{stem}: {len(missing)} cohort samples absent from the covariate matrix "
                         f"(first: {missing[:5]})")
    C = X[[pos[str(sid)] for sid in sample_ids]]
    if np.isnan(C).any():
        raise ValueError(f"{stem}: NaN in the covariate matrix (impute in the builder)")
    if C.shape[1] != len(names):
        raise ValueError(f"{stem}: {C.shape[1]} columns but {len(names)} names")
    return C, names


class CovariateResidualizer(BaseEstimator, TransformerMixin):
    """Regress the LAST n_covariates columns of X out of every other column, with
    coefficients fitted on the data given to fit() (the training partition inside
    the nested procedure); transform() returns the residual columns only."""

    def __init__(self, n_covariates=0):
        self.n_covariates = n_covariates
        self.beta_ = None
        self.r2_ = None

    def _split(self, X):
        n = X.shape[1] - int(self.n_covariates)
        return X[:, :n], np.column_stack([np.ones(X.shape[0]), X[:, n:]])

    def fit(self, X, y=None):
        T, D = self._split(X)
        self.beta_, *_ = np.linalg.lstsq(D, T, rcond=None)
        resid = T - D @ self.beta_
        ss_res = (resid ** 2).sum(axis=0)
        ss_tot = ((T - T.mean(axis=0)) ** 2).sum(axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            self.r2_ = np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, np.nan)
        return self

    def transform(self, X):
        if self.beta_ is None:
            raise RuntimeError("CovariateResidualizer has not been fitted.")
        T, D = self._split(X)
        return T - D @ self.beta_


def _covariate_block_report(best_model, X_train, X_test, y_test, y_pred_proba, label=''):
    """Additive fold-record entry describing what the fitted model does with the
    covariate block (see COVARIATE_BLOCK)."""
    mode, n_cov, names = COVARIATE_BLOCK['mode'], int(COVARIATE_BLOCK['n']), list(COVARIATE_BLOCK['names'])
    out = {'mode': mode, 'n_covariates': n_cov, 'names': names, 'stem': COVARIATE_BLOCK['stem']}
    clf = best_model.named_steps['clf']
    out['classifier_class'] = type(clf).__name__
    if mode == 'augment':
        idx = np.asarray(best_model.named_steps['selector'].indices_)
        n_tss = len(idx) - n_cov
        out['n_tss_selected'] = int(n_tss)
        weights = None
        if hasattr(clf, 'coef_'):
            weights, out['weight_kind'] = np.ravel(clf.coef_), 'coef'
        elif hasattr(clf, 'feature_importances_'):
            weights, out['weight_kind'] = np.ravel(clf.feature_importances_), 'feature_importance'
        if weights is not None and len(weights) == len(idx):
            w_abs = np.abs(weights)
            w_cov, w_tss = w_abs[n_tss:], w_abs[:n_tss]
            order = np.argsort(-w_abs)
            rank = np.empty(len(w_abs), dtype=int)
            rank[order] = np.arange(1, len(w_abs) + 1)
            out['covariate_weights'] = [float(v) for v in weights[n_tss:]]
            out['covariate_abs_rank'] = [int(v) for v in rank[n_tss:]]          # 1 = largest |weight| of all
            out['covariate_abs_quantile_vs_tss'] = [float(np.mean(w_tss < v)) for v in w_cov] if n_tss else [np.nan] * n_cov
            out['tss_abs_weight_median'] = float(np.median(w_tss)) if n_tss else np.nan
            out['tss_abs_weight_max'] = float(w_tss.max()) if n_tss else np.nan
            out['tss_abs_weight_sum'] = float(w_tss.sum()) if n_tss else np.nan
            out['covariate_abs_weight_sum'] = float(w_cov.sum())
        else:
            out['weight_kind'] = None
        # block permutation importance on the held-out fold: shuffle the covariate rows
        # (jointly) / the TSS rows (jointly) and re-score; mean AUROC drop over repeats
        base = roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) == 2 else np.nan
        rng = np.random.default_rng(0)
        n_total = X_test.shape[1]
        drops = {'covariates': [], 'tss': []}
        for _ in range(COVARIATE_PERM_REPEATS):
            perm = rng.permutation(X_test.shape[0])
            for block, cols in (('covariates', slice(n_total - n_cov, n_total)), ('tss', slice(0, n_total - n_cov))):
                Xp = X_test.copy()
                Xp[:, cols] = X_test[perm][:, cols]
                pp = best_model.predict_proba(Xp)[:, 1]
                drops[block].append(base - roc_auc_score(y_test, pp) if np.isfinite(base) else np.nan)
        out['perm_drop_covariates_mean'] = float(np.nanmean(drops['covariates']))
        out['perm_drop_covariates_sd'] = float(np.nanstd(drops['covariates']))
        out['perm_drop_tss_mean'] = float(np.nanmean(drops['tss']))
        out['perm_drop_tss_sd'] = float(np.nanstd(drops['tss']))
        # per-covariate permutation importance (single covariate shuffled)
        per = []
        for j in range(n_cov):
            d = []
            for _ in range(COVARIATE_PERM_REPEATS):
                perm = rng.permutation(X_test.shape[0])
                Xp = X_test.copy()
                Xp[:, n_total - n_cov + j] = X_test[perm, n_total - n_cov + j]
                d.append(base - roc_auc_score(y_test, best_model.predict_proba(Xp)[:, 1]) if np.isfinite(base) else np.nan)
            per.append(float(np.nanmean(d)))
        out['perm_drop_per_covariate'] = per
    elif mode == 'residualize':
        r2 = np.asarray(best_model.named_steps['residualizer'].r2_, dtype=float)
        r2 = r2[np.isfinite(r2)]
        out['residual_r2_median'] = float(np.median(r2)) if r2.size else np.nan
        out['residual_r2_p90'] = float(np.percentile(r2, 90)) if r2.size else np.nan
        out['residual_r2_max'] = float(r2.max()) if r2.size else np.nan
        out['residual_r2_frac_gt_0.1'] = float(np.mean(r2 > 0.1)) if r2.size else np.nan
        out['residual_r2_frac_gt_0.25'] = float(np.mean(r2 > 0.25)) if r2.size else np.nan
        out['n_tss_selected'] = int(len(np.asarray(best_model.named_steps['selector'].indices_)))
    return out


# =================================================================================
# --- Shared evaluation helpers ---
# =================================================================================
# The per-fold / per-cohort metric blocks below used to be copy-pasted into every
# experiment mode. They are defined once here; every mode still produces the exact
# same result-dictionary keys as before, so existing notebooks and figure scripts
# keep working on old and new pickles alike.

def _sens_spec_at_95(fpr_arr, tpr_arr, target_spec=0.95):
    """Sensitivity and specificity at the ROC point closest to the target specificity."""
    if fpr_arr is None or tpr_arr is None or len(fpr_arr) == 0:
        return None, None
    target_fpr = 1 - target_spec
    idx = int(np.nanargmin(np.abs(np.array(fpr_arr) - target_fpr)))
    return float(tpr_arr[idx]), float(1 - fpr_arr[idx])


def _operating_point_reports(y_test, y_pred_proba, y_pred_default, op_thr, sample_ids_test):
    """Test-fold reports at four operating points with TRAIN-fold-derived thresholds.

    op_thr comes from derive_operating_point_thresholds() on the training fold, so
    no threshold is ever tuned on the fold being scored. Returns
    (report_full, confusion_matrix_full, misclassified_sample_ids, full_model_thresholds)
    with the same keys the per-mode copies used to build inline.
    """
    threshold_95spec = op_thr['95_percent_specificity']
    threshold_90spec = op_thr['90_percent_specificity']
    threshold_youden = op_thr['youden_optimal']
    preds = {
        'default_0.5': y_pred_default,
        '95_percent_specificity': (y_pred_proba >= threshold_95spec).astype(int),
        '90_percent_specificity': (y_pred_proba >= threshold_90spec).astype(int),
        'youden_optimal': (y_pred_proba >= threshold_youden).astype(int),
    }
    report_full = {
        key: classification_report(y_test, pred, output_dict=True, zero_division=0)
        for key, pred in preds.items()
    }
    confusion_matrix_full = {
        key: confusion_matrix(y_test, pred).tolist() for key, pred in preds.items()
    }
    misclassified_sample_ids = {
        key: sample_ids_test[y_test != pred].tolist() for key, pred in preds.items()
    }
    # train_*_at_* record where each threshold landed on the TRAIN-fold ROC.
    full_model_thresholds = {
        'default': 0.5,
        '95_percent_specificity': float(threshold_95spec),
        '90_percent_specificity': float(threshold_90spec),
        'youden_optimal': float(threshold_youden),
        'train_fpr_at_95spec': op_thr['train_fpr_at_95spec'],
        'train_tpr_at_95spec': op_thr['train_tpr_at_95spec'],
        'train_fpr_at_90spec': op_thr['train_fpr_at_90spec'],
        'train_tpr_at_90spec': op_thr['train_tpr_at_90spec'],
        # How the thresholds were derived (THRESHOLD_SOURCE_INNER_CV for fitted models).
        'threshold_source': op_thr.get('threshold_source'),
    }
    return report_full, confusion_matrix_full, misclassified_sample_ids, full_model_thresholds


def _clopper_pearson(k, n, alpha=0.05):
    """Exact binomial (Clopper-Pearson) confidence interval for k successes in n."""
    from scipy.stats import beta as _beta
    if n == 0:
        return float('nan'), float('nan')
    lo = 0.0 if k == 0 else float(_beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(_beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def _load_deployment_thresholds(best_model_path):
    """Frozen operating points of a deployed best-model-search model.

    They live in summary_<base>.pkl next to best_model_GridSearchCV_<base>.pkl
    (written by run_best_model_search from the search's own cross-validated scores).
    Returns None for summaries predating that field.
    """
    p = Path(str(best_model_path))
    summary_path = p.with_name(p.name.replace("best_model_GridSearchCV_", "summary_"))
    if not summary_path.exists():
        return None
    with open(summary_path, 'rb') as fh:
        summary = pickle.load(fh)
    return summary.get('deployment_thresholds')


def _fixed_threshold_evaluation(y_score, y_true, strat_labels, thresholds):
    """Score an external cohort at thresholds FROZEN on the training cohort.

    For every operating point in `thresholds` (95%/90% specificity, Youden)
    returns TP/FN/TN/FP with exact binomial CIs — overall and per control
    subgroup (each subgroup's specificity; the cancer class, hence the
    sensitivity, is shared). Nothing here looks at the external labels to
    choose a threshold; this is the deployment view reviewer 4 asked for
    next to the descriptive external-ROC operating points.
    """
    y_score = np.asarray(y_score, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    strat_labels = np.asarray(strat_labels)
    control_groups = [g for g in np.unique(strat_labels[y_true == 0])]

    def _counts(mask, thr):
        yhat = (y_score[mask] >= thr).astype(int)
        yt = y_true[mask]
        tp = int(np.sum((yt == 1) & (yhat == 1))); fn = int(np.sum((yt == 1) & (yhat == 0)))
        tn = int(np.sum((yt == 0) & (yhat == 0))); fp = int(np.sum((yt == 0) & (yhat == 1)))
        sens = tp / (tp + fn) if tp + fn else float('nan')
        spec = tn / (tn + fp) if tn + fp else float('nan')
        return {'tp': tp, 'fn': fn, 'tn': tn, 'fp': fp,
                'sensitivity': sens, 'sensitivity_ci95': _clopper_pearson(tp, tp + fn),
                'specificity': spec, 'specificity_ci95': _clopper_pearson(tn, tn + fp)}

    out = {}
    for name in ('95_percent_specificity', '90_percent_specificity', 'youden_optimal'):
        thr = thresholds.get(name)
        if thr is None or not np.isfinite(thr):
            continue
        entry = {'threshold': float(thr), 'overall': _counts(np.ones_like(y_true, dtype=bool), thr)}
        entry['per_control_group'] = {
            g: _counts((strat_labels == g) | (y_true == 1), thr) for g in control_groups}
        out[name] = entry
    return out


def _per_cancer_type_train_threshold_eval(y_test, y_pred_proba, strat_test, sample_ids_test, op_thr):
    """Per-cancer-type test-fold evaluation at the SAME train-derived thresholds.

    For each cancer type present in the test fold, evaluates the subset (all healthy
    vs this one type) using the fold-level op_thr thresholds unchanged. This mirrors
    single-threshold deployment and never tunes a threshold on the per-type subset.
    ROC/PR curves are kept as raw arrays (historic format of the nested-CV pickles).
    """
    auc_per_type, roc_per_type, pr_per_type = {}, {}, {}
    confusion_matrix_per_type = {}
    classification_report_per_type = {}
    sample_classification_per_type = {}
    healthy_labels_in_test = np.unique(strat_test[y_test == 0])
    unique_cancer_types_in_test = np.unique(strat_test[y_test == 1])

    threshold_95spec = op_thr['95_percent_specificity']
    threshold_90spec = op_thr['90_percent_specificity']
    threshold_youden = op_thr['youden_optimal']

    for cancer_type in unique_cancer_types_in_test:
        mask = np.isin(strat_test, healthy_labels_in_test) | (strat_test == cancer_type)
        y_test_subset = (strat_test[mask] == cancer_type).astype(int)
        y_pred_proba_subset = y_pred_proba[mask]
        sample_ids_subset = sample_ids_test[mask]

        fpr, tpr, thresholds = roc_curve(y_test_subset, y_pred_proba_subset)
        precision, recall, _ = precision_recall_curve(y_test_subset, y_pred_proba_subset)

        auc_per_type[cancer_type] = auc(fpr, tpr)
        roc_per_type[cancer_type] = {'fpr': fpr, 'tpr': tpr}
        pr_per_type[cancer_type] = {'precision': precision, 'recall': recall}

        preds = {
            'default_0.5': (y_pred_proba_subset >= 0.5).astype(int),
            '95_percent_specificity': (y_pred_proba_subset >= threshold_95spec).astype(int),
            '90_percent_specificity': (y_pred_proba_subset >= threshold_90spec).astype(int),
            'youden_optimal': (y_pred_proba_subset >= threshold_youden).astype(int),
        }
        confusion_matrix_per_type[cancer_type] = {
            key: confusion_matrix(y_test_subset, pred).tolist() for key, pred in preds.items()
        }
        classification_report_per_type[cancer_type] = {
            key: classification_report(y_test_subset, pred, output_dict=True, zero_division=0)
            for key, pred in preds.items()
        }
        roc_per_type[cancer_type]['thresholds'] = {
            'default': 0.5,
            '95_percent_specificity': float(threshold_95spec),
            '90_percent_specificity': float(threshold_90spec),
            'youden_optimal': float(threshold_youden),
            'train_fpr_at_95spec': op_thr['train_fpr_at_95spec'],
            'train_tpr_at_95spec': op_thr['train_tpr_at_95spec'],
            'train_fpr_at_90spec': op_thr['train_fpr_at_90spec'],
            'train_tpr_at_90spec': op_thr['train_tpr_at_90spec'],
        }
        sample_classification_per_type[cancer_type] = {
            key: {
                'correct': sample_ids_subset[y_test_subset == pred].tolist(),
                'incorrect': sample_ids_subset[y_test_subset != pred].tolist(),
            }
            for key, pred in preds.items()
        }

    return (auc_per_type, roc_per_type, pr_per_type, confusion_matrix_per_type,
            classification_report_per_type, sample_classification_per_type)


def _per_cancer_type_descriptive_eval(scores, y_binary, strat_labels, sample_ids_array):
    """Per-cancer-type evaluation with DESCRIPTIVE per-subset operating points.

    For each cancer type, evaluates the subset (all healthy vs this one type) and
    reads the 95%/90%-specificity and Youden thresholds off the subset's OWN ROC
    curve. These describe the best achievable operating point on that cohort; they
    are not pre-specified deployment thresholds (see the notes at the call sites).
    ROC/PR curves are stored as plain lists (historic format of these pickles).
    """
    auc_per_type, roc_per_type, pr_per_type = {}, {}, {}
    confusion_per_type, report_per_type = {}, {}
    sample_classification_per_type = {}
    healthy_labels = np.unique(strat_labels[y_binary == 0])
    cancer_types = np.unique(strat_labels[y_binary == 1])

    for cancer_type in cancer_types:
        mask = np.isin(strat_labels, healthy_labels) | (strat_labels == cancer_type)
        y_true_subset = (strat_labels[mask] == cancer_type).astype(int)
        scores_subset = scores[mask]
        sample_ids_subset = sample_ids_array[mask]

        if len(np.unique(y_true_subset)) < 2:
            continue

        fpr, tpr, thresholds = roc_curve(y_true_subset, scores_subset)
        precision, recall, _ = precision_recall_curve(y_true_subset, scores_subset)

        auc_per_type[cancer_type] = auc(fpr, tpr)
        roc_per_type[cancer_type] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist()}
        pr_per_type[cancer_type] = {'precision': precision.tolist(), 'recall': recall.tolist()}

        idx_95spec = np.argmin(np.abs(fpr - 0.05))
        threshold_95spec = thresholds[idx_95spec] if idx_95spec < len(thresholds) else 0.5
        idx_90spec = np.argmin(np.abs(fpr - 0.10))
        threshold_90spec = thresholds[idx_90spec] if idx_90spec < len(thresholds) else 0.5
        idx_youden = np.argmax(tpr - fpr)
        threshold_youden = thresholds[idx_youden] if idx_youden < len(thresholds) else 0.5

        preds = {
            'default_0.5': (scores_subset >= 0.5).astype(int),
            '95_percent_specificity': (scores_subset >= threshold_95spec).astype(int),
            '90_percent_specificity': (scores_subset >= threshold_90spec).astype(int),
            'youden_optimal': (scores_subset >= threshold_youden).astype(int),
        }
        confusion_per_type[cancer_type] = {
            key: confusion_matrix(y_true_subset, pred).tolist() for key, pred in preds.items()
        }
        report_per_type[cancer_type] = {
            key: classification_report(y_true_subset, pred, output_dict=True, zero_division=0)
            for key, pred in preds.items()
        }
        sample_classification_per_type[cancer_type] = {
            key: {
                'correct': sample_ids_subset[y_true_subset == pred].tolist(),
                'incorrect': sample_ids_subset[y_true_subset != pred].tolist(),
            }
            for key, pred in preds.items()
        }
        roc_per_type[cancer_type]['thresholds'] = {
            'default': 0.5,
            '95_percent_specificity': float(threshold_95spec),
            '90_percent_specificity': float(threshold_90spec),
            'youden_optimal': float(threshold_youden),
            'actual_fpr_at_95spec': float(fpr[idx_95spec]) if idx_95spec < len(fpr) else None,
            'actual_tpr_at_95spec': float(tpr[idx_95spec]) if idx_95spec < len(tpr) else None,
            'actual_fpr_at_90spec': float(fpr[idx_90spec]) if idx_90spec < len(fpr) else None,
            'actual_tpr_at_90spec': float(tpr[idx_90spec]) if idx_90spec < len(tpr) else None,
        }

    return (auc_per_type, roc_per_type, pr_per_type, confusion_per_type,
            report_per_type, sample_classification_per_type)


def _best_model_search_passthrough():
    """Number of trailing passthrough columns for the best-model search's selector: the
    covariate block of --augment_covariates (0 without it). --residualize_covariates is a
    fully nested-run construct and is refused here."""
    if COVARIATE_BLOCK['mode'] == 'residualize':
        raise ValueError("--residualize_covariates is only defined for --run_nested_cv_full")
    return int(COVARIATE_BLOCK['n']) if COVARIATE_BLOCK['mode'] == 'augment' else 0


def _refresh_deployment_thresholds(summary_path, X_data, y_labels, stratification_labels,
                                   sample_ids_array, use_selector, use_scaler,
                                   n_top_genes, selection_type):
    """
    --refresh_deployment_thresholds: re-derive the frozen operating points of a
    completed best-model-search configuration from the selected configuration's
    held-out scores on the search's own ten folds, without repeating the grid search.
    The thresholds being replaced are kept as 'deployment_thresholds_previous'.
    """
    with open(summary_path, 'rb') as fh:
        summary = pickle.load(fh)
    prev = summary.get('deployment_thresholds') or {}
    if prev.get('threshold_source') == THRESHOLD_SOURCE_INNER_CV:
        print(f"    deployment thresholds of {summary_path.name} already come from the "
              f"search's own folds - nothing to refresh.")
        return
    stored_ids = summary.get('oof_sample_ids')
    if stored_ids is not None and list(stored_ids) != list(sample_ids_array):
        raise ValueError(f"{summary_path.name}: stored sample order differs from the loaded "
                         f"cohort - use the data arguments of the original search.")
    best_params = dict(summary['best_params'])
    steps = []
    if use_selector:
        steps.append(('selector', TopGeneSelector(n_top_genes=n_top_genes,
                                                  selection_type=selection_type,
                                                  n_passthrough=_best_model_search_passthrough())))
    if use_scaler:
        steps.append(('scaler', StandardScaler()))
    steps.append(('clf', best_params['clf']))
    pipe = Pipeline(steps).set_params(**best_params)
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cv_splits = list(cv.split(X_data, stratification_labels))
    thresholds, scores = inner_cv_operating_thresholds(
        pipe, X_data, y_labels, cv_splits,
        expected_mean_auc=summary.get('best_cv_auc'), label=summary_path.name)
    summary['deployment_thresholds_previous'] = prev
    summary['deployment_thresholds'] = thresholds
    summary['oof_scores'] = scores.tolist()
    summary['oof_sample_ids'] = np.asarray(sample_ids_array).tolist()
    summary['oof_y_true'] = np.asarray(y_labels).tolist()
    summary['oof_auc'] = float(roc_auc_score(y_labels, scores))
    tmp_path = summary_path.with_name(summary_path.name + '.tmp')
    with open(tmp_path, 'wb') as fh:
        pickle.dump(summary, fh)
    os.replace(tmp_path, summary_path)
    print(f"    refreshed {summary_path.name}: thr@95%spec "
          f"{prev.get('95_percent_specificity', float('nan')):.4f} -> "
          f"{thresholds['95_percent_specificity']:.4f}")


def run_best_model_search(all_feature_arrays: dict, y_labels: np.ndarray, stratification_labels: np.ndarray,
                          sample_ids: np.ndarray, CLASSIFIERS: dict, CLF_PARAM_GRIDS: dict, 
                          DATA_PARAM_GRIDS: dict, args: argparse.Namespace, preprocessing_metadata: dict = None):
    """
    Runs a single, comprehensive GridSearchCV FOR EACH feature to find the
    single best combination of (classifier, n_top_genes, hyperparameters).
    
    Refits the best model on all data, evaluates it, and saves the detailed
    results (including per-cancer-type metrics) and the selected gene indices.
    
    Args:
        preprocessing_metadata: Dict containing info about which columns were kept during preprocessing.
                               Saved alongside models for use in external validation.
    """
    
    # Create a dedicated output directory
    best_model_output_dir = Path(args.output_dir) / f"best_model_{args.experiment_name}"
    os.makedirs(best_model_output_dir, exist_ok=True)
    print(f"Saving best models to: {best_model_output_dir}")
    
    # Save preprocessing metadata for external validation
    if preprocessing_metadata is not None:
        metadata_file = best_model_output_dir / "preprocessing_metadata.pkl"
        with open(metadata_file, 'wb') as f:
            pickle.dump(preprocessing_metadata, f)
        print(f"Saved preprocessing metadata to: {metadata_file}")
        print(f"  This will ensure external datasets use identical column filtering")
        
        # Debug: Print what's being saved
        kept_cols = preprocessing_metadata.get('kept_columns_per_feature', {})
        print(f"\n  Saved column counts:")
        for feat, indices in kept_cols.items():
            print(f"     {feat}: {len(indices)} column indices")
        print()
    
    # Convert sample_ids to a numpy array for easier indexing
    sample_ids_array = np.array(sample_ids)

    # --- Run the search for each feature ---
    for feature_name, X_data in all_feature_arrays.items():
        if X_data is None:
            print(f"\n--- Skipping feature '{feature_name}' (no data) ---")
            continue

        # --- Determine whether to use gene selection and scaler ---
        use_selector = not (args.no_gene_selection or args.predefined_feature_matrix)
        use_scaler = not getattr(args, 'skip_scaler', False)
        if use_selector:
            top_genes_list = DATA_PARAM_GRIDS['top_genes']
            if getattr(args, 'best_model_top_genes', None):
                # Resubmission of a subset of the k grid (e.g. only the k=15000 configurations
                # that were OOM-killed at the default memory), into the same output directory.
                top_genes_list = [k for k in top_genes_list if k in set(args.best_model_top_genes)]
                print(f"\n--- k grid restricted to {top_genes_list} (--best_model_top_genes) ---")
        else:
            top_genes_list = [X_data.shape[1]]
            print(f"\n--- No gene selection mode: using all {X_data.shape[1]} features for '{feature_name}' ---")

        # --- Output a model for each number of top genes ---
        for n_top_genes in top_genes_list:

            # Resume: a configuration whose summary is already on disk is not refitted
            # (summary_* is written last, so its presence means the config completed).
            existing = list(best_model_output_dir.glob(f"summary_{feature_name}_{n_top_genes}_genes_*_{args.selection_type}.pkl"))
            if existing:
                print(f"\n--- '{feature_name} | Genes={n_top_genes}' already complete ({existing[0].name}) - skipping ---")
                if getattr(args, 'refresh_deployment_thresholds', False):
                    _refresh_deployment_thresholds(
                        existing[0], X_data, y_labels, stratification_labels,
                        sample_ids_array, use_selector, use_scaler, n_top_genes,
                        args.selection_type)
                continue

            print(f"\n==================================================================")
            print(f"--- Starting Best Model Search for feature: '{feature_name} | Genes={n_top_genes}' ---")
            print(f"--- Data shape: {X_data.shape} ---")
            print(f"==================================================================")
            
            start_time = time.time()

            # --- Build the master parameter grid ---
            # This grid searches over classifiers, n_top_genes, and hyperparameters
            param_grid_list = []
            for clf_name, clf_instance in CLASSIFIERS.items():
                # Create a parameter grid for this classifier
                hyperparams = CLF_PARAM_GRIDS[clf_name].copy()
                hyperparams['clf'] = [clf_instance]
                param_grid_list.append(hyperparams)

            # --- Define the pipeline ---
            pipeline_steps = []
            if use_selector:
                # --augment_covariates: the trailing covariate block bypasses the ranking
                # (n_passthrough), exactly as in the fully nested runs
                pipeline_steps.append(('selector', TopGeneSelector(n_top_genes=n_top_genes, selection_type=args.selection_type,
                                                                   n_passthrough=_best_model_search_passthrough())))
            if use_scaler:
                pipeline_steps.append(('scaler', StandardScaler()))
            pipeline_steps.append(('clf', CLASSIFIERS["Support Vector Machine"]))  # Placeholder, will be replaced by grid search
            pipeline = Pipeline(pipeline_steps)
            
            # Use the same 10-fold stratified CV as the nested CV's outer loop
            cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
            cv_splits = list(cv.split(X_data, stratification_labels))
            
            # GridSearchCV with refit=True does:
            # 1. Cross-validates to find best hyperparameters (using cv_splits)
            # 2. Refits the best model on the ENTIRE dataset (not just best fold)
            # 3. Stores this refitted model as best_estimator_
            # Result: best_estimator_ is trained on ALL data and ready for production
            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid_list,
                scoring='roc_auc',
                cv=cv_splits,
                n_jobs=args.cores,
                verbose=3, # Provide high verbosity to see progress
                refit=True # Refits the best estimator on the whole dataset
            )
            
            try:
                grid_search.fit(X_data, y_labels)
                                
                # --- Extract Basic Results ---
                best_model = grid_search.best_estimator_  # Already trained on ALL data!
                best_params = grid_search.best_params_
                best_score = grid_search.best_score_
                
                # Find the string name of the best classifier
                best_clf_obj = best_params['clf']
                best_clf_name = next((name for name, clf in CLASSIFIERS.items() 
                                      if isinstance(best_clf_obj, type(clf))), "Unknown")
                clf_name_safe = best_clf_name.replace(" ", "_").lower()
                
                # Get the indices from the 'selector' step of the best pipeline (if present)
                if 'selector' in best_model.named_steps:
                    best_gene_indices = best_model.named_steps['selector'].indices_
                else:
                    best_gene_indices = np.arange(X_data.shape[1])

                print(f"   > Completed in {time.time() - start_time:.2f}s")
                print(f"   > Best: {best_clf_name} (CV AUC: {best_score:.4f})")
                
                print(f"  Best CV AUC: {best_score:.4f}")
                print(f"  Best Params: {best_params}")
                print(f"  Selected {len(best_gene_indices)} genes.")
                
                # DEBUG: Show selector indices (if selector exists)
                if use_selector:
                    print(f"\n  DEBUG: TopGeneSelector stored indices (first 20):")
                    print(f"     {best_gene_indices[:20].tolist() if len(best_gene_indices) >= 20 else best_gene_indices.tolist()}")
                    print(f"     -> Are indices sorted? {np.all(best_gene_indices[:-1] <= best_gene_indices[1:])}")
                    print(f"     -> These will be saved in the pipeline pickle file")
                
                # --- Calculate feature selection weights on the full dataset ---
                selection_weights_full = None
                selection_weights_selected = None
                if use_selector:
                    print("  Computing feature selection weights on full dataset...")
                    selection_weights_full = compute_feature_weights(X_data, y_labels, args.selection_type)
                    
                    # Extract weights for only the selected genes
                    if selection_weights_full is not None:
                        selection_weights_selected = selection_weights_full[best_gene_indices]
                        print(f"  Computed selection weights for {len(selection_weights_full)} features.")
                    else:
                        print(f"  No selection weights computed (selection_type={args.selection_type}).")
                
                # --- Run full evaluation on the refitted model ---
                print("  Evaluating refitted model on full dataset...")
                
                # NOTE: This block refits the chosen model on the FULL dataset and reports
                # IN-SAMPLE performance (no held-out set exists here by construction). Any
                # operating-point threshold computed below is necessarily derived from the
                # same data it is scored on; these are in-sample diagnostics for the final
                # deployed model, NOT the generalization estimates. The unbiased
                # generalization estimates come from the nested-CV functions, where
                # thresholds are chosen on the training fold (see
                # derive_operating_point_thresholds).
                y_pred_proba_full = best_model.predict_proba(X_data)[:, 1]
                y_pred_binary_full = best_model.predict(X_data)

                # 1. Calculate overall metrics
                fpr_full, tpr_full, _ = roc_curve(y_labels, y_pred_proba_full)
                auc_full = auc(fpr_full, tpr_full)
                precision_full, recall_full, _ = precision_recall_curve(y_labels, y_pred_proba_full)
                report_full = classification_report(y_labels, y_pred_binary_full, output_dict=True, zero_division=0)
                confusion_matrix_full = confusion_matrix(y_labels, y_pred_binary_full).tolist()

                # 2. Identify misclassified samples
                misclassified_indices = np.where(y_labels != y_pred_binary_full)[0]
                misclassified_sample_ids = sample_ids_array[misclassified_indices].tolist()

                # 3. Per-cancer-type evaluation (all healthy vs one cancer type),
                #    with descriptive operating points read off each subset's ROC.
                (auc_per_type, roc_per_type, pr_per_type, confusion_matrix_per_type,
                 classification_report_per_type, sample_classification_per_type) = \
                    _per_cancer_type_descriptive_eval(
                        y_pred_proba_full, y_labels, stratification_labels, sample_ids_array)

                print(f"  Overall AUC (on full data): {auc_full:.4f}")
                print(f"  Total misclassified samples (on full data): {len(misclassified_sample_ids)}")

                # --- Deployment operating points: CROSS-VALIDATED, not resubstitution ---
                # This refitted model is the one applied to external cohorts, so its
                # operating thresholds must be frozen from the training cohort alone.
                # The in-sample scores above separate the classes (near-)perfectly for
                # flexible classifiers, which makes the 95%-specificity threshold read
                # off them infinite (the 'inf' that reached fig4bc_config.csv). The
                # thresholds below come from the selected configuration's held-out
                # scores on the search's own ten folds (cv_splits) instead, which have
                # a realistic spread and transfer to a new cohort.
                print("  Deriving deployment thresholds from the search's cross-validated scores...")
                deployment_thresholds, oof_scores = inner_cv_operating_thresholds(
                    best_model, X_data, y_labels, cv_splits,
                    expected_mean_auc=best_score, label=f"{feature_name} best-model search")
                oof_auc = float(roc_auc_score(y_labels, oof_scores))
                print(f"  OOF AUC of the selected configuration: {oof_auc:.4f} | "
                      f"thr@95%spec = {deployment_thresholds['95_percent_specificity']:.4f} "
                      f"(train sens {deployment_thresholds['train_tpr_at_95spec']:.3f}), "
                      f"thr@90%spec = {deployment_thresholds['90_percent_specificity']:.4f}")

                # --- Save all results ---
                base_name = f"{feature_name}_{n_top_genes}_genes_{clf_name_safe}_{args.selection_type}"
                
                # 1. Save the full GridSearchCV object
                gs_output_path = best_model_output_dir / f"best_model_GridSearchCV_{base_name}.pkl"
                with open(gs_output_path, 'wb') as f:
                    pickle.dump(grid_search, f)
                print(f"  Saved full GridSearchCV object to: {gs_output_path}")

                # 2. Save just the selected gene indices
                indices_output_path = best_model_output_dir / f"best_model_indices_{base_name}.npy"
                np.save(indices_output_path, best_gene_indices)
                print(f"  Saved best gene indices to: {indices_output_path}")

                # Extract model weights (coefficients, feature importances, or
                # MLP first-layer norms) from the refitted pipeline
                model_weights, model_weights_type = _extract_model_weights(best_model)
                if model_weights is not None:
                    print(f"  Extracted {len(model_weights)} model weights ({model_weights_type}).")
                else:
                    print(f"  Warning: Could not extract weights from {best_clf_name}.")

                # Save model weights separately
                if model_weights is not None:
                    weights_output_path = best_model_output_dir / f"best_model_weights_{base_name}.npy"
                    np.save(weights_output_path, model_weights)
                    print(f"  Saved model weights to: {weights_output_path}")
                
                # Save selection weights separately
                if selection_weights_selected is not None:
                    selection_weights_output_path = best_model_output_dir / f"selection_weights_{base_name}.npy"
                    np.save(selection_weights_output_path, selection_weights_selected)
                    print(f"  Saved selection weights to: {selection_weights_output_path}")

                # 3. Save the comprehensive summary
                summary_output_path = best_model_output_dir / f"summary_{base_name}.pkl"
                summary = {
                    'feature': feature_name,
                    'best_classifier_name': best_clf_name,
                    'selection_type': args.selection_type,
                    'best_cv_auc': best_score,
                    'best_params': best_params,
                    'num_genes_selected': len(best_gene_indices),
                    'gene_indices': best_gene_indices,
                    'gene_indices_file': str(indices_output_path),
                    'gridsearch_object_file': str(gs_output_path),
                    'gridsearch_cv_results_df': pd.DataFrame(grid_search.cv_results_),
                    
                    # Model weights
                    'model_weights': model_weights,
                    'model_weights_type': model_weights_type,
                    'model_weights_file': str(weights_output_path) if model_weights is not None else None,
                    
                    # Selection weights
                    'selection_weights_all_features': selection_weights_full,
                    'selection_weights_selected_features': selection_weights_selected,
                    'selection_weights_file': str(selection_weights_output_path) if selection_weights_selected is not None else None,

                    # Frozen operating points for external application (from the
                    # search's cross-validated scores, see above). Older summaries lack
                    # these keys; external validation then reports no fixed-threshold
                    # block.
                    'deployment_thresholds': deployment_thresholds,
                    'oof_scores': oof_scores.tolist(),
                    'oof_sample_ids': sample_ids_array.tolist(),
                    'oof_y_true': y_labels.tolist(),
                    'oof_auc': oof_auc,

                    # --- New Detailed Evaluation Metrics ---
                    'full_dataset_evaluation': {
                        'description': "Metrics from evaluating the refitted best model on the *entire* dataset.",
                        'auc_full_model': auc_full,
                        'fpr_full_model': fpr_full.tolist(),
                        'tpr_full_model': tpr_full.tolist(),
                        'precision_full_model': precision_full.tolist(),
                        'recall_full_model': recall_full.tolist(),
                        'classification_report_full_model': report_full,
                        'confusion_matrix_full_model': confusion_matrix_full,
                        'misclassified_sample_ids': misclassified_sample_ids,
                        'total_samples_misclassified': len(misclassified_sample_ids),
                        'auc_per_cancer_type': auc_per_type,
                        'roc_per_cancer_type': roc_per_type,
                        'pr_per_cancer_type': pr_per_type,
                        'confusion_matrix_per_cancer_type': confusion_matrix_per_type,
                        'classification_report_per_cancer_type': classification_report_per_type,
                        'sample_classification_per_cancer_type': sample_classification_per_type,
                    }
                }
                with open(summary_output_path, 'wb') as f:
                    pickle.dump(summary, f)
                print(f"  Saved comprehensive summary to: {summary_output_path}")
                
            except Exception as e:
                print(f"!!! ERROR during GridSearchCV for feature '{feature_name}': {e}")
                import traceback
                traceback.print_exc()

    print("\n--- Best Model Search for all features is complete ---")


def get_best_model_feature_name(results_dir: str) -> str:
    """
    Quickly identify which feature was used by the best model WITHOUT loading the full model.
    
    This is optimized for external validation where we want to know which feature to load
    before loading all feature data.
    
    Parameters
    ----------
    results_dir : str
        Path to directory containing best_model_GridSearchCV_*.pkl files
    
    Returns
    -------
    str
        Name of the feature used by the best model (e.g., 'griffin_diff', 'fslr_coverage')
    
    Example
    -------
    >>> feature_needed = get_best_model_feature_name('./best_model_results/best_model_myexp/')
    >>> print(f"Only need to load: {feature_needed}")
    """
    
    results_dir = Path(results_dir)
    
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    # Find all GridSearchCV pickle files
    gridsearch_files = list(results_dir.glob("best_model_GridSearchCV_*.pkl"))
    
    if len(gridsearch_files) == 0:
        raise FileNotFoundError(f"No GridSearchCV files found in {results_dir}")
    
    print(f"\n{'='*90}")
    print(f"IDENTIFYING REQUIRED FEATURE FROM BEST MODEL")
    print(f"{'='*90}")
    print(f"Found {len(gridsearch_files)} model files in {results_dir}")
    
    # Extract just the feature and CV AUC from each file (minimal loading)
    best_auc = -1
    best_feature = None
    
    for pkl_file in gridsearch_files:
        try:
            with open(pkl_file, 'rb') as f:
                gs_result = pickle.load(f)
            
            # Extract feature name from filename
            # Filename format: best_model_GridSearchCV_<feature>_<n_genes>_genes_<classifier>_<selection>.pkl
            # We only want the <feature> part (e.g., 'griffin_diff', 'fslr_coverage', 'mwh')
            full_name = pkl_file.stem.replace('best_model_GridSearchCV_', '')
            
            # Split by '_' and find where the numeric part starts (e.g., '15000_genes')
            parts = full_name.split('_')
            feature_parts = []
            for part in parts:
                if part.isdigit():  # Stop when we hit the n_genes number
                    break
                feature_parts.append(part)
            
            feature_name = '_'.join(feature_parts)
            
            # Get CV AUC
            cv_auc = gs_result.best_score_
            
            if cv_auc > best_auc:
                best_auc = cv_auc
                best_feature = feature_name
                
        except Exception as e:
            print(f"    Error reading {pkl_file.name}: {e}")
            continue
    
    if best_feature is None:
        raise ValueError("Could not identify best model feature")
    
    print(f"\nBest model uses feature: '{best_feature}'")
    print(f"  CV AUC: {best_auc:.4f}")
    print(f"  -> Only this feature will be loaded for external validation")
    print(f"{'='*90}\n")
    
    return best_feature


def analyze_best_models_and_get_best(results_dir: str, verbose: bool = True, 
                                     filter_feature: str = None, filter_n_genes: int = None):
    """
    Analyze all saved best_model_GridSearchCV results and return the overall best model.
    
    This function loads all GridSearchCV pickle files from a directory, extracts key
    metrics, creates an overview DataFrame, and returns the single best model based
    on CV AUC score.
    
    IMPORTANT: The returned best_model.best_estimator_ is already trained on the FULL
    dataset. GridSearchCV with refit=True uses cross-validation to find the best
    hyperparameters, then refits the model on ALL data. You can use it immediately
    for predictions without retraining.
    
    Args:
        filter_feature: If provided, only consider models using this feature
        filter_n_genes: If provided, only consider models using this number of genes
    
    Parameters
    ----------
    results_dir : str
        Path to directory containing best_model_GridSearchCV_*.pkl files
    verbose : bool, default=True
        If True, prints detailed overview table
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'overview_df': pandas DataFrame with all model metrics
        - 'best_model': The GridSearchCV object with the highest CV AUC
                       (best_model.best_estimator_ is trained on full data)
        - 'best_model_info': Dict with metadata about the best model
        - 'best_model_path': Path to the best model pickle file
    
    Example
    -------
    >>> results = analyze_best_models_and_get_best('./best_model_results/best_model_myexp/')
    >>> print(results['overview_df'])
    >>> best_model = results['best_model']
    >>> best_pipeline = best_model.best_estimator_  # Already fitted on ALL data!
    >>> y_pred = best_pipeline.predict(X_new)  # Ready to use!
    """
    
    results_dir = Path(results_dir)
    
    if not results_dir.exists():
        raise ValueError(f"Results directory does not exist: {results_dir}")
    
    # Find all GridSearchCV pickle files
    gridsearch_files = list(results_dir.glob("best_model_GridSearchCV_*.pkl"))
    
    if len(gridsearch_files) == 0:
        raise ValueError(f"No best_model_GridSearchCV_*.pkl files found in {results_dir}")
    
    print(f"\n{'='*90}")
    print(f"ANALYZING BEST MODEL RESULTS")
    print(f"{'='*90}")
    print(f"Found {len(gridsearch_files)} model result files in {results_dir}")
    
    # Extract information from each file
    results_list = []
    
    for pkl_file in sorted(gridsearch_files):
        try:
            # Parse filename to extract metadata
            # Format: best_model_GridSearchCV_{feature}_{n_genes}_genes_{classifier}_{selection_type}.pkl
            filename = pkl_file.stem  # Remove .pkl extension
            parts = filename.replace("best_model_GridSearchCV_", "").split("_")
            
            # Load the GridSearchCV object
            with open(pkl_file, 'rb') as f:
                grid_search = pickle.load(f)
            
            # Extract feature name (may contain underscores)
            # Work backwards: last part is selection_type, then classifier words, then "genes", then n_genes, rest is feature
            # Find "genes" keyword position from the end
            genes_idx = None
            for i in range(len(parts) - 1, -1, -1):
                if parts[i] == "genes":
                    genes_idx = i
                    break
            
            if genes_idx is None or genes_idx < 2:
                print(f"Warning: Could not parse filename format for {pkl_file.name}, skipping.")
                continue
            
            # Extract components
            n_genes = int(parts[genes_idx - 1])
            feature_name = "_".join(parts[:genes_idx - 1])
            
            # The rest after "genes" is classifier + selection_type
            remaining_parts = parts[genes_idx + 1:]

            # Resolve selection_type from filename parts (handles 'mean_diff' and 'fold_change')
            valid_selection_types = {'p_value', 'fold_change', 'weighted', 'mean_diff', 'random'}
            selection_type = None
            classifier_parts = []

            if len(remaining_parts) >= 2:
                last_two = "_".join(remaining_parts[-2:])
                if last_two in valid_selection_types:
                    selection_type = last_two
                    classifier_parts = remaining_parts[:-2]
            if selection_type is None and len(remaining_parts) >= 1:
                last_one = remaining_parts[-1]
                if last_one in valid_selection_types:
                    selection_type = last_one
                    classifier_parts = remaining_parts[:-1]

            if selection_type is None:
                # Fallback: assume last token is selection type
                selection_type = remaining_parts[-1]
                classifier_parts = remaining_parts[:-1]

            classifier_name = " ".join(classifier_parts).title()
            
            # Get GridSearch results
            best_score = grid_search.best_score_
            best_params = grid_search.best_params_
            
            # Extract key hyperparameters (depends on classifier)
            key_hyperparams = {}
            for param_name, param_value in best_params.items():
                if param_name != 'clf':  # Skip the classifier object itself
                    # Clean parameter name (remove 'clf__' prefix)
                    clean_name = param_name.replace('clf__', '')
                    key_hyperparams[clean_name] = param_value
            
            # Count total hyperparameter combinations tested
            n_combinations = len(grid_search.cv_results_['mean_test_score'])
            
            results_list.append({
                'feature': feature_name,
                'n_genes': n_genes,
                'classifier': classifier_name,
                'selection_type': selection_type,
                'cv_auc': best_score,
                'cv_std': grid_search.cv_results_['std_test_score'][grid_search.best_index_],
                'n_combinations_tested': n_combinations,
                'best_params': key_hyperparams,
                'filepath': str(pkl_file),
                'grid_search_object': grid_search
            })
            
        except Exception as e:
            print(f"Warning: Error processing {pkl_file.name}: {e}")
            continue
    
    if len(results_list) == 0:
        raise ValueError("No valid results could be extracted from the files.")
    
    # Create DataFrame for overview
    df = pd.DataFrame(results_list)
    
    # Apply filters if specified
    if filter_feature:
        df = df[df['feature'] == filter_feature]
        if len(df) == 0:
            raise ValueError(f"No models found with feature '{filter_feature}'")
    
    if filter_n_genes:
        df = df[df['n_genes'] == filter_n_genes]
        if len(df) == 0:
            raise ValueError(f"No models found with n_genes={filter_n_genes}")
    
    # Sort by CV AUC (descending)
    df_sorted = df.sort_values('cv_auc', ascending=False).reset_index(drop=True)
    
    # Find the best model
    best_idx = df_sorted.index[0]
    best_model_row = df_sorted.iloc[best_idx]
    best_model_gs = best_model_row['grid_search_object']
    
    # Print overview
    if verbose:
        print(f"\n{'='*90}")
        print("MODEL OVERVIEW (Top 20 by CV AUC)")
        print(f"{'='*90}")
        
        # Create display DataFrame (without grid_search_object column)
        display_df = df_sorted.drop(columns=['grid_search_object', 'filepath']).head(20)
        
        # Format for better display
        display_df['cv_auc'] = display_df['cv_auc'].map('{:.4f}'.format)
        display_df['cv_std'] = display_df['cv_std'].map('{:.4f}'.format)
        display_df['params_summary'] = display_df['best_params'].apply(
            lambda x: ', '.join([f"{k}={v}" for k, v in list(x.items())[:3]])  # Show first 3 params
        )
        
        print(display_df[['feature', 'n_genes', 'classifier', 'selection_type', 'cv_auc', 'cv_std', 'n_combinations_tested']].to_string(index=True))
        
        print(f"\n{'='*90}")
        print("BEST MODEL")
        print(f"{'='*90}")
        print(f"Feature:          {best_model_row['feature']}")
        print(f"N Genes:          {best_model_row['n_genes']}")
        print(f"Classifier:       {best_model_row['classifier']}")
        print(f"Selection Type:   {best_model_row['selection_type']}")
        print(f"CV AUC:           {best_model_row['cv_auc']:.4f} +/- {best_model_row['cv_std']:.4f}")
        print(f"Combinations:     {best_model_row['n_combinations_tested']}")
        print(f"\nBest Hyperparameters:")
        for param_name, param_value in best_model_row['best_params'].items():
            print(f"  {param_name}: {param_value}")
        print(f"\nFile: {best_model_row['filepath']}")
        print(f"{'='*90}\n")
    
    # Return comprehensive results
    return {
        'overview_df': df_sorted,
        'best_model': best_model_gs,
        'best_model_info': {
            'feature': best_model_row['feature'],
            'n_genes': int(best_model_row['n_genes']),
            'classifier': best_model_row['classifier'],
            'selection_type': best_model_row['selection_type'],
            'cv_auc': float(best_model_row['cv_auc']),
            'cv_std': float(best_model_row['cv_std']),
            'best_params': best_model_row['best_params'],
            'n_combinations_tested': int(best_model_row['n_combinations_tested'])
        },
        'best_model_path': best_model_row['filepath']
    }


def _extract_model_weights(pipeline: Pipeline):
    """
    Extract model weights from a fitted pipeline (if available).
    Returns (weights_array_or_None, weights_type_string_or_None).
    """
    clf_step = pipeline.named_steps.get('clf')
    if clf_step is None:
        return None, None

    if hasattr(clf_step, 'coef_'):
        weights = clf_step.coef_.flatten() if clf_step.coef_.ndim > 1 else clf_step.coef_
        return weights, 'coefficients'
    if hasattr(clf_step, 'feature_importances_'):
        return clf_step.feature_importances_, 'feature_importances'
    if hasattr(clf_step, 'coefs_'):
        first_layer_weights = clf_step.coefs_[0]
        weights = np.linalg.norm(first_layer_weights, axis=1)
        return weights, 'mlp_first_layer_norms'

    return None, None


def _build_best_pipeline(best_classifier, best_params: dict, n_genes: int, selection_type: str):
    """
    Build a fresh pipeline matching the best-model configuration.
    """
    clf = clone(best_classifier)
    if best_params:
        clf.set_params(**best_params)
    return Pipeline([
        ('selector', TopGeneSelector(n_top_genes=n_genes, selection_type=selection_type)),
        ('scaler', StandardScaler()),
        ('clf', clf)
    ])


def run_label_shuffle_experiment(
    best_model_results_dir: str,
    all_feature_arrays: dict,
    y_labels: np.ndarray,
    stratification_labels: np.ndarray,
    sample_ids: np.ndarray,
    CLASSIFIERS: dict,
    args: argparse.Namespace,
    filter_feature: str = None,
    filter_n_genes: int = None
):
    """
    Run label-shuffle (permutation) experiment using ONLY the best model configuration.

    This estimates whether model weights/metrics collapse under no-signal labels.
    Produces debug logs, summary tables, and paper-quality plots.
    """

    print("\n" + "="*90)
    print("LABEL SHUFFLE EXPERIMENT (BEST MODEL ONLY)")
    print("="*90)

    best_model_results_dir = Path(best_model_results_dir)
    if not best_model_results_dir.exists():
        print(f"Error: Best model directory not found: {best_model_results_dir}")
        return

    print(f"Best model directory: {best_model_results_dir}")
    if filter_feature:
        print(f"Filter: feature={filter_feature}")
    if filter_n_genes:
        print(f"Filter: n_genes={filter_n_genes}")

    best_results = analyze_best_models_and_get_best(
        str(best_model_results_dir),
        verbose=True,
        filter_feature=filter_feature,
        filter_n_genes=filter_n_genes
    )

    best_info = best_results['best_model_info']
    best_pipeline = best_results['best_model'].best_estimator_

    feature_name = best_info['feature']
    if feature_name not in all_feature_arrays:
        print(f"Error: Feature '{feature_name}' not found in loaded arrays.")
        print(f"   Available: {list(all_feature_arrays.keys())}")
        return

    X = all_feature_arrays[feature_name]
    n_samples, n_features = X.shape

    print(f"\nBest model configuration:")
    print(f"  Feature: {feature_name}")
    print(f"  Classifier: {best_info['classifier']}")
    print(f"  Selection: {best_info['selection_type']}")
    print(f"  N genes: {best_info['n_genes']}")
    print(f"  CV AUC (best): {best_info['cv_auc']:.4f}")
    print(f"  Data shape: {X.shape}")

    # Prepare output directory
    output_dir = Path(args.label_shuffle_output_dir) if args.label_shuffle_output_dir else Path(args.output_dir) / f"label_shuffle_{args.experiment_name}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Extract true model weights and indices
    true_selector = best_pipeline.named_steps.get('selector')
    true_selected_indices = true_selector.indices_ if true_selector is not None else None
    true_weights, true_weights_type = _extract_model_weights(best_pipeline)

    if true_selected_indices is None:
        print("Warning: Best model has no selector indices. Skipping weight comparisons.")

    if true_selected_indices is not None:
        print(f"\nDEBUG: True selector indices (first 20):")
        print(f"   {true_selected_indices[:20].tolist() if len(true_selected_indices) >= 20 else true_selected_indices.tolist()}")
        print(f"   -> Are indices sorted? {np.all(true_selected_indices[:-1] <= true_selected_indices[1:])}")
        true_indices_path = output_dir / "true_selected_indices.npy"
        np.save(true_indices_path, true_selected_indices)
        print(f"   Saved true selected indices to: {true_indices_path}")

    # Full-data AUC for the true model (same data)
    y_true_proba = best_pipeline.predict_proba(X)[:, 1]
    true_full_auc = roc_auc_score(y_labels, y_true_proba)

    print(f"\nTrue model full-data AUC: {true_full_auc:.4f}")

    # Map true weights to full feature space for comparisons
    true_full_weights = None
    if true_weights is not None and true_selected_indices is not None:
        if len(true_weights) != len(true_selected_indices):
            print(f"Warning: True weights length ({len(true_weights)}) does not match indices length ({len(true_selected_indices)}).")
        true_full_weights = np.full(n_features, np.nan)
        true_full_weights[true_selected_indices] = true_weights
        print(f"True weights extracted ({true_weights_type}).")
    else:
        print("Warning: True model weights not available; weight stability plots will be skipped.")

    # Build reusable classifier configuration
    if best_info['classifier'] not in CLASSIFIERS:
        print(f"Error: Classifier '{best_info['classifier']}' not in CLASSIFIERS dictionary.")
        return

    base_classifier = CLASSIFIERS[best_info['classifier']]
    best_params = best_info.get('best_params', {})

    # Experiment parameters
    n_shuffles = args.label_shuffle_n_shuffles
    top_k = min(args.label_shuffle_top_k, best_info['n_genes'])

    print(f"\nPermutation settings:")
    print(f"  Shuffles: {n_shuffles}")
    print(f"  CV folds: {args.label_shuffle_cv_folds}")
    print(f"  Seed: {args.label_shuffle_seed}")
    print(f"  Top-K for stability: {top_k}")

    shuffle_results = []
    shuffle_full_weights = []
    shuffle_selected_indices = []

    # Precompute true top-k indices for comparison
    true_top_k_indices = None
    if true_full_weights is not None:
        true_selected_weights = true_full_weights.copy()
        true_selected_weights[np.isnan(true_selected_weights)] = 0
        true_top_k_indices = np.argsort(np.abs(true_selected_weights))[-top_k:]

    # Compute the OBSERVED (unshuffled) CV AUC using the SAME CV procedure as the
    # permutations, so the permutation p-value compares like-with-like. Comparing the
    # shuffle distribution against GridSearchCV's best_score_ (a different CV scheme,
    # different fold count / inner-loop selection) would invalidate the test.
    observed_cv = StratifiedKFold(n_splits=args.label_shuffle_cv_folds, shuffle=True, random_state=args.label_shuffle_seed)
    observed_fold_aucs = []
    for train_idx, test_idx in observed_cv.split(X, y_labels):
        pipeline_obs = _build_best_pipeline(base_classifier, best_params, best_info['n_genes'], best_info['selection_type'])
        pipeline_obs.fit(X[train_idx], y_labels[train_idx])
        y_obs_proba = pipeline_obs.predict_proba(X[test_idx])[:, 1]
        observed_fold_aucs.append(roc_auc_score(y_labels[test_idx], y_obs_proba))
    observed_cv_auc = float(np.mean(observed_fold_aucs))
    print(f"\nObserved (unshuffled) CV AUC [matched CV scheme]: {observed_cv_auc:.4f}")
    print(f"  (GridSearchCV best_score_ for reference: {best_info['cv_auc']:.4f})")

    for i in range(n_shuffles):
        seed = args.label_shuffle_seed + i
        rng = np.random.RandomState(seed)
        y_perm = rng.permutation(y_labels)

        print(f"\nShuffle {i+1}/{n_shuffles} (seed={seed})")
        print(f"   Label distribution: {format_distribution(y_perm)}")

        # CV AUC on shuffled labels
        cv = StratifiedKFold(n_splits=args.label_shuffle_cv_folds, shuffle=True, random_state=seed)
        fold_aucs = []
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y_perm), start=1):
            pipeline_cv = _build_best_pipeline(base_classifier, best_params, best_info['n_genes'], best_info['selection_type'])
            pipeline_cv.fit(X[train_idx], y_perm[train_idx])
            y_proba = pipeline_cv.predict_proba(X[test_idx])[:, 1]
            fold_auc = roc_auc_score(y_perm[test_idx], y_proba)
            fold_aucs.append(fold_auc)
            print(f"   Fold {fold_idx}: AUC={fold_auc:.4f}")

        cv_auc = float(np.mean(fold_aucs))
        cv_std = float(np.std(fold_aucs))
        print(f"   CV AUC (mean+/-std): {cv_auc:.4f} +/- {cv_std:.4f}")

        # Fit full model for weights + full-data AUC
        pipeline_full = _build_best_pipeline(base_classifier, best_params, best_info['n_genes'], best_info['selection_type'])
        pipeline_full.fit(X, y_perm)

        y_perm_proba = pipeline_full.predict_proba(X)[:, 1]
        full_auc = roc_auc_score(y_perm, y_perm_proba)

        selected_indices = pipeline_full.named_steps['selector'].indices_
        shuffle_weights, shuffle_weights_type = _extract_model_weights(pipeline_full)

        # Map shuffle weights to full space (respect selector ordering)
        shuffle_full = None
        if shuffle_weights is not None and selected_indices is not None:
            if len(shuffle_weights) != len(selected_indices):
                print(f"    Weight/indices length mismatch: {len(shuffle_weights)} vs {len(selected_indices)}")
            shuffle_full = np.full(n_features, np.nan)
            shuffle_full[selected_indices] = shuffle_weights

        # Compute stability metrics
        jaccard = None
        weight_corr = None
        mean_abs_weight = None

        if shuffle_full is not None:
            mean_abs_weight = float(np.nanmean(np.abs(shuffle_full)))

        if true_top_k_indices is not None and shuffle_full is not None:
            shuffle_weights_for_top = shuffle_full.copy()
            shuffle_weights_for_top[np.isnan(shuffle_weights_for_top)] = 0
            shuffle_top_k = np.argsort(np.abs(shuffle_weights_for_top))[-top_k:]
            intersection = len(set(true_top_k_indices).intersection(set(shuffle_top_k)))
            union = len(set(true_top_k_indices).union(set(shuffle_top_k)))
            jaccard = float(intersection / union) if union > 0 else None

        if true_full_weights is not None and shuffle_full is not None:
            overlap_idx = np.where(~np.isnan(true_full_weights) & ~np.isnan(shuffle_full))[0]
            if len(overlap_idx) >= 3:
                weight_corr = float(np.corrcoef(true_full_weights[overlap_idx], shuffle_full[overlap_idx])[0, 1])

        shuffle_results.append({
            'shuffle_idx': i + 1,
            'seed': seed,
            'cv_auc_mean': cv_auc,
            'cv_auc_std': cv_std,
            'full_auc': float(full_auc),
            'n_selected': len(selected_indices) if selected_indices is not None else None,
            'weights_type': shuffle_weights_type,
            'mean_abs_weight': mean_abs_weight,
            'top_k_jaccard_with_true': jaccard,
            'weight_correlation_with_true': weight_corr
        })

        if shuffle_full is not None:
            shuffle_full_weights.append(shuffle_full)
        if selected_indices is not None:
            shuffle_selected_indices.append(selected_indices)

    # Convert to DataFrame
    results_df = pd.DataFrame(shuffle_results)
    results_csv = output_dir / "label_shuffle_results.csv"
    results_df.to_csv(results_csv, index=False)

    # Summary stats
    # Permutation p-value with the (count + 1) / (n + 1) correction so it can never be
    # exactly 0 (an unbiased, valid estimator). Compared against the matched-CV observed
    # statistic, NOT GridSearchCV's best_score_.
    n_ge = int(np.sum(results_df['cv_auc_mean'] >= observed_cv_auc))
    p_value = float((n_ge + 1) / (n_shuffles + 1))
    print(f"\nPermutation p-value (CV AUC >= observed): {p_value:.4f}  "
          f"[{n_ge}/{n_shuffles} permutations >= observed; (count+1)/(n+1) correction]")

    summary = {
        'best_model_info': best_info,
        'true_full_auc': float(true_full_auc),
        'true_cv_auc': observed_cv_auc,
        'observed_cv_auc': observed_cv_auc,
        'gridsearch_cv_auc': float(best_info['cv_auc']),
        'n_shuffles': n_shuffles,
        'cv_auc_mean': float(results_df['cv_auc_mean'].mean()),
        'cv_auc_std': float(results_df['cv_auc_mean'].std()),
        'p_value_cv_auc': p_value,
        'weights_type': true_weights_type,
        'top_k': top_k
    }

    summary_path = output_dir / "label_shuffle_summary.pkl"
    with open(summary_path, 'wb') as f:
        pickle.dump(summary, f)

    if len(shuffle_selected_indices) > 0:
        indices_path = output_dir / "shuffle_selected_indices.npy"
        np.save(indices_path, np.array(shuffle_selected_indices, dtype=object), allow_pickle=True)
        print(f"   Saved per-shuffle selected indices to: {indices_path}")

    # Plotting
    plot_path = output_dir / "label_shuffle_stability.png"
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.25)

    # Panel 1: CV AUC distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(results_df['cv_auc_mean'], bins=20, color='steelblue', edgecolor='black', alpha=0.8)
    ax1.axvline(observed_cv_auc, color='crimson', linestyle='--', linewidth=2, label=f"Observed CV AUC={observed_cv_auc:.3f}")
    ax1.set_title("Shuffled Label CV AUC Distribution", fontweight='bold')
    ax1.set_xlabel("CV AUC (mean across folds)")
    ax1.set_ylabel("Count")
    ax1.legend()
    ax1.grid(alpha=0.2)

    # Panel 2: Full-data AUC distribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(results_df['full_auc'], bins=20, color='slategray', edgecolor='black', alpha=0.8)
    ax2.axvline(true_full_auc, color='crimson', linestyle='--', linewidth=2, label=f"True Full AUC={true_full_auc:.3f}")
    ax2.set_title("Shuffled Label Full-Data AUC", fontweight='bold')
    ax2.set_xlabel("Full-data AUC")
    ax2.set_ylabel("Count")
    ax2.legend()
    ax2.grid(alpha=0.2)

    if true_full_weights is not None and len(shuffle_full_weights) > 0:
        shuffle_full_weights_arr = np.vstack(shuffle_full_weights)

        # Panel 3: Jaccard overlap
        ax3 = fig.add_subplot(gs[1, 0])
        if results_df['top_k_jaccard_with_true'].notna().any():
            ax3.hist(results_df['top_k_jaccard_with_true'].dropna(), bins=20, color='teal', edgecolor='black', alpha=0.8)
            ax3.set_title(f"Top-{top_k} Jaccard Overlap vs True", fontweight='bold')
            ax3.set_xlabel("Jaccard overlap")
            ax3.set_ylabel("Count")
            ax3.grid(alpha=0.2)
        else:
            ax3.text(0.5, 0.5, "No weight-based overlap available", ha='center', va='center')
            ax3.axis('off')

        # Panel 4: Weight correlation
        ax4 = fig.add_subplot(gs[1, 1])
        if results_df['weight_correlation_with_true'].notna().any():
            ax4.hist(results_df['weight_correlation_with_true'].dropna(), bins=20, color='purple', edgecolor='black', alpha=0.8)
            ax4.set_title("Weight Correlation vs True", fontweight='bold')
            ax4.set_xlabel("Correlation")
            ax4.set_ylabel("Count")
            ax4.grid(alpha=0.2)
        else:
            ax4.text(0.5, 0.5, "No weight correlation available", ha='center', va='center')
            ax4.axis('off')

        # Panel 5: Heatmap of shuffle weights for true top-k indices
        ax5 = fig.add_subplot(gs[2, 0])
        if true_top_k_indices is not None:
            heatmap_data = shuffle_full_weights_arr[:, true_top_k_indices].T
            im = ax5.imshow(heatmap_data, aspect='auto', cmap='coolwarm', interpolation='nearest')
            ax5.set_title("Shuffle Weights on True Top-K Features", fontweight='bold')
            ax5.set_xlabel("Shuffle #")
            ax5.set_ylabel("True Top-K Features")
            plt.colorbar(im, ax=ax5, fraction=0.046, pad=0.04)
        else:
            ax5.text(0.5, 0.5, "No true top-K indices available", ha='center', va='center')
            ax5.axis('off')

        # Panel 6: Selection frequency of true top-k features
        ax6 = fig.add_subplot(gs[2, 1])
        if true_top_k_indices is not None:
            freq_counts = []
            for idx in true_top_k_indices:
                freq = np.sum(~np.isnan(shuffle_full_weights_arr[:, idx])) / shuffle_full_weights_arr.shape[0]
                freq_counts.append(freq)
            ax6.bar(np.arange(len(freq_counts)), freq_counts, color='darkorange', edgecolor='black')
            ax6.set_ylim([0, 1])
            ax6.set_title("Selection Frequency of True Top-K", fontweight='bold')
            ax6.set_xlabel("True Top-K Feature Rank")
            ax6.set_ylabel("Selection frequency")
            ax6.grid(axis='y', alpha=0.2)
        else:
            ax6.text(0.5, 0.5, "No selection frequency available", ha='center', va='center')
            ax6.axis('off')
    else:
        ax3 = fig.add_subplot(gs[1:, :])
        ax3.text(0.5, 0.5, "Weight-based analyses unavailable for this classifier", ha='center', va='center', fontsize=12)
        ax3.axis('off')

    fig.suptitle(
        f"Label Shuffle Stability Analysis\nFeature={feature_name} | Classifier={best_info['classifier']} | N={best_info['n_genes']} | Selection={best_info['selection_type']}",
        fontsize=14,
        fontweight='bold',
        y=0.98
    )

    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nLabel shuffle results saved:")
    print(f"   CSV: {results_csv}")
    print(f"   Summary: {summary_path}")
    print(f"   Plot: {plot_path}")
    print("="*90 + "\n")


def validate_best_model_on_external_dataset(
    best_model_results_dir: str,
    external_feature_arrays: dict,
    external_y_labels: np.ndarray,
    external_stratification_labels: np.ndarray,
    external_sample_ids: np.ndarray,
    args: argparse.Namespace,
    training_preprocessing_metadata: dict = None,
    filter_feature: str = None,
    filter_n_genes: int = None,
    validation_mode: str = 'pipeline'
):
    """
    Validates the best model from one dataset on a completely different external dataset.
    
    IMPORTANT: The external dataset filtering (healthy vs. cancer) is determined by
    args.external_metadata_file, NOT args.metadata_file. This ensures that the
    external dataset's labels are correctly identified based on its own metadata,
    not the training dataset's metadata.
    
    Args:
        training_preprocessing_metadata: Metadata from training about which columns were kept.
                                        If provided, external data will be filtered to match.
        filter_feature: If provided, only consider models using this feature
        filter_n_genes: If provided, only consider models using this number of genes
    
    Workflow:
    1. Analyzes all models in best_model_results_dir to find the overall best
    2. Extracts its configuration (feature, n_genes, selection_type)
    3. Applies the FITTED model to the external dataset
    4. Generates comprehensive evaluation metrics
    
    This is true external validation - the model was never trained on external data.
    
    Parameters
    ----------
    best_model_results_dir : str
        Directory containing best_model_GridSearchCV_*.pkl files from training dataset
    external_feature_arrays : dict
        Feature arrays from external dataset {feature_name: array}
    external_y_labels : np.ndarray
        Binary labels for external dataset (0=healthy, 1=cancer)
    external_stratification_labels : np.ndarray
        Detailed labels for stratification (e.g., cancer types)
    external_sample_ids : np.ndarray
        Sample IDs for external dataset
    args : argparse.Namespace
        Command-line arguments (must include args.external_metadata_file)
    
    Returns
    -------
    dict
        Validation results with metrics
    """
    print("\n" + "="*90)
    print("EXTERNAL DATASET VALIDATION")
    print("="*90)
    
    # Step 1: Find the best model from training results
    print("\nStep 1: Analyzing training results to find best model...")
    
    # Apply filters if specified
    filter_desc = []
    if filter_feature:
        filter_desc.append(f"feature={filter_feature}")
    if filter_n_genes:
        filter_desc.append(f"n_genes={filter_n_genes}")
    
    if filter_desc:
        print(f"   Filtering models: {', '.join(filter_desc)}")
    
    best_results = analyze_best_models_and_get_best(
        best_model_results_dir, 
        verbose=True,
        filter_feature=filter_feature,
        filter_n_genes=filter_n_genes
    )
    
    best_info = best_results['best_model_info']
    best_pipeline = best_results['best_model'].best_estimator_
    
    print(f"\nBest model identified:")
    print(f"  Feature: {best_info['feature']}")
    print(f"  N Genes: {best_info['n_genes']}")
    print(f"  Classifier: {best_info['classifier']}")
    print(f"  Selection: {best_info['selection_type']}")
    print(f"  Training CV AUC: {best_info['cv_auc']:.4f}")
    print(f"  Validation Mode: {validation_mode.upper()}")
    
    if validation_mode == 'clinical':
        print(f"\n CLINICAL MODE: Will test without training StandardScaler statistics")
        print(f"   This simulates deployment where only gene selection + coefficients are available")
    
    # Step 2: Get external data for the same feature
    feature_name = best_info['feature']
    X_external = external_feature_arrays.get(feature_name)
    
    if X_external is None:
        raise ValueError(f"Feature '{feature_name}' not found in external dataset. Available: {list(external_feature_arrays.keys())}")
    
    print(f"\nStep 2: Loading external dataset...")
    print(f"  Feature: {feature_name}")
    print(f"  Shape: {X_external.shape}")
    print(f"  Samples: {len(external_sample_ids)} ({np.sum(external_y_labels == 0)} healthy, {np.sum(external_y_labels == 1)} cancer)")
    
    # Step 3: Validate feature alignment
    print(f"\nStep 3: Validating feature alignment...")
    
    # Check if pipeline has a selector step (absent for --no_gene_selection / --predefined_feature_matrix models)
    has_selector = 'selector' in best_pipeline.named_steps
    if has_selector:
        selector = best_pipeline.named_steps['selector']
        selected_indices = selector.indices_
        n_selected = len(selected_indices)
        max_index = selected_indices.max() if len(selected_indices) > 0 else 0
        
        print(f"  Model selected: {n_selected} features (indices: 0-{max_index})")
        print(f"  External data: {X_external.shape[1]} features available")
        
        if max_index >= X_external.shape[1]:
            print(f"\nCRITICAL ERROR: Feature misalignment!")
            print(f"   Model needs index {max_index}, external has {X_external.shape[1]} features")
            print(f"   Missing {max_index - X_external.shape[1] + 1} features")
            print(f"\n   SOLUTION: Ensure preprocessing_metadata.pkl exists in training directory")
            print(f"   and was correctly applied to external data (see output above).")
            raise ValueError(f"Feature misalignment: model needs index {max_index}, external has {X_external.shape[1]} features")
        
        print(f"  Alignment verified")
        
        # DEBUG: Show selector indices ordering
        print(f"\nDEBUG: TopGeneSelector indices (first 20):")
        print(f"   {selected_indices[:20].tolist() if len(selected_indices) >= 20 else selected_indices.tolist()}")
        print(f"   -> These are indices into the {X_external.shape[1]}-column external data")
        print(f"   -> Model coefficient[i] corresponds to feature at selected_indices[i]")
        print(f"   -> Are indices sorted? {np.all(selected_indices[:-1] <= selected_indices[1:])}")
        
        # Verify gene names from bed file
        try:
            bed_file = Path('beds/biomart.bed')
            if bed_file.exists():
                bed_data = pd.read_csv(bed_file, sep='\t', header=None, usecols=[3], names=['gene_name'])
                all_gene_names = bed_data['gene_name'].tolist()
                
                # Get gene names for selected indices (sample first 10 for display)
                selected_gene_names = [all_gene_names[i] for i in selected_indices[:10]]
                
                print(f"\n  Selected gene verification (first 10):")
                for i, (idx, gene) in enumerate(zip(selected_indices[:10], selected_gene_names)):
                    print(f"     [{i}] Index {idx}: {gene}")
                
                # Check if training and external would have same genes at these indices
                if training_preprocessing_metadata:
                    training_dataset = training_preprocessing_metadata.get('dataset', 'unknown')
                    print(f"  Gene names from bed file match training indices")
                    print(f"    (Training: {training_dataset}, External: {args.external_metadata_file})")
            else:
                print(f"\n   Bed file not found at {bed_file}, skipping gene name verification")
        except Exception as e:
            print(f"\n   Could not verify gene names: {e}")
    else:
        # No selector: pipeline uses all features directly (RNA/ATAC models)
        n_selected = X_external.shape[1]
        selected_indices = np.arange(n_selected)
        max_index = n_selected - 1
        print(f"  No gene selector in pipeline (all {n_selected} features used directly)")
        print(f"  External data: {X_external.shape[1]} features available")
        
        if n_selected != X_external.shape[1]:
            raise ValueError(f"Feature count mismatch: model expects {n_selected}, external has {X_external.shape[1]}")
        print(f"  Alignment verified")
    
    # Extract model weights (coefficients / importances / MLP first-layer norms)
    clf_step = best_pipeline.named_steps['clf']
    model_weights, model_weights_type = _extract_model_weights(best_pipeline)

    if model_weights is not None:
        print(f"  Extracted {len(model_weights)} {model_weights_type}")
    
    # Step 4: Apply the fitted model to external data
    print(f"\nStep 4: Applying model to external dataset...")
    
    start_time = time.time()
    
    try:
        if validation_mode == 'pipeline':
            # Standard mode: Apply selector (if present), SKIP pipeline's internal
            # StandardScaler (data is already column-standardized with training stats
            # from standardize_with_training_stats()), then apply classifier directly.
            print(f"  Bypassing pipeline's internal StandardScaler (data already standardized with training stats)")
            if has_selector:
                X_transformed = best_pipeline.named_steps['selector'].transform(X_external)
                print(f"  Applied gene selection: {X_external.shape[1]} -> {X_transformed.shape[1]} features")
            else:
                X_transformed = X_external
            clf_step_pred = best_pipeline.named_steps['clf']
            y_pred_proba = clf_step_pred.predict_proba(X_transformed)[:, 1]
            y_pred = clf_step_pred.predict(X_transformed)
        
        elif validation_mode == 'clinical':
            # Clinical mode: Only gene selection (if applicable), fresh StandardScaler on external data
            print(f"   Clinical mode: Applying gene selection only, then fresh StandardScaler")
            
            # Step 1: Apply gene selection using training indices (if selector exists)
            if has_selector:
                X_external_selected = X_external[:, selected_indices]
                print(f"     Selected {X_external_selected.shape[1]} features from external data")
            else:
                X_external_selected = X_external
                print(f"     No gene selector - using all {X_external_selected.shape[1]} features")
            
            # Step 2: Apply NEW StandardScaler fitted on external data
            scaler_clinical = StandardScaler()
            X_external_scaled = scaler_clinical.fit_transform(X_external_selected)
            print(f"     Applied fresh StandardScaler (mean/std from external data)")
            print(f"     External mean range: [{X_external_selected.mean(axis=0).min():.3f}, {X_external_selected.mean(axis=0).max():.3f}]")
            print(f"     External std range: [{X_external_selected.std(axis=0).min():.3f}, {X_external_selected.std(axis=0).max():.3f}]")
            
            # Step 3: Apply classifier directly
            clf_step = best_pipeline.named_steps['clf']
            y_pred_proba = clf_step.predict_proba(X_external_scaled)[:, 1]
            y_pred = clf_step.predict(X_external_scaled)
            print(f"     Applied classifier: {best_info['classifier']}")
        
        else:
            raise ValueError(f"Unknown validation_mode: {validation_mode}")
            
    except Exception as e:
        print(f"Error applying model: {e}")
        raise
    
    elapsed = time.time() - start_time
    print(f"  Predictions completed in {elapsed:.2f}s")
    
    # Store decision_function scores for SVM (more reliable than Platt-scaled probabilities)
    y_decision_function = None
    if hasattr(clf_step_pred if validation_mode == 'pipeline' else best_pipeline.named_steps['clf'], 'decision_function'):
        try:
            clf_for_df = clf_step_pred if validation_mode == 'pipeline' else best_pipeline.named_steps['clf']
            y_decision_function = clf_for_df.decision_function(X_transformed if validation_mode == 'pipeline' else X_external_scaled).astype(float)
            print(f"  Stored decision_function scores (range: {y_decision_function.min():.4f} to {y_decision_function.max():.4f})")
        except Exception as e:
            print(f"   Could not compute decision_function: {e}")
    
    # Probability degeneracy diagnostic
    proba_std = float(np.std(y_pred_proba))
    proba_range = float(np.ptp(y_pred_proba))
    proba_degenerate = proba_range < 0.01  # All probabilities within 1% of each other
    if proba_degenerate:
        print(f"   PROBABILITY DEGENERACY DETECTED: all predictions within {proba_range:.2e} of each other")
        print(f"     -> Platt scaling has saturated (all samples far from decision boundary)")
        print(f"     -> AUC is still valid (ranking-based) but probabilities are meaningless")
        print(f"     -> Use decision_function scores for threshold analysis instead")
        if y_decision_function is not None:
            df_range = float(np.ptp(y_decision_function))
            print(f"     -> Decision function range: {df_range:.4f} (vs probability range: {proba_range:.2e})")
    
    # Step 4: Calculate comprehensive metrics
    print(f"\nStep 5: Computing evaluation metrics...")
    
    # Use decision_function for ROC if probabilities are degenerate (more numerically stable)
    y_scores_for_roc = y_decision_function if (proba_degenerate and y_decision_function is not None) else y_pred_proba
    
    # Overall metrics
    fpr, tpr, thresholds = roc_curve(external_y_labels, y_scores_for_roc)
    auc_score = auc(fpr, tpr)
    precision_curve, recall_curve, _ = precision_recall_curve(external_y_labels, y_pred_proba)
    
    # Classification report at default threshold
    report = classification_report(external_y_labels, y_pred, output_dict=True, zero_division=0)
    conf_matrix = confusion_matrix(external_y_labels, y_pred).tolist()
    
    # Overall scalar classification metrics at default 0.5 threshold
    overall_accuracy = float(report.get('accuracy', 0))
    overall_precision_scalar = float(report.get('1', {}).get('precision', 0))
    overall_recall_scalar = float(report.get('1', {}).get('recall', 0))  # sensitivity
    overall_f1 = float(report.get('1', {}).get('f1-score', 0))
    overall_specificity = float(report.get('0', {}).get('recall', 0))
    
    # NOTE: For this external cohort the operating-point thresholds below are derived
    # from the external cohort's own ROC. This is a DESCRIPTIVE summary of the best
    # achievable operating point on the external set (e.g. "sensitivity attainable at
    # 95% specificity on this cohort"), NOT a pre-specified deployment threshold. It is
    # not the same as the nested-CV leakage (there the threshold was tuned on held-out
    # folds used for generalization estimates). The frozen deployment view - the
    # training cohort's OOF-derived thresholds applied unchanged - follows below
    # (deployment_thresholds / fixed_threshold_evaluation).
    # Operating point metrics: sensitivity @ 95% specificity (FPR = 0.05)
    target_fpr_95 = 0.05
    idx_95spec = np.argmin(np.abs(fpr - target_fpr_95))
    threshold_95spec = float(thresholds[idx_95spec]) if idx_95spec < len(thresholds) else 0.5
    overall_sens_at_95spec = float(tpr[idx_95spec])
    overall_spec_at_95spec = float(1.0 - fpr[idx_95spec])
    y_pred_95spec = (y_pred_proba >= threshold_95spec).astype(int)
    report_95spec = classification_report(external_y_labels, y_pred_95spec, output_dict=True, zero_division=0)
    conf_matrix_95spec = confusion_matrix(external_y_labels, y_pred_95spec).tolist()
    
    # Operating point metrics: sensitivity @ 90% specificity (FPR = 0.10)
    target_fpr_90 = 0.10
    idx_90spec = np.argmin(np.abs(fpr - target_fpr_90))
    threshold_90spec = float(thresholds[idx_90spec]) if idx_90spec < len(thresholds) else 0.5
    overall_sens_at_90spec = float(tpr[idx_90spec])
    overall_spec_at_90spec = float(1.0 - fpr[idx_90spec])
    y_pred_90spec = (y_pred_proba >= threshold_90spec).astype(int)
    report_90spec = classification_report(external_y_labels, y_pred_90spec, output_dict=True, zero_division=0)
    conf_matrix_90spec = confusion_matrix(external_y_labels, y_pred_90spec).tolist()
    
    # Operating point metrics: Youden's optimal threshold
    youden_index = tpr - fpr
    idx_youden = np.argmax(youden_index)
    threshold_youden = float(thresholds[idx_youden]) if idx_youden < len(thresholds) else 0.5
    overall_sens_at_youden = float(tpr[idx_youden])
    overall_spec_at_youden = float(1.0 - fpr[idx_youden])
    y_pred_youden = (y_pred_proba >= threshold_youden).astype(int)
    report_youden = classification_report(external_y_labels, y_pred_youden, output_dict=True, zero_division=0)
    conf_matrix_youden = confusion_matrix(external_y_labels, y_pred_youden).tolist()
    
    print(f"  Overall AUC: {auc_score:.4f}")
    print(f"  Sensitivity @95% specificity: {overall_sens_at_95spec:.4f} (threshold={threshold_95spec:.4f})")
    print(f"  Sensitivity @90% specificity: {overall_sens_at_90spec:.4f} (threshold={threshold_90spec:.4f})")
    print(f"  Youden optimal: sens={overall_sens_at_youden:.4f}, spec={overall_spec_at_youden:.4f} (threshold={threshold_youden:.4f})")

    # Deployment view: the SAME thresholds the training cohort froze (OOF-derived,
    # saved with the deployed model), applied unchanged to the external cohort.
    # Complements the descriptive operating points above, which are read off the
    # external ROC and therefore use its labels. Applied to predict_proba: that is
    # the scale the thresholds were derived on; when the external probabilities are
    # degenerate (saturated Platt scaling) the block is still reported, flagged.
    deployment_thresholds = _load_deployment_thresholds(best_results['best_model_path'])
    fixed_threshold_evaluation = {}
    if deployment_thresholds is None:
        print("  No deployment thresholds in the deployed model's summary (older best-model "
              "search); fixed-threshold evaluation skipped - rerun the best-model search.")
    else:
        fixed_threshold_evaluation = _fixed_threshold_evaluation(
            y_pred_proba, external_y_labels, external_stratification_labels, deployment_thresholds)
        print(f"\n  Frozen training-cohort thresholds applied to the external cohort"
              f"{' (WARNING: external probabilities degenerate)' if proba_degenerate else ''}:")
        for name, entry in fixed_threshold_evaluation.items():
            o = entry['overall']
            print(f"    {name:<24} thr={entry['threshold']:.4f}  "
                  f"sens={o['sensitivity']:.3f} [{o['sensitivity_ci95'][0]:.3f}-{o['sensitivity_ci95'][1]:.3f}] "
                  f"(TP={o['tp']}, FN={o['fn']})  "
                  f"spec={o['specificity']:.3f} [{o['specificity_ci95'][0]:.3f}-{o['specificity_ci95'][1]:.3f}] "
                  f"(TN={o['tn']}, FP={o['fp']})")
            for g, c in entry['per_control_group'].items():
                print(f"      {g:<22} spec={c['specificity']:.3f} "
                      f"[{c['specificity_ci95'][0]:.3f}-{c['specificity_ci95'][1]:.3f}] "
                      f"(TN={c['tn']}, FP={c['fp']})")

    # Misclassified samples
    external_sample_ids_array = np.array(external_sample_ids)
    misclassified_idx = np.where(external_y_labels != y_pred)[0]
    misclassified_ids = external_sample_ids_array[misclassified_idx].tolist()
    
    # Per-cancer-type evaluation (all healthy vs one cancer type), with descriptive
    # operating points read off each subset's ROC (see the leakage note above).
    healthy_labels = np.unique(external_stratification_labels[external_y_labels == 0])
    (auc_per_type, roc_per_type, pr_per_type, confusion_per_type,
     report_per_type, sample_classification_per_type) = \
        _per_cancer_type_descriptive_eval(
            y_pred_proba, external_y_labels, external_stratification_labels,
            external_sample_ids_array)

    # Detailed Control Group Analysis (Healthy, Cirrhosis, Hepatitis B breakdown)
    print(f"\nAnalyzing control group composition...")
    control_subgroups = {}
    control_subgroup_predictions = {}
    
    for control_label in healthy_labels:
        mask = external_stratification_labels == control_label
        if np.sum(mask) > 0:
            n_samples = np.sum(mask)
            proba_vals = y_pred_proba[mask]
            pred_vals = y_pred[mask]
            
            control_subgroups[control_label] = {
                'n_samples': int(n_samples),
                'mean_proba': float(np.mean(proba_vals)),
                'std_proba': float(np.std(proba_vals)),
                'median_proba': float(np.median(proba_vals)),
                'n_misclassified': int(np.sum(pred_vals == 1)),  # Classified as cancer
                'misclassification_rate': float(np.sum(pred_vals == 1) / n_samples),
                'sample_ids': external_sample_ids_array[mask].tolist(),
                'probabilities': proba_vals.tolist(),
                'predictions': pred_vals.tolist()
            }
            
            print(f"   {control_label}: {n_samples} samples, "
                  f"mean proba: {np.mean(proba_vals):.3f}, "
                  f"misclassified: {np.sum(pred_vals == 1)}/{n_samples} ({np.sum(pred_vals == 1)/n_samples*100:.1f}%)")
    
    # Create confusion matrix per control subgroup + cancer
    print(f"\nCreating detailed confusion matrix (control subgroups + cancer)...")
    
    # Build a multi-class confusion matrix
    # Classes: Healthy, Cirrhosis, Hepatitis B, Cancer (all cancer types combined)
    all_groups = list(healthy_labels) + ['Cancer']
    n_groups = len(all_groups)
    detailed_confusion = np.zeros((n_groups, 2), dtype=int)  # Rows=true groups, Cols=[predicted control, predicted cancer]
    
    for i, group in enumerate(all_groups):
        if group == 'Cancer':
            mask = external_y_labels == 1
        else:
            mask = external_stratification_labels == group
        
        if np.sum(mask) > 0:
            preds = y_pred[mask]
            detailed_confusion[i, 0] = np.sum(preds == 0)  # Predicted as control
            detailed_confusion[i, 1] = np.sum(preds == 1)  # Predicted as cancer
    
    # Step 5: Print and save results
    print(f"\n{'='*90}")
    print("EXTERNAL VALIDATION RESULTS")
    print(f"{'='*90}")
    print(f"\nOverall Performance:")
    print(f"  AUC: {auc_score:.4f}")
    print(f"  Accuracy: {report['accuracy']:.4f}")
    print(f"  Sensitivity (Recall): {report['1']['recall']:.4f}")
    print(f"  Specificity: {report['0']['recall']:.4f}")
    print(f"  Misclassified: {len(misclassified_ids)}/{len(external_sample_ids)} ({len(misclassified_ids)/len(external_sample_ids)*100:.1f}%)")
    
    print(f"\nPer-Cancer-Type AUC:")
    for cancer_type in sorted(auc_per_type.keys()):
        print(f"  {cancer_type}: {auc_per_type[cancer_type]:.4f}")
    
    print(f"\n{'='*90}")
    print("CONTROL SUBGROUP ANALYSIS")
    print(f"{'='*90}")
    print(f"Breakdown of Class 0 (Control) performance:")
    print(f"")
    for group in all_groups[:-1]:  # Exclude 'Cancer'
        idx = all_groups.index(group)
        total = detailed_confusion[idx, :].sum()
        correct = detailed_confusion[idx, 0]
        incorrect = detailed_confusion[idx, 1]
        if total > 0:
            print(f"  {group}:")
            print(f"    Total samples: {total}")
            print(f"    Correctly classified as control: {correct} ({correct/total*100:.1f}%)")
            print(f"    Misclassified as cancer: {incorrect} ({incorrect/total*100:.1f}%)")
            if group in control_subgroups:
                print(f"    Mean prediction probability: {control_subgroups[group]['mean_proba']:.3f}")
    
    # Check consistency across control subgroups
    control_miscl_rates = [
        (detailed_confusion[i, 1] / detailed_confusion[i, :].sum() * 100) 
        if detailed_confusion[i, :].sum() > 0 else 0
        for i in range(len(healthy_labels))
    ]
    
    if len(control_miscl_rates) > 1:
        rate_range = max(control_miscl_rates) - min(control_miscl_rates)
        rate_mean = np.mean(control_miscl_rates)
        rate_std = np.std(control_miscl_rates)
        print(f"\n  Control Subgroup Consistency:")
        print(f"    Mean misclassification rate: {rate_mean:.1f}%")
        print(f"    Std deviation: {rate_std:.1f}%")
        print(f"    Range: {rate_range:.1f}%")
        if rate_range < 10:
            print(f"    Subgroups show similar behavior (range < 10%)")
            print(f"      -> Including all in class 0 appears reasonable")
        elif rate_range < 20:
            print(f"    Moderate variation in subgroups (range 10-20%)")
            print(f"      -> Review detailed plot for specific differences")
        else:
            print(f"    High variation in subgroups (range > 20%)")
            print(f"      -> One or more subgroups may behave differently")
            print(f"      -> Consider separate analysis or exclusion")
    
    print(f"{'='*90}")
    
    # Resolve the output directory
    if args.output_dir:
        output_dir = Path(args.output_dir) / f"external_validation_{args.experiment_name}"
    else:
        output_dir = Path(best_model_results_dir).parent / f"external_validation_{args.experiment_name}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Create weight visualization if available
    weight_plot_path = None
    if model_weights is not None:
        print(f"\nCreating weight visualization...")
        try:
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f'Model Weights Visualization\n{best_info["classifier"]} | {best_info["feature"]} | {n_selected} genes', 
                        fontsize=14, fontweight='bold')
            
            # Plot 1: Weight distribution
            ax = axes[0, 0]
            ax.hist(model_weights, bins=50, edgecolor='black', alpha=0.7)
            ax.set_xlabel('Weight Value')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Weight Distribution ({model_weights_type})')
            ax.axvline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
            ax.grid(True, alpha=0.3)
            
            # Plot 2: Top 20 weights by absolute value
            ax = axes[0, 1]
            top_20_idx = np.argsort(np.abs(model_weights))[-20:][::-1]
            top_20_weights = model_weights[top_20_idx]
            top_20_gene_indices = selected_indices[top_20_idx]
            colors = ['red' if w < 0 else 'blue' for w in top_20_weights]
            bars = ax.barh(range(20), top_20_weights, color=colors, alpha=0.7)
            ax.set_yticks(range(20))
            ax.set_yticklabels([f'Gene {idx}' for idx in top_20_gene_indices], fontsize=8)
            ax.set_xlabel('Weight Value')
            ax.set_title('Top 20 Features by Absolute Weight')
            ax.axvline(0, color='black', linestyle='-', linewidth=1)
            ax.grid(True, alpha=0.3, axis='x')
            ax.invert_yaxis()
            
            # Plot 3: Weights vs feature index (sorted by original index)
            ax = axes[1, 0]
            sort_idx = np.argsort(selected_indices)
            sorted_indices = selected_indices[sort_idx]
            sorted_weights = model_weights[sort_idx]
            ax.scatter(sorted_indices, sorted_weights, alpha=0.6, s=20)
            ax.set_xlabel('Feature Index in Original Array (sorted)')
            ax.set_ylabel('Weight Value')
            ax.set_title('Weight vs Feature Index')
            ax.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
            ax.grid(True, alpha=0.3)
            
            # Plot 4: Cumulative weight contribution
            ax = axes[1, 1]
            sorted_abs_weights = np.sort(np.abs(model_weights))[::-1]
            cumsum = np.cumsum(sorted_abs_weights) / np.sum(np.abs(model_weights))
            ax.plot(range(len(cumsum)), cumsum, linewidth=2)
            ax.set_xlabel('Number of Top Features')
            ax.set_ylabel('Cumulative Absolute Weight Proportion')
            ax.set_title('Cumulative Weight Contribution')
            ax.grid(True, alpha=0.3)
            ax.axhline(0.8, color='red', linestyle='--', alpha=0.5, label='80% threshold')
            ax.axhline(0.9, color='orange', linestyle='--', alpha=0.5, label='90% threshold')
            ax.legend()
            
            # Find how many features contribute to 80% and 90%
            n_80 = np.argmax(cumsum >= 0.8) + 1
            n_90 = np.argmax(cumsum >= 0.9) + 1
            ax.text(0.05, 0.95, f'80% weight: {n_80}/{n_selected} features\n90% weight: {n_90}/{n_selected} features',
                   transform=ax.transAxes, verticalalignment='top', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            weight_plot_path = output_dir / f"weight_visualization_{feature_name}.png"
            plt.savefig(weight_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Weight visualization saved to: {weight_plot_path}")
            
        except Exception as e:
            print(f"  Could not create weight visualization: {e}")
            weight_plot_path = None
    
    # Create control subgroup confusion matrix visualization
    control_confusion_plot_path = None
    if len(control_subgroups) > 0:
        print(f"\nCreating control subgroup confusion matrix...")
        try:
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Display the confusion matrix as a heatmap
            im = ax.imshow(detailed_confusion, cmap='Blues', aspect='auto')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Number of Samples', rotation=270, labelpad=20, fontsize=11)
            
            # Set ticks and labels
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Predicted Control (0)', 'Predicted Cancer (1)'], fontsize=11)
            ax.set_yticks(range(n_groups))
            ax.set_yticklabels(all_groups, fontsize=11)
            
            # Add text annotations
            for i in range(n_groups):
                for j in range(2):
                    value = detailed_confusion[i, j]
                    total = detailed_confusion[i, :].sum()
                    percentage = (value / total * 100) if total > 0 else 0
                    text_color = 'white' if value > detailed_confusion.max() * 0.5 else 'black'
                    ax.text(j, i, f'{value}\n({percentage:.1f}%)', 
                           ha='center', va='center', color=text_color, fontsize=10, fontweight='bold')
            
            # Labels and title
            ax.set_xlabel('Predicted Class', fontsize=12, fontweight='bold')
            ax.set_ylabel('True Class (Control Subgroups + Cancer)', fontsize=12, fontweight='bold')
            ax.set_title(f'Detailed Confusion Matrix: Control Subgroups Analysis\n'
                        f'{best_info["classifier"]} | {best_info["feature"]} | {n_selected} genes',
                        fontsize=13, fontweight='bold', pad=15)
            
            # Add text box with summary statistics
            control_total = detailed_confusion[:len(healthy_labels), :].sum()
            control_correct = detailed_confusion[:len(healthy_labels), 0].sum()
            control_specificity = (control_correct / control_total * 100) if control_total > 0 else 0
            
            cancer_total = detailed_confusion[-1, :].sum()
            cancer_correct = detailed_confusion[-1, 1].sum()
            cancer_sensitivity = (cancer_correct / cancer_total * 100) if cancer_total > 0 else 0
            
            summary_text = (
                f'Overall Specificity (All Controls): {control_specificity:.1f}%\n'
                f'Overall Sensitivity (Cancer): {cancer_sensitivity:.1f}%\n\n'
                f'Control subgroup consistency:\n'
            )
            
            # Check if control subgroups have similar misclassification rates
            control_miscl_rates = [
                (detailed_confusion[i, 1] / detailed_confusion[i, :].sum() * 100) 
                if detailed_confusion[i, :].sum() > 0 else 0
                for i in range(len(healthy_labels))
            ]
            
            if len(control_miscl_rates) > 1:
                rate_std = np.std(control_miscl_rates)
                rate_range = max(control_miscl_rates) - min(control_miscl_rates)
                if rate_range < 10:
                    summary_text += f'Similar (range: {rate_range:.1f}%)'
                else:
                    summary_text += f'Variable (range: {rate_range:.1f}%)'
            
            ax.text(1.35, 0.5, summary_text,
                   transform=ax.transAxes, verticalalignment='center', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, edgecolor='gray'))
            
            plt.tight_layout()
            control_confusion_plot_path = output_dir / f"control_subgroup_confusion_matrix_{feature_name}.png"
            plt.savefig(control_confusion_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Control subgroup confusion matrix saved to: {control_confusion_plot_path}")
            
        except Exception as e:
            print(f"  Could not create control subgroup confusion matrix: {e}")
            import traceback
            traceback.print_exc()
            control_confusion_plot_path = None
    
    # Create diagnostic plots comparing training vs external dataset
    diagnostic_plots = None
    if training_preprocessing_metadata:
        print(f"\nCreating diagnostic plots (training vs external comparison)...")
        try:
            # Load training dataset to compare statistics
            training_metadata_file = training_preprocessing_metadata.get('dataset', '')
            
            if training_metadata_file and Path(training_metadata_file).exists():
                print(f"  Loading training data from: {training_metadata_file}")
                
                # Load training metadata to get sample lists
                training_meta = pd.read_csv(training_metadata_file)
                
                # Filter to same cancer types as used in training (recreate training set)
                # For simplicity, we'll load all available data from training
                training_healthy = training_meta[training_meta['disease'] == 'Healthy']['ID'].tolist()
                training_cancer = training_meta[training_meta['disease'] != 'Healthy']['ID'].tolist()
                
                # Load the same feature from training dataset
                # For predefined matrices (ATAC), load the .npy file directly
                # For CSV features (DD/RNA), use load_feature_in_parallel
                using_predefined = hasattr(args, 'predefined_feature_matrix') and args.predefined_feature_matrix
                
                if using_predefined:
                    # ATAC: Load training predefined matrix
                    print(f"  Loading training predefined matrix for '{feature_name}'...")
                    ext_path = Path(args.predefined_feature_matrix)
                    training_matrix_path = None
                    for suffix in ['_jiang', '_lucas', '_mathios']:
                        if suffix in ext_path.stem:
                            for replacement in ['_pancan', '_brca_crc', '']:
                                cand = ext_path.parent / (ext_path.stem.replace(suffix, replacement) + ext_path.suffix)
                                if cand.exists() and cand != ext_path:
                                    training_matrix_path = cand
                                    break
                        if training_matrix_path:
                            break
                    
                    training_healthy_data = None
                    training_cancer_data = None
                    if training_matrix_path is not None:
                        print(f"  Loading: {training_matrix_path}")
                        full_mat = np.load(training_matrix_path)
                        # Load metadata to split healthy/cancer
                        meta_path = training_matrix_path.with_suffix('.metadata.csv')
                        if meta_path.exists():
                            mat_meta = pd.read_csv(meta_path)
                            healthy_mask = mat_meta['disease'] == 'Healthy'
                            training_healthy_data = full_mat[healthy_mask.values]
                            training_cancer_data = full_mat[~healthy_mask.values]
                            print(f"  Training matrix: {full_mat.shape[0]} samples ({training_healthy_data.shape[0]} healthy, {training_cancer_data.shape[0]} cancer)")
                        else:
                            print(f"  No metadata CSV found for training matrix")
                    else:
                        print(f"  Could not find training predefined matrix")
                else:
                    # CSV features (DD/RNA): load from features_dir
                    features_dir = args.features_dir if hasattr(args, 'features_dir') else './extracted_features/biomart_10kb'
                    print(f"  Loading training feature '{feature_name}' from {features_dir}...")
                    
                    training_healthy_data = load_feature_in_parallel(
                        features_dir, 
                        feature_name, 
                        file_filter=training_healthy,
                        num_cores=args.cores
                    )
                    
                    training_cancer_data = load_feature_in_parallel(
                        features_dir,
                        feature_name,
                        file_filter=training_cancer,
                        num_cores=args.cores
                    )
                
                if training_healthy_data is not None and training_cancer_data is not None:
                    # Apply same preprocessing as external data
                    training_kept_cols = training_preprocessing_metadata.get('kept_columns_per_feature', {})
                    if feature_name in training_kept_cols:
                        kept_indices = np.array(training_kept_cols[feature_name])
                        training_healthy_data = training_healthy_data[:, kept_indices]
                        training_cancer_data = training_cancer_data[:, kept_indices]
                    
                    # Combine and normalize
                    training_combined = np.vstack([training_healthy_data, training_cancer_data])
                    
                    # Row-wise normalize
                    row_means_train = np.nanmean(training_combined, axis=1, keepdims=True)
                    row_stds_train = np.nanstd(training_combined, axis=1, keepdims=True)
                    row_stds_train[row_stds_train == 0] = 1
                    training_combined = np.nan_to_num((training_combined - row_means_train) / row_stds_train)
                    
                    # Calculate feature-level statistics
                    training_feature_means = np.mean(training_combined, axis=0)
                    training_feature_stds = np.std(training_combined, axis=0)
                    
                    # the external array carries the covariate block (--augment_covariates) after
                    # the TSS columns; the diagnostic compares the TSS columns only
                    X_external_tss = X_external[:, :training_combined.shape[1]] if COVARIATE_BLOCK['n'] else X_external
                    external_feature_means = np.mean(X_external_tss, axis=0)
                    external_feature_stds = np.std(X_external_tss, axis=0)

                    # Calculate Spearman correlations
                    
                    # Correlation for means
                    corr_means, pval_means = spearmanr(training_feature_means, external_feature_means)
                    
                    # Correlation for stds
                    corr_stds, pval_stds = spearmanr(training_feature_stds, external_feature_stds)
                    
                    print(f"  Spearman correlation (means): {corr_means:.4f} (p={pval_means:.2e})")
                    print(f"  Spearman correlation (stds):  {corr_stds:.4f} (p={pval_stds:.2e})")
                    
                    # Create diagnostic plots
                    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
                    fig.suptitle(f'Training vs External Dataset Comparison\n{feature_name} | Training: {len(training_combined)} samples, External: {len(X_external)} samples',
                                fontsize=14, fontweight='bold')
                    
                    # Plot 1: Mean comparison scatter
                    ax = axes[0, 0]
                    ax.scatter(training_feature_means, external_feature_means, alpha=0.3, s=5)
                    ax.set_xlabel('Training Feature Means')
                    ax.set_ylabel('External Feature Means')
                    ax.set_title(f'Feature Means Comparison\nSpearman r={corr_means:.4f}, p={pval_means:.2e}')
                    ax.plot([training_feature_means.min(), training_feature_means.max()],
                           [training_feature_means.min(), training_feature_means.max()],
                           'r--', alpha=0.5, label='y=x')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    # Plot 2: Std comparison scatter
                    ax = axes[0, 1]
                    ax.scatter(training_feature_stds, external_feature_stds, alpha=0.3, s=5)
                    ax.set_xlabel('Training Feature Stds')
                    ax.set_ylabel('External Feature Stds')
                    ax.set_title(f'Feature Stds Comparison\nSpearman r={corr_stds:.4f}, p={pval_stds:.2e}')
                    ax.plot([training_feature_stds.min(), training_feature_stds.max()],
                           [training_feature_stds.min(), training_feature_stds.max()],
                           'r--', alpha=0.5, label='y=x')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    # Plot 3: Distribution of means
                    ax = axes[0, 2]
                    ax.hist(training_feature_means, bins=50, alpha=0.5, label='Training', density=True)
                    ax.hist(external_feature_means, bins=50, alpha=0.5, label='External', density=True)
                    ax.set_xlabel('Feature Mean Value')
                    ax.set_ylabel('Density')
                    ax.set_title('Distribution of Feature Means')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    # Plot 4: Distribution of stds
                    ax = axes[1, 0]
                    ax.hist(training_feature_stds, bins=50, alpha=0.5, label='Training', density=True)
                    ax.hist(external_feature_stds, bins=50, alpha=0.5, label='External', density=True)
                    ax.set_xlabel('Feature Std Value')
                    ax.set_ylabel('Density')
                    ax.set_title('Distribution of Feature Stds')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    # Plot 5: Mean difference per feature (sorted)
                    ax = axes[1, 1]
                    mean_diff = external_feature_means - training_feature_means
                    sorted_diff = np.sort(mean_diff)
                    ax.plot(sorted_diff)
                    ax.axhline(0, color='red', linestyle='--', alpha=0.5)
                    ax.set_xlabel('Feature Index (sorted by difference)')
                    ax.set_ylabel('External Mean - Training Mean')
                    ax.set_title(f'Mean Differences (External - Training)\nMedian: {np.median(mean_diff):.4f}')
                    ax.grid(True, alpha=0.3)
                    
                    # Plot 6: Prediction score distributions
                    ax = axes[1, 2]
                    healthy_scores = y_pred_proba[external_y_labels == 0]
                    cancer_scores = y_pred_proba[external_y_labels == 1]
                    ax.hist(healthy_scores, bins=30, alpha=0.5, label=f'Healthy (n={len(healthy_scores)})', density=True)
                    ax.hist(cancer_scores, bins=30, alpha=0.5, label=f'Cancer (n={len(cancer_scores)})', density=True)
                    ax.axvline(0.5, color='red', linestyle='--', label='Threshold=0.5')
                    ax.set_xlabel('Prediction Score')
                    ax.set_ylabel('Density')
                    ax.set_title(f'Prediction Score Distribution\nAUC={auc_score:.4f}')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    diagnostic_plots = output_dir / f"diagnostic_training_vs_external_{feature_name}.png"
                    plt.savefig(diagnostic_plots, dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"  Diagnostic plots saved to: {diagnostic_plots}")
                    
                    # Additional summary statistics
                    print(f"\n  Summary Statistics:")
                    print(f"     Training means: min={training_feature_means.min():.4f}, max={training_feature_means.max():.4f}, median={np.median(training_feature_means):.4f}")
                    print(f"     External means: min={external_feature_means.min():.4f}, max={external_feature_means.max():.4f}, median={np.median(external_feature_means):.4f}")
                    print(f"     Training stds:  min={training_feature_stds.min():.4f}, max={training_feature_stds.max():.4f}, median={np.median(training_feature_stds):.4f}")
                    print(f"     External stds:  min={external_feature_stds.min():.4f}, max={external_feature_stds.max():.4f}, median={np.median(external_feature_stds):.4f}")
                    print(f"     Mean difference range: [{mean_diff.min():.4f}, {mean_diff.max():.4f}]")
                    print(f"     Prediction scores - Healthy: median={np.median(healthy_scores):.4f}, mean={np.mean(healthy_scores):.4f}")
                    print(f"     Prediction scores - Cancer:  median={np.median(cancer_scores):.4f}, mean={np.mean(cancer_scores):.4f}")
                else:
                    print(f"  Could not load training data for comparison")
            else:
                print(f"  Training metadata file not found: {training_metadata_file}")
                
        except Exception as e:
            print(f"  Could not create diagnostic plots: {e}")
            import traceback
            traceback.print_exc()
    
    results = {
        'validation_mode': validation_mode,
        'training_dataset': best_model_results_dir,
        'external_dataset_name': args.experiment_name,
        'best_model_info': best_info,
        'best_model_path': best_results['best_model_path'],
        
        # Feature alignment information
        'selected_feature_indices': selected_indices.tolist(),
        'n_features_selected': n_selected,
        'external_n_features': X_external.shape[1],
        'feature_alignment_validated': True,
        'training_n_features': max_index + 1,  # Minimum features needed from training
        'alignment_note': 'Selected indices refer to positions AFTER blacklist/duplicate removal on training data',
        
        # Model weights
        'model_weights': model_weights.tolist() if model_weights is not None else None,
        'model_weights_type': model_weights_type,
        'weight_plot_path': str(weight_plot_path) if weight_plot_path else None,
        'diagnostic_plot_path': str(diagnostic_plots) if diagnostic_plots else None,
        
        # Control subgroup analysis
        'control_subgroups': control_subgroups,
        'control_group_labels': list(healthy_labels),
        'detailed_confusion_matrix': detailed_confusion.tolist(),
        'detailed_confusion_matrix_labels': all_groups,
        'control_confusion_plot_path': str(control_confusion_plot_path) if control_confusion_plot_path else None,
        
        # External data info
        'external_data_shape': X_external.shape,
        'external_n_samples': len(external_sample_ids),
        'external_n_healthy': int(np.sum(external_y_labels == 0)),
        'external_n_cancer': int(np.sum(external_y_labels == 1)),
        
        # Performance metrics
        'overall_auc': float(auc_score),
        'overall_fpr': fpr.tolist(),
        'overall_tpr': tpr.tolist(),
        'overall_precision_curve': precision_curve.tolist(),
        'overall_recall_curve': recall_curve.tolist(),
        'classification_report': report,
        'confusion_matrix': conf_matrix,
        'misclassified_sample_ids': misclassified_ids,
        'auc_per_cancer_type': auc_per_type,
        'roc_per_cancer_type': roc_per_type,
        'pr_per_cancer_type': pr_per_type,
        'confusion_per_cancer_type': confusion_per_type,
        'report_per_cancer_type': report_per_type,
        'sample_classification_per_cancer_type': sample_classification_per_type,
        'execution_time_seconds': elapsed,
        
        # Scalar classification metrics (default 0.5 threshold)
        'overall_accuracy': overall_accuracy,
        'overall_precision': overall_precision_scalar,
        'overall_recall': overall_recall_scalar,  # = sensitivity at 0.5
        'overall_f1': overall_f1,
        'overall_specificity': overall_specificity,
        
        # Operating point: 95% specificity
        'overall_sens_at_95spec': overall_sens_at_95spec,
        'overall_spec_at_95spec': overall_spec_at_95spec,
        'threshold_95spec': threshold_95spec,
        'classification_report_95spec': report_95spec,
        'confusion_matrix_95spec': conf_matrix_95spec,
        
        # Operating point: 90% specificity
        'overall_sens_at_90spec': overall_sens_at_90spec,
        'overall_spec_at_90spec': overall_spec_at_90spec,
        'threshold_90spec': threshold_90spec,
        'classification_report_90spec': report_90spec,
        'confusion_matrix_90spec': conf_matrix_90spec,
        
        # Operating point: Youden's optimal
        'overall_sens_at_youden': overall_sens_at_youden,
        'overall_spec_at_youden': overall_spec_at_youden,
        'threshold_youden': threshold_youden,
        'classification_report_youden': report_youden,
        'confusion_matrix_youden': conf_matrix_youden,

        # Frozen training-cohort thresholds applied unchanged (deployment view);
        # None / {} when the deployed model predates deployment_thresholds.
        'deployment_thresholds': deployment_thresholds,
        'fixed_threshold_evaluation': fixed_threshold_evaluation,

        # Prediction scores for downstream analysis
        'y_pred_proba': y_pred_proba.tolist(),
        'y_decision_function': y_decision_function.tolist() if y_decision_function is not None else None,
        'y_scores_for_roc': y_scores_for_roc.tolist(),  # decision_function if proba degenerate, else proba
        'proba_degenerate': proba_degenerate,
        'proba_range': proba_range,
        'proba_std': proba_std,
        'y_true': external_y_labels.tolist(),
        'sample_ids': list(external_sample_ids),
        'stratification_labels': list(external_stratification_labels),
    }
    
    output_file = output_dir / f"external_validation_{feature_name}_results.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nResults saved to: {output_file}")
    print(f"{'='*90}\n")
    
    return results


def run_comprehensive_external_validation(
    best_model_results_dir: str,
    external_feature_arrays: dict,
    external_y_labels: np.ndarray,
    external_stratification_labels: np.ndarray,
    external_sample_ids: np.ndarray,
    args: argparse.Namespace,
    training_preprocessing_metadata: dict = None,
    filter_feature: str = None
):
    """
    Comprehensive external validation analyzing generalization across:
    1. Feature types (best model per feature)
    2. Number of genes (best model per top_genes value)
    
    Creates visualizations comparing generalization performance to help identify:
    - Which features generalize better to external datasets
    - Whether models with fewer genes generalize better or worse
    
    Args:
        Same as validate_best_model_on_external_dataset
    
    Returns:
        dict: Results containing per-feature and per-gene-count analysis with plots
    """
    print("\n" + "="*90)
    print("COMPREHENSIVE EXTERNAL VALIDATION ANALYSIS")
    print("="*90)
    print("\nThis will evaluate:")
    print("  1. Best model for each feature type (to identify features that generalize better)")
    print("  2. Best model for each gene count (to assess feature count vs. generalization)")
    
    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir) / f"external_validation_{args.experiment_name}"
    else:
        output_dir = Path(best_model_results_dir).parent / f"external_validation_{args.experiment_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get available features and gene counts from models
    print(f"\nScanning available models in {best_model_results_dir}...")
    results_dir = Path(best_model_results_dir)
    gridsearch_files = list(results_dir.glob("best_model_GridSearchCV_*.pkl"))
    
    if len(gridsearch_files) == 0:
        raise FileNotFoundError(f"No model files found in {results_dir}")
    
    # Extract available features and gene counts
    available_features = set()
    available_gene_counts = set()
    
    for pkl_file in gridsearch_files:
        try:
            with open(pkl_file, 'rb') as f:
                gs = pickle.load(f)
            
            # Extract feature name
            filename = pkl_file.stem.replace('best_model_GridSearchCV_', '')
            parts = filename.split('_')
            
            # Find where gene count starts (first numeric part)
            for i, part in enumerate(parts):
                if part.isdigit():
                    feature_name = '_'.join(parts[:i])
                    n_genes = int(part)
                    available_features.add(feature_name)
                    available_gene_counts.add(n_genes)
                    break
        except Exception as e:
            print(f"  Could not parse {pkl_file.name}: {e}")
            continue
    
    print(f"   Found {len(available_features)} features: {sorted(available_features)}")
    print(f"   Found {len(available_gene_counts)} gene counts: {sorted(available_gene_counts)}")
    
    # Apply feature filter if specified
    if filter_feature:
        if filter_feature in available_features:
            available_features = {filter_feature}
            print(f"\n   Filtered to feature: {filter_feature}")
        else:
            print(f"\n    Requested filter_feature '{filter_feature}' not found in available features: {sorted(available_features)}")
            print(f"   Proceeding with all available features")
    
    # Part 1: Validate best model per feature
    print("\n" + "="*90)
    print("PART 1: GENERALIZATION BY FEATURE TYPE")
    print("="*90)
    
    feature_results = {}
    for feature in sorted(available_features):
        print(f"\n{'-'*90}")
        print(f"Evaluating best {feature} model...")
        print(f"{'-'*90}")
        
        try:
            result = validate_best_model_on_external_dataset(
                best_model_results_dir=best_model_results_dir,
                external_feature_arrays=external_feature_arrays,
                external_y_labels=external_y_labels,
                external_stratification_labels=external_stratification_labels,
                external_sample_ids=external_sample_ids,
                args=args,
                training_preprocessing_metadata=training_preprocessing_metadata,
                filter_feature=feature,
                filter_n_genes=None,
                validation_mode=getattr(args, 'external_validation_mode', 'pipeline')
            )
            feature_results[feature] = result
            print(f"{feature}: AUC = {result['overall_auc']:.4f}")
        except Exception as e:
            print(f"Failed to validate {feature}: {e}")
            feature_results[feature] = None
    
    # Identify best feature from Part 1 results
    best_feature = None
    best_feature_auc = 0
    for feat, res in feature_results.items():
        if res is not None and res['overall_auc'] > best_feature_auc:
            best_feature = feat
            best_feature_auc = res['overall_auc']
    
    if best_feature is None:
        print("No valid feature results, cannot proceed with gene count analysis")
        best_feature = sorted(available_features)[0]  # Fallback to first feature
        print(f"Using fallback feature: {best_feature}")
    
    # Part 2: Validate best model at each gene count (unbiased by feature)
    print("\n" + "="*90)
    print("PART 2: GENERALIZATION BY NUMBER OF GENES")
    print("="*90)
    print("Note: This finds the BEST MODEL at each gene count across ALL features,")
    print("      showing model complexity vs. generalization without feature bias.")
    
    gene_count_results = {}
    for n_genes in sorted(available_gene_counts):
        print(f"\n{'-'*90}")
        print(f"Finding best model with {n_genes} genes across all features...")
        print(f"{'-'*90}")
        
        try:
            result = validate_best_model_on_external_dataset(
                best_model_results_dir=best_model_results_dir,
                external_feature_arrays=external_feature_arrays,
                external_y_labels=external_y_labels,
                external_stratification_labels=external_stratification_labels,
                external_sample_ids=external_sample_ids,
                args=args,
                training_preprocessing_metadata=training_preprocessing_metadata,
                filter_feature=None,  # No feature filter - find best across all
                filter_n_genes=n_genes,
                validation_mode=getattr(args, 'external_validation_mode', 'pipeline')
            )
            gene_count_results[n_genes] = result
            best_feat_this_count = result['best_model_info']['feature']
            print(f"{n_genes} genes: AUC = {result['overall_auc']:.4f} (best feature: {best_feat_this_count})")
        except Exception as e:
            print(f"Failed to validate {n_genes} genes: {e}")
            gene_count_results[n_genes] = None
    
    # Part 3: Create comprehensive visualizations
    print("\n" + "="*90)
    print("PART 3: GENERATING VISUALIZATIONS")
    print("="*90)
    
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(20, 15))
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # --- Plot 1: AUC by Feature Type (Bar Chart) ---
    ax1 = fig.add_subplot(gs[0, :2])
    
    feature_names = []
    feature_aucs = []
    feature_sens = []
    feature_spec = []
    feature_auc_std = []
    feature_sens_std = []
    feature_spec_std = []
    
    for feat in sorted(feature_results.keys()):
        if feature_results[feat] is not None:
            feature_names.append(feat)
            feature_aucs.append(feature_results[feat]['overall_auc'])
            report = feature_results[feat]['classification_report']
            feature_sens.append(report['1']['recall'])
            feature_spec.append(report['0']['recall'])
    
    x_pos = np.arange(len(feature_names))
    ax1.bar(x_pos, feature_aucs, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.set_xlabel('Feature Type', fontsize=12, fontweight='bold')
    ax1.set_ylabel('AUC', fontsize=12, fontweight='bold')
    ax1.set_title('Generalization Performance by Feature Type', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(feature_names, rotation=45, ha='right')
    ax1.set_ylim([0, 1])
    ax1.grid(axis='y', alpha=0.3)
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random')
    
    # Add value labels on bars
    for i, (name, auc) in enumerate(zip(feature_names, feature_aucs)):
        ax1.text(i, auc + 0.02, f'{auc:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax1.legend()
    
    # --- Plot 2: Sensitivity & Specificity by Feature ---
    ax2 = fig.add_subplot(gs[0, 2])
    
    x_pos = np.arange(len(feature_names))
    width = 0.35
    
    ax2.bar(x_pos - width/2, feature_sens, width, label='Sensitivity', alpha=0.7, color='green', edgecolor='black')
    ax2.bar(x_pos + width/2, feature_spec, width, label='Specificity', alpha=0.7, color='orange', edgecolor='black')
    ax2.set_xlabel('Feature Type', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax2.set_title('Sensitivity & Specificity\nby Feature', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=9)
    ax2.set_ylim([0, 1])
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # --- Plot 3: AUC by Number of Genes (Line Plot) ---
    ax3 = fig.add_subplot(gs[1, :2])
    
    gene_counts = []
    gene_aucs = []
    gene_features = []  # Track which feature was best for each gene count
    
    for n_genes in sorted(gene_count_results.keys()):
        if gene_count_results[n_genes] is not None:
            gene_counts.append(n_genes)
            gene_aucs.append(gene_count_results[n_genes]['overall_auc'])
            gene_features.append(gene_count_results[n_genes]['best_model_info']['feature'])
    
    ax3.plot(gene_counts, gene_aucs, marker='o', linewidth=2, markersize=8, color='steelblue', label='Best Model AUC')
    ax3.set_xlabel('Number of Genes', fontsize=12, fontweight='bold')
    ax3.set_ylabel('AUC', fontsize=12, fontweight='bold')
    ax3.set_title('Generalization Performance by Number of Genes\n(Best Model at Each Gene Count)', fontsize=14, fontweight='bold')
    if len(gene_counts) > 1:
        ax3.set_xscale('log')
    ax3.set_ylim([0.5, 1])
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random')
    ax3.legend()
    
    # Add feature labels at each point to show which feature was best
    for i, (gc, auc, feat) in enumerate(zip(gene_counts, gene_aucs, gene_features)):
        ax3.annotate(f'{feat}\n{auc:.3f}', (gc, auc), textcoords="offset points", xytext=(0,10), 
                    ha='center', fontsize=7, alpha=0.7)
    
    # --- Plot 4: Sensitivity & Specificity by Gene Count ---
    ax4 = fig.add_subplot(gs[1, 2])
    
    gene_sens = []
    gene_spec = []
    
    for n_genes in sorted(gene_count_results.keys()):
        if gene_count_results[n_genes] is not None:
            report = gene_count_results[n_genes]['classification_report']
            gene_sens.append(report['1']['recall'])
            gene_spec.append(report['0']['recall'])
    
    ax4.plot(gene_counts, gene_sens, marker='s', linewidth=2, markersize=6, color='green', label='Sensitivity')
    ax4.plot(gene_counts, gene_spec, marker='^', linewidth=2, markersize=6, color='orange', label='Specificity')
    ax4.set_xlabel('Number of Genes', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax4.set_title('Sensitivity & Specificity\nby Gene Count', fontsize=12, fontweight='bold')
    if len(gene_counts) > 1:
        ax4.set_xscale('log')
    ax4.set_ylim([0, 1])
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # --- Plot 5: Training CV AUC vs External AUC (Feature-wise) ---
    ax5 = fig.add_subplot(gs[2, 0])
    
    train_aucs_feat = []
    external_aucs_feat = []
    
    for feat in sorted(feature_results.keys()):
        if feature_results[feat] is not None:
            train_aucs_feat.append(feature_results[feat]['best_model_info']['cv_auc'])
            external_aucs_feat.append(feature_results[feat]['overall_auc'])
    
    ax5.scatter(train_aucs_feat, external_aucs_feat, s=100, alpha=0.7, color='purple', edgecolor='black')
    ax5.plot([0.5, 1], [0.5, 1], 'k--', alpha=0.3, label='Perfect generalization')
    ax5.set_xlabel('Training CV AUC', fontsize=11, fontweight='bold')
    ax5.set_ylabel('External AUC', fontsize=11, fontweight='bold')
    ax5.set_title('Training vs External\n(by Feature)', fontsize=12, fontweight='bold')
    ax5.set_xlim([0.5, 1])
    ax5.set_ylim([0.5, 1])
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    # Label points
    for feat, train_auc, ext_auc in zip(sorted(feature_results.keys()), train_aucs_feat, external_aucs_feat):
        if feature_results[feat] is not None:
            ax5.annotate(feat[:4], (train_auc, ext_auc), fontsize=8, alpha=0.7)
    
    # --- Plot 6: Training CV AUC vs External AUC (Gene count) ---
    ax6 = fig.add_subplot(gs[2, 1])
    
    train_aucs_gene = []
    external_aucs_gene = []
    
    for n_genes in sorted(gene_count_results.keys()):
        if gene_count_results[n_genes] is not None:
            train_aucs_gene.append(gene_count_results[n_genes]['best_model_info']['cv_auc'])
            external_aucs_gene.append(gene_count_results[n_genes]['overall_auc'])
    
    # Color by gene count
    colors = plt.cm.viridis(np.linspace(0, 1, len(gene_counts)))
    for gc, train_auc, ext_auc, color in zip(gene_counts, train_aucs_gene, external_aucs_gene, colors):
        ax6.scatter(train_auc, ext_auc, s=80, alpha=0.7, color=color, edgecolor='black', label=f'{gc}')
    
    ax6.plot([0.5, 1], [0.5, 1], 'k--', alpha=0.3)
    ax6.set_xlabel('Training CV AUC', fontsize=11, fontweight='bold')
    ax6.set_ylabel('External AUC', fontsize=11, fontweight='bold')
    ax6.set_title('Training vs External\n(by Gene Count)', fontsize=12, fontweight='bold')
    ax6.set_xlim([0.5, 1])
    ax6.set_ylim([0.5, 1])
    ax6.grid(True, alpha=0.3)
    
    # --- Plot 7: Generalization Gap (Training - External) ---
    ax7 = fig.add_subplot(gs[2, 2])
    
    gaps_gene = [train - ext for train, ext in zip(train_aucs_gene, external_aucs_gene)]
    
    ax7.plot(gene_counts, gaps_gene, marker='D', linewidth=2, markersize=6, color='crimson')
    ax7.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax7.set_xlabel('Number of Genes', fontsize=11, fontweight='bold')
    ax7.set_ylabel('AUC Gap (Train - External)', fontsize=11, fontweight='bold')
    ax7.set_title('Generalization Gap\nby Gene Count', fontsize=12, fontweight='bold')
    if len(gene_counts) > 1:
        ax7.set_xscale('log')
    ax7.grid(True, alpha=0.3)
    ax7.fill_between(gene_counts, 0, gaps_gene, alpha=0.2, color='crimson')
    
    # --- Plot 8: Confusion matrix for best feature ---
    ax8 = fig.add_subplot(gs[3, :])
    best_feat = None
    best_feat_auc = -1
    best_feat_conf = None
    for feat, res in feature_results.items():
        if res is None:
            continue
        if res['overall_auc'] > best_feat_auc:
            best_feat_auc = res['overall_auc']
            best_feat = feat
            best_feat_conf = res.get('confusion_matrix')

    if best_feat_conf is not None:
        conf_array = np.array(best_feat_conf)
        im = ax8.imshow(conf_array, cmap='Blues', aspect='auto')
        for i in range(2):
            for j in range(2):
                ax8.text(j, i, str(conf_array[i, j]),
                         ha="center", va="center",
                         color="white" if conf_array[i, j] > conf_array.max()/2 else "black",
                         fontsize=14, fontweight='bold')
        ax8.set_xticks([0, 1])
        ax8.set_yticks([0, 1])
        ax8.set_xticklabels(['Healthy', 'Cancer'])
        ax8.set_yticklabels(['Healthy', 'Cancer'])
        ax8.set_xlabel('Predicted', fontweight='bold')
        ax8.set_ylabel('Actual', fontweight='bold')
        ax8.set_title(f'Confusion Matrix (Best Feature: {best_feat})', fontweight='bold')
        fig.colorbar(im, ax=ax8, fraction=0.046, pad=0.04)
    else:
        ax8.text(0.5, 0.5, 'No confusion matrix available', ha='center', va='center')
        ax8.set_axis_off()

    # Overall title
    train_dataset = training_preprocessing_metadata.get('dataset', 'Unknown') if training_preprocessing_metadata else 'Unknown'
    external_dataset = args.external_metadata_file
    fig.suptitle(f'Comprehensive External Validation Analysis\nTraining: {train_dataset} -> External: {external_dataset}', 
                fontsize=16, fontweight='bold', y=0.995)
    
    # Save figure
    plot_path = output_dir / 'comprehensive_external_validation_analysis.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nComprehensive visualization saved to: {plot_path}")
    
    # Save summary results - include detailed metrics for downstream analysis
    def _build_feature_summary(res):
        """Extract comprehensive metrics from a single validation result dict."""
        if res is None:
            return None
        s = {
            'auc': res['overall_auc'],
            'sensitivity': res['classification_report']['1']['recall'],
            'specificity': res['classification_report']['0']['recall'],
            'n_genes': res['best_model_info']['n_genes'],
            'classifier': res['best_model_info']['classifier'],
            'training_cv_auc': res['best_model_info']['cv_auc'],
            # ROC curve data
            'fpr': res.get('overall_fpr'),
            'tpr': res.get('overall_tpr'),
            # Scalar classification metrics (default 0.5 threshold)
            'accuracy': res.get('overall_accuracy'),
            'precision': res.get('overall_precision'),
            'recall': res.get('overall_recall'),
            'f1': res.get('overall_f1'),
            # Operating points
            'sens_at_95spec': res.get('overall_sens_at_95spec'),
            'spec_at_95spec': res.get('overall_spec_at_95spec'),
            'threshold_95spec': float(res['threshold_95spec']) if res.get('threshold_95spec') is not None else None,
            'sens_at_90spec': res.get('overall_sens_at_90spec'),
            'spec_at_90spec': res.get('overall_spec_at_90spec'),
            'threshold_90spec': float(res['threshold_90spec']) if res.get('threshold_90spec') is not None else None,
            'sens_at_youden': res.get('overall_sens_at_youden'),
            'spec_at_youden': res.get('overall_spec_at_youden'),
            'threshold_youden': float(res['threshold_youden']) if res.get('threshold_youden') is not None else None,
            # Per-cancer-type AUC
            'auc_per_cancer_type': res.get('auc_per_cancer_type'),
            'roc_per_cancer_type': res.get('roc_per_cancer_type'),
            # Raw predictions for downstream analysis
            'y_pred_proba': res.get('y_pred_proba'),
            'y_true': res.get('y_true'),
            'sample_ids': res.get('sample_ids'),
            'stratification_labels': res.get('stratification_labels'),
            # Data shape
            'external_n_samples': res.get('external_n_samples'),
            'external_n_healthy': res.get('external_n_healthy'),
            'external_n_cancer': res.get('external_n_cancer'),
            # Classification reports at operating points
            'classification_report': res.get('classification_report'),
            'confusion_matrix': res.get('confusion_matrix'),
            'classification_report_95spec': res.get('classification_report_95spec'),
            'confusion_matrix_95spec': res.get('confusion_matrix_95spec'),
            'classification_report_youden': res.get('classification_report_youden'),
            'confusion_matrix_youden': res.get('confusion_matrix_youden'),
        }
        return s

    summary = {
        'feature_results': {
            feat: _build_feature_summary(res)
            for feat, res in feature_results.items()
        },
        'gene_count_results': {
            n_genes: {
                'auc': res['overall_auc'],
                'sensitivity': res['classification_report']['1']['recall'],
                'specificity': res['classification_report']['0']['recall'],
                'feature': res['best_model_info']['feature'],
                'classifier': res['best_model_info']['classifier'],
                'training_cv_auc': res['best_model_info']['cv_auc'],
                # Also include ROC and operating points for gene count results
                'fpr': res.get('overall_fpr'),
                'tpr': res.get('overall_tpr'),
                'sens_at_95spec': res.get('overall_sens_at_95spec'),
                'sens_at_youden': res.get('overall_sens_at_youden'),
                'spec_at_youden': res.get('overall_spec_at_youden'),
                # Per-sample scores per k (the per-feature result pickle is overwritten
                # by every k evaluation, so this is the only place they survive):
                # lets the figure layer bootstrap the AUROC-vs-k curve and split it by
                # comparison subgroup.
                'y_pred_proba': res.get('y_pred_proba'),
                'y_scores_for_roc': res.get('y_scores_for_roc'),
                'proba_degenerate': res.get('proba_degenerate'),
                'y_true': res.get('y_true'),
                'sample_ids': res.get('sample_ids'),
                'stratification_labels': res.get('stratification_labels'),
                'best_model_path': res.get('best_model_path'),
                'deployment_thresholds': res.get('deployment_thresholds'),
                'fixed_threshold_evaluation': res.get('fixed_threshold_evaluation'),
            } if res else None
            for n_genes, res in gene_count_results.items()
        },
        'training_dataset': train_dataset,
        'external_dataset': external_dataset,
        'plot_path': str(plot_path)
    }
    
    summary_path = output_dir / 'comprehensive_validation_summary.pkl'
    with open(summary_path, 'wb') as f:
        pickle.dump(summary, f)
    
    print(f"Summary results saved to: {summary_path}")
    
    # Print summary table
    print("\n" + "="*90)
    print("SUMMARY: GENERALIZATION BY FEATURE TYPE")
    print("="*90)
    print(f"{'Feature':<15} {'Train AUC':<12} {'External AUC':<12} {'Gap':<10} {'Sensitivity':<12} {'Specificity'}")
    print("-" * 90)
    
    for feat in sorted(feature_results.keys()):
        if feature_results[feat] is not None:
            res = feature_results[feat]
            train_auc = res['best_model_info']['cv_auc']
            ext_auc = res['overall_auc']
            gap = train_auc - ext_auc
            sens = res['classification_report']['1']['recall']
            spec = res['classification_report']['0']['recall']
            print(f"{feat:<15} {train_auc:<12.4f} {ext_auc:<12.4f} {gap:<10.4f} {sens:<12.4f} {spec:.4f}")
    
    print("\n" + "="*90)
    print("SUMMARY: GENERALIZATION BY NUMBER OF GENES")
    print("="*90)
    print(f"{'N Genes':<10} {'Feature':<15} {'Train AUC':<12} {'External AUC':<12} {'Gap':<10} {'Sensitivity':<12} {'Specificity'}")
    print("-" * 90)
    
    for n_genes in sorted(gene_count_results.keys()):
        if gene_count_results[n_genes] is not None:
            res = gene_count_results[n_genes]
            feat = res['best_model_info']['feature']
            train_auc = res['best_model_info']['cv_auc']
            ext_auc = res['overall_auc']
            gap = train_auc - ext_auc
            sens = res['classification_report']['1']['recall']
            spec = res['classification_report']['0']['recall']
            print(f"{n_genes:<10} {feat:<15} {train_auc:<12.4f} {ext_auc:<12.4f} {gap:<10.4f} {sens:<12.4f} {spec:.4f}")
    
    print("\n" + "="*90)
    print("KEY INSIGHTS")
    print("="*90)
    
    # Best generalizing feature
    best_feat = max(feature_results.items(), key=lambda x: x[1]['overall_auc'] if x[1] else 0)
    if best_feat[1]:
        print(f"Best generalizing feature: {best_feat[0]} (AUC: {best_feat[1]['overall_auc']:.4f})")
    
    # Best gene count for generalization
    best_genes = max(gene_count_results.items(), key=lambda x: x[1]['overall_auc'] if x[1] else 0)
    if best_genes[1]:
        print(f"Best gene count: {best_genes[0]} genes (AUC: {best_genes[1]['overall_auc']:.4f}, Feature: {best_genes[1]['best_model_info']['feature']})")
    
    # Smallest generalization gap
    gaps = [(feat, res['best_model_info']['cv_auc'] - res['overall_auc']) 
            for feat, res in feature_results.items() if res]
    if gaps:
        min_gap_feat = min(gaps, key=lambda x: x[1])
        print(f"Smallest generalization gap: {min_gap_feat[0]} (gap: {min_gap_feat[1]:.4f})")
    
    print("="*90)
    
    return summary


def run_within_dataset_generalization_training(
    all_feature_arrays: dict,
    y_labels: np.ndarray,
    stratification_labels: np.ndarray,
    sample_ids: np.ndarray,
    training_cancer_types: list,
    withhold_fraction: float,
    CLASSIFIERS: dict,
    CLF_PARAM_GRIDS: dict,
    DATA_PARAM_GRIDS: dict,
    args: argparse.Namespace,
    split_seed: Optional[int] = None,
    split_index: Optional[int] = None
):
    """
    Phase 1: Within-dataset generalization training with stratified holdout.
    
    Trains models on a subset of cancer types (e.g., BRCA + CRC) with 20% withheld,
    preparing for generalization testing on other cancer types.
    
    Workflow:
    1. Filter to healthy + training_cancer_types only
    2. Stratified split into 80% training / 20% withheld
    3. Save withheld sample IDs for later evaluation
    4. Run best_model_search on 80% training set
    
    Args:
        all_feature_arrays: Feature data
        y_labels: Binary labels (0=healthy, 1=cancer)
        stratification_labels: Detailed cancer type labels
        sample_ids: Sample identifiers
        training_cancer_types: Cancer types to include in training (e.g., ['Breast cancer', 'Colorectal cancer'])
        withhold_fraction: Fraction to withhold for final evaluation (e.g., 0.2 for 20%)
        CLASSIFIERS, CLF_PARAM_GRIDS, DATA_PARAM_GRIDS: Model configuration
        args: Command-line arguments
    
    Returns:
        Path to results directory containing trained models and withheld_sample_ids.pkl
    """
    print("\n" + "="*90)
    print("WITHIN-DATASET GENERALIZATION: TRAINING PHASE")
    print("="*90)
    
    # Create output directory
    # Note: experiment_name already includes descriptive prefix (e.g., "within_gen_...")
    output_dir = Path(args.output_dir) / args.experiment_name
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nConfiguration:")
    print(f"   Training cancer types: {training_cancer_types}")
    print(f"   Withhold fraction: {withhold_fraction*100:.0f}%")
    print(f"   Output directory: {output_dir}")
    if split_index is not None:
        print(f"   Split index: {split_index}")
    if split_seed is not None:
        print(f"   Split seed: {split_seed}")
    
    # Step 1: Filter to healthy + training cancer types
    print(f"\nStep 1: Filtering data to healthy + training cancer types...")
    healthy_mask = y_labels == 0
    training_cancer_mask = np.isin(stratification_labels, training_cancer_types)
    combined_mask = healthy_mask | training_cancer_mask
    
    filtered_features = {k: v[combined_mask] if v is not None else None for k, v in all_feature_arrays.items()}
    filtered_y = y_labels[combined_mask]
    filtered_strat = stratification_labels[combined_mask]
    # Convert sample_ids to numpy array for boolean indexing
    sample_ids_array = np.array(sample_ids)
    filtered_ids = sample_ids_array[combined_mask]  # Keep as numpy array for indexing
    
    print(f"   Original: {len(sample_ids)} samples")
    print(f"   Filtered: {len(filtered_ids)} samples")
    print(f"   Healthy: {np.sum(filtered_y == 0)}")
    for cancer_type in training_cancer_types:
        count = np.sum(filtered_strat == cancer_type)
        print(f"   {cancer_type}: {count}")
    
    # Step 2: Stratified split 80/20
    print(f"\nStep 2: Stratified split ({int((1-withhold_fraction)*100)}% training, {int(withhold_fraction*100)}% withheld)...")
    
    
    # Split indices
    split_random_state = split_seed if split_seed is not None else 42
    train_indices, withheld_indices = train_test_split(
        np.arange(len(filtered_ids)),
        test_size=withhold_fraction,
        stratify=filtered_strat,
        random_state=split_random_state
    )
    
    # Create training and withheld sets
    train_features = {k: v[train_indices] if v is not None else None for k, v in filtered_features.items()}
    train_y = filtered_y[train_indices]
    train_strat = filtered_strat[train_indices]
    train_ids = filtered_ids[train_indices]
    
    withheld_features = {k: v[withheld_indices] if v is not None else None for k, v in filtered_features.items()}
    withheld_y = filtered_y[withheld_indices]
    withheld_strat = filtered_strat[withheld_indices]
    withheld_ids = filtered_ids[withheld_indices]
    
    print(f"   Training set: {len(train_ids)} samples")
    print(f"     Healthy: {np.sum(train_y == 0)}")
    for cancer_type in training_cancer_types:
        count = np.sum(train_strat == cancer_type)
        print(f"     {cancer_type}: {count}")
    
    print(f"   Withheld set: {len(withheld_ids)} samples")
    print(f"     Healthy: {np.sum(withheld_y == 0)}")
    for cancer_type in training_cancer_types:
        count = np.sum(withheld_strat == cancer_type)
        print(f"     {cancer_type}: {count}")
    
    # Step 3: Save withheld information for later evaluation
    print(f"\nStep 3: Saving withheld sample information...")
    withheld_info = {
        'withheld_sample_ids': withheld_ids.tolist(),
        'withheld_y_labels': withheld_y.tolist(),
        'withheld_stratification': withheld_strat.tolist(),
        'training_cancer_types': training_cancer_types,
        'withhold_fraction': withhold_fraction,
        'split_index': int(split_index) if split_index is not None else None,
        'split_seed': int(split_seed) if split_seed is not None else None,
        'training_sample_ids': train_ids.tolist(),
        'training_set_size': len(train_ids),
        'withheld_set_size': len(withheld_ids)
    }
    
    withheld_file = output_dir / "withheld_sample_info.pkl"
    with open(withheld_file, 'wb') as f:
        pickle.dump(withheld_info, f)
    print(f"   Saved to: {withheld_file}")
    
    # Also save as CSV for easy inspection
    withheld_df = pd.DataFrame({
        'sample_id': withheld_ids,
        'binary_label': withheld_y,
        'cancer_type': withheld_strat
    })
    withheld_csv = output_dir / "withheld_samples.csv"
    withheld_df.to_csv(withheld_csv, index=False)
    print(f"   Saved CSV to: {withheld_csv}")
    
    # Step 4: Training-only standardization stats (for withheld/novel evaluation)
    print(f"\nStep 4: Computing training-only standardization stats...")
    training_feature_stats = {
        'feature_means_per_feature': {
            feature: np.nanmean(arr, axis=0).tolist()
            for feature, arr in train_features.items() if arr is not None
        },
        'feature_stds_per_feature': {
            feature: np.nanstd(arr, axis=0).tolist()
            for feature, arr in train_features.items() if arr is not None
        }
    }
    withheld_info['training_feature_stats'] = training_feature_stats

    # Apply training-only standardization to training features
    train_features = standardize_with_training_stats(train_features, training_feature_stats)

    # Step 5: Track preprocessing metadata
    print(f"\nStep 5: Preparing preprocessing metadata...")
    # Note: Preprocessing has already been done in main(), so we just need to record it
    # The kept_columns_per_feature should be passed from main or we can note that training uses the same preprocessing
    preprocessing_metadata = {
        'training_cancer_types': training_cancer_types,
        'withhold_fraction': withhold_fraction,
        'training_set_size': len(train_ids),
        'withheld_set_size': len(withheld_ids),
        'split_index': int(split_index) if split_index is not None else None,
        'split_seed': int(split_seed) if split_seed is not None else None,
        'note': 'Within-dataset generalization: trained on subset of cancer types with holdout'
    }
    
    # Step 6: Run best_model_search on training set
    print(f"\nStep 6: Running best_model_search on {len(train_ids)} training samples...")
    print(f"   This will find the best model using only healthy + {training_cancer_types}")

    prev_skip_scaler = getattr(args, 'skip_scaler', False)
    args.skip_scaler = True
    
    run_best_model_search(
        all_feature_arrays=train_features,
        y_labels=train_y,
        stratification_labels=train_strat,
        sample_ids=train_ids,
        CLASSIFIERS=CLASSIFIERS,
        CLF_PARAM_GRIDS=CLF_PARAM_GRIDS,
        DATA_PARAM_GRIDS=DATA_PARAM_GRIDS,
        args=args,
        preprocessing_metadata=preprocessing_metadata
    )

    args.skip_scaler = prev_skip_scaler
    
    # Step 7: Create training summary visualizations
    print(f"\nStep 7: Creating training summary visualizations...")
    
    
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Plot 1: Training set composition (pie chart)
    ax1 = fig.add_subplot(gs[0, 0])
    composition_data = [np.sum(train_y == 0)]
    composition_labels = ['Healthy']
    for ct in training_cancer_types:
        count = np.sum(train_strat == ct)
        composition_data.append(count)
        composition_labels.append(ct.replace(' cancer', ''))
    
    colors = ['lightgreen'] + list(plt.cm.Set3.colors[:len(training_cancer_types)])
    ax1.pie(composition_data, labels=composition_labels, autopct='%1.1f%%', 
            colors=colors, startangle=90)
    ax1.set_title(f'Training Set Composition\n({len(train_ids)} samples)', fontweight='bold')
    
    # Plot 2: Withheld set composition (pie chart)
    ax2 = fig.add_subplot(gs[0, 1])
    withheld_data = [np.sum(withheld_y == 0)]
    withheld_labels = ['Healthy']
    for ct in training_cancer_types:
        count = np.sum(withheld_strat == ct)
        withheld_data.append(count)
        withheld_labels.append(ct.replace(' cancer', ''))
    
    ax2.pie(withheld_data, labels=withheld_labels, autopct='%1.1f%%',
            colors=colors, startangle=90)
    ax2.set_title(f'Withheld Set Composition\n({len(withheld_ids)} samples)', fontweight='bold')
    
    # Plot 3: Split summary (bar chart)
    ax3 = fig.add_subplot(gs[0, 2])
    split_data = [len(train_ids), len(withheld_ids)]
    split_labels = [f'Training\n{int((1-withhold_fraction)*100)}%', 
                   f'Withheld\n{int(withhold_fraction*100)}%']
    bars = ax3.bar(split_labels, split_data, color=['steelblue', 'coral'], alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Number of Samples', fontweight='bold')
    ax3.set_title('Dataset Split', fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars, split_data):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(val), ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Sample counts by cancer type (horizontal bar)
    ax4 = fig.add_subplot(gs[1, :])
    cancer_types_list = ['Healthy'] + training_cancer_types
    train_counts = [np.sum(train_y == 0)] + [np.sum(train_strat == ct) for ct in training_cancer_types]
    withheld_counts = [np.sum(withheld_y == 0)] + [np.sum(withheld_strat == ct) for ct in training_cancer_types]
    
    y_pos = np.arange(len(cancer_types_list))
    width = 0.35
    
    ax4.barh(y_pos - width/2, train_counts, width, label='Training', color='steelblue', alpha=0.7)
    ax4.barh(y_pos + width/2, withheld_counts, width, label='Withheld', color='coral', alpha=0.7)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels([ct.replace(' cancer', '') for ct in cancer_types_list])
    ax4.set_xlabel('Number of Samples', fontweight='bold')
    ax4.set_title('Detailed Sample Distribution by Type', fontweight='bold')
    ax4.legend()
    ax4.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (train_ct, with_ct) in enumerate(zip(train_counts, withheld_counts)):
        if train_ct > 0:
            ax4.text(train_ct + 2, i - width/2, str(train_ct), va='center', fontsize=9)
        if with_ct > 0:
            ax4.text(with_ct + 2, i + width/2, str(with_ct), va='center', fontsize=9)
    
    fig.suptitle(f'Within-Dataset Generalization: Training Phase\n{", ".join(training_cancer_types)}',
                fontsize=14, fontweight='bold')
    
    plot_path = output_dir / 'training_summary.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   Training summary saved: {plot_path}")
    
    print(f"\nTRAINING PHASE COMPLETE")
    print(f"   Results saved to: {output_dir}")
    print(f"   Withheld samples: {withheld_file}")
    print(f"   Training summary plot: {plot_path}")
    print(f"\n   Next step: Use --run_within_dataset_generalization_eval to evaluate on:")
    print(f"     1. Withheld {withhold_fraction*100:.0f}% ({len(withheld_ids)} samples)")
    print(f"     2. Other PANCAN cancer types (not in training)")
    print("="*90)
    
    return output_dir


def run_within_dataset_generalization_evaluation(
    training_results_dir: str,
    all_feature_arrays: dict,
    y_labels: np.ndarray,
    stratification_labels: np.ndarray,
    sample_ids: np.ndarray,
    args: argparse.Namespace,
    filter_feature: str = None,
    filter_n_genes: int = None
):
    """
    Phase 2: Within-dataset generalization evaluation.
    
    Evaluates the best model from training on:
    1. Withheld samples (20% of healthy + training cancer types) - same distribution test
    2. Other cancer types from PANCAN (generalization to novel cancer types)
    3. Combined evaluation set
    
    Args:
        training_results_dir: Directory containing trained models and withheld_sample_info.pkl
        all_feature_arrays: Full PANCAN feature data
        y_labels: Full PANCAN binary labels
        stratification_labels: Full PANCAN cancer type labels
        sample_ids: Full PANCAN sample IDs
        args: Command-line arguments
        filter_feature: If provided, only evaluate models using this feature
        filter_n_genes: If provided, only evaluate models using this number of genes
    
    Returns:
        Evaluation results dictionary
    """
    print("\n" + "="*90)
    print("WITHIN-DATASET GENERALIZATION: EVALUATION PHASE")
    print("="*90)
    
    training_results_dir = Path(training_results_dir)
    split_dirs = sorted([d for d in training_results_dir.glob("split_*") if (d / "withheld_sample_info.pkl").exists()])
    if split_dirs and not training_results_dir.name.startswith("split_"):
        print(f"\nDetected {len(split_dirs)} split directories; running per-split evaluations...")

        split_summaries = []
        for split_dir in split_dirs:
            print(f"\n=== Split evaluation: {split_dir.name} ===")
            split_summary = run_comprehensive_within_dataset_evaluation(
                training_results_dir=split_dir,
                all_feature_arrays=all_feature_arrays,
                y_labels=y_labels,
                stratification_labels=stratification_labels,
                sample_ids=sample_ids,
                args=args
            )
            split_summaries.append(split_summary)

        def _mean_std_ci(values):
            arr = np.array([v for v in values if v is not None], dtype=float)
            if arr.size == 0:
                return None, None, None
            mean = float(np.mean(arr))
            std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
            if arr.size > 1:
                ci = [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]
            else:
                ci = [mean, mean]
            return mean, std, ci

        def _aggregate_roc(roc_list):
            fpr_grid = np.linspace(0.0, 1.0, 101)
            tprs = []
            for roc in roc_list:
                if not roc:
                    continue
                fpr = np.array(roc.get('fpr') or [])
                tpr = np.array(roc.get('tpr') or [])
                if fpr.size == 0 or tpr.size == 0:
                    continue
                tpr_interp = np.interp(fpr_grid, fpr, tpr)
                tpr_interp[0] = 0.0
                tprs.append(tpr_interp)
            if not tprs:
                return None, None
            tprs = np.vstack(tprs)
            mean_tpr = np.mean(tprs, axis=0)
            lower = np.percentile(tprs, 2.5, axis=0)
            upper = np.percentile(tprs, 97.5, axis=0)
            mean_roc = {'fpr': fpr_grid.tolist(), 'tpr': mean_tpr.tolist()}
            ci = {'fpr': fpr_grid.tolist(), 'tpr_lower': lower.tolist(), 'tpr_upper': upper.tolist()}
            return mean_roc, ci

        no_novel_cancers = bool(split_summaries[0].get('no_novel_cancers')) if split_summaries else False
        use_gene_counts = bool(split_summaries[0].get('use_gene_counts')) if split_summaries else False
        metric_key = 'withheld_auc' if no_novel_cancers else 'novel_auc'

        feature_names = sorted({
            feat
            for s in split_summaries
            for feat in (s.get('feature_results') or {}).keys()
        })
        feature_results = {}
        for feat in feature_names:
            per_split = [
                (s.get('feature_results') or {}).get(feat)
                for s in split_summaries
                if (s.get('feature_results') or {}).get(feat) is not None
            ]
            meta_res = per_split[0] if per_split else {}
            overall_vals = [r.get('overall_auc') for r in per_split if r.get('overall_auc') is not None]
            withheld_vals = [r.get('withheld_auc') for r in per_split if r.get('withheld_auc') is not None]
            novel_vals = [r.get('novel_auc') for r in per_split if r.get('novel_auc') is not None]
            train_vals = [r.get('training_cv_auc') for r in per_split if r.get('training_cv_auc') is not None]
            precision_vals = [r.get('overall_precision') for r in per_split if r.get('overall_precision') is not None]
            recall_vals = [r.get('overall_recall') for r in per_split if r.get('overall_recall') is not None]
            f1_vals = [r.get('overall_f1') for r in per_split if r.get('overall_f1') is not None]
            accuracy_vals = [r.get('overall_accuracy') for r in per_split if r.get('overall_accuracy') is not None]
            overall_sens_vals = [r.get('overall_sens_at_95spec') for r in per_split if r.get('overall_sens_at_95spec') is not None]
            overall_spec_vals = [r.get('overall_spec_at_95spec') for r in per_split if r.get('overall_spec_at_95spec') is not None]
            withheld_sens_vals = [r.get('withheld_sens_at_95spec') for r in per_split if r.get('withheld_sens_at_95spec') is not None]
            withheld_spec_vals = [r.get('withheld_spec_at_95spec') for r in per_split if r.get('withheld_spec_at_95spec') is not None]
            novel_sens_vals = [r.get('novel_sens_at_95spec') for r in per_split if r.get('novel_sens_at_95spec') is not None]
            novel_spec_vals = [r.get('novel_spec_at_95spec') for r in per_split if r.get('novel_spec_at_95spec') is not None]

            overall_mean, overall_std, overall_ci = _mean_std_ci(overall_vals)
            withheld_mean, withheld_std, withheld_ci = _mean_std_ci(withheld_vals)
            novel_mean, novel_std, novel_ci = _mean_std_ci(novel_vals)
            train_mean, train_std, train_ci = _mean_std_ci(train_vals)
            precision_mean, precision_std, precision_ci = _mean_std_ci(precision_vals)
            recall_mean, recall_std, recall_ci = _mean_std_ci(recall_vals)
            f1_mean, f1_std, f1_ci = _mean_std_ci(f1_vals)
            accuracy_mean, accuracy_std, accuracy_ci = _mean_std_ci(accuracy_vals)
            overall_sens_mean, overall_sens_std, _ = _mean_std_ci(overall_sens_vals)
            overall_spec_mean, overall_spec_std, _ = _mean_std_ci(overall_spec_vals)
            withheld_sens_mean, withheld_sens_std, _ = _mean_std_ci(withheld_sens_vals)
            withheld_spec_mean, withheld_spec_std, _ = _mean_std_ci(withheld_spec_vals)
            novel_sens_mean, novel_sens_std, _ = _mean_std_ci(novel_sens_vals)
            novel_spec_mean, novel_spec_std, _ = _mean_std_ci(novel_spec_vals)

            feature_results[feat] = {
                'overall_auc': overall_mean,
                'overall_auc_mean': overall_mean,
                'overall_auc_std': overall_std,
                'overall_auc_ci': overall_ci,
                'withheld_auc': withheld_mean,
                'withheld_auc_mean': withheld_mean,
                'withheld_auc_std': withheld_std,
                'withheld_auc_ci': withheld_ci,
                'novel_auc': novel_mean,
                'novel_auc_mean': novel_mean,
                'novel_auc_std': novel_std,
                'novel_auc_ci': novel_ci,
                'training_cv_auc': train_mean,
                'training_cv_auc_mean': train_mean,
                'training_cv_auc_std': train_std,
                'training_cv_auc_ci': train_ci,
                'overall_precision': precision_mean,
                'overall_precision_mean': precision_mean,
                'overall_precision_std': precision_std,
                'overall_precision_ci': precision_ci,
                'overall_recall': recall_mean,
                'overall_recall_mean': recall_mean,
                'overall_recall_std': recall_std,
                'overall_recall_ci': recall_ci,
                'overall_f1': f1_mean,
                'overall_f1_mean': f1_mean,
                'overall_f1_std': f1_std,
                'overall_f1_ci': f1_ci,
                'overall_accuracy': accuracy_mean,
                'overall_accuracy_mean': accuracy_mean,
                'overall_accuracy_std': accuracy_std,
                'overall_accuracy_ci': accuracy_ci,
                'overall_sens_at_95spec': overall_sens_mean,
                'overall_sens_at_95spec_std': overall_sens_std,
                'overall_spec_at_95spec': overall_spec_mean,
                'overall_spec_at_95spec_std': overall_spec_std,
                'withheld_sens_at_95spec': withheld_sens_mean,
                'withheld_sens_at_95spec_std': withheld_sens_std,
                'withheld_spec_at_95spec': withheld_spec_mean,
                'withheld_spec_at_95spec_std': withheld_spec_std,
                'novel_sens_at_95spec': novel_sens_mean,
                'novel_sens_at_95spec_std': novel_sens_std,
                'novel_spec_at_95spec': novel_spec_mean,
                'novel_spec_at_95spec_std': novel_spec_std,
                'n_genes': meta_res.get('n_genes'),
                'classifier': meta_res.get('classifier'),
                'n_splits': len(per_split)
            }

        gene_counts = sorted({
            n
            for s in split_summaries
            for n in (s.get('gene_count_results') or {}).keys()
        })
        gene_count_results = {}
        for n_genes in gene_counts:
            per_split = [
                (s.get('gene_count_results') or {}).get(n_genes)
                for s in split_summaries
                if (s.get('gene_count_results') or {}).get(n_genes) is not None
            ]
            meta_res = per_split[0] if per_split else {}
            overall_vals = [r.get('overall_auc') for r in per_split if r.get('overall_auc') is not None]
            withheld_vals = [r.get('withheld_auc') for r in per_split if r.get('withheld_auc') is not None]
            novel_vals = [r.get('novel_auc') for r in per_split if r.get('novel_auc') is not None]
            train_vals = [r.get('training_cv_auc') for r in per_split if r.get('training_cv_auc') is not None]
            precision_vals = [r.get('overall_precision') for r in per_split if r.get('overall_precision') is not None]
            recall_vals = [r.get('overall_recall') for r in per_split if r.get('overall_recall') is not None]
            f1_vals = [r.get('overall_f1') for r in per_split if r.get('overall_f1') is not None]
            accuracy_vals = [r.get('overall_accuracy') for r in per_split if r.get('overall_accuracy') is not None]
            overall_sens_vals = [r.get('overall_sens_at_95spec') for r in per_split if r.get('overall_sens_at_95spec') is not None]
            overall_spec_vals = [r.get('overall_spec_at_95spec') for r in per_split if r.get('overall_spec_at_95spec') is not None]
            withheld_sens_vals = [r.get('withheld_sens_at_95spec') for r in per_split if r.get('withheld_sens_at_95spec') is not None]
            withheld_spec_vals = [r.get('withheld_spec_at_95spec') for r in per_split if r.get('withheld_spec_at_95spec') is not None]
            novel_sens_vals = [r.get('novel_sens_at_95spec') for r in per_split if r.get('novel_sens_at_95spec') is not None]
            novel_spec_vals = [r.get('novel_spec_at_95spec') for r in per_split if r.get('novel_spec_at_95spec') is not None]

            overall_mean, overall_std, overall_ci = _mean_std_ci(overall_vals)
            withheld_mean, withheld_std, withheld_ci = _mean_std_ci(withheld_vals)
            novel_mean, novel_std, novel_ci = _mean_std_ci(novel_vals)
            train_mean, train_std, train_ci = _mean_std_ci(train_vals)
            precision_mean, precision_std, precision_ci = _mean_std_ci(precision_vals)
            recall_mean, recall_std, recall_ci = _mean_std_ci(recall_vals)
            f1_mean, f1_std, f1_ci = _mean_std_ci(f1_vals)
            accuracy_mean, accuracy_std, accuracy_ci = _mean_std_ci(accuracy_vals)
            overall_sens_mean, overall_sens_std, _ = _mean_std_ci(overall_sens_vals)
            overall_spec_mean, overall_spec_std, _ = _mean_std_ci(overall_spec_vals)
            withheld_sens_mean, withheld_sens_std, _ = _mean_std_ci(withheld_sens_vals)
            withheld_spec_mean, withheld_spec_std, _ = _mean_std_ci(withheld_spec_vals)
            novel_sens_mean, novel_sens_std, _ = _mean_std_ci(novel_sens_vals)
            novel_spec_mean, novel_spec_std, _ = _mean_std_ci(novel_spec_vals)

            gene_count_results[n_genes] = {
                'overall_auc': overall_mean,
                'overall_auc_mean': overall_mean,
                'overall_auc_std': overall_std,
                'overall_auc_ci': overall_ci,
                'withheld_auc': withheld_mean,
                'withheld_auc_mean': withheld_mean,
                'withheld_auc_std': withheld_std,
                'withheld_auc_ci': withheld_ci,
                'novel_auc': novel_mean,
                'novel_auc_mean': novel_mean,
                'novel_auc_std': novel_std,
                'novel_auc_ci': novel_ci,
                'training_cv_auc': train_mean,
                'training_cv_auc_mean': train_mean,
                'training_cv_auc_std': train_std,
                'training_cv_auc_ci': train_ci,
                'overall_precision': precision_mean,
                'overall_precision_mean': precision_mean,
                'overall_precision_std': precision_std,
                'overall_precision_ci': precision_ci,
                'overall_recall': recall_mean,
                'overall_recall_mean': recall_mean,
                'overall_recall_std': recall_std,
                'overall_recall_ci': recall_ci,
                'overall_f1': f1_mean,
                'overall_f1_mean': f1_mean,
                'overall_f1_std': f1_std,
                'overall_f1_ci': f1_ci,
                'overall_accuracy': accuracy_mean,
                'overall_accuracy_mean': accuracy_mean,
                'overall_accuracy_std': accuracy_std,
                'overall_accuracy_ci': accuracy_ci,
                'overall_sens_at_95spec': overall_sens_mean,
                'overall_sens_at_95spec_std': overall_sens_std,
                'overall_spec_at_95spec': overall_spec_mean,
                'overall_spec_at_95spec_std': overall_spec_std,
                'withheld_sens_at_95spec': withheld_sens_mean,
                'withheld_sens_at_95spec_std': withheld_sens_std,
                'withheld_spec_at_95spec': withheld_spec_mean,
                'withheld_spec_at_95spec_std': withheld_spec_std,
                'novel_sens_at_95spec': novel_sens_mean,
                'novel_sens_at_95spec_std': novel_sens_std,
                'novel_spec_at_95spec': novel_spec_mean,
                'novel_spec_at_95spec_std': novel_spec_std,
                'feature': meta_res.get('feature'),
                'classifier': meta_res.get('classifier'),
                'n_splits': len(per_split)
            }

        best_feature_by_split = [s.get('best_feature') for s in split_summaries if s.get('best_feature')]
        best_feature_varies = len(set(best_feature_by_split)) > 1 if best_feature_by_split else False

        best_feature = None
        best_score = -1
        for feat, res in feature_results.items():
            if res is None:
                continue
            score = res.get(metric_key) or 0
            if score > best_score:
                best_score = score
                best_feature = feat

        # Aggregate ROCs from split-level best-feature ROC curves
        overall_rocs = [s.get('best_feature_roc', {}).get('overall') for s in split_summaries]
        withheld_rocs = [s.get('best_feature_roc', {}).get('withheld') for s in split_summaries]
        novel_rocs = [s.get('best_feature_roc', {}).get('novel') for s in split_summaries]

        overall_roc, overall_ci = _aggregate_roc(overall_rocs)
        withheld_roc, withheld_ci = _aggregate_roc(withheld_rocs)
        novel_roc, novel_ci = _aggregate_roc(novel_rocs)

        # Aggregate best-feature AUC stats
        best_auc_overall_vals = [s.get('best_feature_auc', {}).get('overall') for s in split_summaries]
        best_auc_withheld_vals = [s.get('best_feature_auc', {}).get('withheld') for s in split_summaries]
        best_auc_novel_vals = [s.get('best_feature_auc', {}).get('novel') for s in split_summaries]
        best_overall_mean, best_overall_std, best_overall_ci = _mean_std_ci(best_auc_overall_vals)
        best_withheld_mean, best_withheld_std, best_withheld_ci = _mean_std_ci(best_auc_withheld_vals)
        best_novel_mean, best_novel_std, best_novel_ci = _mean_std_ci(best_auc_novel_vals)

        base_training_types = split_summaries[0].get('training_cancer_types') if split_summaries else None
        eval_n_samples_list = [s.get('eval_n_samples') for s in split_summaries if s.get('eval_n_samples') is not None]
        eval_n_withheld_list = [s.get('eval_n_withheld') for s in split_summaries if s.get('eval_n_withheld') is not None]
        eval_n_novel_list = [s.get('eval_n_novel_cancer') for s in split_summaries if s.get('eval_n_novel_cancer') is not None]
        eval_novel_types = split_summaries[0].get('eval_novel_cancer_types') if split_summaries else None

        summary = {
            'summary_version': '2026-02-07-multisplit',
            'summary_source': str(Path(__file__).resolve()),
            'n_splits': len(split_summaries),
            'split_dirs': [str(d) for d in split_dirs],
            'best_feature_by_split': best_feature_by_split,
            'best_feature_varies': bool(best_feature_varies),
            'training_cancer_types': base_training_types,
            'eval_n_samples': int(np.mean(eval_n_samples_list)) if eval_n_samples_list else None,
            'eval_n_withheld': int(np.mean(eval_n_withheld_list)) if eval_n_withheld_list else None,
            'eval_n_novel_cancer': int(np.mean(eval_n_novel_list)) if eval_n_novel_list else None,
            'eval_novel_cancer_types': eval_novel_types,
            'feature_results': feature_results,
            'gene_count_results': gene_count_results,
            'best_feature': best_feature,
            'best_feature_roc': {
                'overall': overall_roc,
                'withheld': withheld_roc,
                'novel': novel_roc,
                'overall_ci': overall_ci,
                'withheld_ci': withheld_ci,
                'novel_ci': novel_ci
            },
            'best_feature_auc': {
                'overall': best_overall_mean,
                'withheld': best_withheld_mean,
                'novel': best_novel_mean,
                'overall_mean': best_overall_mean,
                'overall_std': best_overall_std,
                'overall_ci': best_overall_ci,
                'withheld_mean': best_withheld_mean,
                'withheld_std': best_withheld_std,
                'withheld_ci': best_withheld_ci,
                'novel_mean': best_novel_mean,
                'novel_std': best_novel_std,
                'novel_ci': best_novel_ci
            },
            'no_novel_cancers': bool(no_novel_cancers),
            'novel_auc_is_fallback': bool(no_novel_cancers),
            'use_gene_counts': bool(split_summaries[0].get('use_gene_counts')) if split_summaries else None,
            'plot_path': None,
            'extra_plot_path': None,
            'roc_plot_path': None
        }

        output_dir = training_results_dir / "comprehensive_evaluation"
        os.makedirs(output_dir, exist_ok=True)
        summary_path = output_dir / 'comprehensive_generalization_summary.pkl'
        with open(summary_path, 'wb') as f:
            pickle.dump(summary, f)

        print(f"\nAggregated multi-split summary saved: {summary_path}")

        # Plot aggregated results with CIs
        try:

            fig = plt.figure(figsize=(20, 12))
            gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

            # Plot 1: AUC by Feature Type (with CI)
            ax1 = fig.add_subplot(gs[0, :2])
            feature_names = []
            novel_aucs_feat = []
            novel_aucs_std = []
            sens_feat = []
            sens_std = []

            metric_label = "Withheld AUC" if no_novel_cancers else "Novel AUC"
            for feat in sorted(feature_results.keys()):
                res = feature_results[feat]
                if res is None:
                    continue
                feature_names.append(feat)
                novel_aucs_feat.append((res.get('withheld_auc') if no_novel_cancers else res.get('novel_auc')) or 0)
                novel_aucs_std.append((res.get('withheld_auc_std') if no_novel_cancers else res.get('novel_auc_std')) or 0)
                sens_feat.append((res.get('withheld_sens_at_95spec') if no_novel_cancers else res.get('novel_sens_at_95spec')) or 0)
                sens_std.append((res.get('withheld_sens_at_95spec_std') if no_novel_cancers else res.get('novel_sens_at_95spec_std')) or 0)

            x_pos = np.arange(len(feature_names))
            ax1.bar(x_pos, novel_aucs_feat, yerr=novel_aucs_std, capsize=4, alpha=0.7, color='steelblue', edgecolor='black')
            ax1.set_xlabel('Feature Type', fontsize=12, fontweight='bold')
            ax1.set_ylabel(metric_label, fontsize=12, fontweight='bold')
            ax1.set_title(f'Generalization Performance by Feature Type ({metric_label})', fontsize=14, fontweight='bold')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(feature_names, rotation=45, ha='right')
            ax1.set_ylim([0, 1])
            ax1.grid(axis='y', alpha=0.3)
            ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)

            # Plot 2: Sens@95%Spec by Feature (with CI)
            ax2 = fig.add_subplot(gs[0, 2])
            ax2.bar(x_pos, sens_feat, yerr=sens_std, capsize=4, color='green', alpha=0.7, edgecolor='black')
            ax2.set_xlabel('Feature Type', fontsize=11, fontweight='bold')
            ax2.set_ylabel('Sensitivity', fontsize=11, fontweight='bold')
            ax2.set_title('Sens @95% Spec', fontsize=12, fontweight='bold')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=9)
            ax2.set_ylim([0, 1])
            ax2.grid(axis='y', alpha=0.3)

            # Plot 3: AUC by Number of Genes (with CI)
            ax3 = fig.add_subplot(gs[1, :2])
            if use_gene_counts and gene_count_results:
                gene_counts = []
                novel_aucs_gene = []
                novel_aucs_gene_std = []
                for n_genes in sorted(gene_count_results.keys()):
                    res = gene_count_results[n_genes]
                    if res is None:
                        continue
                    gene_counts.append(n_genes)
                    novel_aucs_gene.append((res.get('withheld_auc') if no_novel_cancers else res.get('novel_auc')) or 0)
                    novel_aucs_gene_std.append((res.get('withheld_auc_std') if no_novel_cancers else res.get('novel_auc_std')) or 0)
                ax3.plot(gene_counts, novel_aucs_gene, marker='o', linewidth=2, markersize=8, color='steelblue')
                if gene_counts:
                    lower = np.array(novel_aucs_gene) - np.array(novel_aucs_gene_std)
                    upper = np.array(novel_aucs_gene) + np.array(novel_aucs_gene_std)
                    ax3.fill_between(gene_counts, lower, upper, color='steelblue', alpha=0.2, linewidth=0)
                ax3.set_xscale('log')
                ax3.set_ylim([0, 1])
                ax3.set_xlabel('Number of Genes', fontsize=12, fontweight='bold')
                ax3.set_ylabel(metric_label, fontsize=12, fontweight='bold')
                ax3.set_title(f'Performance by Number of Genes ({metric_label})', fontsize=14, fontweight='bold')
                ax3.grid(True, alpha=0.3)
            else:
                ax3.text(0.5, 0.5, 'Not applicable\n(no gene selection)', transform=ax3.transAxes,
                         ha='center', va='center', fontsize=12)
                ax3.set_axis_off()

            # Plot 4: ROC for best feature (with CI)
            ax4 = fig.add_subplot(gs[1, 2])
            roc = summary.get('best_feature_roc') or {}
            if roc.get('overall'):
                ax4.plot(roc['overall']['fpr'], roc['overall']['tpr'], label='Overall', color='steelblue')
            if roc.get('overall_ci'):
                ci = roc['overall_ci']
                ax4.fill_between(ci['fpr'], ci['tpr_lower'], ci['tpr_upper'], color='steelblue', alpha=0.2, linewidth=0)
            if roc.get('withheld'):
                ax4.plot(roc['withheld']['fpr'], roc['withheld']['tpr'], label='Withheld', color='green', linestyle='--')
            if roc.get('withheld_ci'):
                ci = roc['withheld_ci']
                ax4.fill_between(ci['fpr'], ci['tpr_lower'], ci['tpr_upper'], color='green', alpha=0.15, linewidth=0)
            if roc.get('novel'):
                ax4.plot(roc['novel']['fpr'], roc['novel']['tpr'], label='Novel', color='orange', linestyle=':')
            if roc.get('novel_ci'):
                ci = roc['novel_ci']
                ax4.fill_between(ci['fpr'], ci['tpr_lower'], ci['tpr_upper'], color='orange', alpha=0.15, linewidth=0)
            ax4.plot([0, 1], [0, 1], 'k--', alpha=0.3)
            ax4.set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
            ax4.set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
            ax4.set_title('ROC (Best Feature)', fontsize=12, fontweight='bold')
            ax4.set_xlim([0, 1])
            ax4.set_ylim([0, 1])
            ax4.grid(alpha=0.3)
            ax4.legend(fontsize=8)

            # In this multi-split scope the training types live in base_training_types
            # (taken from the first split summary above).
            training_cancer_str = ', '.join([ct.replace(' cancer', '') for ct in (base_training_types or [])])
            fig.suptitle(f'Comprehensive Within-Dataset Generalization (Multi-split)\nTraining: {training_cancer_str}',
                         fontsize=14, fontweight='bold', y=0.98)

            plot_path = output_dir / 'comprehensive_generalization_comparison.png'
            plt.tight_layout()
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Aggregated comparison plot saved: {plot_path}")
        except Exception as e:
            print(f"Failed to plot aggregated results: {e}")

        return summary
    
    # Apply filters if specified
    filter_desc = []
    if filter_feature:
        filter_desc.append(f"feature={filter_feature}")
    if filter_n_genes:
        filter_desc.append(f"n_genes={filter_n_genes}")
    
    if filter_desc:
        print(f"   Filtering models: {', '.join(filter_desc)}")
    
    # Step 1: Load withheld sample information
    print(f"\nStep 1: Loading withheld sample information...")
    withheld_file = training_results_dir / "withheld_sample_info.pkl"
    
    if not withheld_file.exists():
        raise FileNotFoundError(f"Withheld sample file not found: {withheld_file}\n"
                               f"Run training phase first with --run_within_dataset_generalization_train")
    
    with open(withheld_file, 'rb') as f:
        withheld_info = pickle.load(f)
    
    withheld_sample_ids = set(withheld_info['withheld_sample_ids'])
    training_cancer_types = withheld_info['training_cancer_types']
    training_feature_stats = withheld_info.get('training_feature_stats')
    
    print(f"   Loaded withheld info")
    print(f"   Training cancer types: {training_cancer_types}")
    print(f"   Withheld samples: {len(withheld_sample_ids)}")
    
    # Step 2: Split PANCAN into evaluation sets
    print(f"\nStep 2: Creating evaluation sets...")
    
    # Convert sample_ids to array for indexing
    sample_ids_array = np.array(sample_ids)
    
    # Withheld set: samples in withheld_sample_ids
    withheld_mask = np.isin(sample_ids_array, list(withheld_sample_ids))
    
    # Novel cancer types: cancer samples NOT in training_cancer_types and NOT in withheld set
    novel_cancer_mask = (
        (y_labels == 1) &  # Cancer samples
        (~np.isin(stratification_labels, training_cancer_types)) &  # Not training cancer types
        (~withheld_mask)  # Not in withheld set
    )
    
    # Combined evaluation set: withheld + novel cancer types
    eval_mask = withheld_mask | novel_cancer_mask
    
    eval_features = {k: v[eval_mask] if v is not None else None for k, v in all_feature_arrays.items()}
    eval_y = y_labels[eval_mask]
    eval_strat = stratification_labels[eval_mask]
    eval_ids = sample_ids_array[eval_mask]

    if training_feature_stats:
        print("\nApplying training-only standardization to evaluation data")
        eval_features = standardize_with_training_stats(eval_features, training_feature_stats)
    
    # Track which samples are withheld vs novel
    is_withheld = withheld_mask[eval_mask]
    is_novel_cancer = novel_cancer_mask[eval_mask]
    
    print(f"   Evaluation set composition:")
    print(f"     Total: {len(eval_ids)} samples")
    print(f"     Withheld (same distribution): {np.sum(is_withheld)}")
    print(f"       - Healthy: {np.sum(eval_y[is_withheld] == 0)}")
    for cancer_type in training_cancer_types:
        count = np.sum(eval_strat[is_withheld] == cancer_type)
        if count > 0:
            print(f"       - {cancer_type}: {count}")
    
    print(f"     Novel cancer types: {np.sum(is_novel_cancer)}")
    novel_types = np.unique(eval_strat[is_novel_cancer])
    for cancer_type in novel_types:
        count = np.sum(eval_strat[is_novel_cancer] == cancer_type)
        print(f"       - {cancer_type}: {count}")
    
    # Step 3: Find and load best model
    print(f"\nStep 3: Loading best model from training...")
    
    # Look for best_model directory inside training_results_dir
    # Note: Don't use args.experiment_name here as it may differ between training and eval runs
    best_model_dirs = list(training_results_dir.glob("best_model_*"))

    if len(best_model_dirs) == 0:
        # Fallback: look for sibling best_model_* that matches training_results_dir name
        parent_dir = training_results_dir.parent
        sibling_dirs = [d for d in parent_dir.glob("best_model_*") if training_results_dir.name in d.name]
        if len(sibling_dirs) == 0:
            raise FileNotFoundError(f"No best_model_* directory found in {training_results_dir} or sibling dirs in {parent_dir}")
        elif len(sibling_dirs) > 1:
            print(f"Warning: Found {len(sibling_dirs)} matching sibling best_model directories, using first one")
            print(f"   Directories: {[d.name for d in sibling_dirs]}")
        best_model_dir = sibling_dirs[0]
        print(f"Using sibling model directory: {best_model_dir}")
    else:
        if len(best_model_dirs) > 1:
            print(f"Warning: Found {len(best_model_dirs)} best_model directories, using first one")
            print(f"   Directories: {[d.name for d in best_model_dirs]}")
        best_model_dir = best_model_dirs[0]
    print(f"   Using: {best_model_dir.name}")
    
    best_results = analyze_best_models_and_get_best(
        str(best_model_dir), 
        verbose=True,
        filter_feature=filter_feature,
        filter_n_genes=filter_n_genes
    )
    best_info = best_results['best_model_info']
    best_pipeline = best_results['best_model'].best_estimator_
    
    print(f"\n   Best model loaded:")
    print(f"     Feature: {best_info['feature']}")
    print(f"     N Genes: {best_info['n_genes']}")
    print(f"     Classifier: {best_info['classifier']}")
    print(f"     Training CV AUC: {best_info['cv_auc']:.4f}")
    
    # Step 4: Apply model to evaluation set
    print(f"\nStep 4: Evaluating model on {len(eval_ids)} samples...")
    
    feature_name = best_info['feature']
    X_eval = eval_features.get(feature_name)
    
    if X_eval is None:
        raise ValueError(f"Feature '{feature_name}' not found in evaluation data")
    
    y_pred_proba = best_pipeline.predict_proba(X_eval)[:, 1]
    y_pred = best_pipeline.predict(X_eval)
    
    # Step 5: Compute metrics
    print(f"\nStep 5: Computing evaluation metrics...")

    # (uses the module-level _sens_spec_at_95 helper)
    
    # Overall metrics
    fpr, tpr, _ = roc_curve(eval_y, y_pred_proba)
    overall_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(eval_y, y_pred_proba)
    report = classification_report(eval_y, y_pred, output_dict=True, zero_division=0)
    conf_matrix = confusion_matrix(eval_y, y_pred).tolist()
    overall_sens_95, overall_spec_95 = _sens_spec_at_95(fpr, tpr)
    
    # Metrics for withheld subset
    withheld_auc = None
    withheld_report = None
    if np.sum(is_withheld) > 0 and len(np.unique(eval_y[is_withheld])) == 2:
        fpr_w, tpr_w, _ = roc_curve(eval_y[is_withheld], y_pred_proba[is_withheld])
        withheld_auc = auc(fpr_w, tpr_w)
        withheld_report = classification_report(eval_y[is_withheld], y_pred[is_withheld], output_dict=True, zero_division=0)
        withheld_sens_95, withheld_spec_95 = _sens_spec_at_95(fpr_w, tpr_w)
    else:
        withheld_sens_95, withheld_spec_95 = None, None
    
    # Metrics for novel cancer types
    novel_auc = None
    novel_report = None
    if np.sum(is_novel_cancer) > 0:
        # For novel cancer types, we need healthy samples for comparison
        # Combine withheld healthy with novel cancer
        novel_eval_mask = is_withheld & (eval_y == 0) | is_novel_cancer  # Healthy from withheld + novel cancer
        if np.sum(novel_eval_mask) > 0 and len(np.unique(eval_y[novel_eval_mask])) == 2:
            fpr_n, tpr_n, _ = roc_curve(eval_y[novel_eval_mask], y_pred_proba[novel_eval_mask])
            novel_auc = auc(fpr_n, tpr_n)
            novel_report = classification_report(eval_y[novel_eval_mask], y_pred[novel_eval_mask], output_dict=True, zero_division=0)
            novel_sens_95, novel_spec_95 = _sens_spec_at_95(fpr_n, tpr_n)
        else:
            novel_sens_95, novel_spec_95 = None, None
    else:
        novel_sens_95, novel_spec_95 = None, None
    
    # Per-cancer-type metrics
    auc_per_type = {}
    cancer_types_in_eval = np.unique(eval_strat[eval_y == 1])
    healthy_samples_mask = eval_y == 0
    
    for cancer_type in cancer_types_in_eval:
        cancer_mask = eval_strat == cancer_type
        type_mask = healthy_samples_mask | cancer_mask
        
        if np.sum(type_mask) > 0 and len(np.unique(eval_y[type_mask])) == 2:
            fpr_ct, tpr_ct, _ = roc_curve(eval_y[type_mask], y_pred_proba[type_mask])
            auc_per_type[cancer_type] = {
                'auc': auc(fpr_ct, tpr_ct),
                'n_samples': np.sum(cancer_mask),
                'in_training': cancer_type in training_cancer_types
            }
    
    # Step 6: Print and save results
    print(f"\n{'='*90}")
    print("EVALUATION RESULTS")
    print(f"{'='*90}")
    print(f"\nOverall Performance (All {len(eval_ids)} samples):")
    print(f"  AUC: {overall_auc:.4f}")
    print(f"  Accuracy: {report['accuracy']:.4f}")
    print(f"  Sensitivity: {report['1']['recall']:.4f}")
    print(f"  Specificity: {report['0']['recall']:.4f}")
    
    if withheld_auc is not None:
        print(f"\nWithheld Set ({np.sum(is_withheld)} samples - same distribution as training):")
        print(f"  AUC: {withheld_auc:.4f}")
        print(f"  Accuracy: {withheld_report['accuracy']:.4f}")
        print(f"  Sensitivity: {withheld_report['1']['recall']:.4f}")
        print(f"  Specificity: {withheld_report['0']['recall']:.4f}")
    
    if novel_auc is not None:
        print(f"\nNovel Cancer Types ({np.sum(is_novel_cancer)} samples - generalization test):")
        print(f"  AUC: {novel_auc:.4f}")
        print(f"  Accuracy: {novel_report['accuracy']:.4f}")
        print(f"  Sensitivity: {novel_report['1']['recall']:.4f}")
    
    print(f"\nPer-Cancer-Type AUC:")
    for cancer_type in sorted(auc_per_type.keys()):
        info = auc_per_type[cancer_type]
        marker = "training" if info['in_training'] else "novel"
        print(f"  {marker} | {cancer_type}: {info['auc']:.4f} (n={info['n_samples']})")
    
    # Save results
    output_dir = Path(args.output_dir) / f"within_dataset_generalization_eval_{args.experiment_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 6: Create comprehensive evaluation visualizations
    print(f"\nStep 6: Creating comprehensive evaluation visualizations...")
    
    
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    # Plot 1: ROC Curves (Overall, Withheld, Novel)
    ax1 = fig.add_subplot(gs[0, :2])
    
    ax1.plot(fpr, tpr, linewidth=2.5, label=f'Overall (AUC={overall_auc:.3f})', color='steelblue')
    
    if withheld_auc is not None:
        fpr_w, tpr_w, _ = roc_curve(eval_y[is_withheld], y_pred_proba[is_withheld])
        ax1.plot(fpr_w, tpr_w, linewidth=2, label=f'Withheld - Same Distribution (AUC={withheld_auc:.3f})', 
                color='green', linestyle='--')
    
    if novel_auc is not None:
        novel_eval_mask = (is_withheld & (eval_y == 0)) | is_novel_cancer
        fpr_n, tpr_n, _ = roc_curve(eval_y[novel_eval_mask], y_pred_proba[novel_eval_mask])
        ax1.plot(fpr_n, tpr_n, linewidth=2, label=f'Novel Cancer Types (AUC={novel_auc:.3f})', 
                color='orange', linestyle=':')
    
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
    ax1.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax1.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax1.set_title('ROC Curves: Withheld vs Novel Cancer Types', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    
    # Plot 2: Performance Metrics Comparison (bar chart)
    ax2 = fig.add_subplot(gs[0, 2])
    
    metrics_names = ['AUC', 'Accuracy', 'Sens@95%Spec', 'Spec@95%Spec']
    overall_metrics = [overall_auc, report['accuracy'], overall_sens_95 or 0, overall_spec_95 or 0]
    withheld_metrics = [withheld_auc if withheld_auc else 0,
                       withheld_report['accuracy'] if withheld_report else 0,
                       withheld_sens_95 or 0,
                       withheld_spec_95 or 0]
    novel_metrics = [novel_auc if novel_auc else 0,
                    novel_report['accuracy'] if novel_report else 0,
                    novel_sens_95 or 0,
                    novel_spec_95 or 0]
    
    x = np.arange(len(metrics_names))
    width = 0.25
    
    ax2.bar(x - width, overall_metrics, width, label='Overall', color='steelblue', alpha=0.7)
    ax2.bar(x, withheld_metrics, width, label='Withheld', color='green', alpha=0.7)
    ax2.bar(x + width, novel_metrics, width, label='Novel', color='orange', alpha=0.7)
    
    ax2.set_ylabel('Score', fontweight='bold')
    ax2.set_title('Performance Metrics\nComparison', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics_names, rotation=45, ha='right', fontsize=9)
    ax2.legend(fontsize=9)
    ax2.set_ylim([0, 1])
    ax2.grid(axis='y', alpha=0.3)
    
    # Plot 3: AUC by Cancer Type (Training vs Novel)
    ax3 = fig.add_subplot(gs[1, :2])
    
    training_types = [ct for ct in sorted(auc_per_type.keys()) if auc_per_type[ct]['in_training']]
    novel_types_list = [ct for ct in sorted(auc_per_type.keys()) if not auc_per_type[ct]['in_training']]
    
    all_types = training_types + novel_types_list
    all_aucs = [auc_per_type[ct]['auc'] for ct in all_types]
    colors_list = ['green' if ct in training_types else 'orange' for ct in all_types]
    
    bars = ax3.barh(range(len(all_types)), all_aucs, color=colors_list, alpha=0.7, edgecolor='black')
    ax3.set_yticks(range(len(all_types)))
    ax3.set_yticklabels([ct.replace(' cancer', '') for ct in all_types], fontsize=10)
    ax3.set_xlabel('AUC', fontsize=12, fontweight='bold')
    ax3.set_title('Per-Cancer-Type Performance (Green=Training, Orange=Novel)', 
                 fontsize=12, fontweight='bold')
    ax3.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Random')
    ax3.set_xlim([0, 1])
    ax3.grid(axis='x', alpha=0.3)
    
    # Add value labels and sample counts
    for i, (ct, auc_val) in enumerate(zip(all_types, all_aucs)):
        n_samples = auc_per_type[ct]['n_samples']
        ax3.text(auc_val + 0.02, i, f'{auc_val:.3f} (n={n_samples})', 
                va='center', fontsize=9)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='green', alpha=0.7, label='Training Types'),
                      Patch(facecolor='orange', alpha=0.7, label='Novel Types')]
    ax3.legend(handles=legend_elements, loc='lower right')
    
    # Plot 4: Sample Distribution in Evaluation
    ax4 = fig.add_subplot(gs[1, 2])
    
    eval_composition = [
        np.sum(is_withheld & (eval_y == 0)),  # Withheld healthy
        np.sum(is_withheld & (eval_y == 1)),  # Withheld cancer
        np.sum(is_novel_cancer)                # Novel cancer
    ]
    eval_labels = [
        f'Withheld\nHealthy\n({eval_composition[0]})',
        f'Withheld\nCancer\n({eval_composition[1]})',
        f'Novel\nCancer\n({eval_composition[2]})'
    ]
    eval_colors = ['lightgreen', 'lightblue', 'orange']
    
    ax4.pie(eval_composition, labels=eval_labels, autopct='%1.1f%%',
           colors=eval_colors, startangle=90)
    ax4.set_title('Evaluation Set\nComposition', fontweight='bold')
    
    # Plot 5: Prediction Score Distributions
    ax5 = fig.add_subplot(gs[2, 0])
    
    # Separate scores by category
    withheld_healthy_scores = y_pred_proba[is_withheld & (eval_y == 0)]
    withheld_cancer_scores = y_pred_proba[is_withheld & (eval_y == 1)]
    novel_cancer_scores = y_pred_proba[is_novel_cancer]
    
    bins = np.linspace(0, 1, 30)
    ax5.hist(withheld_healthy_scores, bins=bins, alpha=0.5, label='Withheld Healthy', 
            color='green', edgecolor='black')
    ax5.hist(withheld_cancer_scores, bins=bins, alpha=0.5, label='Withheld Cancer', 
            color='blue', edgecolor='black')
    ax5.hist(novel_cancer_scores, bins=bins, alpha=0.5, label='Novel Cancer', 
            color='orange', edgecolor='black')
    
    ax5.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Threshold')
    ax5.set_xlabel('Prediction Score (Cancer Probability)', fontweight='bold')
    ax5.set_ylabel('Frequency', fontweight='bold')
    ax5.set_title('Prediction Score\nDistributions', fontsize=11, fontweight='bold')
    ax5.legend(fontsize=8)
    ax5.grid(axis='y', alpha=0.3)
    
    # Plot 6: Confusion Matrix Heatmap (Overall)
    ax6 = fig.add_subplot(gs[2, 1])
    
    conf_array = np.array(conf_matrix)
    im = ax6.imshow(conf_array, cmap='Blues', aspect='auto')
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax6.text(j, i, str(conf_array[i, j]),
                          ha="center", va="center", color="white" if conf_array[i, j] > conf_array.max()/2 else "black",
                          fontsize=16, fontweight='bold')
    
    ax6.set_xticks([0, 1])
    ax6.set_yticks([0, 1])
    ax6.set_xticklabels(['Healthy', 'Cancer'])
    ax6.set_yticklabels(['Healthy', 'Cancer'])
    ax6.set_xlabel('Predicted', fontweight='bold')
    ax6.set_ylabel('Actual', fontweight='bold')
    ax6.set_title('Confusion Matrix\n(Overall)', fontweight='bold')
    
    # Plot 7: Generalization Summary (Training AUC vs Withheld vs Novel)
    ax7 = fig.add_subplot(gs[2, 2])
    
    comparison_categories = ['Training\nCV', 'Withheld\nSame Dist', 'Novel\nCancer Types']
    comparison_aucs = [
        best_info['cv_auc'],
        withheld_auc if withheld_auc else 0,
        novel_auc if novel_auc else 0
    ]
    comparison_colors = ['steelblue', 'green', 'orange']
    
    bars = ax7.bar(comparison_categories, comparison_aucs, color=comparison_colors, 
                  alpha=0.7, edgecolor='black', linewidth=2)
    ax7.set_ylabel('AUC', fontweight='bold')
    ax7.set_title('Generalization\nPerformance', fontsize=11, fontweight='bold')
    ax7.set_ylim([0, 1])
    ax7.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    ax7.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, comparison_aucs):
        if val > 0:
            ax7.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Calculate and show generalization gap
    if withheld_auc:
        gap_same = best_info['cv_auc'] - withheld_auc
        ax7.text(0.5, 0.15, f'Gap (same dist):\n{gap_same:.3f}', 
                ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    if novel_auc:
        gap_novel = best_info['cv_auc'] - novel_auc
        ax7.text(0.5, 0.05, f'Gap (novel):\n{gap_novel:.3f}',
                ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    # Overall title
    training_cancer_str = ', '.join([ct.replace(' cancer', '') for ct in training_cancer_types])
    fig.suptitle(f'Within-Dataset Generalization Evaluation\nTrained on: {training_cancer_str} | Feature: {best_info["feature"]} | N Genes: {best_info["n_genes"]}',
                fontsize=14, fontweight='bold', y=0.995)
    
    # Save comprehensive plot
    plot_path = output_dir / 'comprehensive_evaluation_analysis.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   Comprehensive evaluation plot saved: {plot_path}")
    
    # Create additional plot: Per-cancer-type detailed breakdown
    fig2, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Top plot: AUC with confidence
    ax_top = axes[0]
    training_aucs = [auc_per_type[ct]['auc'] for ct in training_types]
    novel_aucs = [auc_per_type[ct]['auc'] for ct in novel_types_list]
    
    x_train = np.arange(len(training_types))
    x_novel = np.arange(len(novel_types_list)) + len(training_types) + 0.5
    
    ax_top.bar(x_train, training_aucs, color='green', alpha=0.7, edgecolor='black', 
              label='Training Cancer Types', width=0.8)
    ax_top.bar(x_novel, novel_aucs, color='orange', alpha=0.7, edgecolor='black',
              label='Novel Cancer Types', width=0.8)
    
    ax_top.set_xticks(list(x_train) + list(x_novel))
    ax_top.set_xticklabels([ct.replace(' cancer', '') for ct in training_types + novel_types_list],
                          rotation=45, ha='right')
    ax_top.set_ylabel('AUC', fontsize=12, fontweight='bold')
    ax_top.set_title('Detailed Per-Cancer-Type Performance', fontsize=14, fontweight='bold')
    ax_top.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random')
    ax_top.axhline(y=best_info['cv_auc'], color='steelblue', linestyle=':', alpha=0.7, 
                  label=f'Training CV AUC ({best_info["cv_auc"]:.3f})')
    ax_top.legend()
    ax_top.grid(axis='y', alpha=0.3)
    ax_top.set_ylim([0, 1])
    
    # Add value labels and sample counts
    for i, ct in enumerate(training_types):
        auc_val = auc_per_type[ct]['auc']
        n = auc_per_type[ct]['n_samples']
        ax_top.text(i, auc_val + 0.02, f'{auc_val:.2f}\n(n={n})', 
                   ha='center', va='bottom', fontsize=8)
    
    for i, ct in enumerate(novel_types_list):
        auc_val = auc_per_type[ct]['auc']
        n = auc_per_type[ct]['n_samples']
        ax_top.text(len(training_types) + 0.5 + i, auc_val + 0.02, f'{auc_val:.2f}\n(n={n})',
                   ha='center', va='bottom', fontsize=8)
    
    # Bottom plot: Sample counts
    ax_bottom = axes[1]
    training_counts = [auc_per_type[ct]['n_samples'] for ct in training_types]
    novel_counts = [auc_per_type[ct]['n_samples'] for ct in novel_types_list]
    
    ax_bottom.bar(x_train, training_counts, color='green', alpha=0.7, edgecolor='black', width=0.8)
    ax_bottom.bar(x_novel, novel_counts, color='orange', alpha=0.7, edgecolor='black', width=0.8)
    
    ax_bottom.set_xticks(list(x_train) + list(x_novel))
    ax_bottom.set_xticklabels([ct.replace(' cancer', '') for ct in training_types + novel_types_list],
                             rotation=45, ha='right')
    ax_bottom.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    ax_bottom.set_title('Sample Counts per Cancer Type', fontsize=12, fontweight='bold')
    ax_bottom.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, count in enumerate(training_counts):
        ax_bottom.text(i, count + 1, str(count), ha='center', va='bottom', fontweight='bold')
    for i, count in enumerate(novel_counts):
        ax_bottom.text(len(training_types) + 0.5 + i, count + 1, str(count), 
                      ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    detailed_plot_path = output_dir / 'detailed_cancer_type_analysis.png'
    plt.savefig(detailed_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   Detailed cancer-type plot saved: {detailed_plot_path}")
    
    # Save results dictionary
    results = {
        'training_results_dir': str(training_results_dir),
        'training_cancer_types': training_cancer_types,
        'best_model_info': best_info,
        
        # Evaluation set composition
        'eval_n_samples': len(eval_ids),
        'eval_n_withheld': int(np.sum(is_withheld)),
        'eval_n_novel_cancer': int(np.sum(is_novel_cancer)),
        'novel_cancer_types': novel_types.tolist(),
        
        # Overall metrics
        'overall_auc': float(overall_auc),
        'overall_fpr': fpr.tolist(),
        'overall_tpr': tpr.tolist(),
        'overall_sens_at_95spec': float(overall_sens_95) if overall_sens_95 is not None else None,
        'overall_spec_at_95spec': float(overall_spec_95) if overall_spec_95 is not None else None,
        'overall_precision': precision.tolist(),
        'overall_recall': recall.tolist(),
        'overall_classification_report': report,
        'overall_confusion_matrix': conf_matrix,
        
        # Withheld set metrics
        'withheld_auc': float(withheld_auc) if withheld_auc else None,
        'withheld_report': withheld_report,
        'withheld_sens_at_95spec': float(withheld_sens_95) if withheld_sens_95 is not None else None,
        'withheld_spec_at_95spec': float(withheld_spec_95) if withheld_spec_95 is not None else None,
        
        # Novel cancer types metrics
        'novel_auc': float(novel_auc) if novel_auc else None,
        'novel_report': novel_report,
        'novel_sens_at_95spec': float(novel_sens_95) if novel_sens_95 is not None else None,
        'novel_spec_at_95spec': float(novel_spec_95) if novel_spec_95 is not None else None,
        
        # Per-cancer-type metrics
        'auc_per_cancer_type': auc_per_type,
        
        # Sample tracking
        'withheld_sample_ids': eval_ids[is_withheld].tolist(),
        'novel_cancer_sample_ids': eval_ids[is_novel_cancer].tolist(),
    }
    
    output_file = output_dir / f"generalization_eval_{feature_name}_results.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    
    # Print comprehensive summary
    print(f"\n{'='*90}")
    print("KEY INSIGHTS & SUMMARY")
    print(f"{'='*90}")
    
    print(f"\nOverall Results:")
    print(f"   Total evaluated: {len(eval_ids)} samples")
    print(f"   Overall AUC: {overall_auc:.4f}")
    print(f"   Accuracy: {report['accuracy']:.4f}")
    
    print(f"\nSame Distribution (Withheld Set):")
    if withheld_auc:
        gap_same = best_info['cv_auc'] - withheld_auc
        print(f"   AUC: {withheld_auc:.4f}")
        print(f"   Gap from training: {gap_same:.4f} ({gap_same/best_info['cv_auc']*100:.1f}% performance drop)")
        print(f"   Interpretation: {'Good' if gap_same < 0.05 else 'Moderate' if gap_same < 0.15 else 'Large'} generalization gap")
    else:
        print(f"   Not enough samples for evaluation")
    
    print(f"\nNovel Cancer Types (True Generalization):")
    if novel_auc:
        gap_novel = best_info['cv_auc'] - novel_auc
        print(f"   AUC: {novel_auc:.4f}")
        print(f"   Gap from training: {gap_novel:.4f} ({gap_novel/best_info['cv_auc']*100:.1f}% performance drop)")
        print(f"   Interpretation: Model {'generalizes well' if gap_novel < 0.15 else 'shows moderate generalization' if gap_novel < 0.30 else 'struggles'} to novel cancers")
        print(f"   Novel cancer types tested: {', '.join([ct.replace(' cancer', '') for ct in novel_types.tolist()])}")
    else:
        print(f"   Not enough samples for evaluation")
    
    print(f"\nBest Performing Cancer Types (Top 3):")
    sorted_cancers = sorted(auc_per_type.items(), key=lambda x: x[1]['auc'], reverse=True)[:3]
    for ct, info in sorted_cancers:
        marker = "[training]" if info['in_training'] else "[novel]"
        print(f"   {marker} {ct.replace(' cancer', '')}: AUC={info['auc']:.4f} (n={info['n_samples']})")
    
    print(f"\n Challenging Cancer Types (Bottom 3):")
    bottom_cancers = sorted(auc_per_type.items(), key=lambda x: x[1]['auc'])[:3]
    for ct, info in bottom_cancers:
        marker = "[training]" if info['in_training'] else "[novel]"
        print(f"   {marker} {ct.replace(' cancer', '')}: AUC={info['auc']:.4f} (n={info['n_samples']})")
    
    print(f"\nGeneralization Performance Comparison:")
    print(f"   Training CV AUC:    {best_info['cv_auc']:.4f}")
    if withheld_auc:
        print(f"   Withheld AUC:       {withheld_auc:.4f}  (gap: {best_info['cv_auc'] - withheld_auc:+.4f})")
    if novel_auc:
        print(f"   Novel Cancer AUC:   {novel_auc:.4f}  (gap: {best_info['cv_auc'] - novel_auc:+.4f})")
    
    print(f"\nRecommendations:")
    if withheld_auc and novel_auc:
        if gap_same < 0.05 and gap_novel < 0.15:
            print(f"   Excellent: Model generalizes well to both same distribution and novel cancers")
        elif gap_same < 0.10 and gap_novel < 0.25:
            print(f"   Good: Reasonable generalization, suitable for deployment with monitoring")
        elif gap_novel > gap_same + 0.10:
            print(f"    Model performance drops significantly on novel cancer types")
            print(f"      Consider: Training on more diverse cancer types or feature engineering")
        else:
            print(f"    Moderate generalization gap observed")
            print(f"      Consider: Hyperparameter tuning or regularization adjustments")
    
    print(f"\nOutput Files:")
    print(f"   Results: {output_file}")
    print(f"   Comprehensive plot: {plot_path}")
    print(f"   Detailed analysis: {detailed_plot_path}")
    
    print(f"\nWITHIN-DATASET GENERALIZATION EVALUATION COMPLETE")
    print("="*90)
    
    return results


def run_comprehensive_within_dataset_evaluation(
    training_results_dir: str,
    all_feature_arrays: dict,
    y_labels: np.ndarray,
    stratification_labels: np.ndarray,
    sample_ids: np.ndarray,
    args: argparse.Namespace
):
    """
    Comprehensive within-dataset generalization evaluation analyzing:
    1. Feature types (best model per feature)
    2. Number of genes (best model per top_genes value)
    
    Creates visualizations comparing generalization to novel cancer types to help identify:
    - Which features generalize better to unseen cancer types
    - Whether models with fewer genes generalize better or worse
    - Generalization gap patterns across features and model complexity
    
    Args:
        Same as run_within_dataset_generalization_evaluation
    
    Returns:
        dict: Results containing per-feature and per-gene-count analysis with plots
    """
    print("\n" + "="*90)
    print("COMPREHENSIVE WITHIN-DATASET GENERALIZATION ANALYSIS")
    print("="*90)
    print("\nThis will evaluate:")
    print("  1. Best model for each feature type (generalization to novel cancer types)")
    print("  2. Best model for each gene count (model complexity vs. generalization)")
    
    training_results_dir = Path(training_results_dir)
    use_gene_counts = not (args.no_gene_selection or args.predefined_feature_matrix)
    
    # Setup output directory - save directly in training directory
    output_dir = training_results_dir / "comprehensive_evaluation"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Saving results to: {output_dir}")
    
    # Get available features and gene counts from training models
    print(f"\nScanning available models in training results...")
    # Look for best_model_* subdirectory (created by run_best_model_search)
    best_model_dirs = list(training_results_dir.glob("best_model_*"))

    if len(best_model_dirs) == 0:
        # Fallback: look for sibling best_model_* dirs that match this training dir name
        parent_dir = training_results_dir.parent
        sibling_dirs = [d for d in parent_dir.glob("best_model_*") if training_results_dir.name in d.name]
        if len(sibling_dirs) == 0:
            raise FileNotFoundError(
                f"No best_model_* directory found in {training_results_dir} or sibling dirs in {parent_dir}"
            )
        elif len(sibling_dirs) > 1:
            print(f"Warning: Found {len(sibling_dirs)} matching sibling best_model directories, using first one")
            print(f"   Directories: {[d.name for d in sibling_dirs]}")
        best_model_dir = sibling_dirs[0]
        print(f"Using sibling model directory: {best_model_dir}")
    else:
        if len(best_model_dirs) > 1:
            print(f"Warning: Found {len(best_model_dirs)} best_model directories, using first one")
            print(f"   Directories: {[d.name for d in best_model_dirs]}")
        best_model_dir = best_model_dirs[0]
    print(f"   Using model directory: {best_model_dir.name}")
    
    if not best_model_dir.exists():
        raise FileNotFoundError(f"Best model directory not found: {best_model_dir}")
    
    gridsearch_files = list(best_model_dir.glob("best_model_GridSearchCV_*.pkl"))
    
    if len(gridsearch_files) == 0:
        raise FileNotFoundError(f"No model files found in {best_model_dir}")
    
    # Extract available features and gene counts
    available_features = set()
    available_gene_counts = set()
    
    for pkl_file in gridsearch_files:
        try:
            # Extract feature name and gene count from filename
            filename = pkl_file.stem.replace('best_model_GridSearchCV_', '')
            parts = filename.split('_')
            
            # Find where gene count starts (first numeric part)
            for i, part in enumerate(parts):
                if part.isdigit():
                    feature_name = '_'.join(parts[:i])
                    n_genes = int(part)
                    available_features.add(feature_name)
                    available_gene_counts.add(n_genes)
                    break
        except Exception as e:
            print(f"  Could not parse {pkl_file.name}: {e}")
            continue
    
    print(f"   Found {len(available_features)} features: {sorted(available_features)}")
    print(f"   Found {len(available_gene_counts)} gene counts: {sorted(available_gene_counts)}")
    if not use_gene_counts:
        print("No-gene-selection/predefined mode: gene count analysis will be skipped")
    
    # Prepare evaluation masks once
    print("\nPreparing evaluation masks for within-dataset generalization...")
    withheld_file = training_results_dir / "withheld_sample_info.pkl"
    if not withheld_file.exists():
        raise FileNotFoundError(f"Withheld sample info not found: {withheld_file}")

    with open(withheld_file, 'rb') as f:
        withheld_info = pickle.load(f)

    training_cancer_types = withheld_info['training_cancer_types']
    sample_ids_array = np.array(sample_ids)
    withheld_sample_ids = set(withheld_info['withheld_sample_ids'])
    training_feature_stats = withheld_info.get('training_feature_stats')

    withheld_mask = np.isin(sample_ids_array, list(withheld_sample_ids))
    novel_cancer_mask = (
        (y_labels == 1) &
        (~np.isin(stratification_labels, training_cancer_types)) &
        (~withheld_mask)
    )
    eval_mask = withheld_mask | novel_cancer_mask

    eval_features = {k: v[eval_mask] if v is not None else None for k, v in all_feature_arrays.items()}
    eval_y = y_labels[eval_mask]
    eval_strat = stratification_labels[eval_mask]
    is_withheld = withheld_mask[eval_mask]
    is_novel_cancer = novel_cancer_mask[eval_mask]

    no_novel_cancers = np.sum(is_novel_cancer) == 0
    if no_novel_cancers:
        print("No novel cancer types available in evaluation; using withheld metrics for summaries")

    # Filter available features to those present in evaluation data
    eval_feature_set = {k for k, v in eval_features.items() if v is not None}
    if eval_feature_set:
        missing_features = sorted(set(available_features) - eval_feature_set)
        if missing_features:
            print(f"Skipping {len(missing_features)} feature(s) with no eval data: {missing_features}")
        available_features = {f for f in available_features if f in eval_feature_set}
    else:
        print("No evaluation features available after filtering")
        available_features = set()

    if training_feature_stats:
        print("\nApplying training-only standardization to evaluation data")
        eval_features = standardize_with_training_stats(eval_features, training_feature_stats)

    # (uses the module-level _sens_spec_at_95 helper)

    def _bootstrap_roc_ci(y_true, y_score, n_boot=200, seed=42, fpr_grid=None):
        if y_true is None or y_score is None:
            return None
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)
        if y_true.size == 0 or y_score.size == 0:
            return None
        if fpr_grid is None:
            fpr_grid = np.linspace(0.0, 1.0, 101)
        rng = np.random.RandomState(seed)
        tprs = []
        n = y_true.shape[0]
        for _ in range(n_boot):
            idx = rng.randint(0, n, n)
            y_b = y_true[idx]
            s_b = y_score[idx]
            if len(np.unique(y_b)) < 2:
                continue
            fpr_b, tpr_b, _ = roc_curve(y_b, s_b)
            tpr_interp = np.interp(fpr_grid, fpr_b, tpr_b)
            tpr_interp[0] = 0.0
            tprs.append(tpr_interp)
        if len(tprs) == 0:
            return None
        tprs = np.vstack(tprs)
        lower = np.percentile(tprs, 2.5, axis=0)
        upper = np.percentile(tprs, 97.5, axis=0)
        return {
            'fpr': fpr_grid.tolist(),
            'tpr_lower': lower.tolist(),
            'tpr_upper': upper.tolist()
        }

    def _evaluate_feature_k(feature: str, n_genes: Optional[int] = None):
        try:
            best_results = analyze_best_models_and_get_best(
                str(best_model_dir),
                verbose=False,
                filter_feature=feature,
                filter_n_genes=n_genes if n_genes is not None else None
            )
        except Exception as e:
            print(f"    Skipping {feature} @ {n_genes}: {e}")
            return None

        best_info = best_results['best_model_info']
        best_pipeline = best_results['best_model'].best_estimator_

        X_eval = eval_features.get(feature)
        if X_eval is None:
            print(f"    No eval data for feature '{feature}'")
            return None

        y_pred_proba = best_pipeline.predict_proba(X_eval)[:, 1]
        y_pred = best_pipeline.predict(X_eval)

        # Overall
        fpr, tpr, _ = roc_curve(eval_y, y_pred_proba)
        overall_auc = auc(fpr, tpr)
        report = classification_report(eval_y, y_pred, output_dict=True, zero_division=0)
        overall_precision = float(report['1']['precision']) if report.get('1') else None
        overall_recall = float(report['1']['recall']) if report.get('1') else None
        overall_f1 = float(report['1']['f1-score']) if report.get('1') else None
        overall_accuracy = float(report['accuracy']) if report.get('accuracy') is not None else None
        overall_sens_95, overall_spec_95 = _sens_spec_at_95(fpr, tpr)
        overall_roc_ci = _bootstrap_roc_ci(eval_y, y_pred_proba)

        # Per-cancer-type AUC (healthy vs each cancer type)
        auc_per_cancer_type = {}
        cancer_types_in_eval = np.unique(eval_strat[eval_y == 1])
        healthy_samples_mask = eval_y == 0
        for cancer_type in cancer_types_in_eval:
            cancer_mask = eval_strat == cancer_type
            type_mask = healthy_samples_mask | cancer_mask
            if np.sum(type_mask) > 0 and len(np.unique(eval_y[type_mask])) == 2:
                fpr_ct, tpr_ct, _ = roc_curve(eval_y[type_mask], y_pred_proba[type_mask])
                auc_per_cancer_type[cancer_type] = {
                    'auc': float(auc(fpr_ct, tpr_ct)),
                    'n_samples': int(np.sum(cancer_mask)),
                    'in_training': bool(cancer_type in training_cancer_types)
                }

        # Withheld subset
        withheld_auc = None
        withheld_report = None
        if np.sum(is_withheld) > 0 and len(np.unique(eval_y[is_withheld])) == 2:
            fpr_w, tpr_w, _ = roc_curve(eval_y[is_withheld], y_pred_proba[is_withheld])
            withheld_auc = auc(fpr_w, tpr_w)
            withheld_report = classification_report(eval_y[is_withheld], y_pred[is_withheld], output_dict=True, zero_division=0)
            withheld_sens_95, withheld_spec_95 = _sens_spec_at_95(fpr_w, tpr_w)
            withheld_roc_ci = _bootstrap_roc_ci(eval_y[is_withheld], y_pred_proba[is_withheld])
        else:
            withheld_sens_95, withheld_spec_95 = None, None
            withheld_roc_ci = None

        # Novel cancer types (with withheld healthy for comparison)
        novel_auc = None
        novel_report = None
        fpr_n = None
        tpr_n = None
        if np.sum(is_novel_cancer) > 0:
            novel_eval_mask = (is_withheld & (eval_y == 0)) | is_novel_cancer
            if np.sum(novel_eval_mask) > 0 and len(np.unique(eval_y[novel_eval_mask])) == 2:
                fpr_n, tpr_n, _ = roc_curve(eval_y[novel_eval_mask], y_pred_proba[novel_eval_mask])
                novel_auc = auc(fpr_n, tpr_n)
                novel_report = classification_report(eval_y[novel_eval_mask], y_pred[novel_eval_mask], output_dict=True, zero_division=0)
                novel_sens_95, novel_spec_95 = _sens_spec_at_95(fpr_n, tpr_n)
                novel_roc_ci = _bootstrap_roc_ci(eval_y[novel_eval_mask], y_pred_proba[novel_eval_mask])
            else:
                novel_sens_95, novel_spec_95 = None, None
                novel_roc_ci = None
        else:
            novel_sens_95, novel_spec_95 = None, None
            novel_roc_ci = None

        if no_novel_cancers and novel_auc is None and withheld_auc is not None:
            novel_auc = withheld_auc
            novel_report = withheld_report
            fpr_n, tpr_n = (fpr_w, tpr_w) if withheld_auc is not None else (None, None)
            novel_sens_95, novel_spec_95 = withheld_sens_95, withheld_spec_95
            novel_roc_ci = withheld_roc_ci

        return {
            'best_model_info': best_info,
            'overall_auc': float(overall_auc),
            'withheld_auc': float(withheld_auc) if withheld_auc is not None else None,
            'novel_auc': float(novel_auc) if novel_auc is not None else None,
            'overall_report': report,
            'withheld_report': withheld_report,
            'novel_report': novel_report,
            'overall_roc': {'fpr': fpr.tolist(), 'tpr': tpr.tolist()},
            'withheld_roc': {'fpr': fpr_w.tolist(), 'tpr': tpr_w.tolist()} if withheld_auc is not None else None,
            'novel_roc': {'fpr': fpr_n.tolist(), 'tpr': tpr_n.tolist()} if novel_auc is not None else None,
            'overall_roc_ci': overall_roc_ci,
            'withheld_roc_ci': withheld_roc_ci,
            'novel_roc_ci': novel_roc_ci,
            'overall_precision': overall_precision,
            'overall_recall': overall_recall,
            'overall_f1': overall_f1,
            'overall_accuracy': overall_accuracy,
            'overall_sens_at_95spec': float(overall_sens_95) if overall_sens_95 is not None else None,
            'overall_spec_at_95spec': float(overall_spec_95) if overall_spec_95 is not None else None,
            'auc_per_cancer_type': auc_per_cancer_type,
            'withheld_sens_at_95spec': float(withheld_sens_95) if withheld_sens_95 is not None else None,
            'withheld_spec_at_95spec': float(withheld_spec_95) if withheld_spec_95 is not None else None,
            'novel_sens_at_95spec': float(novel_sens_95) if novel_sens_95 is not None else None,
            'novel_spec_at_95spec': float(novel_spec_95) if novel_spec_95 is not None else None
        }

    # Part 1: Evaluate ALL feature x gene count combinations
    print("\n" + "="*90)
    print("PART 1: GENERALIZATION ACROSS ALL FEATURES x GENE COUNTS")
    print("="*90)

    grid_results = {feat: {} for feat in sorted(available_features)}
    for feature in sorted(available_features):
        print(f"\n{'-'*90}")
        print(f"Evaluating feature: {feature}")
        print(f"{'-'*90}")
        if use_gene_counts:
            for n_genes in sorted(available_gene_counts):
                res = _evaluate_feature_k(feature, n_genes)
                grid_results[feature][n_genes] = res
        else:
            res = _evaluate_feature_k(feature, None)
            grid_results[feature][None] = res

    # Part 2: Summarize by feature (best model per feature)
    print("\n" + "="*90)
    print("PART 2: GENERALIZATION BY FEATURE TYPE (BEST MODEL PER FEATURE)")
    print("="*90)

    feature_results = {}
    for feature, by_k in grid_results.items():
        if no_novel_cancers:
            valid = [v for v in by_k.values() if v and v.get('withheld_auc') is not None]
        else:
            valid = [v for v in by_k.values() if v and v.get('novel_auc') is not None]
        if not valid:
            feature_results[feature] = None
            continue
        if no_novel_cancers:
            best_res = max(valid, key=lambda r: r['withheld_auc'])
        else:
            best_res = max(valid, key=lambda r: r['novel_auc'])
        feature_results[feature] = best_res
        if no_novel_cancers:
            print(f"{feature}: best withheld AUC = {best_res.get('withheld_auc'):.4f}")
        else:
            print(f"{feature}: best novel AUC = {best_res.get('novel_auc'):.4f}")

    # Part 3: Summarize by gene count (best model at each K across all features)
    if use_gene_counts:
        print("\n" + "="*90)
        print("PART 3: GENERALIZATION BY NUMBER OF GENES (BEST MODEL PER K)")
        print("="*90)

        gene_count_results = {}
        for n_genes in sorted(available_gene_counts):
            candidates = []
            for feature in sorted(available_features):
                res = grid_results.get(feature, {}).get(n_genes)
                if res and res.get('novel_auc') is not None:
                    candidates.append(res)
            if candidates:
                best_res = max(candidates, key=lambda r: r['novel_auc'])
                gene_count_results[n_genes] = best_res
                print(f"{n_genes} genes: best novel AUC = {best_res.get('novel_auc'):.4f} (feature={best_res['best_model_info']['feature']})")
            else:
                gene_count_results[n_genes] = None
    else:
        gene_count_results = {}
    
    # Part 4: Create comprehensive comparison visualizations (aligned with external validation layout)
    print("\n" + "="*90)
    print("PART 4: GENERATING COMPREHENSIVE COMPARISON PLOTS")
    print("="*90)


    def _get_report_for_metrics(res):
        if res is None:
            return None
        return res.get('novel_report') or res.get('overall_report')

    def _get_sens_spec_95(res):
        if res is None:
            return 0, 0
        for prefix in ['novel', 'withheld', 'overall']:
            sens = res.get(f'{prefix}_sens_at_95spec')
            spec = res.get(f'{prefix}_spec_at_95spec')
            if sens is not None and spec is not None:
                return sens, spec
        return 0, 0

    # Create main comparison figure (3x3) aligned to external validation
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # --- Plot 1: AUC by Feature Type ---
    ax1 = fig.add_subplot(gs[0, :2])

    feature_names = []
    novel_aucs_feat = []
    feature_auc_std = []
    feature_sens = []
    feature_spec = []
    feature_sens_std = []
    feature_spec_std = []
    train_aucs_feat = []

    metric_label = "Withheld AUC" if no_novel_cancers else "Novel AUC"
    for feat in sorted(feature_results.keys()):
        res = feature_results[feat]
        if res is None:
            continue
        feature_names.append(feat)
        novel_aucs_feat.append((res.get('withheld_auc') if no_novel_cancers else res.get('novel_auc')) or 0)
        feature_auc_std.append((res.get('withheld_auc_std') if no_novel_cancers else res.get('novel_auc_std')) or 0)
        train_aucs_feat.append(res['best_model_info']['cv_auc'])
        sens_95, spec_95 = _get_sens_spec_95(res)
        feature_sens.append(sens_95)
        feature_spec.append(spec_95)
        feature_sens_std.append((res.get('withheld_sens_at_95spec_std') if no_novel_cancers else res.get('novel_sens_at_95spec_std')) or 0)
        feature_spec_std.append((res.get('withheld_spec_at_95spec_std') if no_novel_cancers else res.get('novel_spec_at_95spec_std')) or 0)

    x_pos = np.arange(len(feature_names))
    ax1.bar(x_pos, novel_aucs_feat, yerr=feature_auc_std, capsize=4, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.set_xlabel('Feature Type', fontsize=12, fontweight='bold')
    ax1.set_ylabel(metric_label, fontsize=12, fontweight='bold')
    ax1.set_title(f'Generalization Performance by Feature Type ({metric_label})', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(feature_names, rotation=45, ha='right')
    ax1.set_ylim([0, 1])
    ax1.grid(axis='y', alpha=0.3)
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random')

    for i, auc_val in enumerate(novel_aucs_feat):
        ax1.text(i, auc_val + 0.02, f'{auc_val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax1.legend()

    # --- Plot 2: Sensitivity & Specificity by Feature ---
    ax2 = fig.add_subplot(gs[0, 2])

    width = 0.35
    ax2.bar(x_pos - width/2, feature_sens, width, yerr=feature_sens_std, capsize=4, label='Sens @95% Spec', alpha=0.7, color='green', edgecolor='black')
    ax2.bar(x_pos + width/2, feature_spec, width, yerr=feature_spec_std, capsize=4, label='Spec @95% Spec', alpha=0.7, color='orange', edgecolor='black')
    ax2.set_xlabel('Feature Type', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax2.set_title('Sensitivity @95% Spec\nby Feature', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=9)
    ax2.set_ylim([0, 1])
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    # --- Plot 3: AUC by Number of Genes ---
    ax3 = fig.add_subplot(gs[1, :2])

    gene_counts = []
    novel_aucs_gene = []
    gene_features = []
    train_aucs_gene = []
    gene_sens = []
    gene_spec = []
    gene_auc_std = []
    gene_sens_std = []
    gene_spec_std = []

    if use_gene_counts:
        for n_genes in sorted(gene_count_results.keys()):
            res = gene_count_results[n_genes]
            if res is None:
                continue
            gene_counts.append(n_genes)
            novel_aucs_gene.append((res.get('withheld_auc') if no_novel_cancers else res.get('novel_auc')) or 0)
            gene_auc_std.append((res.get('withheld_auc_std') if no_novel_cancers else res.get('novel_auc_std')) or 0)
            train_aucs_gene.append(res['best_model_info']['cv_auc'])
            gene_features.append(res['best_model_info']['feature'])
            sens_95, spec_95 = _get_sens_spec_95(res)
            gene_sens.append(sens_95)
            gene_spec.append(spec_95)
            gene_sens_std.append((res.get('withheld_sens_at_95spec_std') if no_novel_cancers else res.get('novel_sens_at_95spec_std')) or 0)
            gene_spec_std.append((res.get('withheld_spec_at_95spec_std') if no_novel_cancers else res.get('novel_spec_at_95spec_std')) or 0)

        ax3.plot(gene_counts, novel_aucs_gene, marker='o', linewidth=2, markersize=8, color='steelblue', label='Best Model Novel AUC')
        if len(gene_auc_std) == len(novel_aucs_gene) and len(gene_counts) > 0:
            lower = np.array(novel_aucs_gene) - np.array(gene_auc_std)
            upper = np.array(novel_aucs_gene) + np.array(gene_auc_std)
            ax3.fill_between(gene_counts, lower, upper, color='steelblue', alpha=0.2, linewidth=0)
        ax3.set_xlabel('Number of Genes', fontsize=12, fontweight='bold')
        ax3.set_ylabel(metric_label, fontsize=12, fontweight='bold')
        ax3.set_title(f'Generalization Performance by Number of Genes\n(Best Model at Each Gene Count) ({metric_label})', fontsize=14, fontweight='bold')
        ax3.set_xscale('log')
        ax3.set_ylim([0, 1])
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random')
        ax3.legend()

        for gc, auc_val, feat in zip(gene_counts, novel_aucs_gene, gene_features):
            ax3.annotate(f'{feat}', (gc, auc_val), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=7, alpha=0.7)
    else:
        ax3.text(0.5, 0.5, 'Not applicable\n(no gene selection)', transform=ax3.transAxes,
                 ha='center', va='center', fontsize=12)
        ax3.set_axis_off()

    # --- Plot 4: Sensitivity & Specificity by Gene Count ---
    ax4 = fig.add_subplot(gs[1, 2])

    if use_gene_counts:
        ax4.plot(gene_counts, gene_sens, marker='s', linewidth=2, markersize=6, color='green', label='Sens @95% Spec')
        ax4.plot(gene_counts, gene_spec, marker='^', linewidth=2, markersize=6, color='orange', label='Spec @95% Spec')
        if len(gene_sens_std) == len(gene_sens) and len(gene_counts) > 0:
            ax4.fill_between(gene_counts, np.array(gene_sens) - np.array(gene_sens_std), np.array(gene_sens) + np.array(gene_sens_std), color='green', alpha=0.15, linewidth=0)
        if len(gene_spec_std) == len(gene_spec) and len(gene_counts) > 0:
            ax4.fill_between(gene_counts, np.array(gene_spec) - np.array(gene_spec_std), np.array(gene_spec) + np.array(gene_spec_std), color='orange', alpha=0.15, linewidth=0)
        ax4.set_xlabel('Number of Genes', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Score', fontsize=11, fontweight='bold')
        ax4.set_title('Sensitivity @95% Spec\nby Gene Count', fontsize=12, fontweight='bold')
        ax4.set_xscale('log')
        ax4.set_ylim([0, 1])
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Not applicable\n(no gene selection)', transform=ax4.transAxes,
                 ha='center', va='center', fontsize=12)
        ax4.set_axis_off()

    # --- Plot 5: Training CV AUC vs Evaluation AUC (Feature-wise) ---
    ax5 = fig.add_subplot(gs[2, 0])

    ax5.scatter(train_aucs_feat, novel_aucs_feat, s=100, alpha=0.7, color='purple', edgecolor='black')
    ax5.plot([0.5, 1], [0.5, 1], 'k--', alpha=0.3, label='Perfect generalization')
    ax5.set_xlabel('Training CV AUC', fontsize=11, fontweight='bold')
    ax5.set_ylabel(metric_label, fontsize=11, fontweight='bold')
    ax5.set_title(f'Training vs {metric_label}\n(by Feature)', fontsize=12, fontweight='bold')
    ax5.set_xlim([0.5, 1])
    ax5.set_ylim([0.5, 1])
    ax5.grid(True, alpha=0.3)
    ax5.legend()

    for feat, train_auc, novel_auc in zip(feature_names, train_aucs_feat, novel_aucs_feat):
        ax5.annotate(feat[:4], (train_auc, novel_auc), fontsize=8, alpha=0.7)

    # --- Plot 6: Training CV AUC vs Novel AUC (Gene count) ---
    ax6 = fig.add_subplot(gs[2, 1])

    if use_gene_counts:
        colors = plt.cm.viridis(np.linspace(0, 1, len(gene_counts)))
        for gc, train_auc, novel_auc, color in zip(gene_counts, train_aucs_gene, novel_aucs_gene, colors):
            ax6.scatter(train_auc, novel_auc, s=80, alpha=0.7, color=color, edgecolor='black')

        ax6.plot([0.5, 1], [0.5, 1], 'k--', alpha=0.3)
        ax6.set_xlabel('Training CV AUC', fontsize=11, fontweight='bold')
        ax6.set_ylabel('Novel AUC', fontsize=11, fontweight='bold')
        ax6.set_title('Training vs Novel\n(by Gene Count)', fontsize=12, fontweight='bold')
        ax6.set_xlim([0.5, 1])
        ax6.set_ylim([0.5, 1])
        ax6.grid(True, alpha=0.3)
    else:
        ax6.text(0.5, 0.5, 'Not applicable\n(no gene selection)', transform=ax6.transAxes,
                 ha='center', va='center', fontsize=12)
        ax6.set_axis_off()

    # --- Plot 7: Generalization Gap (Training - Novel) ---
    ax7 = fig.add_subplot(gs[2, 2])

    if use_gene_counts:
        gaps_gene = [train - novel for train, novel in zip(train_aucs_gene, novel_aucs_gene)]
        ax7.plot(gene_counts, gaps_gene, marker='D', linewidth=2, markersize=6, color='crimson')
        ax7.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax7.set_xlabel('Number of Genes', fontsize=11, fontweight='bold')
        ax7.set_ylabel('AUC Gap (Train - Novel)', fontsize=11, fontweight='bold')
        ax7.set_title('Generalization Gap\nby Gene Count', fontsize=12, fontweight='bold')
        ax7.set_xscale('log')
        ax7.grid(True, alpha=0.3)
        ax7.fill_between(gene_counts, 0, gaps_gene, alpha=0.2, color='crimson')
    else:
        ax7.text(0.5, 0.5, 'Not applicable\n(no gene selection)', transform=ax7.transAxes,
                 ha='center', va='center', fontsize=12)
        ax7.set_axis_off()

    # Overall title
    training_cancer_str = ', '.join([ct.replace(' cancer', '') for ct in training_cancer_types])
    fig.suptitle(f'Comprehensive Within-Dataset Generalization Analysis\nTraining Cancer Types: {training_cancer_str}',
                fontsize=16, fontweight='bold', y=0.995)

    # Save main plot
    plot_path = output_dir / 'comprehensive_generalization_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nComprehensive comparison plot saved: {plot_path}")

    # --- ROC curves for best feature (withheld vs novel) ---
    best_feature = None
    best_feature_auc = -1
    for feat, res in feature_results.items():
        if res is None:
            continue
        if (res.get('novel_auc') or 0) > best_feature_auc:
            best_feature_auc = res.get('novel_auc') or 0
            best_feature = feat

    roc_plot_path = None
    best_res = None
    if best_feature and feature_results.get(best_feature):
        best_res = feature_results[best_feature]
        fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
        overall_roc = best_res.get('overall_roc')
        withheld_roc = best_res.get('withheld_roc')
        novel_roc = best_res.get('novel_roc')

        if overall_roc:
            ax_roc.plot(overall_roc['fpr'], overall_roc['tpr'], linewidth=2.5,
                        label=f"Overall (AUC={best_res.get('overall_auc', 0):.3f})", color='steelblue')
        if withheld_roc:
            ax_roc.plot(withheld_roc['fpr'], withheld_roc['tpr'], linewidth=2,
                        label=f"Withheld (AUC={best_res.get('withheld_auc', 0):.3f})", color='green', linestyle='--')
        if novel_roc:
            ax_roc.plot(novel_roc['fpr'], novel_roc['tpr'], linewidth=2,
                        label=f"Novel (AUC={best_res.get('novel_auc', 0):.3f})", color='orange', linestyle=':')

        ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
        ax_roc.set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
        ax_roc.set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
        ax_roc.set_title(f'ROC: Withheld vs Novel (Best Feature: {best_feature})', fontsize=12, fontweight='bold')
        ax_roc.set_xlim([0, 1])
        ax_roc.set_ylim([0, 1])
        ax_roc.grid(alpha=0.3)
        ax_roc.legend(loc='lower right', fontsize=9)

        roc_plot_path = output_dir / 'within_gen_withheld_vs_novel_roc.png'
        plt.savefig(roc_plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig_roc)

        print(f"ROC comparison plot saved: {roc_plot_path}")

    # --- Extra plots to support "all features work best" ---
    extra_plot_path = None
    novel_csv = None
    withheld_csv = None
    frac_within = 0.0
    if use_gene_counts:
        feature_names_sorted = sorted(available_features)
        gene_counts_sorted = sorted(available_gene_counts)
        novel_matrix = np.full((len(feature_names_sorted), len(gene_counts_sorted)), np.nan)
        withheld_matrix = np.full((len(feature_names_sorted), len(gene_counts_sorted)), np.nan)

        for i, feat in enumerate(feature_names_sorted):
            for j, k in enumerate(gene_counts_sorted):
                res = grid_results.get(feat, {}).get(k)
                if res is None:
                    continue
                novel_matrix[i, j] = res.get('novel_auc', np.nan)
                withheld_matrix[i, j] = res.get('withheld_auc', np.nan)

        novel_df = pd.DataFrame(novel_matrix, index=feature_names_sorted, columns=gene_counts_sorted)
        withheld_df = pd.DataFrame(withheld_matrix, index=feature_names_sorted, columns=gene_counts_sorted)
        novel_csv = output_dir / 'within_gen_novel_auc_matrix.csv'
        withheld_csv = output_dir / 'within_gen_withheld_auc_matrix.csv'
        novel_df.to_csv(novel_csv)
        withheld_df.to_csv(withheld_csv)

        fig_extra, axes = plt.subplots(1, 2, figsize=(18, 6))

        # Heatmap of novel AUC across features x K
        im = axes[0].imshow(novel_matrix, aspect='auto', interpolation='nearest', vmin=0.5, vmax=1)
        axes[0].set_title('Novel AUC Heatmap (Feature x K)', fontsize=12, fontweight='bold')
        axes[0].set_yticks(np.arange(len(feature_names_sorted)))
        axes[0].set_yticklabels(feature_names_sorted, fontsize=9)
        axes[0].set_xticks(np.arange(len(gene_counts_sorted)))
        axes[0].set_xticklabels(gene_counts_sorted, rotation=45, ha='right', fontsize=8)
        axes[0].set_xlabel('Number of Genes', fontsize=10, fontweight='bold')
        axes[0].set_ylabel('Feature Type', fontsize=10, fontweight='bold')
        plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

        # Performance band across features vs K
        best_by_k = np.nanmax(novel_matrix, axis=0)
        median_by_k = np.nanmedian(novel_matrix, axis=0)
        p75_by_k = np.nanpercentile(novel_matrix, 75, axis=0)
        axes[1].plot(gene_counts_sorted, best_by_k, marker='o', color='steelblue', label='Best across features')
        axes[1].plot(gene_counts_sorted, median_by_k, marker='s', color='gray', label='Median across features')
        axes[1].fill_between(gene_counts_sorted, median_by_k, p75_by_k, color='gray', alpha=0.2, label='Median-75th')
        axes[1].set_xscale('log')
        axes[1].set_ylim([0, 1])
        axes[1].set_xlabel('Number of Genes', fontsize=10, fontweight='bold')
        axes[1].set_ylabel('Novel AUC', fontsize=10, fontweight='bold')
        axes[1].set_title('Novel AUC vs K (Across Features)', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(fontsize=8)

        best_overall = np.nanmax(novel_matrix)
        within_eps = np.sum(novel_matrix >= (best_overall - 0.01))
        total_configs = np.sum(~np.isnan(novel_matrix))
        frac_within = (within_eps / total_configs) if total_configs > 0 else 0
        axes[1].text(0.02, 0.05, f'Configs within 0.01 of best: {within_eps}/{total_configs} ({frac_within:.1%})',
                     transform=axes[1].transAxes, fontsize=9,
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        extra_plot_path = output_dir / 'comprehensive_generalization_heatmap.png'
        plt.tight_layout()
        plt.savefig(extra_plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Extra generalization plots saved: {extra_plot_path}")
    
    # Save summary results
    summary = {
        'summary_version': '2026-02-07',
        'summary_source': str(Path(__file__).resolve()),
        'n_splits': 1,
        'best_feature_by_split': [best_feature] if best_feature else [],
        'best_feature_varies': False,
        'feature_results': {
            feat: {
                'withheld_auc': res.get('withheld_auc'),
                'novel_auc': res.get('novel_auc'),
                'overall_auc': res.get('overall_auc'),
                'overall_precision': res.get('overall_precision'),
                'overall_recall': res.get('overall_recall'),
                'overall_f1': res.get('overall_f1'),
                'overall_accuracy': res.get('overall_accuracy'),
                'overall_sens_at_95spec': res.get('overall_sens_at_95spec'),
                'overall_spec_at_95spec': res.get('overall_spec_at_95spec'),
                'withheld_sens_at_95spec': res.get('withheld_sens_at_95spec'),
                'withheld_spec_at_95spec': res.get('withheld_spec_at_95spec'),
                'novel_sens_at_95spec': res.get('novel_sens_at_95spec'),
                'novel_spec_at_95spec': res.get('novel_spec_at_95spec'),
                'n_genes': res['best_model_info']['n_genes'],
                'classifier': res['best_model_info']['classifier'],
                'training_cv_auc': res['best_model_info']['cv_auc']
            } if res else None
            for feat, res in feature_results.items()
        },
        'gene_count_results': {
            n_genes: {
                'withheld_auc': res.get('withheld_auc'),
                'novel_auc': res.get('novel_auc'),
                'overall_auc': res.get('overall_auc'),
                'overall_precision': res.get('overall_precision'),
                'overall_recall': res.get('overall_recall'),
                'overall_f1': res.get('overall_f1'),
                'overall_accuracy': res.get('overall_accuracy'),
                'overall_sens_at_95spec': res.get('overall_sens_at_95spec'),
                'overall_spec_at_95spec': res.get('overall_spec_at_95spec'),
                'withheld_sens_at_95spec': res.get('withheld_sens_at_95spec'),
                'withheld_spec_at_95spec': res.get('withheld_spec_at_95spec'),
                'novel_sens_at_95spec': res.get('novel_sens_at_95spec'),
                'novel_spec_at_95spec': res.get('novel_spec_at_95spec'),
                'feature': res['best_model_info']['feature'],
                'classifier': res['best_model_info']['classifier'],
                'training_cv_auc': res['best_model_info']['cv_auc']
            } if res else None
            for n_genes, res in gene_count_results.items()
        },
        'training_cancer_types': withheld_info['training_cancer_types'],
        'eval_n_samples': int(len(eval_y)),
        'eval_n_withheld': int(np.sum(is_withheld)),
        'eval_n_novel_cancer': int(np.sum(is_novel_cancer)),
        'eval_novel_cancer_types': sorted(np.unique(stratification_labels[eval_mask & novel_cancer_mask])),
        'no_novel_cancers': bool(no_novel_cancers),
        'novel_auc_is_fallback': bool(no_novel_cancers),
        'plot_path': str(plot_path),
        'extra_plot_path': str(extra_plot_path) if extra_plot_path else None,
        'roc_plot_path': str(roc_plot_path) if roc_plot_path else None,
        'best_feature': best_feature,
        'best_feature_roc': {
            'overall': best_res.get('overall_roc') if best_res else None,
            'withheld': best_res.get('withheld_roc') if best_res else None,
            'novel': best_res.get('novel_roc') if best_res else None,
            'overall_ci': best_res.get('overall_roc_ci') if best_res else None,
            'withheld_ci': best_res.get('withheld_roc_ci') if best_res else None,
            'novel_ci': best_res.get('novel_roc_ci') if best_res else None
        },
        'best_feature_auc': {
            'overall': best_res.get('overall_auc') if best_res else None,
            'withheld': best_res.get('withheld_auc') if best_res else None,
            'novel': best_res.get('novel_auc') if best_res else None,
            'overall_mean': best_res.get('overall_auc') if best_res else None,
            'overall_std': 0.0 if best_res else None,
            'withheld_mean': best_res.get('withheld_auc') if best_res else None,
            'withheld_std': 0.0 if best_res else None,
            'novel_mean': best_res.get('novel_auc') if best_res else None,
            'novel_std': 0.0 if best_res else None
        },
        'best_feature_auc_per_cancer_type': best_res.get('auc_per_cancer_type') if best_res else None,
        'novel_auc_matrix_csv': str(novel_csv) if novel_csv else None,
        'withheld_auc_matrix_csv': str(withheld_csv) if withheld_csv else None,
        'fraction_within_0.01_of_best': float(frac_within),
        'use_gene_counts': use_gene_counts
    }
    
    summary_path = output_dir / 'comprehensive_generalization_summary.pkl'
    with open(summary_path, 'wb') as f:
        pickle.dump(summary, f)
    
    print(f"Summary results saved: {summary_path}")
    
    # Print comprehensive summary tables
    print("\n" + "="*90)
    print("COMPREHENSIVE SUMMARY: GENERALIZATION BY FEATURE TYPE")
    print("="*90)
    print(f"{'Feature':<15} {'Train AUC':<11} {'Withheld AUC':<13} {metric_label:<11} {'Gap (Novel)':<12} {'N Genes'}")
    print("-" * 90)
    
    for feat in sorted(feature_results.keys()):
        if feature_results[feat] is not None:
            res = feature_results[feat]
            train_auc = res['best_model_info']['cv_auc']
            with_auc = res.get('withheld_auc', 0) or 0
            novel_auc = (res.get('withheld_auc') if no_novel_cancers else res.get('novel_auc')) or 0
            gap = train_auc - novel_auc
            n_genes = res['best_model_info']['n_genes']
            print(f"{feat:<15} {train_auc:<11.4f} {with_auc:<13.4f} {novel_auc:<11.4f} {gap:<12.4f} {n_genes}")
    
    print("\n" + "="*90)
    print("COMPREHENSIVE SUMMARY: GENERALIZATION BY NUMBER OF GENES")
    print("="*90)
    print(f"{'N Genes':<10} {'Feature':<15} {'Train AUC':<11} {'Withheld AUC':<13} {metric_label:<11} {'Gap (Novel)'}")
    print("-" * 90)
    
    for n_genes in sorted(gene_count_results.keys()):
        if gene_count_results[n_genes] is not None:
            res = gene_count_results[n_genes]
            feat = res['best_model_info']['feature']
            train_auc = res['best_model_info']['cv_auc']
            with_auc = res.get('withheld_auc', 0) or 0
            novel_auc = (res.get('withheld_auc') if no_novel_cancers else res.get('novel_auc')) or 0
            gap = train_auc - novel_auc
            print(f"{n_genes:<10} {feat:<15} {train_auc:<11.4f} {with_auc:<13.4f} {novel_auc:<11.4f} {gap:.4f}")
    
    print("\n" + "="*90)
    print("KEY INSIGHTS")
    print("="*90)
    
    # Best feature for novel cancer generalization
    non_empty_features = [(feat, res) for feat, res in feature_results.items() if res]
    if non_empty_features:
        best_feat_novel = max(non_empty_features,
                             key=lambda x: x[1].get('novel_auc', 0) if x[1] else 0)
        print(f"Best feature for novel cancer types: {best_feat_novel[0]} (Novel AUC: {best_feat_novel[1].get('novel_auc'):.4f})")
    else:
        print("No valid feature results available for novel cancer types")
    
    # Best gene count for novel cancer generalization
    if use_gene_counts:
        non_empty_gene_counts = [(n, res) for n, res in gene_count_results.items() if res]
        if non_empty_gene_counts:
            best_genes_novel = max(non_empty_gene_counts,
                                   key=lambda x: x[1].get('novel_auc', 0) if x[1] else 0)
            print(f"Best gene count for novel cancer types: {best_genes_novel[0]} genes (Novel AUC: {best_genes_novel[1].get('novel_auc'):.4f})")
        else:
            print("No valid gene-count results available for novel cancer types")
    else:
        print("Gene-count insights skipped (no gene selection)")
    
    # Smallest generalization gap to novel cancers
    gap_metric_key = 'withheld_auc' if no_novel_cancers else 'novel_auc'
    gaps_novel = [(feat, res['best_model_info']['cv_auc'] - (res.get(gap_metric_key, 0) or 0))
                 for feat, res in feature_results.items() if res and res.get(gap_metric_key)]
    if gaps_novel:
        min_gap_feat = min(gaps_novel, key=lambda x: x[1])
        print(f"Smallest generalization gap to novel cancers: {min_gap_feat[0]} (gap: {min_gap_feat[1]:.4f})")
    
    # Overall averages
    train_aucs_all = [res['best_model_info']['cv_auc'] for res in feature_results.values() if res] + \
                     train_aucs_gene
    withheld_aucs_all = [res.get('withheld_auc') for res in feature_results.values() if res and res.get('withheld_auc') is not None] + \
                        [res.get('withheld_auc') for res in gene_count_results.values() if res and res.get('withheld_auc') is not None]
    novel_aucs_all = [res.get(gap_metric_key) for res in feature_results.values() if res and res.get(gap_metric_key) is not None] + \
                     [res.get(gap_metric_key) for res in gene_count_results.values() if res and res.get(gap_metric_key) is not None]

    avg_train = np.mean(train_aucs_all) if train_aucs_all else 0
    avg_withheld = np.mean(withheld_aucs_all) if withheld_aucs_all else 0
    avg_novel = np.mean(novel_aucs_all) if novel_aucs_all else 0

    print(f"\nAverage Performance:")
    print(f"   Training CV AUC:       {avg_train:.4f}")
    print(f"   Withheld AUC:          {avg_withheld:.4f} (gap: {avg_train - avg_withheld:+.4f})")
    print(f"   {metric_label}:      {avg_novel:.4f} (gap: {avg_train - avg_novel:+.4f})")
    
    print("="*90)
    
    return summary


def run_within_dataset_generalization_compare(compare_dirs: list, args: argparse.Namespace):
    """
    Compare DD/RNA/ATAC within-dataset generalization results in a single set of plots.
    Expects each directory to contain comprehensive_generalization_summary.pkl
    produced by run_comprehensive_within_dataset_evaluation.
    """
    print("\n" + "="*90)
    print("WITHIN-DATASET GENERALIZATION COMPARISON")
    print("="*90)

    def _resolve_summary_path(path_str: str) -> Path:
        p = Path(path_str)
        if p.is_dir():
            if (p / 'comprehensive_generalization_summary.pkl').exists():
                return p / 'comprehensive_generalization_summary.pkl'
            if (p / 'comprehensive_evaluation' / 'comprehensive_generalization_summary.pkl').exists():
                return p / 'comprehensive_evaluation' / 'comprehensive_generalization_summary.pkl'
        return p

    def _infer_label(path_str: str) -> str:
        lower = path_str.lower()
        if 'rna' in lower:
            return 'rna'
        if 'atac' in lower:
            return 'atac'
        return 'dd'

    summaries = {}
    for entry in compare_dirs:
        summary_path = _resolve_summary_path(entry)
        if not summary_path.exists():
            print(f"Summary not found: {summary_path}")
            continue
        with open(summary_path, 'rb') as f:
            summary = pickle.load(f)
        label = _infer_label(str(summary_path))
        summaries[label] = summary
        print(f"Loaded {label} summary: {summary_path}")

    if len(summaries) == 0:
        print("No summaries loaded. Provide valid compare dirs.")
        return

    output_dir = Path(args.output_dir) / "within_gen_comparison"
    os.makedirs(output_dir, exist_ok=True)

    labels = sorted(summaries.keys())
    best_novel = []
    best_withheld = []
    best_overall = []
    best_sens95 = []

    for label in labels:
        s = summaries[label]
        aucs = s.get('best_feature_auc', {})
        best_overall.append(aucs.get('overall') or 0)
        best_withheld.append(aucs.get('withheld') or 0)
        best_novel.append(aucs.get('novel') or 0)

        # Sens@95%Spec from best feature if available
        sens95 = 0
        best_feat = s.get('best_feature')
        feat_results = s.get('feature_results', {})
        if best_feat and feat_results.get(best_feat):
            sens95 = feat_results[best_feat].get('novel_sens_at_95spec') or 0
        best_sens95.append(sens95)


    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    x = np.arange(len(labels))
    width = 0.25
    axes[0, 0].bar(x - width, best_overall, width, label='Overall', color='steelblue')
    axes[0, 0].bar(x, best_withheld, width, label='Withheld', color='green')
    axes[0, 0].bar(x + width, best_novel, width, label='Novel', color='orange')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(labels)
    axes[0, 0].set_ylim([0, 1])
    axes[0, 0].set_ylabel('AUC', fontweight='bold')
    axes[0, 0].set_title('Best Feature AUCs', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)

    axes[0, 1].bar(x, best_sens95, color='purple', alpha=0.8)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(labels)
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].set_ylabel('Sensitivity', fontweight='bold')
    axes[0, 1].set_title('Novel Sens @95% Spec (Best Feature)', fontweight='bold')
    axes[0, 1].grid(axis='y', alpha=0.3)

    # ROC curves: withheld vs novel per type
    ax_w = axes[1, 0]
    ax_n = axes[1, 1]
    for label in labels:
        s = summaries[label]
        roc = s.get('best_feature_roc', {})
        if roc.get('withheld'):
            ax_w.plot(roc['withheld']['fpr'], roc['withheld']['tpr'], linewidth=2, label=label)
        if roc.get('novel'):
            ax_n.plot(roc['novel']['fpr'], roc['novel']['tpr'], linewidth=2, label=label)

    for ax, title in [(ax_w, 'Withheld ROC (Best Feature)'), (ax_n, 'Novel ROC (Best Feature)')]:
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xlabel('False Positive Rate', fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontweight='bold')
        ax.set_title(title, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle('Within-Dataset Generalization Comparison (DD/RNA/ATAC)', fontweight='bold')
    out_path = output_dir / 'within_gen_dd_rna_atac_comparison.png'
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Comparison plot saved: {out_path}")


def run_baseline_aggregate_study(
    all_feature_arrays: dict,
    y_labels: np.ndarray,
    stratification_labels: np.ndarray,
    sample_ids: np.ndarray,
    DATA_PARAM_GRIDS: dict,
    args: argparse.Namespace
):
    """
    Run baseline aggregate study: select top genes, aggregate them from raw files,
    and create ROC curves without using a classifier.
    
    This tests if a simple aggregated biomarker (sum/mean of top genes) performs
    comparably to complex ML models.
    
    Iterates through all top_genes values from DATA_PARAM_GRIDS to compare performance
    across different numbers of selected genes.
    """
    print("\n" + "="*90)
    print("BASELINE AGGREGATE STUDY")
    print("="*90)
    
    # Create output directory
    baseline_output_dir = Path(args.output_dir) / f"baseline_aggregate_{args.experiment_name}"
    os.makedirs(baseline_output_dir, exist_ok=True)
    print(f"Output directory: {baseline_output_dir}")
    
    # Configuration
    feature_name = args.baseline_feature
    selection_type = args.baseline_selection
    features_dir = args.features_dir
    top_genes_list = DATA_PARAM_GRIDS['top_genes']
    
    print(f"\nStudy Configuration:")
    print(f"   Feature Type: {feature_name}")
    print(f"   Selection Method: {selection_type}")
    print(f"   Top Genes Values: {top_genes_list}")
    print(f"   Features Directory: {features_dir}")
    print(f"   Note: Aggregation method is implicit in feature type")
    
    # Get feature data
    X = all_feature_arrays.get(feature_name)
    if X is None:
        print(f"Error: Feature '{feature_name}' not found in feature arrays.")
        return
    
    print(f"   Data Shape: {X.shape}")
    print(f"   Total Samples: {len(sample_ids)}")

    # main() passes sample_ids as a plain list; fold indexing below needs an array
    sample_ids = np.asarray(sample_ids)

    # Setup cross-validation
    outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    all_results = []  # Store results for all n_genes configurations
    study_start_time = time.time()
    
    print(f"\nStarting baseline aggregate study for {len(top_genes_list)} gene count configurations...")
    
    # Iterate through all top_genes values
    for n_top_genes in top_genes_list:
        print(f"\n" + "="*80)
        print(f"TESTING WITH {n_top_genes} TOP GENES")
        print("="*80)
        
        fold_results = []
        config_start_time = time.time()
        
        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, stratification_labels)):
            print(f"\n{'='*80}")
            print(f"Fold {fold_idx + 1}/{outer_cv.get_n_splits()}")
            print(f"{'='*80}")
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y_labels[train_idx], y_labels[test_idx]
            stratification_train = stratification_labels[train_idx]
            stratification_test = stratification_labels[test_idx]
            sample_ids_train = sample_ids[train_idx]
            sample_ids_test = sample_ids[test_idx]
            
            print(f"  Train: {len(train_idx)} samples, Test: {len(test_idx)} samples")
            print(f"  Train distribution: {format_distribution(stratification_train)}")
            print(f"  Test distribution:  {format_distribution(stratification_test)}")
            
            # Step 1: Select top genes from training data
            print(f"  Selecting top {n_top_genes} genes using '{selection_type}' method...")
            top_gene_indices = select_top_genes(X_train, y_train, n_top_genes, selection_type=selection_type)
            print(f"  Selected {len(top_gene_indices)} genes")
            print(f"  Gene indices (first 10): {top_gene_indices[:10].tolist() if len(top_gene_indices) > 0 else []}")
            
            if len(top_gene_indices) == 0:
                print(f"   No genes selected, skipping fold")
                continue
            
            # Step 2: Aggregate selected genes from raw data files for TRAIN set
            print(f"  Aggregating selected genes from raw files (train set)...")
            print(f"  Train samples (first 5): {sample_ids_train[:5].tolist()}")
            print(f"  Features directory: {features_dir}")
            print(f"  Feature type: {feature_name} (aggregation implicit in feature)")
            
            train_aggregated = aggregate_feature_from_raw_data(
                sample_ids_train.tolist(),
                top_gene_indices,
                feature_name,
                features_dir
            )
            
            print(f"  Train aggregated shape: {train_aggregated.shape}")
            print(f"  Train aggregated range: [{np.nanmin(train_aggregated):.4f}, {np.nanmax(train_aggregated):.4f}]")
            print(f"  Train aggregated mean: {np.nanmean(train_aggregated):.4f}, std: {np.nanstd(train_aggregated):.4f}")
            
            # Step 3: Aggregate selected genes from raw data files for TEST set
            print(f"  Aggregating selected genes from raw files (test set)...")
            print(f"  Test samples (first 5): {sample_ids_test[:5].tolist()}")
            
            test_aggregated = aggregate_feature_from_raw_data(
                sample_ids_test.tolist(),
                top_gene_indices,
                feature_name,
                features_dir
            )
            
            print(f"  Test aggregated shape: {test_aggregated.shape}")
            print(f"  Test aggregated range: [{np.nanmin(test_aggregated):.4f}, {np.nanmax(test_aggregated):.4f}]")
            print(f"  Test aggregated mean: {np.nanmean(test_aggregated):.4f}, std: {np.nanstd(test_aggregated):.4f}")
            
            # Check for NaN values
            train_nan_count = np.sum(np.isnan(train_aggregated))
            test_nan_count = np.sum(np.isnan(test_aggregated))
            if train_nan_count > 0 or test_nan_count > 0:
                print(f"   Warning: {train_nan_count} NaN in train, {test_nan_count} NaN in test")
            else:
                print(f"  No NaN values in aggregated data")
            
            # Step 4: Use aggregated values directly as "prediction scores"
            # Higher aggregated value = more likely to be cancer (or vice versa, depending on feature)
            y_score = test_aggregated
            
            print(f"  Using aggregated values as prediction scores (no ML classifier)")
            print(f"  y_test distribution: {np.sum(y_test == 0)} healthy, {np.sum(y_test == 1)} cancer")
            print(f"  y_score (first 10): {y_score[:10].tolist()}")
            
            # Calculate ROC curve on the TEST fold (for AUC / ROC-curve reporting only;
            # AUC is threshold-independent so it is computed directly on the test fold)
            fpr_full, tpr_full, thresholds_full = roc_curve(y_test, y_score)
            auc_full = auc(fpr_full, tpr_full)
            precision_full, recall_full, _ = precision_recall_curve(y_test, y_score)

            print(f"  Overall AUC: {auc_full:.4f}")
            print(f"  ROC curve points: {len(fpr_full)}")
            print(f"  Threshold range: [{thresholds_full.min():.4f}, {thresholds_full.max():.4f}]")

            # Calculate predictions at multiple operating points.
            # Thresholds are chosen on the TRAIN fold (train_aggregated) and applied
            # unchanged to the TEST fold (y_score) to avoid test-set leakage.
            print(f"  Calculating predictions at 4 operating points (thresholds from TRAIN fold)...")
            op_thr = derive_operating_point_thresholds(y_train, train_aggregated)

            # At 95% specificity (threshold chosen on TRAIN fold)
            threshold_95spec = op_thr['95_percent_specificity']
            y_pred_95spec = (y_score >= threshold_95spec).astype(int)
            print(f"  95% specificity: threshold={threshold_95spec:.4f} (train FPR={op_thr['train_fpr_at_95spec']:.4f}, train TPR={op_thr['train_tpr_at_95spec']:.4f})")

            # At 90% specificity (threshold chosen on TRAIN fold)
            threshold_90spec = op_thr['90_percent_specificity']
            y_pred_90spec = (y_score >= threshold_90spec).astype(int)
            print(f"  90% specificity: threshold={threshold_90spec:.4f} (train FPR={op_thr['train_fpr_at_90spec']:.4f}, train TPR={op_thr['train_tpr_at_90spec']:.4f})")

            # At Youden's optimal (threshold chosen on TRAIN fold)
            threshold_youden = op_thr['youden_optimal']
            y_pred_youden = (y_score >= threshold_youden).astype(int)
            print(f"  Youden optimal: threshold={threshold_youden:.4f} (train FPR={op_thr['train_fpr_at_youden']:.4f}, train TPR={op_thr['train_tpr_at_youden']:.4f})")

            # At median (default) - median of TRAIN scores
            threshold_median = op_thr['median']
            y_pred_median = (y_score >= threshold_median).astype(int)
            print(f"  Median threshold: {threshold_median:.4f} (from TRAIN fold)")
            
            # Generate reports and confusion matrices
            report_full = {
                'median_threshold': classification_report(y_test, y_pred_median, output_dict=True, zero_division=0),
                '95_percent_specificity': classification_report(y_test, y_pred_95spec, output_dict=True, zero_division=0),
                '90_percent_specificity': classification_report(y_test, y_pred_90spec, output_dict=True, zero_division=0),
                'youden_optimal': classification_report(y_test, y_pred_youden, output_dict=True, zero_division=0),
            }
            
            confusion_matrix_full = {
                'median_threshold': confusion_matrix(y_test, y_pred_median).tolist(),
                '95_percent_specificity': confusion_matrix(y_test, y_pred_95spec).tolist(),
                '90_percent_specificity': confusion_matrix(y_test, y_pred_90spec).tolist(),
                'youden_optimal': confusion_matrix(y_test, y_pred_youden).tolist(),
            }
            
            # Track misclassified samples at each operating point
            misclassified_sample_ids = {
                'median_threshold': sample_ids_test[y_test != y_pred_median].tolist(),
                '95_percent_specificity': sample_ids_test[y_test != y_pred_95spec].tolist(),
                '90_percent_specificity': sample_ids_test[y_test != y_pred_90spec].tolist(),
                'youden_optimal': sample_ids_test[y_test != y_pred_youden].tolist(),
            }
            
            # Store thresholds (operating points chosen on the TRAIN fold).
            # train_*_at_* record where each threshold landed on the TRAIN-fold ROC.
            full_model_thresholds = {
                'median': float(threshold_median),
                '95_percent_specificity': float(threshold_95spec),
                '90_percent_specificity': float(threshold_90spec),
                'youden_optimal': float(threshold_youden),
                'train_fpr_at_95spec': op_thr['train_fpr_at_95spec'],
                'train_tpr_at_95spec': op_thr['train_tpr_at_95spec'],
                'train_fpr_at_90spec': op_thr['train_fpr_at_90spec'],
                'train_tpr_at_90spec': op_thr['train_tpr_at_90spec'],
            }
            
            # Per-cancer-type evaluation
            auc_per_type, roc_per_type, pr_per_type = {}, {}, {}
            confusion_matrix_per_type = {}
            classification_report_per_type = {}
            sample_classification_per_type = {}
            
            healthy_labels_in_test = np.unique(stratification_test[y_test == 0])
            unique_cancer_types_in_test = np.unique(stratification_test[y_test == 1])
            
            print(f"  Evaluating per-cancer-type metrics for {len(unique_cancer_types_in_test)} cancer types")
            print(f"  Cancer types: {unique_cancer_types_in_test.tolist()}")
            
            for cancer_type in unique_cancer_types_in_test:
                mask = np.isin(stratification_test, healthy_labels_in_test) | (stratification_test == cancer_type)
                y_test_subset = (stratification_test[mask] == cancer_type).astype(int)
                y_score_subset = y_score[mask]
                sample_ids_subset = sample_ids_test[mask]
                
                fpr, tpr, thresholds = roc_curve(y_test_subset, y_score_subset)
                precision, recall, _ = precision_recall_curve(y_test_subset, y_score_subset)
                
                auc_per_type[cancer_type] = auc(fpr, tpr)
                roc_per_type[cancer_type] = {'fpr': fpr, 'tpr': tpr}
                pr_per_type[cancer_type] = {'precision': precision, 'recall': recall}
                
                # Calculate confusion matrices at different thresholds (same as overall)
                y_pred_median_subset = (y_score_subset >= threshold_median).astype(int)
                y_pred_95spec_subset = (y_score_subset >= threshold_95spec).astype(int)
                y_pred_90spec_subset = (y_score_subset >= threshold_90spec).astype(int)
                y_pred_youden_subset = (y_score_subset >= threshold_youden).astype(int)
                
                confusion_matrix_per_type[cancer_type] = {
                    'median_threshold': confusion_matrix(y_test_subset, y_pred_median_subset).tolist(),
                    '95_percent_specificity': confusion_matrix(y_test_subset, y_pred_95spec_subset).tolist(),
                    '90_percent_specificity': confusion_matrix(y_test_subset, y_pred_90spec_subset).tolist(),
                    'youden_optimal': confusion_matrix(y_test_subset, y_pred_youden_subset).tolist(),
                }
                
                classification_report_per_type[cancer_type] = {
                    'median_threshold': classification_report(y_test_subset, y_pred_median_subset, output_dict=True, zero_division=0),
                    '95_percent_specificity': classification_report(y_test_subset, y_pred_95spec_subset, output_dict=True, zero_division=0),
                    '90_percent_specificity': classification_report(y_test_subset, y_pred_90spec_subset, output_dict=True, zero_division=0),
                    'youden_optimal': classification_report(y_test_subset, y_pred_youden_subset, output_dict=True, zero_division=0),
                }
                
                sample_classification_per_type[cancer_type] = {
                    'median_threshold': {
                        'correct': sample_ids_subset[y_test_subset == y_pred_median_subset].tolist(),
                        'incorrect': sample_ids_subset[y_test_subset != y_pred_median_subset].tolist(),
                    },
                    '95_percent_specificity': {
                        'correct': sample_ids_subset[y_test_subset == y_pred_95spec_subset].tolist(),
                        'incorrect': sample_ids_subset[y_test_subset != y_pred_95spec_subset].tolist(),
                    },
                    '90_percent_specificity': {
                        'correct': sample_ids_subset[y_test_subset == y_pred_90spec_subset].tolist(),
                        'incorrect': sample_ids_subset[y_test_subset != y_pred_90spec_subset].tolist(),
                    },
                    'youden_optimal': {
                        'correct': sample_ids_subset[y_test_subset == y_pred_youden_subset].tolist(),
                        'incorrect': sample_ids_subset[y_test_subset != y_pred_youden_subset].tolist(),
                    },
                }
            
            # Store fold results
            fold_results.append({
                'feature': feature_name,
                'top_genes': n_top_genes,
                'selection_type': selection_type,
                'fold': fold_idx + 1,
                'num_genes_used': len(top_gene_indices),
                
                # Sample and gene tracking for coverage aggregation experiments
                'train_sample_ids': sample_ids_train.tolist(),
                'test_sample_ids': sample_ids_test.tolist(),
                'selected_gene_indices': top_gene_indices.tolist(),
                
                'misclassified_sample_ids': misclassified_sample_ids,
                'y_pred_proba': y_score.tolist(),  # Aggregated scores used as prediction probabilities
                'y_test': y_test.tolist(),  # True labels for verification
                
                # Overall metrics
                'auc_full_model': auc_full,
                'fpr_full_model': fpr_full,
                'tpr_full_model': tpr_full,
                'precision_full_model': precision_full,
                'recall_full_model': recall_full,
                'classification_report_full_model': report_full,
                'confusion_matrix_full_model': confusion_matrix_full,
                'full_model_thresholds': full_model_thresholds,
                
                # Per-cancer-type metrics
                'auc_per_cancer_type': auc_per_type,
                'roc_per_cancer_type': roc_per_type,
                'pr_per_cancer_type': pr_per_type,
                'confusion_matrix_per_cancer_type': confusion_matrix_per_type,
                'classification_report_per_cancer_type': classification_report_per_type,
                'sample_classification_per_type': sample_classification_per_type,
            })
        
        config_elapsed = time.time() - config_start_time
        
        # Save results for this n_genes configuration
        output_filename = f"baseline_aggregate_{feature_name}_{n_top_genes}_genes_{selection_type}.pkl"
        output_path = baseline_output_dir / output_filename
        
        print(f"\n  Saving results for {n_top_genes} genes configuration...")
        print(f"  Total folds completed: {len(fold_results)}")
        print(f"  Output file: {output_filename}")
        
        results_df = pd.DataFrame(fold_results)
        with open(output_path, 'wb') as f:
            pickle.dump(results_df, f)
        
        # Print summary for this configuration
        aucs = [r['auc_full_model'] for r in fold_results]
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        min_auc = np.min(aucs)
        max_auc = np.max(aucs)
        print(f"\nCompleted {n_top_genes} genes in {config_elapsed:.2f}s")
        print(f"   AUC statistics:")
        print(f"     Mean: {mean_auc:.4f} +/- {std_auc:.4f}")
        print(f"     Range: [{min_auc:.4f}, {max_auc:.4f}]")
        print(f"     Per-fold AUCs: {[f'{auc:.3f}' for auc in aucs]}")
        print(f"   Saved to: {output_path}")
        
        # Store results for final summary
        all_results.append({
            'n_genes': n_top_genes,
            'mean_auc': mean_auc,
            'std_auc': std_auc,
            'results_file': output_path
        })
    
    # Print final summary across all configurations
    study_elapsed = time.time() - study_start_time
    print(f"\n" + "="*90)
    print("BASELINE AGGREGATE STUDY COMPLETE")
    print("="*90)
    print(f"Total time: {study_elapsed:.2f}s ({study_elapsed/60:.2f} min)")
    print(f"\nSummary Across All Gene Counts:")
    print(f"{'N Genes':<12} {'Mean AUC':<12} {'Std AUC':<12} {'Output File'}")
    print("-" * 90)
    for result in all_results:
        print(f"{result['n_genes']:<12} {result['mean_auc']:<12.4f} {result['std_auc']:<12.4f} {result['results_file'].name}")
    
    # Find best configuration
    best_config = max(all_results, key=lambda x: x['mean_auc'])
    print(f"\nBest Configuration: {best_config['n_genes']} genes with AUC = {best_config['mean_auc']:.4f} +/- {best_config['std_auc']:.4f}")
    print(f"\n{'='*90}\n")


# =================================================================================
# --- Extended per-fold reporting (covariate-panel runs) ---
# =================================================================================
# The per-classifier nested-CV site below used to store one threshold-free number
# per fold (AUROC) plus confusion matrices at four operating points. For the
# covariate panels - and the RNA / ATAC comparators, which share this code path -
# that is not enough to understand WHY a model is good, so every fold record now
# also carries:
#   test_metrics                threshold-free test-fold metrics: AUROC, AUPRC,
#                               partial AUROC at FPR <= 0.05 / 0.10 (McClish
#                               standardized), Brier score, log loss, sensitivity
#                               at fixed TEST specificities (descriptive, read off
#                               the test ROC - not an operating point), class-wise
#                               score quartiles
#   operating_point_metrics     sensitivity / specificity / PPV / NPV / F1 /
#                               balanced accuracy / MCC implied by the stored
#                               confusion matrices at the four operating points
#   train_metrics               the same threshold-free block for (a) the selected
#                               configuration's inner-CV scores on the training
#                               partition (the scores its thresholds come from) and
#                               (b) its resubstitution scores - their gap to the
#                               test fold is the overfitting diagnostic
#   y_train, inner_cv_train_scores, resubstitution_train_scores
#                               per training sample (aligned with train_sample_ids)
#   inner_cv_results            every hyperparameter candidate of the classifier's
#                               grid with its mean / sd / rank / per-split inner
#                               score: the landscape the winner was picked from
#   fitted_model_summary        plain-type description of the fitted pipeline
#                               (scaler statistics, coefficients, feature
#                               importances, support-vector counts)
#   feature_names               column names of the predefined matrix (None when
#                               unknown or when the width does not match)
# Every key is NEW; no previously stored key changed. Pickles written before this
# carry none of them and the exporters treat their absence as "not available".
# With --store_fitted_models the fitted pipelines of the ten folds are written to a
# SEPARATE models_<classifier>_<feature>_all_features.pkl next to the results
# pickle, so readers that cannot import a classifier's library are not affected.

TEST_SPEC_GRID = (0.80, 0.90, 0.95, 0.98, 0.99)
PARTIAL_AUC_FPR = (0.05, 0.10)


def _safe_div(a, b):
    return float(a) / float(b) if b else float('nan')


def _confusion_metrics(cm, threshold):
    """Metrics implied by a stored [[tn, fp], [fn, tp]] confusion matrix."""
    (tn, fp), (fn, tp) = [[int(v) for v in row] for row in cm]
    sens, spec = _safe_div(tp, tp + fn), _safe_div(tn, tn + fp)
    denom = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        'threshold': float(threshold) if threshold is not None else float('nan'),
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp, 'n_flagged': tp + fp,
        'sensitivity': sens, 'specificity': spec,
        'ppv': _safe_div(tp, tp + fp), 'npv': _safe_div(tn, tn + fn),
        'f1': _safe_div(2 * tp, 2 * tp + fp + fn),
        'accuracy': _safe_div(tp + tn, tp + tn + fp + fn),
        'balanced_accuracy': (sens + spec) / 2 if np.isfinite(sens) and np.isfinite(spec) else float('nan'),
        'mcc': float((tp * tn - fp * fn) / denom) if denom > 0 else float('nan'),
    }


def _threshold_free_metrics(y, scores):
    """AUROC, AUPRC, partial AUROC, Brier, log loss, sensitivity at fixed
    specificities and class-wise score quartiles of one set of scores (NaN
    where a class is absent)."""
    y = np.asarray(y, dtype=int)
    s = np.asarray(scores, dtype=float)
    both = len(np.unique(y)) == 2
    out = {'n': int(len(y)), 'n_pos': int((y == 1).sum()), 'prevalence': _safe_div((y == 1).sum(), len(y)),
           'auroc': float(roc_auc_score(y, s)) if both else float('nan'),
           'auprc': float(average_precision_score(y, s)) if both else float('nan')}
    for f in PARTIAL_AUC_FPR:
        out[f'partial_auroc_fpr_{f:.2f}'] = float(roc_auc_score(y, s, max_fpr=f)) if both else float('nan')
    pr = np.clip(s, 1e-15, 1 - 1e-15)
    out['brier'] = float(np.mean((s - y) ** 2))
    out['log_loss'] = float(-np.mean(y * np.log(pr) + (1 - y) * np.log(1 - pr)))
    for spec in TEST_SPEC_GRID:
        key = f'sens_at_spec_{spec:.2f}'
        if both:
            fpr, tpr, _ = roc_curve(y, s)
            ok = fpr <= (1 - spec) + 1e-12
            out[key] = float(tpr[ok].max()) if ok.any() else 0.0
        else:
            out[key] = float('nan')
    for cls, name in ((0, 'healthy'), (1, 'cancer')):
        v = s[y == cls]
        for q, qn in ((25, 'q25'), (50, 'median'), (75, 'q75')):
            out[f'score_{qn}_{name}'] = float(np.percentile(v, q)) if v.size else float('nan')
    return out


def _jsonable(v):
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    return str(v)


def _inner_cv_landscape(grid_search):
    """Every hyperparameter candidate of the GridSearchCV with its inner-fold scores."""
    cv = grid_search.cv_results_
    split_keys = sorted(k for k in cv if k.startswith('split') and k.endswith('_test_score'))
    return [{
        'params': {k.replace('clf__', ''): _jsonable(v) for k, v in cv['params'][i].items()},
        'mean_test_score': float(cv['mean_test_score'][i]),
        'std_test_score': float(cv['std_test_score'][i]),
        'rank_test_score': int(cv['rank_test_score'][i]),
        'split_test_scores': [float(cv[k][i]) for k in split_keys],
        'mean_fit_time': float(cv['mean_fit_time'][i]),
        'is_best': bool(i == grid_search.best_index_),
    } for i in range(len(cv['params']))]


def _fitted_model_summary(pipeline, feature_names):
    """Plain-type description of a fitted scaler + classifier pipeline."""
    out = {'feature_names': list(feature_names) if feature_names is not None else None}
    scaler = pipeline.named_steps.get('scaler') if hasattr(pipeline, 'named_steps') else None
    if scaler is not None and hasattr(scaler, 'mean_'):
        out['scaler_mean'] = [float(v) for v in scaler.mean_]
        out['scaler_scale'] = [float(v) for v in scaler.scale_]
    clf = pipeline.named_steps['clf'] if hasattr(pipeline, 'named_steps') else pipeline
    out['classifier_class'] = type(clf).__name__
    if hasattr(clf, 'coef_'):
        out['coef'] = [float(v) for v in np.ravel(clf.coef_)]
        out['intercept'] = [float(v) for v in np.ravel(getattr(clf, 'intercept_', []))]
    if hasattr(clf, 'feature_importances_'):
        out['feature_importances'] = [float(v) for v in np.ravel(clf.feature_importances_)]
    if hasattr(clf, 'n_support_'):
        out['n_support'] = [int(v) for v in clf.n_support_]
    if hasattr(clf, 'n_iter_'):
        try:
            out['n_iter'] = int(np.ravel(clf.n_iter_)[0])
        except (TypeError, ValueError, IndexError):
            pass
    return out


def _extended_fold_report(best_model, grid_search, X_train, y_train, y_test, test_scores,
                          inner_cv_scores, op_thr, confusion_matrix_full, feature_names=None):
    """The additive per-fold record documented above."""
    resub = best_model.predict_proba(X_train)[:, 1]
    # 'default_0.5' is the classifier's own predict() rule (0.5 on predict_proba for
    # most classifiers, the decision function's sign for SVC) - no single threshold
    thr_of = {'default_0.5': None,
              '95_percent_specificity': op_thr.get('95_percent_specificity'),
              '90_percent_specificity': op_thr.get('90_percent_specificity'),
              'youden_optimal': op_thr.get('youden_optimal')}
    names = list(feature_names) if feature_names is not None and len(feature_names) == X_train.shape[1] else None
    return {
        'test_metrics': _threshold_free_metrics(y_test, test_scores),
        'operating_point_metrics': {key: _confusion_metrics(cm, thr_of.get(key))
                                    for key, cm in confusion_matrix_full.items()},
        'train_metrics': {
            'inner_cv': _threshold_free_metrics(y_train, inner_cv_scores),
            'resubstitution': _threshold_free_metrics(y_train, resub),
            'inner_cv_best_score_mean_over_splits': float(grid_search.best_score_),
        },
        'y_train': [int(v) for v in np.asarray(y_train)],
        'inner_cv_train_scores': [float(v) for v in inner_cv_scores],
        'resubstitution_train_scores': [float(v) for v in resub],
        'inner_cv_results': _inner_cv_landscape(grid_search),
        'fitted_model_summary': _fitted_model_summary(best_model, names),
        'feature_names': names,
    }


def run_single_experiment_no_selection(params: dict):
    """
    Worker function to run nested CV WITHOUT gene selection - uses all features.
    An experiment is defined by a unique combination of classifier and feature.
    """
    # Unpack parameters
    clf_name = params['classifier']
    feature_name = params['feature']
    all_feature_arrays = params['feature_data']
    y_labels = params['labels']
    stratification_labels = params['stratification_labels']
    sample_ids = np.array(params['sample_ids']) 
    output_dir = params['output_dir']
    
    # Get classifier, its parameter grid, and the specific feature matrix
    classifier = params['classifiers'][clf_name]
    param_grid = params['clf_param_grids'][clf_name]
    X = all_feature_arrays[feature_name]
    
    # Define output file path
    clf_name_safe = clf_name.replace(" ", "_").lower()
    output_filename = f"results_{clf_name_safe}_{feature_name}_all_features.pkl"
    output_path = os.path.join(output_dir, output_filename)
    models_path = os.path.join(output_dir, f"models_{clf_name_safe}_{feature_name}_all_features.pkl")

    def _panel_done():
        """A complete result for this cohort (10 folds covering exactly these samples)."""
        try:
            with open(output_path, 'rb') as fh:
                df = pickle.load(fh)
            held_out = set(str(s) for ids in df['test_sample_ids'] for s in ids)
            ok = len(df) == 10 and held_out == set(str(s) for s in sample_ids)
            if ok and params.get('store_fitted_models'):
                with open(models_path, 'rb') as fh:
                    pickle.load(fh)
            return bool(ok)
        except Exception:
            return False

    # Resume / cross-cluster claim: a complete result is not recomputed
    # (delete it to force), and a result another worker is computing is waited for.
    if _panel_done():
        print(f"--- {output_filename} already complete for this cohort - skipping ---")
        return output_path
    claim = _WorkClaim(output_path, output_filename)
    if not claim.try_acquire():
        print(f"--- {output_filename}: claimed by {claim.holder()} - waiting ---")
        if claim.wait(_panel_done) == 'done':
            return output_path

    print(f"\n======================================================================================")
    print(f"STARTING (NO SELECTION): {clf_name_safe} | Feature: '{feature_name}' | All {X.shape[1]} features")
    print(f"======================================================================================")

    outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    fold_results = []
    fitted_models = {}
    report_feature_names = params.get('feature_names')
    start_time = time.time()

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, stratification_labels)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_labels[train_idx], y_labels[test_idx]
        stratification_train = stratification_labels[train_idx]
        stratification_test = stratification_labels[test_idx]
        sample_ids_train = sample_ids[train_idx]  # Get sample IDs for the train set
        sample_ids_test = sample_ids[test_idx]  # Get sample IDs for the test set

        # --- Print Outer Loop Distribution ---
        print(f"\n[Outer Loop] Fold {fold_idx + 1}/{outer_cv.get_n_splits()}:")
        print(f"  Train distribution: {format_distribution(stratification_train)}")
        print(f"  Test distribution:  {format_distribution(stratification_test)}")
        print(f"  Num of 0's (Healthy) in train: {np.sum(y_train == 0)}, Num of 1's (Cancer) in train: {np.sum(y_train == 1)}")
        print(f"  Num of 0's (Healthy) in test: {np.sum(y_test == 0)}, Num of 1's (Cancer) in test: {np.sum(y_test == 1)}")

        # No gene selection - use all features
        pipeline = Pipeline([('scaler', StandardScaler()), ('clf', classifier)])
        
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        inner_splits = list(inner_cv.split(X_train, stratification_train))

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring='roc_auc',
            cv=inner_splits,
            n_jobs=1
        )
        
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        
        y_pred_proba_full = best_model.predict_proba(X_test)[:, 1]
        y_pred_binary_full = best_model.predict(X_test)

        # Calculate overall metrics on the TEST fold (AUC is threshold-independent)
        fpr_full, tpr_full, thresholds_full = roc_curve(y_test, y_pred_proba_full)
        auc_full = auc(fpr_full, tpr_full)
        precision_full, recall_full, _ = precision_recall_curve(y_test, y_pred_proba_full)

        # Operating-point thresholds chosen on the TRAIN fold (from the selected
        # configuration's scores on the inner validation folds, not resubstitution),
        # applied to the TEST fold
        op_thr, inner_cv_scores = inner_cv_operating_thresholds(
            best_model, X_train, y_train, inner_splits,
            expected_mean_auc=grid_search.best_score_, label=f"fold {fold_idx + 1}")

        # Reports / confusion matrices / misclassified IDs at the four operating
        # points. Thresholds were derived from the TRAIN fold only (op_thr above).
        y_pred_default = y_pred_binary_full  # sklearn's default decision rule
        (report_full, confusion_matrix_full, misclassified_sample_ids,
         full_model_thresholds) = _operating_point_reports(
            y_test, y_pred_proba_full, y_pred_default, op_thr, sample_ids_test)

        # --- Per-cancer-type evaluation (same train-derived thresholds) ---
        unique_cancer_types_in_test = np.unique(stratification_test[y_test == 1])
        print(f"  [Outer Loop] Fold {fold_idx + 1}/{outer_cv.get_n_splits()}: Overall AUC = {auc_full:.3f}. Evaluating on {len(unique_cancer_types_in_test)} cancer types.")
        (auc_per_type, roc_per_type, pr_per_type, confusion_matrix_per_type,
         classification_report_per_type, sample_classification_per_type) = \
            _per_cancer_type_train_threshold_eval(
                y_test, y_pred_proba_full, stratification_test, sample_ids_test, op_thr)

        # Extended reporting (additive keys; see _extended_fold_report above)
        extended = _extended_fold_report(
            best_model, grid_search, X_train, y_train, y_test, y_pred_proba_full,
            inner_cv_scores, op_thr, confusion_matrix_full, feature_names=report_feature_names)
        if params.get('store_fitted_models'):
            fitted_models[fold_idx + 1] = best_model

        fold_results.append({
            'feature': feature_name,
            'top_genes': 'all',
            'selection_type': 'none',
            'fold': fold_idx + 1,
            'classifier': clf_name,
            'best_params': grid_search.best_params_,
            # Mean inner-CV AUC of the winning hyperparameters: the ONLY score that may be
            # used to compare classifiers/configs per fold without touching the test fold
            'inner_cv_best_score': float(grid_search.best_score_),
            'num_genes_used': X.shape[1],
            
            # Sample and gene tracking for coverage aggregation experiments
            'train_sample_ids': sample_ids_train.tolist(),
            'test_sample_ids': sample_ids_test.tolist(),
            'selected_gene_indices': list(range(X.shape[1])),  # All features used (no selection)
            
            'misclassified_sample_ids': misclassified_sample_ids,
            'y_pred_proba': y_pred_proba_full.tolist(),  # Prediction probabilities for all test samples
            'y_test': y_test.tolist(),  # True labels for verification

            # Overall model metrics
            'auc_full_model': auc_full,
            'fpr_full_model': fpr_full,
            'tpr_full_model': tpr_full,
            'precision_full_model': precision_full,
            'recall_full_model': recall_full,
            'classification_report_full_model': report_full,
            'confusion_matrix_full_model': confusion_matrix_full,
            'full_model_thresholds': full_model_thresholds,

            # Per-cancer-type metrics
            'auc_per_cancer_type': auc_per_type,
            'roc_per_cancer_type': roc_per_type,
            'pr_per_cancer_type': pr_per_type,
            'confusion_matrix_per_cancer_type': confusion_matrix_per_type,
            'classification_report_per_cancer_type': classification_report_per_type,
            'sample_classification_per_type': sample_classification_per_type,

            # Extended per-fold reporting: new keys only
            **extended,
        })

    elapsed = time.time() - start_time
    results_df = pd.DataFrame(fold_results)
    
    _atomic_pickle(output_path, results_df)

    if fitted_models:
        _atomic_pickle(models_path, {'classifier': clf_name, 'feature': feature_name,
                                     'feature_names': report_feature_names, 'models': fitted_models})
        print(f"--- Fitted fold models saved to {models_path} ---")
    claim.release()

    print(f"--- Finished and saved to {output_path} in {elapsed:.2f}s ---")
    return output_path

def run_single_experiment(params: dict):
    """
    Worker function to run the full nested CV pipeline for a single experiment configuration.
    An experiment is defined by a unique combination of classifier, feature, and top_genes count.
    
    Each fold result now includes sample and gene tracking for coverage aggregation experiments:
    - 'train_sample_ids': List of sample IDs in outer fold training set
    - 'test_sample_ids': List of sample IDs in outer fold test set
    - 'selected_gene_indices': Indices of genes selected by feature selection
    - 'inner_fold_tracking': List of dicts tracking inner CV splits:
        - 'inner_fold': Inner fold number (1-3)
        - 'inner_train_sample_ids': Sample IDs in inner training set
        - 'inner_test_sample_ids': Sample IDs in inner validation set
    
    Usage for coverage aggregation experiments:
    For each outer fold, use 'train_sample_ids' + 'selected_gene_indices' to:
    1. Aggregate BAM coverage across selected genes for training samples
    2. Calculate TSS features (RCOV, Griffin, MWH, etc.) on aggregated coverage
    3. Compare to ML model performance on same fold
    
    This enables testing whether aggregated-coverage-then-feature-calculation
    outperforms feature-calculation-then-ML on low-coverage samples.
    """
    # Unpack parameters
    clf_name = params['classifier']
    feature_name = params['feature']
    n_top_genes = params['top_genes']
    all_feature_arrays = params['feature_data']
    y_labels = params['labels']
    stratification_labels = params['stratification_labels']
    sample_ids = np.array(params['sample_ids']) 
    output_dir = params['output_dir']
    selection_type = params['selection_type']
    
    # Get classifier, its parameter grid, and the specific feature matrix
    classifier = params['classifiers'][clf_name]
    param_grid = params['clf_param_grids'][clf_name]
    X = all_feature_arrays[feature_name]
    
    # Define output file path
    clf_name_safe = clf_name.replace(" ", "_").lower()
    output_filename = f"results_{clf_name_safe}_{feature_name}_{n_top_genes}_genes_{selection_type}.pkl"
    output_path = os.path.join(output_dir, output_filename)

    print(f"\n======================================================================================")
    print(f"STARTING: {clf_name_safe} | Feature: '{feature_name}' | Top Genes: {n_top_genes}")
    print(f"======================================================================================")

    outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    fold_results = []
    start_time = time.time()

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, stratification_labels)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_labels[train_idx], y_labels[test_idx]
        stratification_train = stratification_labels[train_idx]
        stratification_test = stratification_labels[test_idx]
        sample_ids_test = sample_ids[test_idx] # Get sample IDs for the test set
        sample_ids_train = sample_ids[train_idx] # Get sample IDs for the train set

        # --- Print Outer Loop Distribution ---
        print(f"\n[Outer Loop] Fold {fold_idx + 1}/{outer_cv.get_n_splits()}:")
        print(f"  Train distribution: {format_distribution(stratification_train)}")
        print(f"  Test distribution:  {format_distribution(stratification_test)}")
        print(f"  Num of 0's (Healthy) in train: {np.sum(y_train == 0)}, Num of 1's (Cancer) in train: {np.sum(y_train == 1)}")
        print(f"  Num of 0's (Healthy) in test: {np.sum(y_test == 0)}, Num of 1's (Cancer) in test: {np.sum(y_test == 1)}")

        # TSS ranking + top-k selection is fitted INSIDE every training partition it
        # serves (selector as the first pipeline step, exactly as in
        # run_nested_experiment_full): the inner hyperparameter search refits it on
        # each inner-training split (for the search and for the threshold scores of
        # the winner alike), and the winning refit on its own training data. The outer test fold never
        # influences gene selection, and inner validation scores no longer see genes
        # ranked with their own samples.
        pipeline = Pipeline([
            ('selector', TopGeneSelector(n_top_genes=n_top_genes, selection_type=selection_type)),
            ('scaler', StandardScaler()),
            ('clf', classifier),
        ])

        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        inner_splits = list(inner_cv.split(X_train, stratification_train))

        # --- Track Inner Loop Sample IDs for Coverage Aggregation ---
        inner_fold_tracking = []
        for inner_fold_idx, (inner_train_idx, inner_test_idx) in enumerate(inner_splits):
            inner_fold_tracking.append({
                'inner_fold': inner_fold_idx + 1,
                'inner_train_sample_ids': sample_ids_train[inner_train_idx].tolist(),
                'inner_test_sample_ids': sample_ids_train[inner_test_idx].tolist(),
                # Gene selection is refit on each inner-training split by the selector.
            })

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring='roc_auc',
            cv=inner_splits,
            n_jobs=1
        )

        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        # Genes chosen by the final refit on the whole outer-training partition
        # (same computation as the previous outer-level select_top_genes call).
        top_gene_indices = np.asarray(best_model.named_steps['selector'].indices_)
        if len(top_gene_indices) == 0:
            # TopGeneSelector.transform falls back to a single zero column here,
            # which silently yields constant predictions. Fail loudly instead.
            raise RuntimeError(
                f"Fold {fold_idx + 1}: selector chose 0 genes for feature '{feature_name}' "
                f"(k={n_top_genes}, selection_type={selection_type}).")

        y_pred_proba_full = best_model.predict_proba(X_test)[:, 1]
        y_pred_binary_full = best_model.predict(X_test)

        # Calculate overall metrics on the TEST fold (AUC is threshold-independent)
        fpr_full, tpr_full, thresholds_full = roc_curve(y_test, y_pred_proba_full)
        auc_full = auc(fpr_full, tpr_full)
        precision_full, recall_full, _ = precision_recall_curve(y_test, y_pred_proba_full)

        # Operating-point thresholds chosen on the TRAIN fold (from the selected
        # configuration's scores on the inner validation folds, not resubstitution),
        # applied to the TEST fold. best_model is the full pipeline, so gene selection
        # is refit inside each inner-training split here as in the search.
        op_thr, _ = inner_cv_operating_thresholds(
            best_model, X_train, y_train, inner_splits,
            expected_mean_auc=grid_search.best_score_, label=f"fold {fold_idx + 1}")

        # Reports / confusion matrices / misclassified IDs at the four operating
        # points. Thresholds were derived from the TRAIN fold only (op_thr above).
        y_pred_default = y_pred_binary_full  # sklearn's default decision rule
        (report_full, confusion_matrix_full, misclassified_sample_ids,
         full_model_thresholds) = _operating_point_reports(
            y_test, y_pred_proba_full, y_pred_default, op_thr, sample_ids_test)

        # --- Per-cancer-type evaluation (same train-derived thresholds) ---
        unique_cancer_types_in_test = np.unique(stratification_test[y_test == 1])
        print(f"  [Outer Loop] Fold {fold_idx + 1}/{outer_cv.get_n_splits()}: Overall AUC = {auc_full:.3f}. Evaluating on {len(unique_cancer_types_in_test)} cancer types.")
        (auc_per_type, roc_per_type, pr_per_type, confusion_matrix_per_type,
         classification_report_per_type, sample_classification_per_type) = \
            _per_cancer_type_train_threshold_eval(
                y_test, y_pred_proba_full, stratification_test, sample_ids_test, op_thr)

        fold_results.append({
            'feature': feature_name,
            'top_genes': n_top_genes,
            'selection_type': selection_type,
            'fold': fold_idx + 1,
            'classifier': clf_name,
            'best_params': grid_search.best_params_,
            # Mean inner-CV AUC of the winning hyperparameters: the ONLY score that may be
            # used to compare classifiers/configs per fold without touching the test fold
            'inner_cv_best_score': float(grid_search.best_score_),
            'num_genes_used': len(top_gene_indices),
            
            # Sample and gene tracking for coverage aggregation experiments
            'train_sample_ids': sample_ids_train.tolist(),
            'test_sample_ids': sample_ids_test.tolist(),
            'selected_gene_indices': top_gene_indices.tolist(),
            'inner_fold_tracking': inner_fold_tracking,  # Inner CV splits with sample IDs
            
            'misclassified_sample_ids': misclassified_sample_ids,
            'y_pred_proba': y_pred_proba_full.tolist(),  # Prediction probabilities for all test samples
            'y_test': y_test.tolist(),  # True labels for verification

            # Overall model metrics
            'auc_full_model': auc_full,
            'fpr_full_model': fpr_full,
            'tpr_full_model': tpr_full,
            'precision_full_model': precision_full,
            'recall_full_model': recall_full,
            'classification_report_full_model': report_full,
            'confusion_matrix_full_model': confusion_matrix_full,
            'full_model_thresholds': full_model_thresholds,

            # Per-cancer-type metrics
            'auc_per_cancer_type': auc_per_type,
            'roc_per_cancer_type': roc_per_type,
            'pr_per_cancer_type': pr_per_type,
            'confusion_matrix_per_cancer_type': confusion_matrix_per_type,
            'classification_report_per_cancer_type': classification_report_per_type,
            'sample_classification_per_type': sample_classification_per_type,
        })

    elapsed = time.time() - start_time
    results_df = pd.DataFrame(fold_results)
    
    with open(output_path, 'wb') as f:
        pickle.dump(results_df, f)
        
    print(f"--- Finished and saved to {output_path} in {elapsed:.2f}s ---")
    return output_path


# =================================================================================
# --- Fully Nested Selection (feature + ranking function + k + classifier) ---
# =================================================================================

def _nested_full_grid_sizes(CLASSIFIERS, CLF_PARAM_GRIDS, classifier_names,
                            top_genes_list, selection_types, feature_names,
                            n_outer=10, n_inner=3):
    """
    Exact fit accounting for run_nested_experiment_full, computed from the grids
    themselves so the printed estimate cannot drift from what is actually run.

    Returns a dict with the per-classifier candidate counts and the implied number of
    inner-loop model fits: features x ranking functions x k values x classifier
    hyperparameters x inner folds x outer folds.
    """
    per_clf = {}
    for clf_name in classifier_names:
        n = 1
        for values in CLF_PARAM_GRIDS[clf_name].values():
            n *= len(list(values))
        per_clf[clf_name] = n
    hp_total = sum(per_clf.values())

    n_data = len(top_genes_list) * len(selection_types)
    cand_per_feature = n_data * hp_total                    # GridSearchCV candidates
    fits_per_feature_fold = cand_per_feature * n_inner      # one fit per candidate per inner fold
    fits_per_outer_fold = fits_per_feature_fold * len(feature_names)
    inner_fits = fits_per_outer_fold * n_outer

    return {
        'per_classifier_hyperparams': per_clf,
        'hyperparams_total': hp_total,
        'n_features': len(feature_names),
        'n_selection_types': len(selection_types),
        'n_top_genes': len(top_genes_list),
        'candidates_per_feature': cand_per_feature,
        'fits_per_feature_per_outer_fold': fits_per_feature_fold,
        'fits_per_outer_fold': fits_per_outer_fold,
        'inner_fits_total': inner_fits,
        # One refit of the winning config per outer fold, plus its inner-fold models
        # fitted once more to place the operating point (inner_cv_train_scores).
        'final_refits': n_outer,
        'oof_fits': n_outer * n_inner,
        'total_fits': inner_fits + n_outer + n_outer * n_inner,
    }


def print_nested_full_cost_estimate(sizes, seconds_per_fit=None):
    """Human-readable version of _nested_full_grid_sizes()."""
    print("\n" + "=" * 90)
    print("COMPUTE COST ESTIMATE - run_nested_experiment_full")
    print("=" * 90)
    print("Classifier hyperparameter combinations:")
    for clf_name, n in sizes['per_classifier_hyperparams'].items():
        print(f"    {clf_name:<26} {n:>6,}")
    print(f"    {'TOTAL':<26} {sizes['hyperparams_total']:>6,}")
    print()
    print(f"  features                              : {sizes['n_features']:>6,}")
    print(f"  ranking functions                     : {sizes['n_selection_types']:>6,}")
    print(f"  k (top_genes) values                  : {sizes['n_top_genes']:>6,}")
    print(f"  classifier hyperparameters            : {sizes['hyperparams_total']:>6,}")
    print(f"  -> GridSearchCV candidates per feature: {sizes['candidates_per_feature']:>12,}")
    print(f"  x 3 inner folds                       : {sizes['fits_per_feature_per_outer_fold']:>12,}  (per feature, per outer fold)")
    print(f"  x {sizes['n_features']} feature(s)                        : {sizes['fits_per_outer_fold']:>12,}  (per outer fold)")
    print(f"  x 10 outer folds                      : {sizes['inner_fits_total']:>12,}  INNER FITS")
    print()
    print(f"  + final refit of winner (1/outer fold): {sizes['final_refits']:>12,}")
    print(f"  + winner's inner-fold fits (op. point): {sizes['oof_fits']:>12,}")
    print(f"  = TOTAL MODEL FITS                    : {sizes['total_fits']:>12,}")
    if seconds_per_fit:
        core_hours = sizes['total_fits'] * seconds_per_fit / 3600.0
        print(f"\n  At {seconds_per_fit:.3f} s/fit -> {core_hours:,.1f} core-hours "
              f"({core_hours / 32:,.1f} h on 32 cores at perfect scaling)")
    print("=" * 90 + "\n")


def _nested_full_inner_splits(strat_train, y_train, fold_idx):
    """
    Inner CV splits of one outer-training partition (local indices), shared by the
    inner search, the operating-point thresholds and --nested_full_refresh_thresholds
    so that all three see the same folds.

    Inner stratification: fine-grained labels when every class is large enough for
    3 folds, else fall back to the binary labels (and, if even a binary class has
    fewer than 3 members, shrink the inner fold count). Only small-subset runs
    (--nested_full_sizesweep) ever take the fallbacks; full-cohort runs are unchanged.
    """
    inner_strat = strat_train
    min_class = np.unique(inner_strat, return_counts=True)[1].min()
    if min_class < 3:
        print(f"  NOTE: smallest stratification class in the training partition has "
              f"{min_class} member(s); inner CV stratifies on the binary label instead.")
        inner_strat = y_train
        min_class = np.unique(inner_strat, return_counts=True)[1].min()
    n_inner = int(min(3, min_class))
    if n_inner < 2:
        raise RuntimeError(
            f"Fold {fold_idx + 1}: cannot build an inner CV - a class has only "
            f"{min_class} training sample(s).")
    if n_inner < 3:
        print(f"  NOTE: inner CV reduced to {n_inner} folds (smallest class: {min_class}).")
    inner_cv = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=42)
    return list(inner_cv.split(np.zeros((len(y_train), 1)), inner_strat))


def _nested_full_one_fold(fold_idx, train_idx, test_idx, all_feature_arrays, y_labels,
                          stratification_labels, sample_ids, CLASSIFIERS, CLF_PARAM_GRIDS,
                          feature_names, top_genes_list, selection_types, classifier_names,
                          n_jobs, cache_dir):
    """
    One outer fold of the fully nested procedure. Returns (fold_record, oof_rows).

    Everything supervised - TSS ranking included - is fitted inside the inner-training
    partitions; the outer test partition is touched exactly once, at the end.
    """
    fold_start = time.time()
    y_train, y_test = y_labels[train_idx], y_labels[test_idx]
    strat_train = stratification_labels[train_idx]
    strat_test = stratification_labels[test_idx]
    sample_ids_test = sample_ids[test_idx]

    print(f"\n{'=' * 90}")
    print(f"[Outer fold {fold_idx + 1}/10] train={len(train_idx)} test={len(test_idx)} | "
          f"train +{int(np.sum(y_train == 1))}/-{int(np.sum(y_train == 0))} | "
          f"test +{int(np.sum(y_test == 1))}/-{int(np.sum(y_test == 0))}")
    print(f"{'=' * 90}")

    def _make_pipeline():
        # covariate block (COVARIATE_BLOCK, set in main): 'augment' keeps the last n
        # columns past the ranking; 'residualize' regresses them out of the TSSs first
        cov_mode, n_cov = COVARIATE_BLOCK['mode'], int(COVARIATE_BLOCK['n'])
        steps = []
        if cov_mode == 'residualize':
            steps.append(('residualizer', CovariateResidualizer(n_covariates=n_cov)))
        steps += [
            ('selector', TopGeneSelector(n_passthrough=n_cov if cov_mode == 'augment' else 0)),
            ('scaler', StandardScaler()),
            ('clf', CLASSIFIERS[classifier_names[0]]),  # placeholder, set by the grid
        ]
        return Pipeline(steps, memory=cache_dir) if cache_dir else Pipeline(steps)

    def _build_param_grid():
        """One dict per classifier, searching k, ranking function and that classifier's
        hyperparameters jointly (same shape as run_best_model_search)."""
        grid = []
        for clf_name in classifier_names:
            d = {k: list(v) for k, v in CLF_PARAM_GRIDS[clf_name].items()}
            d['clf'] = [CLASSIFIERS[clf_name]]
            d['selector__n_top_genes'] = list(top_genes_list)
            d['selector__selection_type'] = list(selection_types)
            grid.append(d)
        return grid

    def _resolve_clf_name(clf_obj):
        for name, inst in CLASSIFIERS.items():
            if inst is clf_obj:
                return name
        for name, inst in CLASSIFIERS.items():
            if type(inst) is type(clf_obj):
                return name
        return type(clf_obj).__name__

    inner_splits = _nested_full_inner_splits(strat_train, y_train, fold_idx)

    # --- Inner search, once per feature, on this fold's training partition only ---
    per_feature = {}
    for feature_name in feature_names:
        X_train = all_feature_arrays[feature_name][train_idx]
        t0 = time.time()
        grid_search = GridSearchCV(
            estimator=_make_pipeline(),
            param_grid=_build_param_grid(),
            scoring='roc_auc',
            cv=inner_splits,
            n_jobs=n_jobs,
            refit=False,      # the winner is refitted once, after the feature is chosen
            error_score=np.nan,
        )
        grid_search.fit(X_train, y_train)
        score = grid_search.best_score_
        score = -np.inf if score is None or not np.isfinite(score) else float(score)
        per_feature[feature_name] = {'best_score': score, 'best_params': grid_search.best_params_}
        print(f"  [{feature_name:<16}] inner best roc_auc = {score:.4f} "
              f"({len(grid_search.cv_results_['params']):,} candidates, {time.time() - t0:.1f}s)")

    # --- Feature chosen on inner scores only ---
    best_feature = max(feature_names, key=lambda f: per_feature[f]['best_score'])
    best_params = per_feature[best_feature]['best_params']
    inner_best_score = per_feature[best_feature]['best_score']
    if not np.isfinite(inner_best_score):
        raise RuntimeError(f"Fold {fold_idx + 1}: every candidate failed on every feature.")

    clf_name_sel = _resolve_clf_name(best_params.get('clf'))
    clf_hyperparams = {k: v for k, v in best_params.items() if k.startswith('clf__')}
    sel_type = best_params.get('selector__selection_type')
    n_top = best_params.get('selector__n_top_genes')
    print(f"  -> selected: feature={best_feature} | selection_type={sel_type} | "
          f"k={n_top} | clf={clf_name_sel} | inner best_score_={inner_best_score:.4f}")

    # --- Refit the winning configuration on the whole outer-training partition ---
    X_best = all_feature_arrays[best_feature]
    X_train_best, X_test_best = X_best[train_idx], X_best[test_idx]
    best_model = clone(_make_pipeline()).set_params(**best_params)
    best_model.fit(X_train_best, y_train)

    selected_indices = best_model.named_steps['selector'].indices_
    selected_indices = np.asarray(selected_indices).tolist() if selected_indices is not None else []
    if len(selected_indices) == 0:
        # TopGeneSelector falls back to a single zero column here, which silently yields
        # constant predictions and AUC 0.5. Fail loudly instead.
        raise RuntimeError(
            f"Fold {fold_idx + 1}: selector chose 0 genes on feature '{best_feature}' "
            f"(k={n_top}, matrix has {X_best.shape[1]} columns)."
        )

    # --- One prediction on the untouched outer test partition ---
    y_pred_proba = best_model.predict_proba(X_test_best)[:, 1]
    fpr_full, tpr_full, _ = roc_curve(y_test, y_pred_proba)
    auc_full = auc(fpr_full, tpr_full)
    precision_full, recall_full, _ = precision_recall_curve(y_test, y_pred_proba)

    # --- Operating point from TRAINING data only: the winning configuration's
    # held-out scores on the inner validation folds it was selected on ---
    op_thr, _ = inner_cv_operating_thresholds(
        best_model, X_train_best, y_train, inner_splits,
        expected_mean_auc=inner_best_score, label=f"fold {fold_idx + 1}")

    y_pred_default = (y_pred_proba >= 0.5).astype(int)
    report_full, confusion_full, _, full_model_thresholds = _operating_point_reports(
        y_test, y_pred_proba, y_pred_default, op_thr, sample_ids_test)

    fold_elapsed = time.time() - fold_start
    print(f"  -> held-out AUC = {auc_full:.4f}   (fold took {fold_elapsed:.1f}s)")

    fold_record = {
        'fold': fold_idx + 1,
        'feature': best_feature,
        'selection_type': sel_type,
        'n_top_genes': n_top,
        'classifier': clf_name_sel,
        'clf_hyperparams': clf_hyperparams,
        'inner_best_score': inner_best_score,
        'test_auc': float(auc_full),
        'selected_gene_indices': selected_indices,
        'num_genes_used': len(selected_indices),
        'inner_best_score_per_feature': {f: per_feature[f]['best_score'] for f in feature_names},
        # Width of every candidate matrix: how many regions k was selected from.
        # Lets the comparator region sweep (RNA regions, ATAC TFs) report "k of N"
        # without re-loading the features.
        'n_features_available': {f: int(all_feature_arrays[f].shape[1]) for f in feature_names},
        'train_sample_ids': sample_ids[train_idx].tolist(),
        'test_sample_ids': sample_ids_test.tolist(),
        'y_test': y_test.tolist(),
        'y_pred_proba': y_pred_proba.tolist(),
        'fpr_full_model': fpr_full,
        'tpr_full_model': tpr_full,
        'precision_full_model': precision_full,
        'recall_full_model': recall_full,
        'classification_report_full_model': report_full,
        'confusion_matrix_full_model': confusion_full,
        'full_model_thresholds': full_model_thresholds,
        'fold_elapsed_seconds': fold_elapsed,
    }
    if COVARIATE_BLOCK['mode']:
        fold_record['covariate_block'] = _covariate_block_report(
            best_model, X_train_best, X_test_best, y_test, y_pred_proba, label=f"fold {fold_idx + 1}")
        cb = fold_record['covariate_block']
        print(f"  -> covariate block ({cb['mode']}, {cb['n_covariates']} covariates): "
              + (f"perm drop covariates {cb['perm_drop_covariates_mean']:+.3f} / TSS {cb['perm_drop_tss_mean']:+.3f}, "
                 f"TSS selected {cb['n_tss_selected']}" if cb['mode'] == 'augment'
                 else f"median R2 of TSS on covariates {cb['residual_r2_median']:.3f}"))

    oof_rows = [
        {
            'sample_id': sid,
            'y_true': int(y_true_i),
            'cancer_type': strat_i if int(y_true_i) == 1 else None,
            'stratification_label': strat_i,
            'y_pred_proba': float(proba_i),
            'outer_fold': fold_idx + 1,
        }
        for sid, y_true_i, strat_i, proba_i
        in zip(sample_ids_test, y_test, strat_test, y_pred_proba)
    ]

    return fold_record, oof_rows


def _nested_full_finalize(output_dir, fold_records, oof_rows, sizes, meta, elapsed, n_samples):
    """Assemble the three requested outputs, write them to disk and print the table."""

    fold_records = sorted(fold_records, key=lambda r: r['fold'])
    per_fold_df = pd.DataFrame(fold_records)
    oof_df = pd.DataFrame(oof_rows).sort_values(['outer_fold', 'sample_id']).reset_index(drop=True)

    # The outer CV is a partition, so every sample must appear exactly once.
    if len(oof_df) != n_samples:
        raise RuntimeError(f"OOF table has {len(oof_df)} rows for {n_samples} samples.")
    dupes = oof_df['sample_id'][oof_df['sample_id'].duplicated()].tolist()
    if dupes:
        raise RuntimeError(f"OOF table contains duplicated sample_ids: {dupes[:10]}")

    aucs = per_fold_df['test_auc'].to_numpy(dtype=float)
    summary = {
        'n_outer_folds': len(per_fold_df),
        'mean_test_auc': float(np.mean(aucs)),
        'std_test_auc': float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
        'min_test_auc': float(np.min(aucs)),
        'max_test_auc': float(np.max(aucs)),
        'oof_auc_pooled': float(roc_auc_score(oof_df['y_true'], oof_df['y_pred_proba'])),
        'n_samples': int(n_samples),
        'grid_sizes': sizes,
        'elapsed_seconds': elapsed,
        'seconds_per_fit_observed': elapsed / max(sizes['total_fits'], 1),
    }
    summary.update(meta)

    per_fold_path = output_dir / "nested_full_per_fold.pkl"
    with open(per_fold_path, 'wb') as fh:
        pickle.dump(per_fold_df, fh)

    # Flat CSV of the selection table (the large arrays stay in the pickle).
    table_cols = ['fold', 'feature', 'selection_type', 'n_top_genes', 'classifier',
                  'clf_hyperparams', 'inner_best_score', 'test_auc', 'num_genes_used']
    per_fold_csv = per_fold_df[table_cols].copy()
    per_fold_csv['clf_hyperparams'] = per_fold_csv['clf_hyperparams'].apply(
        lambda d: "; ".join(f"{k.replace('clf__', '')}={v}" for k, v in sorted(d.items()))
    )
    per_fold_csv.to_csv(output_dir / "nested_full_per_fold.csv", index=False)

    # Selected TSS indices, one entry per fold (kept separate: they can be long).
    with open(output_dir / "nested_full_selected_indices.pkl", 'wb') as fh:
        pickle.dump({r['fold']: r['selected_gene_indices'] for r in fold_records}, fh)

    oof_df.to_csv(output_dir / "nested_full_oof_predictions.csv", index=False)
    with open(output_dir / "nested_full_summary.json", 'w') as fh:
        json.dump(summary, fh, indent=2, default=str)

    print("\n" + "=" * 90)
    print("FULLY NESTED SELECTION - PER-FOLD RESULTS")
    print("=" * 90)
    print(per_fold_csv.to_string(index=False))
    print("\n" + "-" * 90)
    print(f"Held-out AUC over {summary['n_outer_folds']} outer folds: "
          f"mean = {summary['mean_test_auc']:.4f}, sd = {summary['std_test_auc']:.4f} "
          f"(min {summary['min_test_auc']:.4f}, max {summary['max_test_auc']:.4f})")
    print(f"Pooled out-of-fold AUC over all {summary['n_samples']} samples: "
          f"{summary['oof_auc_pooled']:.4f}")
    n_distinct = len(per_fold_csv[['feature', 'selection_type', 'n_top_genes', 'classifier']]
                     .drop_duplicates())
    print(f"Distinct configurations selected across folds: {n_distinct}"
          f"{'  (unanimous)' if n_distinct == 1 else ''}")
    print(f"Total compute time: {elapsed / 60:.1f} min "
          f"({summary['seconds_per_fit_observed'] * 1000:.1f} ms per fit)")
    print("-" * 90)
    print("Saved:")
    for name in ["nested_full_per_fold.pkl", "nested_full_per_fold.csv",
                 "nested_full_selected_indices.pkl", "nested_full_oof_predictions.csv",
                 "nested_full_summary.json"]:
        print(f"  {output_dir / name}")
    print("=" * 90 + "\n")

    return {'per_fold': per_fold_df, 'oof_predictions': oof_df,
            'summary': summary, 'output_dir': output_dir}


def _nested_full_aggregate_dir(out_dir, n_samples, fallback_sizes=None, fallback_meta=None,
                               expected_folds=10):
    """
    Combine the per-fold partials written by only_fold in one directory into the final
    outputs. Used for a plain run, for each k of a --nested_full_ksweep, and for each
    fraction of a --nested_full_sizesweep (where expected_folds may be below 10 because
    small subsets cannot sustain a 10-fold stratified outer partition).
    """
    def _search_space_key(meta):
        # 'label_permutation_seed' is absent in older folds; a fold without the key and a fold
        # with None ran the same search (no permutation)
        return {k: v for k, v in (meta or {}).items() if not (k == 'label_permutation_seed' and v is None)}

    out_dir = Path(out_dir)
    partials = sorted(out_dir.glob("nested_full_fold_*.pkl"))
    if not partials:
        raise FileNotFoundError(f"No nested_full_fold_*.pkl partials in {out_dir}")
    fold_records, oof_rows, elapsed = [], [], 0.0
    part_sizes, part_meta = None, None
    for p in partials:
        with open(p, 'rb') as fh:
            part = pickle.load(fh)
        fold_records.append(part['fold_record'])
        oof_rows.extend(part['oof_rows'])
        elapsed += part['fold_record'].get('fold_elapsed_seconds', 0.0)
        # Describe what the folds actually ran with, not what this aggregation job
        # happened to be given on the command line.
        if part_meta is None:
            part_sizes, part_meta = part.get('sizes'), part.get('meta')
        elif _search_space_key(part.get('meta')) != _search_space_key(part_meta):
            raise RuntimeError(
                f"{p.name} was produced with a different search space than "
                f"{partials[0].name}:\n  {part.get('meta')}\n  vs\n  {part_meta}"
            )
    found = sorted(r['fold'] for r in fold_records)
    missing = [f for f in range(1, int(expected_folds) + 1) if f not in found]
    if missing:
        raise RuntimeError(f"Missing outer fold partial(s): {missing} (found {found})")
    print(f"Aggregating {len(fold_records)} fold partials from {out_dir}")
    return _nested_full_finalize(out_dir, fold_records, oof_rows,
                                 part_sizes or fallback_sizes, part_meta or fallback_meta,
                                 elapsed, n_samples)


def _ksweep_derive_tuned_arm(output_dir, per_k_results, sizes, meta, n_samples):
    """
    The "k tuned as a hyperparameter" arm, derived from a completed k-sweep.

    Every ksweep_k<k>/ point is a complete nested selection at fixed k on the same
    outer folds and the same seeded inner splits, so per outer fold the k with the
    highest inner-CV best score is precisely what one fully nested run over the whole
    k grid would have selected (GridSearchCV's argmax over the joint grid equals the
    argmax over the per-k argmaxes; the winning configuration's refit and prediction
    are deterministic). The derived arm therefore IS the fully nested result for the
    k grid, at no extra compute - verified on the brca_crc data-driven pair (10/10
    folds identical to the standalone run: k, scores, thresholds).

    Ties between k values (identical inner scores) go to the smallest k. Writes the
    standard result files into <output_dir>/ksweep_tuned/ so downstream code treats
    it like any other nested_full directory; the summary carries
    derived_from_ksweep=True and the per-fold inner scores of every k.
    """
    tuned_dir = Path(output_dir) / "ksweep_tuned"
    os.makedirs(tuned_dir, exist_ok=True)
    ks = sorted(per_k_results)
    folds = sorted(per_k_results[ks[0]]['per_fold']['fold'].astype(int))
    fold_records, oof_rows, elapsed = [], [], 0.0
    print("=" * 110)
    print("k TUNED AS A HYPERPARAMETER - per outer fold, the k with the best inner-CV score")
    print("=" * 110)
    for fold in folds:
        inner_by_k = {}
        for k in ks:
            pf = per_k_results[k]['per_fold']
            row = pf[pf['fold'].astype(int) == fold]
            if len(row) != 1:
                raise RuntimeError(f"ksweep_k{k:05d}: fold {fold} appears {len(row)} times.")
            inner_by_k[k] = float(row.iloc[0]['inner_best_score'])
            elapsed += float(row.iloc[0].get('fold_elapsed_seconds', 0.0))
        best_k = max(ks, key=lambda k: (inner_by_k[k], -k))   # ties -> smallest k
        pf = per_k_results[best_k]['per_fold']
        rec = pf[pf['fold'].astype(int) == fold].iloc[0].to_dict()
        rec['ksweep_inner_scores'] = inner_by_k
        rec['ksweep_k_selected'] = int(best_k)
        fold_records.append(rec)
        oof = per_k_results[best_k]['oof_predictions']
        oof_rows.extend(oof[oof['outer_fold'].astype(int) == fold].to_dict('records'))
        runners = sorted(ks, key=lambda k: -inner_by_k[k])[1:3]
        print(f"  fold {fold:>2}: k = {best_k:>6} (inner {inner_by_k[best_k]:.4f}, "
              f"test AUC {rec['test_auc']:.4f}; runner-up "
              + ", ".join(f"k={k} {inner_by_k[k]:.4f}" for k in runners) + ")")
    tuned_meta = dict(meta, top_genes_searched=[int(k) for k in ks],
                      derived_from_ksweep=True,
                      tie_break="smallest k on equal inner score")
    print(f"\nWriting the derived arm to {tuned_dir} (same layout as a plain nested_full run)")
    return _nested_full_finalize(tuned_dir, fold_records, oof_rows, sizes, tuned_meta,
                                 elapsed, n_samples)


def _sizesweep_subsets(fractions, stratification_labels, seed):
    """
    Deterministic, nested, stratified sample subsets for --nested_full_sizesweep.

    Per stratum (fine-grained disease label, so healthy fraction AND cancer-type
    composition are preserved) the sample indices are shuffled ONCE with `seed`;
    fraction f keeps the first max(1, round(f * n_stratum)) of that fixed order.
    Because every subset is a prefix of the same shuffle:
      - subsets nest: subset(0.8) is a strict subset of subset(0.9), so the curve
        reflects removing samples, not redrawing cohorts;
      - each subset depends only on (seed, cohort), NOT on which fractions a given
        submission runs - fractions can be submitted in chunks and still agree.
    The max(1, ...) floor keeps every stratum represented (a rare cancer type is
    never silently dropped from the task at small fractions).

    Returns {fraction: sorted global index array}.
    """
    rng = np.random.default_rng(seed)
    stratification_labels = np.asarray(stratification_labels)
    perms = []
    for s in pd.unique(stratification_labels):  # first-appearance order: stable
        idx = np.flatnonzero(stratification_labels == s)
        perms.append(rng.permutation(idx))
    out = {}
    for f in fractions:
        if not 0.0 < f <= 1.0:
            raise ValueError(f"sizesweep fractions must be in (0, 1], got {f}")
        keep = np.concatenate([p[:max(1, int(round(f * len(p))))] for p in perms])
        out[f] = np.sort(keep)
    return out


def _sizesweep_dir_name(fraction):
    """'sizesweep_f090' for 0.9 - fixed width, sorts with the sweep."""
    return f"sizesweep_f{int(round(fraction * 100)):03d}"


def _refresh_thresholds_one_partial(partial_path, all_feature_arrays, y_labels,
                                    stratification_labels, sample_ids, CLASSIFIERS):
    """
    Worker of --nested_full_refresh_thresholds: re-derive the operating-point
    thresholds of ONE stored fold partial from the winning configuration's scores on
    the inner validation folds, without repeating the inner search.

    The stored winner (feature, ranking function, k, classifier, hyperparameters) is
    refitted on the stored training partition in the stored sample order and must
    reproduce the stored test-fold predictions before anything is rewritten - that
    proves the loaded cohort and features are those of the original run. The stored
    predictions and AUROC are left untouched; only the thresholds and the reports /
    confusion matrices that depend on them change. The thresholds being replaced are
    kept in the record as 'full_model_thresholds_previous'.
    """
    partial_path = Path(partial_path)
    tag = f"{partial_path.parent.name}/{partial_path.name}"
    with open(partial_path, 'rb') as fh:
        part = pickle.load(fh)
    rec = part['fold_record']
    prev = dict(rec['full_model_thresholds'])
    out = {'partial': tag, 'fold': int(rec['fold']), 'classifier': rec['classifier'],
           'threshold_95spec_previous': float(prev['95_percent_specificity']),
           'threshold_95spec': float(prev['95_percent_specificity']), 'status': 'unchanged'}
    if prev.get('threshold_source') == THRESHOLD_SOURCE_INNER_CV:
        out['status'] = 'already inner-CV'
        return out

    id_to_idx = {sid: i for i, sid in enumerate(np.asarray(sample_ids))}
    train_ids, test_ids = list(rec['train_sample_ids']), list(rec['test_sample_ids'])
    missing = [sid for sid in train_ids + test_ids if sid not in id_to_idx]
    if missing:
        raise KeyError(
            f"{tag}: {len(missing)} stored sample ID(s) are absent from the loaded cohort "
            f"(e.g. {missing[:3]}). Use the data arguments of the original run.")
    if rec['feature'] not in all_feature_arrays or rec['classifier'] not in CLASSIFIERS:
        raise KeyError(f"{tag}: feature '{rec['feature']}' or classifier "
                       f"'{rec['classifier']}' is not available in this invocation.")
    train_idx = np.array([id_to_idx[sid] for sid in train_ids])
    test_idx = np.array([id_to_idx[sid] for sid in test_ids])
    y_train, y_test = y_labels[train_idx], y_labels[test_idx]
    if not np.array_equal(y_test, np.asarray(rec['y_test'])):
        raise ValueError(f"{tag}: stored y_test does not match the current labels of the "
                         f"stored test samples - cohort/labels differ from the original run.")

    X = all_feature_arrays[rec['feature']]
    pipe = Pipeline([('selector', TopGeneSelector()),
                     ('scaler', StandardScaler()),
                     ('clf', CLASSIFIERS[rec['classifier']])])
    pipe = clone(pipe).set_params(
        selector__n_top_genes=int(rec['n_top_genes']),
        selector__selection_type=rec['selection_type'],
        **dict(rec['clf_hyperparams']))
    pipe.fit(X[train_idx], y_train)
    stored_proba = np.asarray(rec['y_pred_proba'], dtype=float)
    proba = pipe.predict_proba(X[test_idx])[:, 1]
    if not np.allclose(proba, stored_proba, atol=1e-6):
        raise RuntimeError(
            f"{tag}: refit test predictions deviate from the stored ones (max diff "
            f"{np.abs(proba - stored_proba).max():.3e}) - the refit does not reproduce "
            f"the stored fold model, thresholds NOT rewritten.")

    inner_splits = _nested_full_inner_splits(
        stratification_labels[train_idx], y_train, int(rec['fold']) - 1)
    op_thr, _ = inner_cv_operating_thresholds(
        pipe, X[train_idx], y_train, inner_splits,
        expected_mean_auc=rec.get('inner_best_score'), label=tag)
    report_full, confusion_full, _, full_model_thresholds = _operating_point_reports(
        y_test, stored_proba, (stored_proba >= 0.5).astype(int), op_thr,
        np.asarray(test_ids))

    rec['full_model_thresholds_previous'] = prev
    rec['classification_report_full_model'] = report_full
    rec['confusion_matrix_full_model'] = confusion_full
    rec['full_model_thresholds'] = full_model_thresholds
    tmp_path = partial_path.with_name(partial_path.name + '.tmp')
    with open(tmp_path, 'wb') as fh:
        pickle.dump(part, fh)
    os.replace(tmp_path, partial_path)
    out.update(threshold_95spec=float(full_model_thresholds['95_percent_specificity']),
               status='refreshed')
    return out


def _nested_full_refresh_thresholds(output_dir, all_feature_arrays, y_labels,
                                    stratification_labels, sample_ids, CLASSIFIERS,
                                    n_jobs=1):
    """
    Re-derive the operating-point thresholds of every stored fold partial of one fully
    nested experiment (the plain run and any ksweep_k* / sizesweep_f* subdirectories)
    with inner_cv_operating_thresholds(), i.e. from the winner's scores on the inner
    validation folds. Costs 1 + n_inner fits per fold instead of a new inner search.
    Aggregate afterwards (same call with --nested_full_aggregate) to rebuild the tables.
    """
    output_dir = Path(output_dir)
    dirs = [output_dir] + sorted(output_dir.glob("ksweep_k*")) + sorted(output_dir.glob("sizesweep_f*"))
    partials = [p for d in dirs for p in sorted(d.glob("nested_full_fold_*.pkl"))]
    if not partials:
        raise FileNotFoundError(f"No nested_full_fold_*.pkl partials under {output_dir}")
    print(f"Refreshing operating-point thresholds of {len(partials)} fold partial(s) "
          f"under {output_dir} (inner-CV scores of the stored winning configuration)")
    rows = Parallel(n_jobs=max(1, min(int(n_jobs), len(partials))), verbose=5)(
        delayed(_refresh_thresholds_one_partial)(
            p, all_feature_arrays, y_labels, stratification_labels, sample_ids, CLASSIFIERS)
        for p in partials)
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "threshold_refresh_log.csv", index=False)
    print(table.to_string(index=False))
    print(f"{int((table['status'] == 'refreshed').sum())} partial(s) rewritten, "
          f"{int((table['status'] != 'refreshed').sum())} already current. "
          f"Log: {output_dir / 'threshold_refresh_log.csv'}")
    return table


# ---------------------------------------------------------------------------
# Cross-cluster work claims. The same experiment may be submitted on two
# clusters that mount the same storage. Every unit
# of work that writes one file - a fold partial, a per-k / per-fraction partial, a
# covariate-panel result pickle, an aggregate - is claimed through a lock file next
# to its output (<output>.lock: atomic O_EXCL creation, holder written inside,
# touched by a heartbeat thread every minute). A worker that meets a live claim
# moves on to its next unit and comes back at the end to wait for the result; a
# claim without heartbeat for NESTED_FULL_LOCK_STALE_SECONDS (default 20 min: the
# holder was killed) is taken over. Outputs are written to a temporary name and
# renamed into place, so a reader never sees a half-written file.
# ---------------------------------------------------------------------------
WORK_CLAIM_STALE_S = float(os.environ.get('NESTED_FULL_LOCK_STALE_SECONDS', 20 * 60))
WORK_CLAIM_HEARTBEAT_S = float(os.environ.get('NESTED_FULL_LOCK_HEARTBEAT_SECONDS', 60))
WORK_CLAIM_POLL_S = float(os.environ.get('NESTED_FULL_LOCK_POLL_SECONDS', 60))
WORK_CLAIM_VERIFY_S = float(os.environ.get('NESTED_FULL_LOCK_VERIFY_SECONDS', 2))
WORK_CLAIM_MAX_WAIT_S = float(os.environ.get('NESTED_FULL_LOCK_MAX_WAIT_SECONDS', 12 * 3600))


def _atomic_pickle(path, obj):
    """Write `obj` to `path` through a temporary file and an atomic rename."""
    path = Path(path)
    tmp = path.with_name(f"{path.name}.{socket.gethostname()}.{os.getpid()}.tmp")
    with open(tmp, 'wb') as fh:
        pickle.dump(obj, fh)
    os.replace(tmp, path)


class _WorkClaim:
    """Claim on one unit of work whose output file is `target` (lock: <target>.lock)."""

    def __init__(self, target, label=None):
        self.target = Path(target)
        self.lock = self.target.with_name(self.target.name + '.lock')
        self.label = label or self.target.name
        self.token = (f"{socket.gethostname()} job={os.environ.get('SLURM_JOB_ID', '-')} "
                      f"pid={os.getpid()} since={time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.held = False
        self._stop = None
        self._thread = None

    def holder(self):
        try:
            return self.lock.read_text().strip()
        except OSError:
            return None

    def age_s(self):
        try:
            return time.time() - self.lock.stat().st_mtime
        except OSError:
            return None

    def try_acquire(self):
        """True when this worker now holds the claim (heartbeat running)."""
        for _ in range(3):
            try:
                fd = os.open(self.lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                age = self.age_s()
                if age is None:                 # vanished between the two calls: retry
                    continue
                if age > WORK_CLAIM_STALE_S:
                    print(f"  claim on {self.label}: stale lock ({age / 60:.0f} min without "
                          f"heartbeat, holder: {self.holder()}) - taking over")
                    try:
                        os.remove(self.lock)
                    except OSError:
                        pass
                    continue
                return False
            with os.fdopen(fd, 'w') as fh:
                fh.write(self.token)
            time.sleep(WORK_CLAIM_VERIFY_S)     # a simultaneous takeover may have replaced the file
            if self.holder() != self.token:
                return False
            self.held = True
            self._stop = threading.Event()
            self._thread = threading.Thread(target=self._heartbeat, daemon=True)
            self._thread.start()
            return True
        return False

    def _heartbeat(self):
        while not self._stop.wait(WORK_CLAIM_HEARTBEAT_S):
            try:
                os.utime(self.lock, None)
            except OSError:
                pass

    def release(self):
        if self._stop is not None:
            self._stop.set()
            self._thread.join(timeout=5)
            self._stop = None
        if self.held:
            try:
                if self.holder() == self.token:
                    os.remove(self.lock)
            except OSError:
                pass
            self.held = False

    def wait(self, is_done):
        """Block until `is_done()` (another worker delivered the output) or until this
        worker acquires the claim (the holder released it without output, or died).
        Returns 'done' or 'acquired'."""
        t0, last = time.time(), None
        while True:
            if is_done():
                return 'done'
            if self.try_acquire():
                return 'acquired'
            waited = time.time() - t0
            if waited > WORK_CLAIM_MAX_WAIT_S:
                raise RuntimeError(f"{self.label}: waited {waited / 3600:.1f} h for "
                                   f"{self.holder()} - giving up")
            if last is None or waited - last >= 600:
                age = self.age_s()
                print(f"  waiting for {self.label}: held by {self.holder()} "
                      f"(heartbeat {age / 60:.0f} min ago; {waited / 60:.0f} min waited)"
                      if age is not None else f"  waiting for {self.label}")
                last = waited
            time.sleep(WORK_CLAIM_POLL_S)


def _sizesweep_partial_ok(path, frac, subsample_seed):
    try:
        with open(path, 'rb') as fh:
            prev = pickle.load(fh)
        return bool(np.isclose(prev['meta']['sample_fraction'], frac)
                    and prev['meta']['subsample_seed'] == int(subsample_seed))
    except Exception:
        return False


def _nested_full_existing_partial(path, fold_number, meta_expected, expected_test_ids):
    """Resume support for the per-fold steps of run_nested_experiment_full.

    Returns the stored partial when `path` holds a readable pickle written for this
    outer fold (identical held-out sample IDs, so the cohort and the split are the
    same) with the same search grid (features, ranking functions, k values,
    classifiers) and the same covariate block; returns None otherwise, so the fold
    is (re)computed. A truncated pickle (task killed mid-write) fails to load and
    is recomputed. Delete a partial to force recomputation.
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        with open(path, 'rb') as fh:
            prev = pickle.load(fh)
        rec, pmeta = prev['fold_record'], prev['meta']
    except Exception as e:
        print(f"\nExisting partial {path} unreadable ({e!r}) - recomputing.")
        return None
    same_fold = (int(rec.get('fold', -1)) == int(fold_number)
                 and list(rec.get('test_sample_ids', [])) == [str(x) for x in expected_test_ids])
    grid_keys = ('features_searched', 'selection_types_searched',
                 'top_genes_searched', 'classifiers_searched')
    same_grid = all(list(pmeta.get(k, [])) == list(meta_expected[k]) for k in grid_keys)
    cb = rec.get('covariate_block') or {}          # absent in partials written before the block existed = no block
    same_cov = (cb.get('mode') == COVARIATE_BLOCK['mode']
                and list(cb.get('names') or []) == list(COVARIATE_BLOCK['names']))
    if same_fold and same_grid and same_cov:
        print(f"\nPartial for outer fold {fold_number} already exists ({path}) - skipping. "
              f"Delete it to force recomputation.")
        return prev
    why = ("different fold / cohort (held-out IDs differ)" if not same_fold
           else "different search grid" if not same_grid else "different covariate block")
    print(f"\nExisting partial {path}: {why} - recomputing.")
    return None


def run_nested_experiment_full(all_feature_arrays: dict,
                               y_labels: np.ndarray,
                               stratification_labels: np.ndarray,
                               sample_ids,
                               CLASSIFIERS: dict,
                               CLF_PARAM_GRIDS: dict,
                               DATA_PARAM_GRIDS: dict,
                               args: argparse.Namespace,
                               feature_names=None,
                               top_genes_list=None,
                               selection_types=None,
                               classifier_names=None,
                               cache_dir=None,
                               estimate_only=False,
                               only_fold=None,
                               aggregate=False,
                               ksweep=False,
                               sizesweep=False,
                               sizesweep_fractions=None,
                               subsample_seed=42,
                               refresh_thresholds=False):
    """
    Fully nested model selection in a single pipeline.

    run_single_experiment() fixes feature / ranking function / k / classifier outside the
    cross-validation and tunes only the classifier hyperparameters inside it; each
    combination is a separate job, and the choice between those jobs is made afterwards
    on the same outer folds they are scored on. This function moves that entire choice
    inside the outer loop, so the reported held-out performance estimates the whole
    selection procedure rather than one hand-picked configuration.

    Structure
    ---------
    Outer loop  : StratifiedKFold(n_splits=10, shuffle=True, random_state=42) on
                  stratification_labels - the same partition run_single_experiment uses,
                  and identical across features (StratifiedKFold splits on y alone).
    Inner loop  : StratifiedKFold(n_splits=3, shuffle=True, random_state=42) on the
                  outer-training partition.
    Pipeline    : TopGeneSelector -> StandardScaler -> classifier. Because the selector is
                  a pipeline step, TSS ranking is refitted inside every inner-training
                  partition; no supervised step ever sees the fold it is scored on.
    Inner search: one GridSearchCV per feature, searching jointly over
                  selector__n_top_genes, selector__selection_type, the classifier and its
                  hyperparameters (a param_grid list with one dict per classifier).
    Feature     : cannot live in a sklearn param grid because it changes X, so it is
                  handled by running the inner search once per feature on that fold's
                  training partition and keeping the feature with the highest inner
                  best_score_. Selection therefore uses inner scores only.
    Refit       : the winning full configuration is refitted on the entire outer-training
                  partition and predicts once on the untouched outer test partition.
    Operating pt: thresholds come from the winning configuration's held-out scores on
                  the inner validation folds (inner_cv_operating_thresholds(): the
                  nested cross-validated training predictions, same inner splits as
                  the selection), then are applied unchanged to the test fold. No
                  threshold is computed from test labels.

    run_single_experiment() is left untouched; both remain runnable for comparison.

    Parallelism
    -----------
    The ten outer folds are independent, so the expensive path is a SLURM array job:
    `only_fold=N` runs outer fold N alone and writes nested_full_fold_NN.pkl, and
    `aggregate=True` combines the ten partials into the final tables. Within a fold,
    GridSearchCV already parallelises across candidates on args.cores.

    Parameters
    ----------
    feature_names, top_genes_list, selection_types, classifier_names
        Restrict the search space (used for cheap test runs). None means the full grid.
    cache_dir
        If given, used as the Pipeline `memory` cache. The selector's fit depends only on
        (inner-training partition, selection_type, n_top_genes), so caching collapses the
        repeated ranking recomputation across the classifier hyperparameters sharing it -
        a large saving for 'p_value'/'weighted', which run mannwhitneyu per gene.
    estimate_only
        Print the fit-count estimate and return without running anything.
    only_fold
        1-based outer fold index. Runs just that fold and writes a partial result.
    aggregate
        Combine previously written per-fold partials into the final outputs.
    ksweep
        Run one complete nested selection per k instead of a single selection over the
        whole k grid, writing each k into its own ksweep_k<k>/ subdirectory. Feature,
        ranking function and hyperparameters are still chosen inside the fold; only k is
        held fixed. This yields a k trend in which every point is leakage-free, unlike a
        curve read off per-configuration outer-fold scores. Combine with only_fold (the
        k loop runs inside one task, so the feature matrices are loaded once) and then
        with aggregate to produce nested_full_ksweep.csv.

        The aggregate additionally derives the "k tuned as a hyperparameter" arm from
        the sweep (ksweep_tuned/, same files as a plain run): per outer fold it keeps
        the k whose inner-CV best score is highest. Because every k was searched on
        the same seeded inner splits, this is exactly what a single fully nested run
        over the whole k grid selects (GridSearchCV's argmax over the joint grid is
        the argmax over the per-k argmaxes); verified on the brca_crc data-driven
        pair, where all ten folds reproduce the standalone run's k, predictions and
        thresholds bit for bit. A separate full run is therefore not needed to
        obtain the tuned arm - only to cross-check it.

    Comparator region sweep (reviewer: "the number of TSSs was tuned as a
    hyperparameter while the comparator methods use fixed numbers of regions")
        The procedure is agnostic to what the columns are, so the same fully nested
        selection - k included - runs on the comparators' own region sets:
          RNA  : --no_gene_selection --features_dir extracted_features/zhu_all_after_lift
                 --nested_full_features rcov (the 792 Zhu et al. regions, feature
                 locked to the manuscript's rcov), k grid up to 792;
          ATAC : --predefined_feature_matrix extracted_features/ulz_atac_tfbs_features.npy
                 (504 Ulz TF-level features, each already the aggregate over that
                 TF's top-1000 sites), k grid up to 504.
        k values beyond a region set's width collapse onto a single "all regions"
        point (see the k-grid note in the body). slurm/nested_full.sh FEATURE_SET=rna|atac
        assembles these arguments; src/figures/export/export_figS13.py collects the
        sweeps, the tuned arms and the fixed-region manuscript arms.
    sizesweep
        Sample-size robustness sweep (reviewer request): run one COMPLETE fully nested
        selection per sample fraction, on a deterministic nested stratified subset of
        the cohort (see _sizesweep_subsets), writing each fraction into its own
        sizesweep_f<pct>/ subdirectory. Every point of the resulting learning curve is
        leakage-free: nothing outside the fraction's subset is ever touched, and all
        selection still happens inside each outer fold. Small subsets cannot sustain a
        10-fold outer partition, so the fold count per fraction is min(10, smaller
        binary class count), stratified on the disease labels while every stratum can
        appear in every fold (binary labels otherwise); array tasks whose fold index
        exceeds it skip that fraction. Combine with only_fold (the fraction loop runs
        inside one task) and then with aggregate to produce nested_full_sizesweep.csv.
    sizesweep_fractions
        Fractions to run (default 0.9 .. 0.1). Because subsets are prefixes of one
        seeded shuffle, fractions may be split across submissions and still agree.
    subsample_seed
        Seed for the per-stratum shuffle that defines the subsets (default 42).

    Returns
    -------
    dict with 'per_fold' (DataFrame), 'oof_predictions' (DataFrame), 'summary' (dict) and
    'output_dir'; the estimate dict when estimate_only=True; the partial path when
    only_fold is set.
    """

    # The four ranking functions. 'random' is the negative control used elsewhere in this
    # script, not a ranking function, so it is not in the selection grid by default; pass
    # selection_types explicitly to include it.
    RANKING_FUNCTIONS = ['mean_diff', 'p_value', 'fold_change', 'weighted']

    sample_ids = np.asarray(sample_ids)
    stratification_labels = np.asarray(stratification_labels)
    y_labels = np.asarray(y_labels)

    if feature_names is None:
        feature_names = list(DATA_PARAM_GRIDS['feature'])
    feature_names = [f for f in feature_names if all_feature_arrays.get(f) is not None]
    if not feature_names:
        raise ValueError("run_nested_experiment_full: no usable feature arrays.")

    if top_genes_list is None:
        top_genes_list = list(DATA_PARAM_GRIDS['top_genes'])
    if selection_types is None:
        selection_types = list(RANKING_FUNCTIONS)
    if classifier_names is None:
        classifier_names = list(CLASSIFIERS.keys())

    unknown = [c for c in classifier_names if c not in CLASSIFIERS]
    if unknown:
        raise ValueError(f"Unknown classifier(s) {unknown}. Available: {list(CLASSIFIERS)}")

    if sizesweep and ksweep:
        raise ValueError("--nested_full_sizesweep and --nested_full_ksweep are mutually exclusive.")

    # k grid vs region-set width. TopGeneSelector caps k at the matrix width, so
    # every k >= width is the same "all regions" candidate: searching several of
    # them wastes fits and, in a ksweep, draws duplicate points. Keep the smallest
    # such k (its label is preserved - the data-driven grid's 15000 still means
    # "all 14,224 TSSs") and drop the larger ones. For the data-driven TSS set
    # this is a no-op; it matters for the comparator region sets (792 Zhu
    # regions, 504 Ulz TFs) run through the same procedure.
    widths = {f: int(all_feature_arrays[f].shape[1]) for f in feature_names}
    max_width = max(widths.values())
    top_genes_list = sorted({int(k) for k in top_genes_list})
    beyond = [k for k in top_genes_list if k >= max_width]
    if len(beyond) > 1:
        print(f"NOTE: k values {beyond[1:]} exceed the widest region set "
              f"({max_width} columns) and would duplicate k={beyond[0]} "
              f"(= all regions); dropped from the grid.")
        top_genes_list = [k for k in top_genes_list if k < max_width] + [beyond[0]]
    print("Region-set width per feature (k is selected from these): "
          + ", ".join(f"{f}={w:,}" for f, w in widths.items()))
    if sizesweep_fractions is None:
        sizesweep_fractions = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    sizesweep_fractions = sorted({round(float(f), 4) for f in sizesweep_fractions}, reverse=True)

    sizes = _nested_full_grid_sizes(
        CLASSIFIERS, CLF_PARAM_GRIDS, classifier_names,
        top_genes_list, selection_types, feature_names,
    )
    print_nested_full_cost_estimate(sizes)
    if estimate_only:
        return sizes

    output_dir = Path(args.output_dir) / f"nested_full_{args.experiment_name}"
    os.makedirs(output_dir, exist_ok=True)

    n_samples = all_feature_arrays[feature_names[0]].shape[0]
    meta = {
        'features_searched': list(feature_names),
        'selection_types_searched': list(selection_types),
        'top_genes_searched': [int(k) for k in top_genes_list],
        'classifiers_searched': list(classifier_names),
        'label_permutation_seed': getattr(args, 'nested_full_permute_labels', None),
    }

    # ---------------------------------------------------------------- threshold refresh
    # Stored fold partials whose thresholds predate inner_cv_operating_thresholds():
    # re-derive them from the stored winning configuration (no new inner search), then
    # fall through to the aggregate mode if that was requested as well.
    if refresh_thresholds:
        refresh_table = _nested_full_refresh_thresholds(
            output_dir, all_feature_arrays, y_labels, stratification_labels, sample_ids,
            CLASSIFIERS, n_jobs=args.cores)
        if not aggregate:
            return {'threshold_refresh': refresh_table}

    # ---------------------------------------------------------------- aggregate mode
    def _run_aggregate():
        if sizesweep:
            subdirs = sorted(output_dir.glob("sizesweep_f*"), reverse=True)
            if not subdirs:
                raise FileNotFoundError(f"No sizesweep_f* subdirectories in {output_dir}")
            # Fractions aggregate independently, so an incomplete fraction (array
            # tasks timed out / still queued) is SKIPPED with a warning instead of
            # failing the whole sweep: complete fractions become inspectable while
            # the rest still runs. Rerunning the array resumes the missing folds
            # (existing partials are skipped) and a later aggregate fills the gaps.
            sweep_rows, skipped = [], []
            for sub_dir in subdirs:
                partials = sorted(sub_dir.glob("nested_full_fold_*.pkl"))
                sub_meta, present = None, set()
                for p in partials:
                    try:
                        with open(p, 'rb') as fh:
                            part = pickle.load(fh)
                    except Exception as e:
                        print(f"WARNING: {p} unreadable ({e!r}) - treated as missing "
                              f"(a rerun of that array task recomputes it).")
                        continue
                    present.add(int(part['fold_record']['fold']))
                    if sub_meta is None:
                        sub_meta = part['meta']
                if sub_meta is None:
                    print(f"WARNING: no readable fold partials in {sub_dir.name} - "
                          f"fraction skipped.")
                    skipped.append((sub_dir.name, "no readable partials"))
                    continue
                missing = sorted(set(range(1, int(sub_meta['n_outer_folds']) + 1))
                                 - present)
                if missing:
                    print(f"WARNING: {sub_dir.name} (fraction "
                          f"{sub_meta['sample_fraction']:g}): missing outer fold(s) "
                          f"{missing} of {sub_meta['n_outer_folds']} - fraction "
                          f"skipped. Resubmit the array to fill them in.")
                    skipped.append((sub_dir.name, f"missing folds {missing}"))
                    continue
                frac = sub_meta['sample_fraction']
                print(f"\n{'#' * 90}\n#  sample fraction = {frac:g} "
                      f"(n = {sub_meta['n_samples_subset']}, "
                      f"{sub_meta['n_outer_folds']} outer folds)\n{'#' * 90}")
                res = _nested_full_aggregate_dir(
                    sub_dir, sub_meta['n_samples_subset'],
                    expected_folds=sub_meta['n_outer_folds'])
                pf = res['per_fold']
                distinct = pf[['feature', 'selection_type', 'classifier']].drop_duplicates()
                sweep_rows.append({
                    'sample_fraction': frac,
                    'n_samples': sub_meta['n_samples_subset'],
                    'n_outer_folds': sub_meta['n_outer_folds'],
                    'subsample_seed': sub_meta['subsample_seed'],
                    'mean_test_auc': res['summary']['mean_test_auc'],
                    'std_test_auc': res['summary']['std_test_auc'],
                    'oof_auc_pooled': res['summary']['oof_auc_pooled'],
                    'mean_inner_score': float(pf['inner_best_score'].mean()),
                    'n_distinct_configs': len(distinct),
                    'modal_feature': pf['feature'].mode().iloc[0],
                    'feature_votes': "; ".join(f"{k_}={v}" for k_, v in
                                               pf['feature'].value_counts().items()),
                    'modal_selection_type': pf['selection_type'].mode().iloc[0],
                    'modal_classifier': pf['classifier'].mode().iloc[0],
                    'mean_num_genes_used': float(pf['num_genes_used'].mean()),
                })

            if not sweep_rows:
                raise RuntimeError(
                    f"No complete fraction in {output_dir}: every sizesweep_f* "
                    f"subdirectory is missing fold partials. Resubmit the array and "
                    f"aggregate again once at least one fraction has all its folds.")

            sweep_df = (pd.DataFrame(sweep_rows)
                        .sort_values('sample_fraction', ascending=False)
                        .reset_index(drop=True))
            sweep_path = output_dir / "nested_full_sizesweep.csv"
            sweep_df.to_csv(sweep_path, index=False)

            print("\n" + "=" * 110)
            print("FULLY NESTED SAMPLE-SIZE SWEEP - every point is a complete nested "
                  "selection on a nested stratified subset")
            print("=" * 110)
            show = ['sample_fraction', 'n_samples', 'n_outer_folds', 'mean_test_auc',
                    'std_test_auc', 'oof_auc_pooled', 'n_distinct_configs',
                    'modal_feature', 'modal_classifier']
            print(sweep_df[show].to_string(index=False))
            print(f"\nSaved -> {sweep_path}")
            if skipped:
                print("\n" + "!" * 110)
                print(f"INCOMPLETE SWEEP: {len(skipped)} fraction(s) skipped - the CSV "
                      f"above covers only the complete ones.")
                for name, why in skipped:
                    print(f"  {name}: {why}")
                print("Resubmit the sweep array (existing partials are kept) and "
                      "aggregate again for the full curve.")
                print("!" * 110)
            print("=" * 110 + "\n")
            return {'sizesweep': sweep_df, 'output_dir': output_dir,
                    'skipped_fractions': skipped}

        if not ksweep:
            return _nested_full_aggregate_dir(output_dir, n_samples, sizes, meta)

        subdirs = sorted(output_dir.glob("ksweep_k*"))
        if not subdirs:
            raise FileNotFoundError(f"No ksweep_k* subdirectories in {output_dir}")
        sweep_rows = []
        per_k_results = {}
        for sub_dir in subdirs:
            k_val = int(sub_dir.name.replace("ksweep_k", ""))
            print(f"\n{'#' * 90}\n#  k = {k_val}\n{'#' * 90}")
            res = _nested_full_aggregate_dir(sub_dir, n_samples, sizes, meta)
            per_k_results[k_val] = res
            pf = res['per_fold']
            distinct = pf[['feature', 'selection_type', 'classifier']].drop_duplicates()
            sweep_rows.append({
                'k': k_val,
                'mean_test_auc': res['summary']['mean_test_auc'],
                'std_test_auc': res['summary']['std_test_auc'],
                'oof_auc_pooled': res['summary']['oof_auc_pooled'],
                'mean_inner_score': float(pf['inner_best_score'].mean()),
                'n_distinct_configs': len(distinct),
                'modal_feature': pf['feature'].mode().iloc[0],
                'feature_votes': "; ".join(f"{k_}={v}" for k_, v in
                                           pf['feature'].value_counts().items()),
                'modal_selection_type': pf['selection_type'].mode().iloc[0],
                'mean_num_genes_used': float(pf['num_genes_used'].mean()),
            })

        sweep_df = pd.DataFrame(sweep_rows).sort_values('k').reset_index(drop=True)
        sweep_path = output_dir / "nested_full_ksweep.csv"
        sweep_df.to_csv(sweep_path, index=False)

        print("\n" + "=" * 110)
        print("FULLY NESTED k-SWEEP - every point is a complete nested selection at fixed k")
        print("=" * 110)
        show = ['k', 'mean_test_auc', 'std_test_auc', 'mean_inner_score',
                'n_distinct_configs', 'modal_feature', 'modal_selection_type']
        print(sweep_df[show].to_string(index=False))
        print(f"\nSaved -> {sweep_path}")
        print("=" * 110 + "\n")

        tuned = _ksweep_derive_tuned_arm(output_dir, per_k_results, sizes, meta, n_samples)
        return {'ksweep': sweep_df, 'ksweep_tuned': tuned, 'output_dir': output_dir}

    if aggregate:
        # Cross-cluster: the aggregates of the same run (one per cluster) run one after
        # the other instead of writing the same files at the same time.
        claim = _WorkClaim(output_dir / "nested_full_aggregate", f"aggregate of {output_dir.name}")
        claim.wait(lambda: False)
        try:
            return _run_aggregate()
        finally:
            claim.release()

    # ---------------------------------------------------------------- cache guard
    # Pipeline(memory=...) makes joblib hash the transformer on every fit. When
    # model_hpc.py is run as a script, TopGeneSelector.__module__ == "__main__", and the
    # loky workers GridSearchCV(n_jobs>1) spawns have their own __main__ that does not
    # contain the class -- so the hash raises PicklingError and EVERY fit fails
    # ("All the N fits failed"). Caching is only an optimisation, so drop it rather than
    # let the run die. It works when model_hpc is imported as a module, or with n_jobs=1.
    if cache_dir and TopGeneSelector.__module__ == "__main__" and args.cores != 1:
        print("NOTE: pipeline caching disabled. TopGeneSelector is defined in '__main__' "
              "(model_hpc.py run as a script), which joblib cannot hash inside the "
              f"n_jobs={args.cores} workers; every fit would fail. The TSS ranking will be "
              "recomputed per candidate instead (the slow p_value / weighted rankings are "
              "memoised in-process, see select_top_genes). To use the cache, import model_hpc "
              "as a module or set --cores 1.")
        cache_dir = None

    # ---------------------------------------------------------------- input guards
    # A feature matrix with no columns (or with NaNs) silently produces degenerate
    # predictions rather than an error, so check before spending any compute.
    for f in feature_names:
        Xf = all_feature_arrays[f]
        if Xf.ndim != 2 or Xf.shape[1] == 0:
            raise ValueError(f"Feature '{f}' has shape {Xf.shape}: no columns to select from.")
        if Xf.shape[0] != n_samples:
            raise ValueError(f"Feature '{f}' has {Xf.shape[0]} rows, expected {n_samples}.")
        n_nan = int(np.isnan(Xf).sum())
        if n_nan:
            raise ValueError(
                f"Feature '{f}' contains {n_nan} NaN values "
                f"({int(np.isnan(Xf).any(axis=0).sum())} of {Xf.shape[1]} columns affected). "
                f"StandardScaler and the classifiers cannot consume NaN."
            )

    print(f"Saving fully-nested results to: {output_dir}")
    print(f"  features           : {feature_names}")
    print(f"  ranking functions  : {selection_types}")
    print(f"  top_genes          : {top_genes_list}")
    print(f"  classifiers        : {classifier_names}")
    print(f"  pipeline cache     : {cache_dir if cache_dir else 'disabled'}")

    # The outer partition depends only on n_samples and stratification_labels, so it is
    # identical for every feature (and matches run_single_experiment's outer folds).
    # The sizesweep builds its own per-fraction partitions instead (a full-cohort
    # 10-fold split may not even be constructible for every task).
    if not sizesweep:
        outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        outer_splits = list(outer_cv.split(np.zeros((n_samples, 1)), stratification_labels))

    # ---------------------------------------------------------------- single-fold mode
    if only_fold is not None:
        if not 1 <= int(only_fold) <= 10:
            raise ValueError(f"only_fold must be in 1..10, got {only_fold}")
        fold_idx = int(only_fold) - 1

        if sizesweep:
            # One complete nested selection per sample fraction. The fraction loop runs
            # inside this one task (the feature matrices are loaded once); subsets are
            # nested prefixes of a single seeded per-stratum shuffle, so every array
            # task derives the identical subsets independently.
            subsets = _sizesweep_subsets(sizesweep_fractions, stratification_labels,
                                         subsample_seed)
            written = []
            for frac in sizesweep_fractions:
                sub_idx = subsets[frac]
                sub_strat = stratification_labels[sub_idx]
                sub_y = y_labels[sub_idx]
                # Outer fold count from the BINARY class counts (each outer fold needs
                # both classes); stratify on the fine-grained disease labels only while
                # every stratum can appear in every fold, else on the binary label
                # (mirrors the inner-CV fallback in _nested_full_one_fold).
                y_counts = np.unique(sub_y, return_counts=True)[1]
                n_folds_f = int(min(10, y_counts.min()))
                if n_folds_f < 2:
                    raise ValueError(
                        f"sizesweep fraction {frac:g}: smallest class has "
                        f"{y_counts.min()} sample(s); no outer CV possible.")
                if fold_idx + 1 > n_folds_f:
                    print(f"\nfraction {frac:g}: only {n_folds_f} outer folds possible "
                          f"(smallest class has {y_counts.min()} samples) - "
                          f"outer fold {fold_idx + 1} does not exist, skipping.")
                    continue
                # Resume support: fractions are deterministic given (seed, cohort), so a
                # partial written by an earlier (timed-out or chunked) submission is
                # still valid - skip it instead of recomputing. A truncated pickle
                # (task killed mid-write) fails to load and is recomputed.
                sub_dir = output_dir / _sizesweep_dir_name(frac)
                existing = sub_dir / f"nested_full_fold_{fold_idx + 1:02d}.pkl"
                if existing.exists():
                    try:
                        with open(existing, 'rb') as fh:
                            prev = pickle.load(fh)
                        if np.isclose(prev['meta']['sample_fraction'], frac) and \
                                prev['meta']['subsample_seed'] == int(subsample_seed):
                            print(f"\nfraction {frac:g}: partial for outer fold "
                                  f"{fold_idx + 1} already exists ({existing}) - skipping. "
                                  f"Delete it to force recomputation.")
                            written.append(existing)
                            continue
                        print(f"\nfraction {frac:g}: existing partial has different "
                              f"meta ({prev['meta'].get('subsample_seed')}) - recomputing.")
                    except Exception as e:
                        print(f"\nfraction {frac:g}: existing partial unreadable "
                              f"({e!r}) - recomputing.")
                # Cross-cluster claim: another worker on this fraction / fold -> wait for it.
                os.makedirs(sub_dir, exist_ok=True)
                claim = _WorkClaim(existing, f"fraction {frac:g} fold {fold_idx + 1}")
                if not claim.try_acquire():
                    print(f"  fraction {frac:g} fold {fold_idx + 1}: claimed by {claim.holder()} - waiting")
                    if claim.wait(lambda e=existing: _sizesweep_partial_ok(e, frac, subsample_seed)) == 'done':
                        written.append(existing)
                        continue
                strat_counts = np.unique(sub_strat, return_counts=True)[1]
                if strat_counts.min() >= n_folds_f:
                    outer_strat, outer_strat_kind = sub_strat, "disease"
                else:
                    outer_strat, outer_strat_kind = sub_y, "binary"
                    print(f"fraction {frac:g}: smallest stratum has {strat_counts.min()} "
                          f"sample(s) < {n_folds_f} folds - outer CV stratifies on the "
                          f"binary label.")
                sub_splits = list(StratifiedKFold(
                    n_splits=n_folds_f, shuffle=True, random_state=42
                ).split(np.zeros((len(sub_idx), 1)), outer_strat))
                tr_local, te_local = sub_splits[fold_idx]
                print(f"\n{'#' * 90}\n#  sample fraction = {frac:g}   "
                      f"(n = {len(sub_idx)}, {n_folds_f} outer folds, "
                      f"outer fold {fold_idx + 1})\n{'#' * 90}")
                rec, rows = _nested_full_one_fold(
                    fold_idx, sub_idx[tr_local], sub_idx[te_local],
                    all_feature_arrays, y_labels, stratification_labels, sample_ids,
                    CLASSIFIERS, CLF_PARAM_GRIDS,
                    feature_names, top_genes_list, selection_types, classifier_names,
                    args.cores, cache_dir,
                )
                f_meta = dict(
                    meta,
                    sample_fraction=float(frac),
                    n_samples_subset=int(len(sub_idx)),
                    n_outer_folds=n_folds_f,
                    outer_stratification=outer_strat_kind,
                    subsample_seed=int(subsample_seed),
                    subset_class_counts={str(s): int(c) for s, c in
                                         zip(*np.unique(sub_strat, return_counts=True))},
                )
                p = sub_dir / f"nested_full_fold_{fold_idx + 1:02d}.pkl"
                _atomic_pickle(p, {'fold_record': rec, 'oof_rows': rows,
                                   'sizes': sizes, 'meta': f_meta})
                claim.release()
                written.append(p)
            print(f"\nWrote {len(written)} per-fraction partials for outer fold "
                  f"{fold_idx + 1} under {output_dir}")
            print("Run with --nested_full_sizesweep --nested_full_aggregate once all "
                  "array tasks are done.")
            return {'partial_paths': written}

        train_idx, test_idx = outer_splits[fold_idx]

        if ksweep:
            # One complete nested selection per k. Feature, ranking function and the SVM
            # hyperparameters are still chosen inside this fold's inner CV; only k is held
            # fixed, so each point of the resulting trend is leakage-free. The k loop runs
            # inside this one task so the feature matrices are loaded once, not once per k.
            written, deferred = [], []

            def _k_partial(k_val):
                sub_dir = output_dir / f"ksweep_k{int(k_val):05d}"
                return (sub_dir / f"nested_full_fold_{fold_idx + 1:02d}.pkl",
                        dict(meta, top_genes_searched=[int(k_val)]))

            def _k_done(k_val):
                p, k_meta = _k_partial(k_val)
                return _nested_full_existing_partial(p, fold_idx + 1, k_meta, sample_ids[test_idx]) is not None

            def _k_compute(k_val):
                p, k_meta = _k_partial(k_val)
                k_sizes = _nested_full_grid_sizes(
                    CLASSIFIERS, CLF_PARAM_GRIDS, classifier_names,
                    [k_val], selection_types, feature_names)
                print(f"\n{'#' * 90}\n#  k = {k_val}   (outer fold {fold_idx + 1})\n{'#' * 90}")
                rec, rows = _nested_full_one_fold(
                    fold_idx, train_idx, test_idx, all_feature_arrays, y_labels,
                    stratification_labels, sample_ids, CLASSIFIERS, CLF_PARAM_GRIDS,
                    feature_names, [k_val], selection_types, classifier_names,
                    args.cores, cache_dir,
                )
                _atomic_pickle(p, {'fold_record': rec, 'oof_rows': rows,
                                   'sizes': k_sizes, 'meta': k_meta})
                return p

            for k_val in top_genes_list:
                p, _ = _k_partial(k_val)
                # Resume: a k whose partial was written by an earlier submission of this
                # array task (killed, timed out, credential expired) is not recomputed.
                if _k_done(k_val):
                    written.append(p)
                    continue
                os.makedirs(p.parent, exist_ok=True)
                # Cross-cluster claim: a k another worker is computing is deferred.
                claim = _WorkClaim(p, f"fold {fold_idx + 1} k={k_val}")
                if not claim.try_acquire():
                    print(f"  fold {fold_idx + 1} k={k_val}: claimed by {claim.holder()} - deferred")
                    deferred.append(k_val)
                    continue
                try:
                    written.append(_k_compute(k_val))
                finally:
                    claim.release()
            for k_val in deferred:          # come back for the units another worker claimed
                p, _ = _k_partial(k_val)
                claim = _WorkClaim(p, f"fold {fold_idx + 1} k={k_val}")
                if claim.wait(lambda k=k_val: _k_done(k)) == 'acquired':
                    try:
                        _k_compute(k_val)
                    finally:
                        claim.release()
                written.append(p)
            print(f"\nWrote {len(written)} per-k partials for outer fold {fold_idx + 1} "
                  f"under {output_dir}")
            print("Run with --nested_full_ksweep --nested_full_aggregate once all "
                  "10 folds are done.")
            return {'partial_paths': written}

        part_path = output_dir / f"nested_full_fold_{fold_idx + 1:02d}.pkl"
        # Resume: rerunning the array after a kill / time-out / credential expiry
        # only recomputes the folds without a valid partial.
        prev = _nested_full_existing_partial(part_path, fold_idx + 1, meta, sample_ids[test_idx])
        if prev is not None:
            return {'fold_record': prev['fold_record'], 'partial_path': part_path}
        # Cross-cluster claim: if another worker (other cluster) is on this fold, wait
        # for its partial instead of computing it twice.
        claim = _WorkClaim(part_path, f"fold {fold_idx + 1}")
        if not claim.try_acquire():
            print(f"  fold {fold_idx + 1}: claimed by {claim.holder()} - waiting for its partial")
            done = claim.wait(lambda: _nested_full_existing_partial(
                part_path, fold_idx + 1, meta, sample_ids[test_idx]) is not None)
            if done == 'done':
                with open(part_path, 'rb') as fh:
                    prev = pickle.load(fh)
                return {'fold_record': prev['fold_record'], 'partial_path': part_path}
        try:
            fold_record, oof_rows = _nested_full_one_fold(
                fold_idx, train_idx, test_idx, all_feature_arrays, y_labels,
                stratification_labels, sample_ids, CLASSIFIERS, CLF_PARAM_GRIDS,
                feature_names, top_genes_list, selection_types, classifier_names,
                args.cores, cache_dir,
            )
            _atomic_pickle(part_path, {'fold_record': fold_record, 'oof_rows': oof_rows,
                                       'sizes': sizes, 'meta': meta})
        finally:
            claim.release()
        print(f"\nWrote partial result for outer fold {fold_idx + 1} -> {part_path}")
        print("Run with --nested_full_aggregate once all 10 folds are done.")
        return {'fold_record': fold_record, 'partial_path': part_path}

    # ---------------------------------------------------------------- all folds serially
    if sizesweep or ksweep:
        raise ValueError(
            "The sizesweep/ksweep modes run per outer fold: submit with "
            "--nested_full_only_fold N (the slurm wrapper's array does this) and combine "
            "with --nested_full_aggregate afterwards. A serial all-folds sweep is not "
            "implemented.")
    start_time = time.time()
    fold_records, oof_rows = [], []
    for fold_idx, (train_idx, test_idx) in enumerate(outer_splits):
        rec, rows = _nested_full_one_fold(
            fold_idx, train_idx, test_idx, all_feature_arrays, y_labels,
            stratification_labels, sample_ids, CLASSIFIERS, CLF_PARAM_GRIDS,
            feature_names, top_genes_list, selection_types, classifier_names,
            args.cores, cache_dir,
        )
        fold_records.append(rec)
        oof_rows.extend(rows)

    return _nested_full_finalize(output_dir, fold_records, oof_rows, sizes, meta,
                                 time.time() - start_time, n_samples)


def run_cross_depth_eval(all_feature_arrays: dict,
                         kept_columns_per_feature: dict,
                         original_n_features: int,
                         blacklist_file_used,
                         y_labels: np.ndarray,
                         stratification_labels: np.ndarray,
                         sample_ids,
                         CLASSIFIERS: dict,
                         args: argparse.Namespace):
    """
    Cross-depth evaluation of a completed fully nested run (reviewer arm: do
    FUTURE samples need to be sequenced deep, or can a model trained on the
    deep cohort score shallow samples?).

    For every outer fold of the source run (--cross_depth_source_dir), the
    stored winning configuration (feature, ranking function, k, classifier,
    hyperparameters) is refitted on the stored full-depth training partition -
    the fitted pipelines are not pickled by the nested run, but every
    classifier is seeded, so the refit is deterministic and CHEAP (one fit per
    fold). Reproduction is then PROVEN, not assumed: the refit must reproduce
    the stored selected TSS indices and the stored test-fold predictions
    before any cross-depth number is produced. Finally the SAME model scores
    the SAME test samples loaded from each downsampled feature directory
    (--cross_depth_eval_dirs name=dir ...), and the stored train-derived
    95%-specificity threshold is applied unchanged. Everything is paired:
    same folds, same models, same test samples - only the test depth changes.

    Test-depth matrices are rebuilt through the exact preprocessing order of
    main() (blacklist removal -> per-sample row z-score -> duplicate-column
    selection), and the rebuild path is self-checked by reconstructing the
    full-depth matrices from --features_dir and requiring them to equal the
    in-memory training matrices.

    Writes to <output_dir>/cross_depth_<experiment_name>/:
      cross_depth_predictions.csv  one row per (depth, fold, sample) - depth
                                   'full' is the reproduced source prediction
      cross_depth_per_fold.csv     AUC + operating point per (depth, fold)
      cross_depth_summary.json     per-depth summaries + provenance + checks
    """
    source_dir = Path(args.cross_depth_source_dir)
    with open(source_dir / "nested_full_per_fold.pkl", 'rb') as fh:
        per_fold = pickle.load(fh)
    print(f"Cross-depth evaluation of {source_dir} ({len(per_fold)} folds)")

    eval_dirs = {}
    for spec in args.cross_depth_eval_dirs:
        name, _, d = spec.partition('=')
        if not name or not d:
            raise ValueError(f"--cross_depth_eval_dirs expects name=dir, got {spec!r}")
        eval_dirs[name] = d
    if not eval_dirs:
        raise ValueError("--cross_depth_eval_dirs: at least one name=dir required")

    sample_ids = np.asarray(sample_ids)
    y_labels = np.asarray(y_labels)
    stratification_labels = np.asarray(stratification_labels)
    id_to_idx = {sid: i for i, sid in enumerate(sample_ids)}
    for _, r in per_fold.iterrows():
        missing = [s for s in list(r['train_sample_ids']) + list(r['test_sample_ids'])
                   if s not in id_to_idx]
        if missing:
            raise ValueError(
                f"fold {r['fold']}: {len(missing)} source-run sample(s) not in the "
                f"current cohort (e.g. {missing[:5]}) - run with the same metadata/"
                f"cancer-type selection as the source run.")

    features_needed = sorted(set(per_fold['feature']))
    print(f"features used by the source run: {features_needed}")

    # Post-blacklist original-index set: replicates main()'s preprocessing ORDER
    # (blacklist removal happens BEFORE the per-sample row z-score, so the z-score
    # statistics depend on it; duplicate-column selection comes after).
    if blacklist_file_used:
        removed = np.load(blacklist_file_used)
        post_blacklist = np.delete(np.arange(original_n_features), removed)
    else:
        post_blacklist = np.arange(original_n_features)

    def load_depth_matrix(feature, folder):
        raw = load_feature_in_parallel(folder, feature, file_filter=list(sample_ids),
                                       num_cores=args.cores, exact=True)
        if raw is None:
            raise RuntimeError(f"could not load feature '{feature}' from {folder}")
        if raw.shape[1] != original_n_features:
            raise ValueError(f"{folder}/{feature}: {raw.shape[1]} columns, "
                             f"expected {original_n_features}")
        arr = raw[:, post_blacklist]
        mu = np.nanmean(arr, axis=1, keepdims=True)
        sd = np.nanstd(arr, axis=1, keepdims=True)
        sd[sd == 0] = 1.0
        arr = np.nan_to_num((arr - mu) / sd)
        pos = np.searchsorted(post_blacklist, np.asarray(kept_columns_per_feature[feature]))
        return arr[:, pos]

    # ---- self-check: the rebuild path must reproduce the training matrices ----
    for f in features_needed:
        rebuilt = load_depth_matrix(f, args.features_dir)
        if not np.allclose(rebuilt, all_feature_arrays[f], atol=1e-6):
            raise RuntimeError(
                f"self-check failed: '{f}' rebuilt from {args.features_dir} does not "
                f"reproduce the in-memory training matrix (max diff "
                f"{np.abs(rebuilt - all_feature_arrays[f]).max():.3e}). The eval-dir "
                f"preprocessing would not be aligned - aborting.")
    print("self-check passed: full-depth matrices reconstruct exactly; "
          "eval-dir preprocessing is aligned")

    eval_arrays = {name: {f: load_depth_matrix(f, d) for f in features_needed}
                   for name, d in eval_dirs.items()}
    for name in eval_dirs:
        print(f"loaded eval depth '{name}': {eval_dirs[name]}")

    out_dir = Path(args.output_dir) / f"cross_depth_{args.experiment_name}"
    os.makedirs(out_dir, exist_ok=True)

    pred_rows, fold_rows, selection_from_store = [], [], []
    for _, r in per_fold.sort_values('fold').iterrows():
        fold = int(r['fold'])
        train_idx = np.array([id_to_idx[s] for s in r['train_sample_ids']])
        test_idx = np.array([id_to_idx[s] for s in r['test_sample_ids']])
        feat = r['feature']
        X = all_feature_arrays[feat]
        y_tr, y_te = y_labels[train_idx], y_labels[test_idx]

        pipe = Pipeline([('selector', TopGeneSelector()),
                         ('scaler', StandardScaler()),
                         ('clf', CLASSIFIERS[r['classifier']])])
        pipe = clone(pipe).set_params(
            selector__n_top_genes=int(r['n_top_genes']),
            selector__selection_type=r['selection_type'],
            **dict(r['clf_hyperparams']))
        pipe.fit(X[train_idx], y_tr)

        # ---- reproduction checks: refit must equal the stored source model ----
        sel = np.sort(np.asarray(pipe.named_steps['selector'].indices_))
        stored_order = np.asarray(r['selected_gene_indices'], dtype=int)
        stored_sel = np.sort(stored_order)
        if not np.array_equal(sel, stored_sel):
            # On the no_deep ksweep_tuned sources one TSS at the k = 10,000
            # boundary differed (BRCA/CRC fold 2, PANCAN fold 9) - a near-tie in the
            # ranking score, not a different model. The stored selection IS the source
            # model's, so refit on it (stored column order, which seeded tree models
            # depend on); the prediction check below still has to pass.
            new, lost = np.setdiff1d(sel, stored_sel), np.setdiff1d(stored_sel, sel)
            print(f"  fold {fold}: refit ranking differs from the stored selection "
                  f"({new.size} new {new[:5].tolist()}, {lost.size} missing {lost[:5].tolist()}); "
                  f"refitting on the stored selection")
            pipe = clone(pipe)
            pipe.steps[0] = ('selector', _StoredIndexSelector(stored_order))
            pipe.fit(X[train_idx], y_tr)
            selection_from_store.append({'fold': fold, 'n_new': int(new.size), 'n_missing': int(lost.size),
                                         'new': new.tolist(), 'missing': lost.tolist()})
        proba_full = pipe.predict_proba(X[test_idx])[:, 1]
        stored_proba = np.asarray(r['y_pred_proba'], dtype=float)
        if not np.allclose(proba_full, stored_proba, atol=1e-6):
            raise RuntimeError(
                f"fold {fold}: refit test predictions deviate from the stored ones "
                f"(max diff {np.abs(proba_full - stored_proba).max():.3e}) - "
                f"refit does not reproduce the source model, aborting.")

        thr = float(r['full_model_thresholds']['95_percent_specificity'])

        def emit(depth, scores):
            auc_d = float(roc_auc_score(y_te, scores))
            yhat = (scores >= thr).astype(int)
            tp = int(np.sum((y_te == 1) & (yhat == 1)))
            fn = int(np.sum((y_te == 1) & (yhat == 0)))
            tn = int(np.sum((y_te == 0) & (yhat == 0)))
            fp = int(np.sum((y_te == 0) & (yhat == 1)))
            fold_rows.append({
                'fold': fold, 'depth': depth, 'auc': auc_d,
                'sensitivity_95spec': tp / (tp + fn) if tp + fn else np.nan,
                'specificity_95spec': tn / (tn + fp) if tn + fp else np.nan,
                'feature': feat, 'classifier': r['classifier'],
            })
            for sid, yt, st, s in zip(r['test_sample_ids'], y_te,
                                      stratification_labels[test_idx], scores):
                pred_rows.append({'fold': fold, 'sample_id': sid, 'y_true': int(yt),
                                  'stratification_label': st, 'depth': depth,
                                  'score': float(s), 'threshold_95spec': thr})
            return auc_d

        auc_full = emit('full', proba_full)
        msg = f"  fold {fold:>2} ({feat}, {r['classifier']}): full={auc_full:.4f}"
        for name in eval_dirs:
            scores = pipe.predict_proba(eval_arrays[name][feat][test_idx])[:, 1]
            msg += f"  {name}={emit(name, scores):.4f}"
        print(msg)

    pred_df = pd.DataFrame(pred_rows)
    fold_df = pd.DataFrame(fold_rows)
    pred_df.to_csv(out_dir / "cross_depth_predictions.csv", index=False)
    fold_df.to_csv(out_dir / "cross_depth_per_fold.csv", index=False)

    summary = {'source_dir': str(source_dir), 'eval_dirs': eval_dirs,
               'features_dir_full': args.features_dir,
               'features_used': features_needed, 'n_folds': int(len(per_fold)),
               'reproduction_checks': 'passed (selected indices + test predictions)' if not selection_from_store
                                      else 'passed (test predictions; stored selection used where the refit ranking differed)',
               'selection_from_store': selection_from_store,
               'per_depth': {}}
    print("\n" + "=" * 90)
    print("CROSS-DEPTH EVALUATION - full-depth-trained models on downsampled test folds")
    print("=" * 90)
    for depth in ['full'] + list(eval_dirs):
        sub = fold_df[fold_df['depth'] == depth]
        summary['per_depth'][depth] = {
            'mean_auc': float(sub['auc'].mean()),
            'std_auc': float(sub['auc'].std(ddof=0)),
            'mean_sensitivity_95spec': float(sub['sensitivity_95spec'].mean()),
            'mean_specificity_95spec': float(sub['specificity_95spec'].mean()),
        }
        s = summary['per_depth'][depth]
        print(f"  {depth:>10}: AUC {s['mean_auc']:.4f} ± {s['std_auc']:.4f}   "
              f"sens@95 {s['mean_sensitivity_95spec']:.3f}   "
              f"spec {s['mean_specificity_95spec']:.3f}")
    with open(out_dir / "cross_depth_summary.json", 'w') as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f"\nSaved -> {out_dir}")
    print("=" * 90 + "\n")
    return {'predictions': pred_df, 'per_fold': fold_df, 'summary': summary,
            'output_dir': out_dir}


# =================================================================================
# --- Main Execution Function ---
# =================================================================================

def _recompute_inner_scores_for_pkl(pkl_path, all_feature_arrays, y_labels,
                                    stratification_labels, id_to_idx, CLASSIFIERS):
    """
    Worker: reconstruct the mean inner-CV AUC (== GridSearchCV.best_score_) for every
    outer fold stored in one nested-CV results pickle.

    Older pickles saved best_params but not best_score_. Because every split is seeded
    (outer StratifiedKFold(10, shuffle, rs=42); inner StratifiedKFold(3, shuffle, rs=42))
    and train_sample_ids/selected_gene_indices are stored per fold, the inner score of the
    winning config can be reproduced exactly with 3 fits per fold - no grid rerun needed.
    Pickles that already carry 'inner_cv_best_score' are read out without refitting.
    """
    with open(pkl_path, 'rb') as f:
        results_df = pickle.load(f)

    rows = []
    for _, fold_row in results_df.iterrows():
        clf_name = fold_row['classifier']
        feature_name = fold_row['feature']
        record = {
            'source_pkl': pkl_path.name,
            'classifier': clf_name,
            'feature': feature_name,
            'top_genes': fold_row.get('top_genes', 'all'),
            'selection_type': fold_row.get('selection_type', 'none'),
            'fold': int(fold_row['fold']),
            'test_auc': float(fold_row['auc_full_model']),
            'inner_cv_auc': np.nan,
            'recomputed': False,
        }

        stored = fold_row.get('inner_cv_best_score', None)
        if stored is not None and np.isfinite(stored):
            record['inner_cv_auc'] = float(stored)
            rows.append(record)
            continue

        if clf_name not in CLASSIFIERS:
            print(f"{pkl_path.name} fold {record['fold']}: classifier '{clf_name}' not "
                  f"available in this environment - skipping recompute.")
            rows.append(record)
            continue
        if feature_name not in all_feature_arrays:
            raise KeyError(
                f"{pkl_path.name}: feature '{feature_name}' not in the loaded feature arrays "
                f"({sorted(all_feature_arrays.keys())}). Invoke --recompute_inner_scores with "
                f"the same data arguments as the original nested-CV run."
            )

        train_ids = fold_row['train_sample_ids']
        test_ids = fold_row['test_sample_ids']
        missing = [s for s in train_ids + test_ids if s not in id_to_idx]
        if missing:
            raise KeyError(
                f"{pkl_path.name} fold {record['fold']}: {len(missing)} sample IDs from the "
                f"stored fold are absent from the loaded cohort (e.g. {missing[:3]}). The data "
                f"arguments must match the original run (same manifest/cohort filters)."
            )
        train_idx = np.array([id_to_idx[s] for s in train_ids])
        test_idx = np.array([id_to_idx[s] for s in test_ids])

        # Guard against silently recomputing on different data than the original run
        if not np.array_equal(y_labels[test_idx], np.asarray(fold_row['y_test'])):
            raise ValueError(
                f"{pkl_path.name} fold {record['fold']}: stored y_test does not match current "
                f"labels for the stored test sample IDs - cohort/labels differ from the "
                f"original run. Refusing to recompute inner scores on mismatched data."
            )

        gene_idx = np.asarray(fold_row['selected_gene_indices'], dtype=int)
        X_train = all_feature_arrays[feature_name][train_idx][:, gene_idx]
        y_train = y_labels[train_idx]
        strat_train = stratification_labels[train_idx]

        pipeline = Pipeline([('scaler', StandardScaler()),
                             ('clf', clone(CLASSIFIERS[clf_name]))])
        pipeline.set_params(**fold_row['best_params'])

        # Identical splits to the original GridSearchCV call
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        inner_splits = list(inner_cv.split(X_train, strat_train))
        scores = cross_val_score(pipeline, X_train, y_train, scoring='roc_auc',
                                 cv=inner_splits, n_jobs=1, error_score=np.nan)
        record['inner_cv_auc'] = float(np.nanmean(scores))
        record['recomputed'] = True
        rows.append(record)

    return rows


def run_nested_classifier_selection(results_dir, all_feature_arrays, y_labels,
                                    stratification_labels, sample_ids, CLASSIFIERS,
                                    n_jobs=1):
    """
    Post-hoc, leakage-free classifier selection over an existing nested-CV results
    directory (one results_*.pkl per classifier/feature combination).

    Step 1 recovers each fold's mean inner-CV AUC for the stored winning hyperparameters
    (see _recompute_inner_scores_for_pkl) and writes inner_cv_scores.csv.
    Step 2 then selects per OUTER fold on INNER scores only, at two scopes:
      - within_feature: classifier chosen per fold, feature held fixed
      - global (ALL_FEATURES): feature AND classifier chosen jointly per fold -
        the fully nested protocol of run_nested_experiment_full, applied post-hoc
    and stitches the winners' stored test-fold predictions into OOF prediction files,
    so no choice ever sees the outer test fold.

    Writes into results_dir:
      - inner_cv_scores.csv                            (all configs, all folds)
      - nested_selection_summary.csv                   (per-fold winners, both scopes)
      - nested_selection_predictions_<feature>.csv     (per-feature OOF predictions)
      - nested_selection_predictions_ALL_FEATURES.csv  (joint feature+classifier OOF)

    Caveat: gene-selection-variant pickles from older runs fitted the TSS
    ranking on the full outer-train fold, so their inner scores are slightly optimistic
    relative to the selector-inside-pipeline protocol; comparisons among such pickles
    are still like-for-like. run_single_experiment now nests the selector in the
    pipeline (like nested_full), so newer pickles carry fully nested inner scores.
    The recompute fallback below reconstructs the OLD protocol (stored gene indices,
    scaler+clf pipeline) and is only reached for old pickles, which never store
    'inner_cv_best_score'.
    """
    results_dir = Path(results_dir)
    pkl_files = sorted(results_dir.glob("results_*.pkl"))
    if not pkl_files:
        raise FileNotFoundError(f"No results_*.pkl files found in {results_dir}")

    id_to_idx = {s: i for i, s in enumerate(sample_ids)}
    print(f"--- Recomputing inner-CV scores for {len(pkl_files)} result files "
          f"in {results_dir} (n_jobs={n_jobs}) ---")
    per_file_rows = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_recompute_inner_scores_for_pkl)(
            p, all_feature_arrays, y_labels, stratification_labels, id_to_idx, CLASSIFIERS)
        for p in pkl_files
    )
    scores_df = pd.DataFrame([r for rows in per_file_rows for r in rows])
    scores_path = results_dir / "inner_cv_scores.csv"
    scores_df.to_csv(scores_path, index=False)
    print(f"Inner-CV scores written to {scores_path} "
          f"({int(scores_df['recomputed'].sum())} folds refitted, "
          f"{int((~scores_df['recomputed'] & scores_df['inner_cv_auc'].notna()).sum())} read from pickles)")

    # --- Per-fold selection on inner scores only ---
    pkl_cache = {}

    def _fold_row(source_pkl, fold):
        if source_pkl not in pkl_cache:
            with open(results_dir / source_pkl, 'rb') as f:
                pkl_cache[source_pkl] = pickle.load(f)
        df = pkl_cache[source_pkl]
        return df[df['fold'] == fold].iloc[0]

    summary_rows = []

    def _select_and_stitch(candidates, tag, scope):
        """Per OUTER fold, pick the candidate config with the highest INNER score and
        stitch that config's stored test-fold predictions into an OOF prediction file.
        `candidates` spans whatever axes are in it - classifier only (within-feature
        scope) or classifier x feature x k x ranking jointly (global scope)."""
        pred_rows, local_summary = [], []
        for fold, fold_group in candidates.groupby('fold'):
            winner = fold_group.loc[fold_group['inner_cv_auc'].idxmax()]
            src = _fold_row(winner['source_pkl'], fold)
            thr95 = src['full_model_thresholds'].get('95_percent_specificity', np.nan)
            for sid, y_t, score in zip(src['test_sample_ids'], src['y_test'], src['y_pred_proba']):
                pred_rows.append({'sample_id': sid, 'fold': fold, 'y_true': y_t,
                                  'score': score, 'threshold_95spec': thr95,
                                  'feature': winner['feature'],
                                  'classifier': winner['classifier']})
            local_summary.append({
                'scope': scope, 'group': tag, 'fold': fold,
                'chosen_feature': winner['feature'],
                'chosen_top_genes': winner['top_genes'],
                'chosen_selection_type': winner['selection_type'],
                'chosen_classifier': winner['classifier'],
                'inner_cv_auc': winner['inner_cv_auc'],
                'test_auc': winner['test_auc'],
                'n_candidates': len(fold_group),
            })
        pred_df = pd.DataFrame(pred_rows)
        pooled_auc = roc_auc_score(pred_df['y_true'], pred_df['score'])
        pred_path = results_dir / f"nested_selection_predictions_{tag}.csv"
        pred_df.to_csv(pred_path, index=False)
        mean_fold_auc = float(np.mean([r['test_auc'] for r in local_summary]))
        chosen = pd.Series([f"{r['chosen_feature']}|{r['chosen_classifier']}"
                            for r in local_summary]).value_counts().to_dict()
        print(f"  [{tag}] nested selection over {len(local_summary)} folds: "
              f"pooled OOF AUC = {pooled_auc:.4f}, mean fold AUC = {mean_fold_auc:.4f}, "
              f"chosen = {chosen} -> {pred_path.name}")
        return local_summary

    valid = scores_df[scores_df['inner_cv_auc'].notna()]
    if valid.empty:
        raise RuntimeError(f"No usable inner scores in {results_dir} - nothing to select.")
    group_keys = ['feature', 'top_genes', 'selection_type']

    # Within-feature scope: classifier chosen per fold, feature held fixed
    for (feature_name, top_genes, sel_type), group in valid.groupby(group_keys):
        tag = feature_name if (str(top_genes) == 'all' and sel_type == 'none') \
            else f"{feature_name}_{top_genes}_{sel_type}"
        summary_rows += _select_and_stitch(group, tag, scope='within_feature')

    # Global scope: feature AND classifier (and k / ranking where present) chosen
    # jointly per fold on inner scores - the fully nested protocol of
    # run_nested_experiment_full, applied post-hoc. Identical to in-run selection:
    # same inner scores, same argmax, and the winner's outer-test predictions were
    # already computed without ever informing the choice.
    if valid.groupby(group_keys).ngroups > 1:
        summary_rows += _select_and_stitch(valid, 'ALL_FEATURES', scope='global')

    summary_df = pd.DataFrame(summary_rows)
    summary_path = results_dir / "nested_selection_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Per-fold selection summary written to {summary_path}")
    return summary_df


def main(args):
    """
    Main function to orchestrate data loading, preprocessing, and parallel model training.
    """
    # --- Define Constants ---
    CLASSIFIERS = {
        "Support Vector Machine": SVC(probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=10000),
        "Multi-Layer Perceptron": MLPClassifier(random_state=42, early_stopping=True, validation_fraction=0.1)
    }
    if XGBOOST_AVAILABLE:
        # n_jobs=1: parallelism belongs to GridSearchCV / joblib (matches RF, which also
        # runs single-threaded per fit); tree_method="hist" for speed on wide matrices
        CLASSIFIERS["XGBoost"] = XGBClassifier(random_state=42, n_jobs=1, tree_method="hist", eval_metric="logloss")
    else:
        print("xgboost is not installed in this environment - 'XGBoost' will be skipped. "
              "Rebuild comp.sif after `pixi lock` picks up the xgboost dependency in pyproject.toml.")

    CLF_PARAM_GRIDS = {
        # SVM: 6 C x 2 kernels x 2 gamma = 24 fits, of which 6 are duplicates
        # (gamma is ignored by the linear kernel), i.e. 18 distinct models
        # Lower C = stronger regularization (wider margin, more misclassifications allowed)
        "Support Vector Machine": {
            "clf__C": np.logspace(-3, 2, 6),  # 0.001 to 100 (favoring lower C)
            "clf__kernel": ["linear", "rbf"],
            "clf__gamma": ["scale", "auto"]  # Only used for RBF
        },

        # RF: 2 n_estimators x 3 max_depth x 2 min_samples_leaf x 2 max_features = 24 combinations
        # n_estimators is a stability knob, not a complexity knob (performance plateaus with
        # more trees), so two values spanning the range suffice; max_depth 15/20 were dropped
        # because with min_samples_leaf >= 5 and n ~ 300 samples trees rarely grow past depth
        # ~10, making 10/15/20/None nearly identical models
        # More regularization via: deeper min_samples_leaf, limited depth
        "Random Forest": {
            "clf__n_estimators": [200, 500],
            "clf__max_depth": [5, 10, None],
            "clf__min_samples_leaf": [5, 10],  # Higher = more regularization
            "clf__max_features": ["sqrt", "log2"]  # Feature subsampling = regularization
        },

        # XGB: 2 n_estimators x 2 learning_rate x 3 max_depth x 2 colsample_bytree = 24 combinations
        # Sized to match the other grids. Regularization suited to n ~ 300-400: shallow trees
        # (depth 2-4), shrinkage coupled to ensemble size, and column subsampling (0.3 helps when
        # many selected genes are correlated/noisy); subsample fixed at 0.8 (standard row bagging,
        # rarely worth tuning at this n). reg_lambda stays at its default (1.0).
        "XGBoost": {
            "clf__n_estimators": [100, 300],
            "clf__learning_rate": [0.05, 0.1],
            "clf__max_depth": [2, 3, 4],
            "clf__colsample_bytree": [0.3, 1.0],
            "clf__subsample": [0.8]
        },

        # KNN: 5 neighbors x 2 weights x 2 metrics = 20 combinations
        # Higher K = more regularization (smoother decision boundaries)
        "K-Nearest Neighbors": {
            "clf__n_neighbors": [5, 7, 11, 15, 21],  # Higher values for noisy data
            "clf__weights": ["uniform", "distance"],
            "clf__metric": ["euclidean", "manhattan"]
        },
        
        # LR: 5 C x 3 l1_ratio = 15 combinations
        # Lower C = stronger regularization; higher L1 = more feature selection
        "Logistic Regression": {
            "clf__C": np.logspace(-3, 1, 5),  # 0.001 to 10 (stronger regularization)
            "clf__penalty": ["elasticnet"],
            "clf__solver": ["saga"],
            "clf__l1_ratio": [0.1, 0.5, 0.9] 
        },
        
        # MLP: 6 architectures x 4 alpha x 2 learning_rate = 48 combinations
        # Higher alpha = stronger L2 regularization; early_stopping also helps
        "Multi-Layer Perceptron": {
            "clf__hidden_layer_sizes": [
                (100,),           # Single layer for few features
                (200,),           # Single layer, more neurons
                (100, 50),        # Two layers, moderate
                (256, 128),       # Two layers, larger
                (512, 256),       # Two layers, very large for many features
                (256, 128, 64)    # Three layers for complex patterns
            ],
            "clf__alpha": np.logspace(-4, -1, 4),  # 0.0001 to 0.1 (stronger regularization)
            "clf__learning_rate_init": [0.001, 0.01],
            "clf__max_iter": [2000]  # More iterations for deeper networks
        },
    }

    DATA_PARAM_GRIDS = {
        "feature": ["fslr_coverage", "fslr", "griffin_diff", "rcov", "mwh"],
        "top_genes": [10, 25, 50, 100, 250, 500, 1000, 1500, 2500, 5000, 10000, 15000]
    }

    # --- Create Output Directory ---
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load and Preprocess Data ---
    print("--- Loading and preprocessing data ---")
    
    # ------------------------------------------------------------------ STAGE 1
    # Cohort selection: pick the metadata file that drives this run and derive
    # the healthy / cancer disease filters for the dataset it belongs to.
    # Determine which metadata file to use based on mode
    if args.validate_external:
        if args.external_metadata_file is None:
            print("Error: --validate_external requires --external_metadata_file")
            print("   Example: --external_metadata_file accessory_files/jiang_metadata.csv")
            return
        active_metadata_file = args.external_metadata_file
        print(f"External validation mode: Using external dataset metadata from {active_metadata_file}")
    else:
        active_metadata_file = args.metadata_file
        print(f"Training/nested CV mode: Using metadata from {active_metadata_file}")
    
    if 'cris' in active_metadata_file.lower():
        metadata = pd.read_csv(active_metadata_file)
        healthy_to_filter = ['Healthy']

        # --- Process cancer types argument ---
        selected_cancers = args.cancer_types
        if 'PANCAN' in [c.upper() for c in selected_cancers]:
            cancer_types_to_filter = list(CANCER_MAP.values())
            print("PANCAN option selected. Including all available cancer types.")
        else:
            try:
                cancer_types_to_filter = [CANCER_MAP[c.lower()] for c in selected_cancers]
            except KeyError as e:
                print(f"Error: Invalid cancer type '{e.args[0]}'. Please choose from {list(CANCER_MAP.keys())} or PANCAN.")
                return

    elif 'jiang' in active_metadata_file.lower():
        metadata = pd.read_csv(active_metadata_file)
        healthy_to_filter = ['Healthy', 'Cirrhosis', 'Hepatitis B']
        cancer_types_to_filter = ['Liver cancer']

    elif 'lucas' in active_metadata_file.lower():
        metadata = pd.read_csv(active_metadata_file)
        healthy_to_filter = ['No baseline cancer', 'no lung cancer, other cancer', 'Benign', 'no lung cancer, later lung cancer', 'no lung cancer, other cancer, later lung cancer', 'no lung cancer, prior lung cancer', 'no lung cancer, prior lung cancer, later lung cancer']
        cancer_types_to_filter = ['lung cancer', 'lung cancer, other cancer', 'met to lung, other cancer', 'lung cancer, prior lung cancer', 'met to lung', 'lung cancer, later lung cancer']

    elif 'mathios' in active_metadata_file.lower():
        metadata = pd.read_csv(active_metadata_file)
        healthy_to_filter = ['healthy']
        cancer_types_to_filter = ['cancer']
    
    else:
        print(f"Error: Unknown metadata file '{active_metadata_file}'")
        print(f"   Expected one of: cris, jiang, lucas, mathios")
        print(f"   Please check your metadata file path or add support for this dataset.")
        return

    # Optional narrowing of class 0 (e.g. Jiang HCC vs healthy-only instead of
    # HCC vs all 135 non-HCC comparators). Only a subset of the dataset's own
    # control labels is accepted, so a typo cannot silently relabel cancers.
    if args.control_groups:
        unknown = [g for g in args.control_groups if g not in healthy_to_filter]
        if unknown:
            print(f"Error: --control_groups {unknown} are not control labels of this dataset "
                  f"(allowed: {healthy_to_filter})")
            return
        healthy_to_filter = list(args.control_groups)
        print(f"Class 0 restricted to control group(s): {', '.join(healthy_to_filter)}")

    # Optional explicit exclusion list (--exclude_samples): IDs dropped from
    # the metadata before any class split, so they are absent from both classes,
    # from the fold generator and from predefined-matrix rows (applied again
    # there, since those matrices carry their own sample lists). The panel of
    # normals is untouched. Sensitivity runs without the 30 deep-sequenced
    # healthy samples use accessory_files/cris_exclude_deep_healthy.txt.
    excluded_sample_ids = []
    if getattr(args, 'exclude_samples', None):
        with open(args.exclude_samples) as fh:
            excluded_sample_ids = [ln.split('#', 1)[0].strip() for ln in fh]
        excluded_sample_ids = sorted({s for s in excluded_sample_ids if s})
        present = metadata['ID'].isin(excluded_sample_ids)
        print(f"\n--exclude_samples {args.exclude_samples}: {len(excluded_sample_ids)} IDs listed, "
              f"{int(present.sum())} present in the metadata and EXCLUDED from this run:")
        for _, r in metadata[present].iterrows():
            print(f"     {r['ID']} ({r.get('sample_name', '?')}, {r['disease']})")
        metadata = metadata[~present].reset_index(drop=True)

    # Manifest-driven selection: when the metadata file carries a 'role'
    # column (accessory_files/cris_manifest.csv, built by
    # src/comp/build_cris_manifest.py), the manifest is the single source of
    # truth - role == 'model' selects samples, the panel of normals is
    # role == 'panel_of_normals', excluded records never enter, and feature
    # files are resolved by exact sample ID (no substring fallback).
    manifest_mode = 'role' in metadata.columns
    if manifest_mode:
        print("Manifest metadata detected ('role' column): role-based selection, exact-ID file matching")
        pending_mask = (metadata['role'] == 'model') & metadata.get(
            'gcfix_pending', pd.Series(False, index=metadata.index)).fillna(False).astype(bool)
        if int(pending_mask.sum()):
            print(f"{int(pending_mask.sum())} model-role sample(s) pending GC-correction recovery are EXCLUDED from this run:")
            for _, r in metadata[pending_mask].iterrows():
                print(f"     {r['ID']} ({r.get('sample_name', '?')}, {r['disease']})")
            print("   (regenerate the manifest after recovery to include them)")
        model_df = metadata[(metadata['role'] == 'model') & ~pending_mask]
        healthy_files_df = model_df[model_df['disease'].isin(healthy_to_filter)]
    else:
        print(f"Filtering out healthy samples with conditions: {', '.join(healthy_to_filter)}")
        if 'cris' in active_metadata_file.lower():  # Special handling for CRIS healthy samples since we use the first 50 for Panel-of-Normals
            healthy_files_df = metadata[metadata['disease'] == 'Healthy'].iloc[args.pon_n:]
        else:
            healthy_files_df = metadata[metadata['disease'].isin(healthy_to_filter)]
    healthy_files_df = healthy_files_df.drop_duplicates(subset='ID')

    healthy_file_names = healthy_files_df['ID'].tolist()
    print(f"Filtering for cancer types: {', '.join(cancer_types_to_filter)}")
    if manifest_mode:
        cancer_files_df = model_df[model_df['disease'].isin(cancer_types_to_filter)]
    else:
        cancer_files_df = metadata[metadata['disease'].isin(cancer_types_to_filter)]
    cancer_files_df = cancer_files_df.drop_duplicates(subset='ID')
    cancer_file_names = cancer_files_df['ID'].tolist()
    print(f"Found {len(cancer_file_names)} cancer samples.")
    print(f"Found {len(healthy_file_names)} healthy samples.")

    # Create a list of sample IDs so that after classification we can determine which samples were used
    # Note: This will be overridden if using a predefined matrix (sample order comes from predefined metadata)
    combined_df = pd.concat([healthy_files_df, cancer_files_df], ignore_index=True)
    sample_ids = combined_df['ID'].tolist()
    
    # Track whether we're using a predefined matrix (for sample_id override later)
    using_predefined_matrix = False
    predefined_sample_ids_healthy = []
    predefined_sample_ids_cancer = []
    
    # Initialize feature arrays (will be populated by loading or predefined matrix)
    feature_arrays_healthy = {}
    feature_arrays_cancer = {}
    
    # ------------------------------------------------------------------ STAGE 2
    # External-validation setup: load the training run's preprocessing metadata
    # so the external cohort can be filtered/standardized exactly like training.
    # Load training preprocessing metadata EARLY if doing external validation
    training_preprocessing_metadata = None
    required_feature_only = None  # For external validation optimization
    
    if args.validate_external:
        if args.external_model_dir is None:
            print("Error: --validate_external requires --external_model_dir")
            return
        
        training_metadata_file = Path(args.external_model_dir) / "preprocessing_metadata.pkl"
        if training_metadata_file.exists():
            with open(training_metadata_file, 'rb') as f:
                training_preprocessing_metadata = pickle.load(f)
            print(f"\nLoaded training preprocessing metadata")
            print(f"   Dataset: {training_preprocessing_metadata.get('dataset', 'N/A')}")
            print(f"   Blacklist: {training_preprocessing_metadata.get('blacklist_file_used', 'None')}")
            
            # Debug: Show what indices are being loaded
            kept_cols = training_preprocessing_metadata.get('kept_columns_per_feature', {})
            print(f"\n   Loaded column indices from training:")
            for feat, indices in kept_cols.items():
                print(f"      {feat}: {len(indices)} column indices (will select these from external data)")
            
            print(f"   -> External data will use training's exact column filtering\n")
        else:
            print(f"\n WARNING: No preprocessing metadata at {training_metadata_file}")
            print(f"   External validation may fail due to feature misalignment!\n")
        
        # OPTIMIZATION: Identify which feature is needed BEFORE loading all data
        # Skip this optimization if doing comprehensive validation (need all features)
        filter_feat = getattr(args, 'filter_feature', None)
        if args.comprehensive_external_validation:
            if filter_feat:
                # Comprehensive but filtered to one feature: only load that feature
                required_feature_only = filter_feat
                print(f"Comprehensive validation mode with filter: Loading only '{filter_feat}' feature\n")
            else:
                print(f"Comprehensive validation mode: Loading ALL features for multi-feature analysis\n")
                required_feature_only = None
        else:
            if filter_feat:
                required_feature_only = filter_feat
                print(f"Using --filter_feature: Only loading '{filter_feat}' feature for external validation\n")
            else:
                try:
                    required_feature_only = get_best_model_feature_name(args.external_model_dir)
                    print(f"OPTIMIZATION: Only loading '{required_feature_only}' feature for external validation")
                    print(f"   (Skipping other features to save time and memory)\n")
                except Exception as e:
                    print(f"Could not identify required feature: {e}")
                    print(f"   Falling back to loading all features\n")
                    required_feature_only = None

    # ------------------------------------------------------------------ STAGE 3
    # Feature loading: either one predefined sample x feature .npy matrix (ATAC),
    # or one column per feature type out of the per-sample *_BIN.csv files.
    # If a predefined feature matrix is provided, load it and build healthy/cancer splits
    if args.predefined_feature_matrix:
        print("\n" + "="*90)
        print("LOADING PREDEFINED FEATURE MATRIX")
        print("="*90)
        matrix_path = Path(args.predefined_feature_matrix)
        metadata_path = Path(args.predefined_metadata_csv) if args.predefined_metadata_csv else None

        # Prefer pancan ATAC matrix when running within-gen and pancan files exist
        if matrix_path.name == "ulz_atac_tfbs_features.npy":
            pancan_matrix = matrix_path.with_name("ulz_atac_tfbs_features_pancan.npy")
            pancan_meta = matrix_path.with_name("ulz_atac_tfbs_features_pancan.metadata.csv")
            use_pancan = pancan_matrix.exists() and pancan_meta.exists() and (
                args.run_within_dataset_generalization_train or
                args.run_within_dataset_generalization_eval or
                args.comprehensive_within_dataset_generalization
            )
            if use_pancan:
                print("Detected ATAC pancan matrix for within-gen; switching to pancan files")
                print(f"   Pancan matrix: {pancan_matrix}")
                print(f"   Pancan metadata: {pancan_meta}")
                matrix_path = pancan_matrix
                metadata_path = pancan_meta

        print(f"Matrix file: {matrix_path}")
        
        if metadata_path is None:
            print("Error: When using --predefined_feature_matrix you must also provide --predefined_metadata_csv")
            return

        print(f"Metadata CSV: {metadata_path}")
        
        pre_matrix = np.load(matrix_path)
        print(f"Loaded matrix with shape: {pre_matrix.shape} (samples x features)")
        print(f"  Matrix data type: {pre_matrix.dtype}")
        print(f"  Matrix value range: [{np.nanmin(pre_matrix):.4f}, {np.nanmax(pre_matrix):.4f}]")
        print(f"  Matrix mean: {np.nanmean(pre_matrix):.4f}, std: {np.nanstd(pre_matrix):.4f}")

        # --- ATAC column alignment for external validation ---
        # When the external predefined matrix has fewer columns (TFs) than training,
        # align by TF name to create a padded matrix matching training's column space.
        if args.validate_external and training_preprocessing_metadata is not None:
            training_n_features = training_preprocessing_metadata.get('original_n_features')
            if training_n_features is not None and pre_matrix.shape[1] != training_n_features:
                print(f"\nATAC column alignment needed: external has {pre_matrix.shape[1]} cols, training had {training_n_features}")
                
                # Load external TF names (alongside external matrix)
                external_tf_path = matrix_path.with_suffix('.tf_names.csv')
                if not external_tf_path.exists():
                    # Try alternate naming: replace .npy with .tf_names.csv
                    external_tf_path = matrix_path.parent / (matrix_path.stem + '.tf_names.csv')
                
                # Load training TF names from preprocessing metadata or auto-detect
                training_tf_names = training_preprocessing_metadata.get('predefined_tf_names')
                if training_tf_names is None:
                    # Auto-detect: look for tf_names.csv files matching the feature name
                    # Try common locations based on known naming convention
                    training_tf_candidates = [
                        matrix_path.parent / f"{matrix_path.stem.rsplit('_', 1)[0]}_brca_crc.tf_names.csv",
                        matrix_path.parent / f"{matrix_path.stem.rsplit('_', 1)[0]}.tf_names.csv",
                    ]
                    # Also try all tf_names files in extracted_features/ that match base name
                    base_feature_name = matrix_path.stem
                    for s in ['_jiang', '_lucas', '_mathios', '_cris']:
                        if base_feature_name.endswith(s):
                            base_feature_name = base_feature_name[:-len(s)]
                            break
                    training_tf_candidates.append(matrix_path.parent / f"{base_feature_name}_brca_crc.tf_names.csv")
                    training_tf_candidates.append(matrix_path.parent / f"{base_feature_name}.tf_names.csv")
                    
                    for candidate in training_tf_candidates:
                        if candidate.exists():
                            train_tf_df = pd.read_csv(candidate)
                            training_tf_names = train_tf_df.iloc[:, 0].tolist()
                            print(f"  Auto-detected training TF names from: {candidate}")
                            print(f"    ({len(training_tf_names)} TFs)")
                            break
                
                if external_tf_path.exists() and training_tf_names is not None:
                    ext_tf_df = pd.read_csv(external_tf_path)
                    external_tf_names = ext_tf_df.iloc[:, 0].tolist()
                    print(f"  Loaded external TF names from: {external_tf_path}")
                    print(f"    External: {len(external_tf_names)} TFs, Training: {len(training_tf_names)} TFs")
                    
                    # Verify dimensions match
                    assert len(external_tf_names) == pre_matrix.shape[1], \
                        f"TF names ({len(external_tf_names)}) != matrix cols ({pre_matrix.shape[1]})"
                    assert len(training_tf_names) == training_n_features, \
                        f"Training TF names ({len(training_tf_names)}) != training features ({training_n_features})"
                    
                    # Build mapping: training_col_idx -> external_col_idx (or -1)
                    ext_tf_to_idx = {name: i for i, name in enumerate(external_tf_names)}
                    overlap_count = sum(1 for tf in training_tf_names if tf in ext_tf_to_idx)
                    print(f"    Overlap: {overlap_count}/{len(training_tf_names)} training TFs found in external")
                    
                    # Create padded matrix with training's column count
                    padded = np.zeros((pre_matrix.shape[0], training_n_features), dtype=pre_matrix.dtype)
                    for train_idx, tf_name in enumerate(training_tf_names):
                        ext_idx = ext_tf_to_idx.get(tf_name)
                        if ext_idx is not None:
                            padded[:, train_idx] = pre_matrix[:, ext_idx]
                    
                    print(f"  Padded matrix: {pre_matrix.shape} -> {padded.shape}")
                    print(f"    {overlap_count} columns filled from external data, "
                          f"{training_n_features - overlap_count} columns zero-padded")
                    pre_matrix = padded
                elif not external_tf_path.exists():
                    print(f"   External TF names file not found at {external_tf_path}")
                    print(f"     Cannot align columns - validation may fail!")
                elif training_tf_names is None:
                    print(f"   Training TF names not found in preprocessing metadata or auto-detect")
                    print(f"     Cannot align columns - validation may fail!")

        meta_df = pd.read_csv(metadata_path)
        print(f"Loaded metadata CSV with {len(meta_df)} rows and columns: {list(meta_df.columns)}")

        # Normalize column name for sample ID
        if 'sample_id' in meta_df.columns:
            id_col = 'sample_id'
        elif 'ID' in meta_df.columns:
            id_col = 'ID'
        else:
            print("Error: predefined metadata CSV must contain a 'sample_id' or 'ID' column")
            return

        sample_order = meta_df[id_col].astype(str).tolist()
        print(f"Using '{id_col}' column for sample IDs ({len(sample_order)} samples)")
        print(f"  First 5 sample IDs: {sample_order[:5]}")
        
        if pre_matrix.shape[0] != len(sample_order):
            print(f"Error: Number of rows in predefined matrix ({pre_matrix.shape[0]}) does not match number of sample IDs in metadata ({len(sample_order)})")
            return

        # Resolve disease labels (prefer predefined metadata if present)
        merged_meta = pd.DataFrame({'ID': sample_order})
        disease_col = None
        for col in meta_df.columns:
            if col.lower() == 'disease':
                disease_col = col
                break

        if disease_col is not None:
            merged_meta['disease'] = meta_df[disease_col].astype(str)
            print(f"\nUsing disease labels from predefined metadata column '{disease_col}'")
        else:
            merged_meta = merged_meta.merge(metadata[['ID', 'disease']], on='ID', how='left')
            print(f"\nMerged with main metadata to recover disease labels")

        if merged_meta['disease'].isna().any():
            missing = merged_meta['disease'].isna().sum()
            missing_ids = merged_meta.loc[merged_meta['disease'].isna(), 'ID'].tolist()
            print(f"Missing disease labels for {missing} sample(s)")
            print(f"   Missing IDs (first 10): {missing_ids[:10]}")

        disease_counts = merged_meta['disease'].value_counts()
        print(f"  Disease distribution in predefined matrix:")
        for disease, count in disease_counts.items():
            print(f"    {disease}: {count}")

        # Debug: show training types present for within-gen
        if (args.run_within_dataset_generalization_train or args.run_within_dataset_generalization_eval or args.comprehensive_within_dataset_generalization) and args.generalization_training_types:
            print("\nWithin-gen debug: training types in predefined matrix")
            for ct in args.generalization_training_types:
                ct_count = int((merged_meta['disease'] == ct).sum())
                print(f"   {ct}: {ct_count} sample(s)")
            healthy_count = int((merged_meta['disease'].fillna('') == 'Healthy').sum())
            print(f"   Healthy: {healthy_count} sample(s)")

        # Optionally remove Panel-of-Normal samples for CRIS datasets
        pon_removed_count = 0
        if 'cris' in active_metadata_file.lower():
            print(f"\nCRIS dataset detected - checking for Panel-of-Normal samples to remove")
            if manifest_mode:
                print("  PoN configuration: role == 'panel_of_normals' from the manifest")
                pon_ids = metadata[metadata['role'] == 'panel_of_normals']['ID'].tolist()
            else:
                print(f"  PoN configuration: First {args.pon_n} healthy samples from main metadata")
                pon_ids = metadata[metadata['disease'] == 'Healthy'].iloc[:args.pon_n]['ID'].tolist()
            if pon_ids:
                print(f"  PoN sample ID range: {pon_ids[0]} to {pon_ids[-1]}")
            
            pon_present = [s for s in sample_order if s in pon_ids]
            if pon_present:
                print(f"  Found {len(pon_present)} PoN samples in predefined matrix")
                print(f"  PoN samples to remove: {pon_present[:10]}{'...' if len(pon_present)>10 else ''}")
                # Keep rows that are NOT in pon_present
                keep_mask = [s not in pon_present for s in sample_order]
                pre_matrix = pre_matrix[np.array(keep_mask), :]
                merged_meta = merged_meta.loc[keep_mask].reset_index(drop=True)
                sample_order = merged_meta['ID'].tolist()
                pon_removed_count = len(pon_present)
                print(f"  Removed {pon_removed_count} PoN samples")
                print(f"  Matrix shape after PoN removal: {pre_matrix.shape}")
            else:
                print(f"  No PoN samples found in predefined matrix (already excluded or not present)")
        else:
            print(f"\nNon-CRIS dataset - skipping Panel-of-Normal removal")

        # --exclude_samples: the predefined matrix carries its own sample list, so
        # the exclusion is applied to its rows as well (same mechanics as the PoN removal)
        if excluded_sample_ids:
            excl_present = [s for s in sample_order if s in set(excluded_sample_ids)]
            if excl_present:
                keep_mask = [s not in set(excl_present) for s in sample_order]
                pre_matrix = pre_matrix[np.array(keep_mask), :]
                merged_meta = merged_meta.loc[keep_mask].reset_index(drop=True)
                sample_order = merged_meta['ID'].tolist()
                print(f"\n--exclude_samples: removed {len(excl_present)} row(s) from the predefined matrix "
                      f"-> shape {pre_matrix.shape}")
            else:
                print(f"\n--exclude_samples: none of the listed IDs are rows of the predefined matrix")

        # Apply row-standardization (skipped for heterogeneous scalar panels, where a
        # per-sample z-score across features mixes units; the pipeline's StandardScaler
        # still column-standardizes on training folds)
        if args.predefined_no_row_standardize:
            print(f"\nSkipping row-wise standardization (--predefined_no_row_standardize)")
            pre_matrix = pre_matrix.astype(float)
            n_nan = int(np.sum(np.isnan(pre_matrix)))
            if n_nan:
                raise ValueError(
                    f"Predefined matrix contains {n_nan} NaN value(s). Without row "
                    "standardization there is no nan_to_num step; impute when building the matrix.")
        else:
            print(f"\nApplying row-wise z-score standardization to predefined matrix")
            print(f"  Before standardization - mean: {np.nanmean(pre_matrix):.4f}, std: {np.nanstd(pre_matrix):.4f}")
            mat = pre_matrix.astype(float).copy()
            row_means = np.nanmean(mat, axis=1, keepdims=True)
            row_stds = np.nanstd(mat, axis=1, keepdims=True)
            row_stds[row_stds == 0] = 1.0
            pre_matrix = (mat - row_means) / row_stds
            pre_matrix = np.nan_to_num(pre_matrix)  # Replace NaN with 0, inf with large finite numbers
            print(f"  After standardization  - mean: {np.nanmean(pre_matrix):.4f}, std: {np.nanstd(pre_matrix):.4f}")
            print(f"  NaN values remaining: {np.sum(np.isnan(pre_matrix))}")
            print(f"  Row standardization complete")


        # Build healthy / cancer splits using the merged_meta diseases and previously selected cancer_types_to_filter
        print(f"\nSplitting predefined matrix into healthy and cancer subsets")
        print(f"  Healthy filter criteria: {healthy_to_filter}")
        print(f"  Cancer filter criteria: {cancer_types_to_filter}")
        
        disease_lc = merged_meta['disease'].fillna('').str.lower()
        healthy_mask = disease_lc.isin([d.lower() for d in healthy_to_filter])
        cancer_mask = disease_lc.isin([d.lower() for d in cancer_types_to_filter]) if len(cancer_types_to_filter) > 0 else np.array([False]*len(disease_lc))

        healthy_mat = pre_matrix[np.where(healthy_mask)[0], :] if healthy_mask.any() else np.empty((0, pre_matrix.shape[1]))
        cancer_mat = pre_matrix[np.where(cancer_mask)[0], :] if cancer_mask.any() else np.empty((0, pre_matrix.shape[1]))

        print(f"  Healthy subset: {healthy_mat.shape[0]} samples x {healthy_mat.shape[1]} features")
        print(f"  Cancer subset:  {cancer_mat.shape[0]} samples x {cancer_mat.shape[1]} features")
        
        # Get sample IDs for each subset for logging
        healthy_sample_ids = merged_meta.loc[healthy_mask, 'ID'].tolist() if healthy_mask.any() else []
        cancer_sample_ids = merged_meta.loc[cancer_mask, 'ID'].tolist() if cancer_mask.any() else []
        print(f"  Healthy sample IDs (first 5): {healthy_sample_ids[:5]}")
        print(f"  Cancer sample IDs (first 5):  {cancer_sample_ids[:5]}")

        # Derive feature name for predefined matrices
        # For external validation: use the key from training's preprocessing metadata
        # For training: derive from matrix filename, stripping dataset suffixes
        if args.validate_external and training_preprocessing_metadata is not None:
            training_kept_cols = training_preprocessing_metadata.get('kept_columns_per_feature', {})
            if training_kept_cols:
                predefined_feature_name = list(training_kept_cols.keys())[0]
                print(f"  Using feature name from training metadata: '{predefined_feature_name}'")
            else:
                predefined_feature_name = "predefined_matrix"
        else:
            # Derive from filename: strip known dataset suffixes to get base feature name
            predefined_feature_name = matrix_path.stem
            for suffix in ['_brca_crc', '_pancan', '_jiang', '_lucas', '_mathios', '_cris']:
                if predefined_feature_name.endswith(suffix):
                    predefined_feature_name = predefined_feature_name[:-len(suffix)]
                    break
            print(f"  Derived feature name from filename: '{predefined_feature_name}'")
        tss_features = [predefined_feature_name]
        feature_arrays_healthy[predefined_feature_name] = healthy_mat
        feature_arrays_cancer[predefined_feature_name] = cancer_mat
        predefined_sample_ids_healthy = healthy_sample_ids
        predefined_sample_ids_cancer = cancer_sample_ids
        using_predefined_matrix = True

    else:
        # Load features from folder (optionally optimized to a single required feature)
        features_dir = args.features_dir
        tss_features = [required_feature_only] if required_feature_only else DATA_PARAM_GRIDS['feature']
        if args.feature_subset:
            requested = set(args.feature_subset)
            tss_features = [feat for feat in tss_features if feat in requested]
            if len(tss_features) == 0:
                print(f"Error: --feature_subset {args.feature_subset} does not match any available features")
                print(f"   Available features: {DATA_PARAM_GRIDS['feature']}")
                return
        print(f"\nLoading features from: {features_dir}")
        print(f"   Features: {tss_features}")

        for feature_to_load in tss_features:
            print(f"  -> Loading '{feature_to_load}'")
            feature_arrays_healthy[feature_to_load] = load_feature_in_parallel(
                features_dir,
                feature_to_load,
                file_filter=healthy_file_names,
                num_cores=args.cores,
                exact=manifest_mode
            )
            feature_arrays_cancer[feature_to_load] = load_feature_in_parallel(
                features_dir,
                feature_to_load,
                file_filter=cancer_file_names,
                num_cores=args.cores,
                exact=manifest_mode
            )

    # ------------------------------------------------------------------ STAGE 4
    # Column filtering: blacklist removal for internal runs, or the training
    # run's exact kept-column indices for external validation.
    # Track which columns are kept after preprocessing (for external validation)
    kept_columns_per_feature = {}  # Will store which columns survive preprocessing
    original_n_features = None  # Track original feature count before any filtering
    blacklist_file_used = None  # Track which blacklist file was used
    
    # For external validation: apply training's column filtering directly (skip blacklist/duplicate removal)
    if args.validate_external and training_preprocessing_metadata is not None:
        print("\nApplying training's preprocessing to external data (skipping blacklist/duplicate steps)")
        training_kept_cols = training_preprocessing_metadata.get('kept_columns_per_feature', {})
        
        print(f"   Training metadata contains {len(training_kept_cols)} features")
        for feat_name, indices_list in training_kept_cols.items():
            print(f"      {feat_name}: {len(indices_list)} column indices")
        
        for feature in tss_features:
            if feature in training_kept_cols:
                kept_indices = np.array(training_kept_cols[feature])
                
                print(f"\n   Applying {len(kept_indices)} training indices to '{feature}'")
                
                if feature_arrays_healthy.get(feature) is not None:
                    original_shape = feature_arrays_healthy[feature].shape
                    feature_arrays_healthy[feature] = feature_arrays_healthy[feature][:, kept_indices]
                    print(f"      Healthy: {original_shape[1]} -> {feature_arrays_healthy[feature].shape[1]} columns")
                
                if feature_arrays_cancer.get(feature) is not None:
                    original_shape = feature_arrays_cancer[feature].shape
                    feature_arrays_cancer[feature] = feature_arrays_cancer[feature][:, kept_indices]
                    print(f"      Cancer: {original_shape[1]} -> {feature_arrays_cancer[feature].shape[1]} columns")
                
                # Store the same indices as training
                kept_columns_per_feature[feature] = kept_indices
        
        # Copy metadata from training
        original_n_features = training_preprocessing_metadata.get('original_n_features')
        blacklist_file_used = training_preprocessing_metadata.get('blacklist_file_used')
        
        # Check if training had final feature counts recorded
        training_final_counts = training_preprocessing_metadata.get('final_n_features_per_feature', {})
        if training_final_counts:
            print(f"\n   Training's final feature counts:")
            for feat, count in training_final_counts.items():
                print(f"      {feat}: {count} columns")
        
        print("   Training's preprocessing applied to external data")
    
    # Remove blacklisted regions (only when NOT in no_gene_selection mode, not using predefined matrix, and NOT external validation)
    elif (not args.no_gene_selection) and (not args.predefined_feature_matrix):
        to_be_removed = None
        _bl_default = "accessory_files/final_tss_indices_to_remove_hg38.npy"
        if getattr(args, 'blacklist_indices', None) and args.blacklist_indices != _bl_default:
            # an explicit exclusion file for another region set with the same layout
            to_be_removed = np.load(args.blacklist_indices)
            blacklist_file_used = args.blacklist_indices
        elif 'cris' in active_metadata_file.lower() or 'jiang' in active_metadata_file.lower():
            to_be_removed = np.load("accessory_files/final_tss_indices_to_remove_hg38.npy")
            blacklist_file_used = "accessory_files/final_tss_indices_to_remove_hg38.npy"
        elif ('lucas' in active_metadata_file.lower() or 'mathios' in active_metadata_file.lower()):
            to_be_removed = np.load("accessory_files/final_tss_indices_to_remove_hg19.npy")
            blacklist_file_used = "accessory_files/final_tss_indices_to_remove_hg19.npy"
        else:
            print("Skipping blacklist removal since there is not a predefined blacklist for this dataset.")
        
        if to_be_removed is not None:
            print(f"\nApplying blacklist removal from {blacklist_file_used}")
            print(f"   Removing {len(to_be_removed)} regions")
            
            for feature in tss_features:
                if feature_arrays_healthy[feature] is not None:
                    # Store original feature count
                    if original_n_features is None:
                        original_n_features = feature_arrays_healthy[feature].shape[1]
                    
                    # Track which columns we keep
                    all_indices = np.arange(original_n_features)
                    kept_mask = np.ones(original_n_features, dtype=bool)
                    kept_mask[to_be_removed] = False
                    kept_columns_per_feature[feature] = np.where(kept_mask)[0]
                    
                    # Apply removal
                    feature_arrays_healthy[feature] = np.delete(feature_arrays_healthy[feature], to_be_removed, axis=1)
                    print(f"   {feature}: {original_n_features} -> {len(kept_columns_per_feature[feature])} columns")
                    
                if feature_arrays_cancer[feature] is not None:
                    feature_arrays_cancer[feature] = np.delete(feature_arrays_cancer[feature], to_be_removed, axis=1)
            print("   Blacklist removal complete")
    else:
        print("\nSkipping blacklist removal (no_gene_selection mode or predefined matrix)")
        # No filtering - all columns are kept
        for feature in tss_features:
            if feature_arrays_healthy.get(feature) is not None:
                if original_n_features is None:
                    original_n_features = feature_arrays_healthy[feature].shape[1]
                kept_columns_per_feature[feature] = np.arange(original_n_features)
                print(f"   {feature}: All {original_n_features} columns kept")

    # ------------------------------------------------------------------ STAGE 5
    # Normalization: row-wise z-score per sample; for external validation also
    # column-wise standardization with the TRAINING cohort's means/stds.
    # Normalize data (skip if using predefined matrix that may already be normalized)
    if args.predefined_feature_matrix:
        feature_arrays_healthy_normalized = feature_arrays_healthy
        feature_arrays_cancer_normalized = feature_arrays_cancer
    else:
        print("\nApplying normalize_feature_arrays() to loaded TSS features")
        feature_arrays_healthy_normalized = normalize_feature_arrays(feature_arrays_healthy)
        feature_arrays_cancer_normalized = normalize_feature_arrays(feature_arrays_cancer)
        print("All features normalized.")

    # If external validation and training stats are available, standardize using training mean/std
    if args.validate_external and training_preprocessing_metadata is not None:
        if 'feature_means_per_feature' not in training_preprocessing_metadata or \
           'feature_stds_per_feature' not in training_preprocessing_metadata:
            # Training stats missing from pkl - compute from training data
            print("\nTraining column-wise stats missing from preprocessing metadata")
            print("   Computing from training data (for consistent standardization)...")

            computed_means = {}
            computed_stds = {}

            if using_predefined_matrix:
                # ATAC: Load training predefined matrix
                ext_path = Path(args.predefined_feature_matrix)
                training_matrix_path = None
                for suffix in ['_jiang', '_lucas', '_mathios']:
                    if suffix in ext_path.stem:
                        for replacement in ['_pancan', '_brca_crc', '']:
                            cand = ext_path.parent / (ext_path.stem.replace(suffix, replacement) + ext_path.suffix)
                            if cand.exists() and cand != ext_path:
                                training_matrix_path = cand
                                break
                    if training_matrix_path:
                        break

                if training_matrix_path is not None:
                    print(f"   Loading training ATAC matrix: {training_matrix_path}")
                    train_mat = np.load(training_matrix_path)
                    print(f"   Shape: {train_mat.shape}")

                    # Apply row z-score (same as predefined matrix loading)
                    mat = train_mat.astype(float).copy()
                    row_means_t = np.nanmean(mat, axis=1, keepdims=True)
                    row_stds_t = np.nanstd(mat, axis=1, keepdims=True)
                    row_stds_t[row_stds_t == 0] = 1.0
                    train_mat = np.nan_to_num((mat - row_means_t) / row_stds_t)

                    # Apply training column filtering
                    kept_cols = training_preprocessing_metadata.get('kept_columns_per_feature', {})
                    for feat_name, indices in kept_cols.items():
                        indices_arr = np.array(indices)
                        if indices_arr.max() < train_mat.shape[1]:
                            filtered = train_mat[:, indices_arr]
                            computed_means[feat_name] = np.nanmean(filtered, axis=0).tolist()
                            computed_stds[feat_name] = np.nanstd(filtered, axis=0).tolist()
                            print(f"   '{feat_name}': computed from {filtered.shape[0]} samples x {filtered.shape[1]} features")
                else:
                    print(f"    Could not find training ATAC matrix (tried _brca_crc / base variants)")
            else:
                # CSV features (DD/RNA): Load training CSVs from features_dir
                training_metadata_path = training_preprocessing_metadata.get('dataset', args.metadata_file)
                if training_metadata_path and Path(training_metadata_path).exists():
                    train_meta = pd.read_csv(training_metadata_path)
                    if 'ID' in train_meta.columns:
                        train_file_names = train_meta['ID'].tolist()
                        print(f"   Loading training CSVs ({len(train_file_names)} samples) from {args.features_dir}")

                        kept_cols = training_preprocessing_metadata.get('kept_columns_per_feature', {})
                        for feat_name in tss_features:
                            if feat_name not in kept_cols:
                                continue
                            print(f"   Loading training '{feat_name}'...")
                            train_array = load_feature_in_parallel(
                                args.features_dir, feat_name,
                                file_filter=train_file_names,
                                num_cores=args.cores
                            )
                            if train_array is not None:
                                # Apply row normalization (same as normalize_feature_arrays)
                                row_means_t = np.nanmean(train_array, axis=1, keepdims=True)
                                row_stds_t = np.nanstd(train_array, axis=1, keepdims=True)
                                row_stds_t[row_stds_t == 0] = 1.0
                                train_array = np.nan_to_num((train_array - row_means_t) / row_stds_t)

                                # Apply column filtering
                                indices_arr = np.array(kept_cols[feat_name])
                                if indices_arr.max() < train_array.shape[1]:
                                    filtered = train_array[:, indices_arr]
                                    computed_means[feat_name] = np.nanmean(filtered, axis=0).tolist()
                                    computed_stds[feat_name] = np.nanstd(filtered, axis=0).tolist()
                                    print(f"   '{feat_name}': computed from {filtered.shape[0]} samples x {filtered.shape[1]} features")
                    else:
                        print(f"    Training metadata has no 'ID' column; cannot load training CSVs")
                else:
                    print(f"    Training metadata file not found: {training_metadata_path}")

            if computed_means and computed_stds:
                training_preprocessing_metadata['feature_means_per_feature'] = computed_means
                training_preprocessing_metadata['feature_stds_per_feature'] = computed_stds
                print(f"   Training column-wise stats computed for {len(computed_means)} feature(s)")

                # Save back to pkl so future runs are instant
                pkl_path = Path(args.external_model_dir) / "preprocessing_metadata.pkl"
                if pkl_path.exists():
                    with open(pkl_path, 'wb') as f:
                        pickle.dump(training_preprocessing_metadata, f)
                    print(f"   Saved training stats to {pkl_path} for future runs")

        if 'feature_means_per_feature' in training_preprocessing_metadata and \
           'feature_stds_per_feature' in training_preprocessing_metadata:
            print("\nApplying training mean/std standardization to external data")
            feature_arrays_healthy_normalized = standardize_with_training_stats(
                feature_arrays_healthy_normalized,
                training_preprocessing_metadata
            )
            feature_arrays_cancer_normalized = standardize_with_training_stats(
                feature_arrays_cancer_normalized,
                training_preprocessing_metadata
            )
        else:
            print("\n Training mean/std still unavailable after fallback; proceeding without column standardization")

    # ------------------------------------------------------------------ STAGE 6
    # Assemble the final arrays: stack healthy+cancer, drop duplicate columns,
    # record preprocessing metadata, and build label / stratification arrays.
    # Combine healthy and cancer data
    print("\nCombining healthy and cancer data into unified feature arrays")
    all_feature_arrays = {}
    for feature in tss_features:
        healthy_data = feature_arrays_healthy_normalized.get(feature)
        cancer_data = feature_arrays_cancer_normalized.get(feature)
        if healthy_data is not None and cancer_data is not None:
            combined = np.vstack([healthy_data, cancer_data])
            all_feature_arrays[feature] = combined
            print(f"  Feature '{feature}': {healthy_data.shape[0]} healthy + {cancer_data.shape[0]} cancer = {combined.shape[0]} total samples x {combined.shape[1]} features")

    # Per feature, check for duplicate columns and remove them (skip for external validation)
    if args.validate_external and training_preprocessing_metadata is not None:
        print("\nSkipping duplicate removal (already handled by training's column filtering)")
        unique_feature_arrays = all_feature_arrays
    else:
        print("\nChecking for and removing duplicate feature columns")
        unique_feature_arrays = {}
        for feature, array in all_feature_arrays.items():
            if array is None:
                continue
            original_cols = array.shape[1]
            _, unique_indices = np.unique(array, axis=1, return_index=True)
            sorted_unique_indices = np.sort(unique_indices)
            unique_array = array[:, sorted_unique_indices]
            unique_feature_arrays[feature] = unique_array
            
            # Update kept_columns to reflect duplicate removal
            # CRITICAL: Must use sorted indices to match the array column order
            if feature in kept_columns_per_feature:
                kept_columns_per_feature[feature] = kept_columns_per_feature[feature][sorted_unique_indices]
            
            if original_cols != unique_array.shape[1]:
                print(f"  Feature '{feature}': Reduced from {original_cols} to {unique_array.shape[1]} unique columns")
            else:
                print(f"  Feature '{feature}': All {unique_array.shape[1]} columns are unique")
    
    # Replace the old dictionary and feature list with the unique ones
    all_feature_arrays = unique_feature_arrays
    DATA_PARAM_GRIDS['feature'] = list(all_feature_arrays.keys())

    # --- Covariate block (--augment_covariates / --residualize_covariates): appended
    # to every TSS matrix AFTER row standardization and duplicate removal, aligned by
    # sample ID; see COVARIATE_BLOCK. Not combined with predefined matrices.
    cov_stem = getattr(args, 'augment_covariates', None) or getattr(args, 'residualize_covariates', None)
    # External validation of a model trained with a covariate block: the external
    # cohort's block (same covariates, that cohort's panel stem) must be given, is
    # appended in the same position and standardized with the training statistics of
    # those columns (the TSS columns were standardized above).
    trained_block = (training_preprocessing_metadata or {}).get('covariate_block') or {}
    if args.validate_external and trained_block.get('mode'):
        if trained_block['mode'] != 'augment':
            raise ValueError(f"external validation of a '{trained_block['mode']}' covariate model is not defined")
        if not getattr(args, 'augment_covariates', None):
            raise ValueError("the deployed model was trained with --augment_covariates "
                             f"{trained_block.get('stem')} ({', '.join(trained_block.get('names', []))}); "
                             "pass --augment_covariates <external cohort's stem of the same panel>")
    elif args.validate_external and cov_stem:
        raise ValueError("--augment_covariates given, but the deployed model was trained without a covariate block")
    if cov_stem:
        if getattr(args, 'augment_covariates', None) and getattr(args, 'residualize_covariates', None):
            raise ValueError("--augment_covariates and --residualize_covariates are mutually exclusive")
        if args.predefined_feature_matrix:
            raise ValueError("the covariate block is defined for TSS feature matrices, not predefined matrices")
        cov_mode = 'augment' if getattr(args, 'augment_covariates', None) else 'residualize'
        C_block, cov_names = load_covariate_block(cov_stem, sample_ids)
        if args.validate_external:
            if list(cov_names) != list(trained_block.get('names', [])):
                raise ValueError(f"covariate names of {cov_stem} {cov_names} differ from the deployed model's "
                                 f"{trained_block.get('names')}")
            means_map = training_preprocessing_metadata.get('feature_means_per_feature', {})
            stds_map = training_preprocessing_metadata.get('feature_stds_per_feature', {})
        for feature, arr in all_feature_arrays.items():
            if arr.shape[0] != C_block.shape[0]:
                raise ValueError(f"{feature}: {arr.shape[0]} rows vs {C_block.shape[0]} covariate rows")
            block = C_block
            if args.validate_external:
                n_cov = C_block.shape[1]
                means, stds = np.array(means_map[feature]), np.array(stds_map[feature])
                if len(means) != arr.shape[1] + n_cov:
                    raise ValueError(f"{feature}: training stats have {len(means)} columns, expected "
                                     f"{arr.shape[1]} TSS + {n_cov} covariate columns")
                mu, sd = means[-n_cov:], stds[-n_cov:].copy()
                sd[sd == 0] = 1.0
                block = (C_block - mu) / sd
                print(f"  '{feature}': covariate block standardized with the training statistics "
                      f"(means {np.round(mu, 3).tolist()})")
            all_feature_arrays[feature] = np.hstack([arr, block])
        COVARIATE_BLOCK.update(mode=cov_mode, n=int(C_block.shape[1]), names=cov_names, stem=str(cov_stem))
        print(f"\nCovariate block ({cov_mode}): {C_block.shape[1]} covariates from {cov_stem} appended to every "
              f"feature matrix as passthrough columns - {', '.join(cov_names)}")
    
    total_columns = sum(arr.shape[1] for arr in all_feature_arrays.values() if arr is not None)
    print(f"\nProceeding with {len(all_feature_arrays)} feature type(s), totaling {total_columns} columns across all features.")
    
    # Compute per-feature training mean/std (for external standardization)
    feature_means_per_feature = {
        feature: np.nanmean(arr, axis=0).tolist()
        for feature, arr in all_feature_arrays.items() if arr is not None
    }
    feature_stds_per_feature = {
        feature: np.nanstd(arr, axis=0).tolist()
        for feature, arr in all_feature_arrays.items() if arr is not None
    }

    # Store comprehensive preprocessing metadata for external validation
    preprocessing_metadata = {
        'original_n_features': original_n_features,
        'kept_columns_per_feature': {k: v.tolist() for k, v in kept_columns_per_feature.items()},
        'blacklist_file_used': blacklist_file_used,
        'blacklist_applied': blacklist_file_used is not None,
        'duplicates_removed': True,
        'final_n_features_per_feature': {feature: arr.shape[1] for feature, arr in all_feature_arrays.items() if arr is not None},
        'feature_means_per_feature': feature_means_per_feature,
        'feature_stds_per_feature': feature_stds_per_feature,
        'note': 'External datasets must use kept_columns_per_feature to select the EXACT same columns from raw data',
        'dataset': active_metadata_file,
        # --exclude_samples bookkeeping (empty list when the flag was not used)
        'exclude_samples_file': getattr(args, 'exclude_samples', None),
        'excluded_sample_ids': excluded_sample_ids,
        # covariate block (empty when neither --augment_covariates nor --residualize_covariates)
        'covariate_block': dict(COVARIATE_BLOCK),
        # derived-task labels: always None (option removed); key kept for the result format
        'task_controls': getattr(args, 'task_controls', None),
        'task_cases': getattr(args, 'task_cases', None),
    }
    
    # If using predefined matrix, save TF names for column alignment in external validation
    if args.predefined_feature_matrix:
        tf_names_path = Path(args.predefined_feature_matrix).with_suffix('.tf_names.csv')
        if not tf_names_path.exists():
            tf_names_path = Path(args.predefined_feature_matrix).parent / (Path(args.predefined_feature_matrix).stem + '.tf_names.csv')
        if tf_names_path.exists():
            tf_df = pd.read_csv(tf_names_path)
            preprocessing_metadata['predefined_tf_names'] = tf_df.iloc[:, 0].tolist()
            print(f"   Saved {len(preprocessing_metadata['predefined_tf_names'])} TF names in preprocessing metadata")
    
    print(f"\nPreprocessing metadata summary (for external validation):")
    if original_n_features:
        print(f"   Original features: {original_n_features}")
    print(f"   Blacklist applied: {preprocessing_metadata['blacklist_applied']}")
    if blacklist_file_used:
        print(f"   Blacklist file: {blacklist_file_used}")
    for feature in all_feature_arrays.keys():
        if feature in kept_columns_per_feature:
            n_kept = len(kept_columns_per_feature[feature])
            n_actual_cols = all_feature_arrays[feature].shape[1]
            print(f"   {feature}: {n_kept} indices saved -> {n_actual_cols} columns in array (indices: [{kept_columns_per_feature[feature][0]}...{kept_columns_per_feature[feature][-1]}])")
            if n_kept != n_actual_cols:
                print(f"      WARNING: Mismatch! Saved {n_kept} indices but array has {n_actual_cols} columns!")
    
    # Store column indices for external validation
    # This allows external datasets to use the EXACT same feature columns
    column_indices_per_feature = {}
    for feature in all_feature_arrays.keys():
        if all_feature_arrays[feature] is not None:
            column_indices_per_feature[feature] = {
                'n_columns': all_feature_arrays[feature].shape[1],
                'blacklist_applied': not args.no_gene_selection and not args.predefined_feature_matrix,
                'duplicates_removed': True,  # We always check for duplicates
                'note': 'External datasets must use these exact column counts for proper alignment'
            }
    
    print(f"\nColumn indices info (for external validation reproducibility):")
    for feature, info in column_indices_per_feature.items():
        print(f"   {feature}: {info['n_columns']} columns (blacklist={info['blacklist_applied']})")

    n_healthy = feature_arrays_healthy_normalized[tss_features[0]].shape[0] if feature_arrays_healthy_normalized.get(tss_features[0]) is not None else 0
    n_cancer = feature_arrays_cancer_normalized[tss_features[0]].shape[0] if feature_arrays_cancer_normalized.get(tss_features[0]) is not None else 0
    y_labels = np.array([0] * n_healthy + [1] * n_cancer)
    
    # Override sample_ids if using predefined matrix (order is guaranteed by predefined metadata)
    if using_predefined_matrix:
        sample_ids = predefined_sample_ids_healthy + predefined_sample_ids_cancer
        print(f"\nUsing sample IDs from predefined matrix metadata")
        print(f"   Total: {len(sample_ids)} samples ({len(predefined_sample_ids_healthy)} healthy + {len(predefined_sample_ids_cancer)} cancer)")
        print(f"   First 5 sample IDs: {sample_ids[:5]}")
    
    print(f"\nTotal samples: {len(y_labels)} ({n_healthy} healthy, {n_cancer} cancer)")

    # Create detailed labels for stratification
    if using_predefined_matrix:
        # Build stratification labels from the predefined metadata (merged_meta already has correct order)
        healthy_strat_labels = merged_meta.loc[healthy_mask, 'disease'].tolist() if healthy_mask.any() else []
        cancer_strat_labels = merged_meta.loc[cancer_mask, 'disease'].tolist() if cancer_mask.any() else []
        stratification_labels = np.array(healthy_strat_labels + cancer_strat_labels)
        print(f"Using stratification labels from predefined matrix metadata")
    else:
        healthy_strat_labels = healthy_files_df['disease'].tolist()
        cancer_strat_labels = cancer_files_df['disease'].tolist()
        stratification_labels = np.array(healthy_strat_labels + cancer_strat_labels)
    
    print(f"Stratification will be based on {len(np.unique(stratification_labels))} unique classes.")
    unique_strat, strat_counts = np.unique(stratification_labels, return_counts=True)
    for label, count in zip(unique_strat, strat_counts):
        print(f"  {label}: {count} samples")
    
    print("\n" + "="*90)
    print("DATA PREPARATION SUMMARY")
    print("="*90)
    print(f"Mode: {'PREDEFINED MATRIX' if using_predefined_matrix else 'TSS FEATURE EXTRACTION'}")
    print(f"Features loaded: {list(all_feature_arrays.keys())}")
    print(f"Total samples: {len(sample_ids)} ({n_healthy} healthy + {n_cancer} cancer)")
    print(f"Sample ordering: Healthy first, then cancer (as required by downstream code)")
    print(f"Binary labels (y_labels): {np.sum(y_labels == 0)} zeros (healthy), {np.sum(y_labels == 1)} ones (cancer)")
    if using_predefined_matrix:
        print(f"Panel-of-Normal handling: {'Applied (removed first ' + str(args.pon_n) + ' healthy)' if 'cris' in active_metadata_file.lower() else 'Not applicable'}")
        print(f"normalize_feature_arrays(): Skipped (predefined matrix used as-is)")
    else:
        print(f"normalize_feature_arrays(): Applied")
        print(f"Blacklist removal: {'Applied' if (not args.no_gene_selection) else 'Skipped'}")
    print("="*90)
    
    print("\n--- Data preparation complete ---")

    # ------------------------------------------------------------------ STAGE 7
    # Mode dispatch: each requested mode runs on the arrays prepared above.
    # --- 1. Run Best Model Search (if requested) ---
    if args.run_best_model_search:
        print("\n--- STARTING BEST MODEL PER-FEATURE SEARCH ---")
        run_best_model_search(
            all_feature_arrays=all_feature_arrays,
            y_labels=y_labels,
            stratification_labels=stratification_labels,
            sample_ids=sample_ids,
            CLASSIFIERS=CLASSIFIERS,
            CLF_PARAM_GRIDS=CLF_PARAM_GRIDS,
            DATA_PARAM_GRIDS=DATA_PARAM_GRIDS,
            args=args,
            preprocessing_metadata=preprocessing_metadata
        )
        print("--- BEST MODEL PER-FEATURE SEARCH COMPLETE ---")

    # --- 1b. Label Shuffle Experiment (if requested) ---
    if args.run_label_shuffle_experiment:
        print("\n--- STARTING LABEL SHUFFLE EXPERIMENT ---")

        # Resolve the best-model directory automatically based on the selected cancer
        # types (an explicit --label_shuffle_model_dir always wins). When restricted to
        # breast+colorectal we want the brca_crc best-model search; otherwise PANCAN.
        label_shuffle_model_dir = args.label_shuffle_model_dir
        if label_shuffle_model_dir is None:
            output_dir = Path(args.output_dir)
            selected = [c.lower() for c in args.cancer_types]
            is_brca_crc = ('pancan' not in selected) and set(selected) == {'breast', 'colorectal'}

            # Default (PANCAN) directory uses the current experiment name.
            default_dir = output_dir / f"best_model_{args.experiment_name}"

            if is_brca_crc:
                # Prefer the brca_crc best-model search dir (e.g. best_model_best_model_search_brca_crc).
                brca_candidates = sorted(
                    d for d in output_dir.glob("best_model_*brca_crc*")
                    if d.is_dir() and any(d.glob("best_model_GridSearchCV_*.pkl"))
                )
                if len(brca_candidates) == 1:
                    label_shuffle_model_dir = brca_candidates[0]
                elif len(brca_candidates) > 1:
                    label_shuffle_model_dir = brca_candidates[0]
                    print(f"Found {len(brca_candidates)} brca_crc best-model dirs, using first: {label_shuffle_model_dir.name}")
                    print(f"   Candidates: {[d.name for d in brca_candidates]}")
                else:
                    label_shuffle_model_dir = default_dir
                    print(f"Cancer types restricted to breast+colorectal but no "
                          f"'best_model_*brca_crc*' dir with GridSearchCV files found in {output_dir}.")
                    print(f"   Falling back to default (likely PANCAN): {default_dir.name}")
            else:
                label_shuffle_model_dir = default_dir

            print(f"Auto-resolved label-shuffle best-model dir: {label_shuffle_model_dir}")

        run_label_shuffle_experiment(
            best_model_results_dir=str(label_shuffle_model_dir),
            all_feature_arrays=all_feature_arrays,
            y_labels=y_labels,
            stratification_labels=stratification_labels,
            sample_ids=np.array(sample_ids),
            CLASSIFIERS=CLASSIFIERS,
            args=args,
            filter_feature=args.label_shuffle_filter_feature,
            filter_n_genes=args.label_shuffle_filter_n_genes
        )
        print("--- LABEL SHUFFLE EXPERIMENT COMPLETE ---")

    # --- 1c. Post-hoc nested classifier selection on an existing results dir ---
    if args.recompute_inner_scores:
        print("\n--- RECOMPUTING INNER-CV SCORES + NESTED CLASSIFIER SELECTION ---")
        run_nested_classifier_selection(
            results_dir=args.recompute_inner_scores,
            all_feature_arrays=all_feature_arrays,
            y_labels=y_labels,
            stratification_labels=stratification_labels,
            sample_ids=list(sample_ids),
            CLASSIFIERS=CLASSIFIERS,
            n_jobs=args.cores,
        )
        print("--- NESTED CLASSIFIER SELECTION COMPLETE ---")

    # --- 2. Run Nested CV (if requested) ---
    if args.run_nested_cv:
        print("\n--- STARTING NESTED CROSS-VALIDATION ---")
        
        # Create dedicated output dir for nested CV results
        nested_cv_output_dir = Path(args.output_dir) / args.experiment_name
        os.makedirs(nested_cv_output_dir, exist_ok=True)
        print(f"Saving nested CV results to: {nested_cv_output_dir}")

        # --- Create All Experiment Combinations ---
        print("--- Generating nested CV experiment configurations ---")
        classifier_names = list(CLASSIFIERS.keys())
        
        # Check if we should skip gene selection (predefined matrix or explicit no_gene_selection flag)
        skip_gene_selection = args.no_gene_selection or args.predefined_feature_matrix is not None
        
        if skip_gene_selection:
            # No gene selection: just iterate over classifiers and features
            feature_names = list(all_feature_arrays.keys())
            
            predefined_feature_names = None
            if args.predefined_feature_matrix:
                print(f"--- Using PREDEFINED MATRIX: No gene selection will be performed ---")
                _pm = Path(args.predefined_feature_matrix)
                for _cand in (_pm.with_suffix('.tf_names.csv'), _pm.parent / (_pm.stem + '.tf_names.csv')):
                    if _cand.exists():
                        predefined_feature_names = pd.read_csv(_cand).iloc[:, 0].tolist()
                        break
            else:
                print(f"--- Using NO GENE SELECTION mode ---")
            
            experiment_configs = [
                {
                    'classifier': clf,
                    'feature': feat,
                    'feature_data': all_feature_arrays,
                    'labels': y_labels,
                    'stratification_labels': stratification_labels,
                    'sample_ids': sample_ids,
                    'output_dir': nested_cv_output_dir,
                    'classifiers': CLASSIFIERS,
                    'clf_param_grids': CLF_PARAM_GRIDS,
                    # extended reporting: column names of a predefined matrix (if known)
                    'feature_names': predefined_feature_names,
                    'store_fitted_models': args.store_fitted_models,
                }
                for clf, feat in itertools.product(classifier_names, feature_names)
            ]
            
            print(f"--- Starting {len(experiment_configs)} nested CV experiments (NO gene selection) on {args.cores} cores ---")
            Parallel(n_jobs=args.cores, verbose=10)(
                delayed(run_single_experiment_no_selection)(config) for config in experiment_configs
            )
        else:
            # Standard mode with gene selection
            feature_names = DATA_PARAM_GRIDS['feature']
            top_genes_list = DATA_PARAM_GRIDS['top_genes']

            # Create the Cartesian product of all parameters
            param_combinations = list(itertools.product(classifier_names, feature_names, top_genes_list))

            experiment_configs = [
                {
                    'classifier': clf,
                    'feature': feat,
                    'top_genes': n_genes,
                    'feature_data': all_feature_arrays,
                    'labels': y_labels,
                    'stratification_labels': stratification_labels,
                    'sample_ids': sample_ids,
                    'output_dir': nested_cv_output_dir, # Use dedicated subdir
                    'selection_type': args.selection_type,
                    'classifiers': CLASSIFIERS,
                    'clf_param_grids': CLF_PARAM_GRIDS
                }
                for clf, feat, n_genes in param_combinations
            ]
            
            print(f"--- Starting {len(experiment_configs)} nested CV experiments on {args.cores} cores ---")
            Parallel(n_jobs=args.cores, verbose=10)(
                delayed(run_single_experiment)(config) for config in experiment_configs
            )
        
        print("\n--- All nested CV experiments completed successfully ---")

    # --- 2b. Run Fully Nested CV (if requested) ---
    if args.run_nested_cv_full:
        print("\n--- STARTING FULLY NESTED CROSS-VALIDATION (joint selection) ---")
        nested_y, nested_strat = np.asarray(y_labels), np.asarray(stratification_labels)
        if args.nested_full_permute_labels is not None:
            # Label-permutation null of the COMPLETE procedure (reviewer 4, additional
            # comment 6): diagnosis labels are shuffled over the samples once, before the
            # outer folds are drawn, and everything (feature, ranking, k, classifier,
            # hyperparameters) is selected again on the shuffled labels. y and the
            # stratification labels move together, so the folds stay stratified.
            order = np.random.RandomState(args.nested_full_permute_labels).permutation(len(nested_y))
            nested_y, nested_strat = nested_y[order], nested_strat[order]
            print(f"LABEL PERMUTATION seed {args.nested_full_permute_labels}: "
                  f"{int((nested_y != np.asarray(y_labels)).sum())} of {len(nested_y)} binary labels changed")
        run_nested_experiment_full(
            all_feature_arrays=all_feature_arrays,
            y_labels=nested_y,
            stratification_labels=nested_strat,
            sample_ids=sample_ids,
            CLASSIFIERS=CLASSIFIERS,
            CLF_PARAM_GRIDS=CLF_PARAM_GRIDS,
            DATA_PARAM_GRIDS=DATA_PARAM_GRIDS,
            args=args,
            feature_names=args.nested_full_features,
            top_genes_list=args.nested_full_top_genes,
            selection_types=args.nested_full_selection_types,
            classifier_names=args.nested_full_classifiers,
            cache_dir=args.nested_full_cache_dir,
            estimate_only=args.nested_full_estimate_only,
            only_fold=args.nested_full_only_fold,
            aggregate=args.nested_full_aggregate,
            ksweep=args.nested_full_ksweep,
            sizesweep=args.nested_full_sizesweep,
            sizesweep_fractions=args.nested_full_sizesweep_fractions,
            subsample_seed=args.nested_full_subsample_seed,
            refresh_thresholds=args.nested_full_refresh_thresholds,
        )
        print("--- FULLY NESTED CROSS-VALIDATION COMPLETE ---")

    if args.run_cross_depth_eval:
        print("\n--- STARTING CROSS-DEPTH EVALUATION ---")
        run_cross_depth_eval(
            all_feature_arrays=all_feature_arrays,
            kept_columns_per_feature=kept_columns_per_feature,
            original_n_features=original_n_features,
            blacklist_file_used=blacklist_file_used,
            y_labels=y_labels,
            stratification_labels=stratification_labels,
            sample_ids=sample_ids,
            CLASSIFIERS=CLASSIFIERS,
            args=args,
        )
        print("--- CROSS-DEPTH EVALUATION COMPLETE ---")

    if args.run_baseline_aggregate:
        print("\n--- STARTING BASELINE AGGREGATE STUDY ---")
        run_baseline_aggregate_study(
            all_feature_arrays=all_feature_arrays,
            y_labels=y_labels,
            stratification_labels=stratification_labels,
            sample_ids=sample_ids,
            DATA_PARAM_GRIDS=DATA_PARAM_GRIDS,
            args=args
        )
        print("--- BASELINE AGGREGATE STUDY COMPLETE ---")
    
    # --- 5. Within-Dataset Generalization Training ---
    if args.run_within_dataset_generalization_train:
        print("\n" + "="*90)
        print("WITHIN-DATASET GENERALIZATION TRAINING")
        print("="*90)
        
        # Parse training cancer types from args
        training_cancer_types = args.generalization_training_types
        if not training_cancer_types:
            print("Error: --generalization_training_types is required")
            print("   Example: --generalization_training_types \"Breast cancer\" \"Colorectal cancer\"")
            return
        
        # Get feature type (should be single type now)
        feature_types = args.within_gen_feature_types
        if len(feature_types) != 1:
            print(f"Warning: Multiple feature types specified. Using first: {feature_types[0]}")
        feature_type = feature_types[0]
        
        print(f"Training types: {training_cancer_types}")
        print(f"Feature type: {feature_type}")
        print(f"Withhold fraction: {args.generalization_withhold_fraction}")
        print()
        
        # Update experiment name to include feature type
        original_exp_name = args.experiment_name
        base_exp_name = f"{original_exp_name}_{feature_type}"
        args.experiment_name = base_exp_name

        # Multi-split training (default 1)
        n_splits = int(getattr(args, 'within_gen_n_splits', 1) or 1)
        split_seed = int(getattr(args, 'within_gen_split_seed', 42) or 42)

        if n_splits <= 1:
            run_within_dataset_generalization_training(
                all_feature_arrays=all_feature_arrays,
                y_labels=y_labels,
                stratification_labels=stratification_labels,
                sample_ids=sample_ids,
                training_cancer_types=training_cancer_types,
                withhold_fraction=args.generalization_withhold_fraction,
                CLASSIFIERS=CLASSIFIERS,
                CLF_PARAM_GRIDS=CLF_PARAM_GRIDS,
                DATA_PARAM_GRIDS=DATA_PARAM_GRIDS,
                args=args,
                split_seed=split_seed,
                split_index=1
            )
        else:
            base_output_dir = Path(args.output_dir) / base_exp_name
            os.makedirs(base_output_dir, exist_ok=True)
            print(f"\nRunning {n_splits} stratified splits (holdout={args.generalization_withhold_fraction:.2f})")
            print(f"   Base output dir: {base_output_dir}")
            for split_idx in range(n_splits):
                split_args = copy.copy(args)
                split_args.output_dir = str(base_output_dir)
                split_args.experiment_name = f"split_{split_idx + 1:02d}"
                print(f"\n--- Split {split_idx + 1}/{n_splits} ---")
                run_within_dataset_generalization_training(
                    all_feature_arrays=all_feature_arrays,
                    y_labels=y_labels,
                    stratification_labels=stratification_labels,
                    sample_ids=sample_ids,
                    training_cancer_types=training_cancer_types,
                    withhold_fraction=args.generalization_withhold_fraction,
                    CLASSIFIERS=CLASSIFIERS,
                    CLF_PARAM_GRIDS=CLF_PARAM_GRIDS,
                    DATA_PARAM_GRIDS=DATA_PARAM_GRIDS,
                    args=split_args,
                    split_seed=split_seed + split_idx,
                    split_index=split_idx + 1
                )
        
        print(f"\n{feature_type.upper()} training complete")
        print("="*90)
    
    # --- 6. Within-Dataset Generalization Evaluation ---
    if args.run_within_dataset_generalization_eval:
        print("\n" + "="*90)
        print("WITHIN-DATASET GENERALIZATION EVALUATION")
        print("="*90)
        
        if args.generalization_training_dir is None:
            print("Error: --generalization_training_dir is required")
            print("   Example: --generalization_training_dir results/within_gen_Breast_cancer_Colorectal_cancer_dd")
            return
        
        # Get feature type (should be single type now)
        feature_types = args.within_gen_feature_types
        if len(feature_types) != 1:
            print(f"Warning: Multiple feature types specified. Using first: {feature_types[0]}")
        feature_type = feature_types[0]
        
        print(f"Feature type: {feature_type}")
        print(f"Training directory: {args.generalization_training_dir}")
        print(f"Comprehensive analysis: {args.comprehensive_within_dataset_generalization}")
        print()
        
        # The training directory already has the feature type suffix
        training_dir = args.generalization_training_dir
        
        if not Path(training_dir).exists():
            print(f"Error: {training_dir} not found.")
            return
        
        # Always use multi-split-aware evaluation function.
        # run_within_dataset_generalization_evaluation detects split_* subdirs,
        # calls run_comprehensive_within_dataset_evaluation per split, and
        # aggregates results (mean +/- std) across splits. If no splits are
        # found it falls back to single-dir evaluation.
        print("Running within-dataset generalization evaluation...")
        run_within_dataset_generalization_evaluation(
            training_results_dir=training_dir,
            all_feature_arrays=all_feature_arrays,
            y_labels=y_labels,
            stratification_labels=stratification_labels,
            sample_ids=sample_ids,
            args=args
        )
        
        print(f"\n{feature_type.upper()} evaluation complete")
        print("="*90)

    # --- 6b. Within-Dataset Generalization Comparison (DD/RNA/ATAC) ---
    if args.run_within_dataset_generalization_compare:
        if not args.generalization_compare_dirs:
            print("Error: --generalization_compare_dirs is required")
            print("   Example: --generalization_compare_dirs results/within_gen_*_dd results/within_gen_*_rna results/within_gen_*_atac")
            return
        run_within_dataset_generalization_compare(
            compare_dirs=args.generalization_compare_dirs,
            args=args
        )
    
    # --- 7. External Validation ---
    if args.validate_external:
        print("\n--- STARTING EXTERNAL DATASET VALIDATION ---")
        
        # Check if comprehensive analysis is requested
        filter_feat = getattr(args, 'filter_feature', None)
        if filter_feat:
            print(f"\nFeature filter active: only using '{filter_feat}' models")
        
        if args.comprehensive_external_validation:
            print("\nRunning comprehensive external validation (all features & gene counts)...")
            run_comprehensive_external_validation(
                best_model_results_dir=args.external_model_dir,
                external_feature_arrays=all_feature_arrays,
                external_y_labels=y_labels,
                external_stratification_labels=stratification_labels,
                external_sample_ids=sample_ids,
                args=args,
                training_preprocessing_metadata=training_preprocessing_metadata,
                filter_feature=filter_feat
            )
        else:
            # Standard validation (single best model only)
            validate_best_model_on_external_dataset(
                best_model_results_dir=args.external_model_dir,
                external_feature_arrays=all_feature_arrays,
                external_y_labels=y_labels,
                external_stratification_labels=stratification_labels,
                external_sample_ids=sample_ids,
                args=args,
                training_preprocessing_metadata=training_preprocessing_metadata,
                filter_feature=filter_feat,
                validation_mode=getattr(args, 'external_validation_mode', 'pipeline')
            )
        print("--- EXTERNAL VALIDATION COMPLETE ---")

    # Check if at least one mode was specified
    if not any([
        args.run_best_model_search,
        args.run_label_shuffle_experiment,
        args.run_nested_cv,
        args.run_nested_cv_full,
        args.run_cross_depth_eval,
        args.run_baseline_aggregate,
        args.validate_external,
        args.run_within_dataset_generalization_train,
        args.run_within_dataset_generalization_eval
    ]):
        print("\nWarning: No analysis mode specified.")
        print("Please specify at least one mode:")
        print("  --run_best_model_search")
        print("  --run_label_shuffle_experiment")
        print("  --run_nested_cv")
        print("  --run_nested_cv_full")
        print("  --run_baseline_aggregate")
        print("  --validate_external")
        print("  --run_within_dataset_generalization_train")
        print("  --run_within_dataset_generalization_eval")
        print("Exiting.")



# =================================================================================
# --- Command-Line Interface & Main Entry Point ---
# =================================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run nested cross-validation for genomic classifiers on an HPC.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--run_best_model_search",
        action='store_true',
        default=False,
        help="Run a single comprehensive GridSearchCV to find the best model (clf+genes) per feature."
    )
    parser.add_argument(
        "--best_model_top_genes",
        nargs='+',
        type=int,
        default=None,
        help=("Restrict --run_best_model_search to these k values (default: the full 12-value grid). "
              "Configurations whose summary_* already exists in the output directory are skipped, "
              "so a subset can be resubmitted into the same directory (e.g. only k=15000 after an OOM).")
    )

    parser.add_argument(
        "--refresh_deployment_thresholds",
        action='store_true',
        default=False,
        help=("With --run_best_model_search: for configurations already complete in the "
              "output directory, re-derive the frozen deployment thresholds from the "
              "selected configuration's held-out scores on the search's own ten folds "
              "(10 fits, no new grid search) and rewrite the summary. Rerun the external "
              "validation afterwards so its fixed-threshold evaluation uses them.")
    )

    parser.add_argument(
        "--run_label_shuffle_experiment",
        action='store_true',
        default=False,
        help="Run label-shuffle (permutation) experiment using the best model configuration."
    )

    parser.add_argument(
        "--label_shuffle_model_dir",
        type=str,
        default=None,
        help="Directory containing best_model_GridSearchCV_*.pkl files. Defaults to output_dir/best_model_<experiment_name>."
    )

    parser.add_argument(
        "--label_shuffle_output_dir",
        type=str,
        default=None,
        help="Output directory for label shuffle results. Defaults to output_dir/label_shuffle_<experiment_name>."
    )

    parser.add_argument(
        "--label_shuffle_n_shuffles",
        type=int,
        default=50,
        help="Number of label shuffles to run (default: 50)."
    )

    parser.add_argument(
        "--label_shuffle_seed",
        type=int,
        default=42,
        help="Random seed for label shuffles. Each shuffle uses seed+index."
    )

    parser.add_argument(
        "--label_shuffle_cv_folds",
        type=int,
        default=5,
        help="Number of CV folds for shuffled-label AUC (default: 5)."
    )

    parser.add_argument(
        "--label_shuffle_top_k",
        type=int,
        default=50,
        help="Top-K features for stability comparisons (default: 50)."
    )

    parser.add_argument(
        "--label_shuffle_filter_feature",
        type=str,
        default=None,
        help="Optional feature filter for best-model selection (e.g., 'griffin_diff')."
    )

    parser.add_argument(
        "--label_shuffle_filter_n_genes",
        type=int,
        default=None,
        help="Optional n_genes filter for best-model selection (e.g., 15000)."
    )

    parser.add_argument(
        "--run_nested_cv",
        action='store_true',
        default=False,
        help="Run the full nested cross-validation experiments."
    )

    parser.add_argument(
        "--recompute_inner_scores",
        type=str,
        default=None,
        metavar="RESULTS_DIR",
        help="Path to an existing nested-CV results directory (results_*.pkl files). "
             "Reconstructs each fold's mean inner-CV AUC for the stored winning "
             "hyperparameters (exact reproduction of GridSearchCV.best_score_ via the "
             "seeded splits; ~3 fits per fold per pickle, no grid rerun), then performs "
             "per-outer-fold classifier selection on inner scores ONLY and writes "
             "stitched out-of-fold predictions. Must be invoked with the same data "
             "arguments (manifest, cancer types, feature dirs) as the original run; "
             "mismatches are detected and abort."
    )

    parser.add_argument(
        "--run_nested_cv_full",
        action='store_true',
        default=False,
        help=("Run FULLY nested CV: feature, ranking function, top_genes and classifier "
              "are all selected inside each outer fold (run_nested_experiment_full).")
    )

    parser.add_argument(
        "--nested_full_features",
        type=str,
        nargs='+',
        default=None,
        help="Restrict --run_nested_cv_full to these feature types (default: all five)."
    )

    parser.add_argument(
        "--nested_full_top_genes",
        type=int,
        nargs='+',
        default=None,
        help="Restrict --run_nested_cv_full to these k values (default: the full 12-value grid)."
    )

    parser.add_argument(
        "--nested_full_selection_types",
        type=str,
        nargs='+',
        choices=['p_value', 'fold_change', 'weighted', 'mean_diff', 'random'],
        default=None,
        help=("Ranking functions searched by --run_nested_cv_full "
              "(default: mean_diff p_value fold_change weighted; 'random' is a control).")
    )

    parser.add_argument(
        "--nested_full_classifiers",
        type=str,
        nargs='+',
        default=None,
        help="Restrict --run_nested_cv_full to these classifiers (default: all five)."
    )

    parser.add_argument(
        "--augment_covariates",
        default=None,
        metavar="STEM",
        help=("Append the covariate matrix <STEM>.npy (+ .metadata.csv, .tf_names.csv; a panel from "
              "src/comp/build_scalar_panel_matrix.py) to every TSS feature matrix. The covariates bypass "
              "the TSS ranking (always in the model) and enter the classifier next to the selected TSSs. "
              "Fully nested runs (fold records gain 'covariate_block'), the best-model search (the block is "
              "column-standardized with the TSSs and recorded in preprocessing_metadata.pkl) and external "
              "validation of such a model (give the external cohort's stem of the same panel, e.g. "
              "extracted_features/scalar_panel_pipeline4_jiang).")
    )
    parser.add_argument(
        "--residualize_covariates",
        default=None,
        metavar="STEM",
        help=("Regress the covariates of <STEM> out of every TSS column inside each training partition "
              "(OLS fitted on the training samples, applied to train and test) before ranking and "
              "classification; the classifier sees residual TSS features only. Mutually exclusive with "
              "--augment_covariates.")
    )
    parser.add_argument(
        "--nested_full_cache_dir",
        type=str,
        default=None,
        help=("Optional joblib cache dir for the Pipeline. Caches the TSS-ranking fit so it "
              "is not recomputed for every classifier hyperparameter sharing it. Use fast "
              "local scratch, not a network share.")
    )

    parser.add_argument(
        "--nested_full_estimate_only",
        action='store_true',
        default=False,
        help="Print the --run_nested_cv_full compute-cost estimate and exit without fitting."
    )

    parser.add_argument(
        "--nested_full_only_fold",
        type=int,
        default=None,
        help=("Run ONLY this outer fold (1-10) of --run_nested_cv_full and write a partial "
              "result. Used by the SLURM array job so the 10 folds run in parallel.")
    )

    parser.add_argument(
        "--nested_full_aggregate",
        action='store_true',
        default=False,
        help=("Combine the per-fold partials written by --nested_full_only_fold into the "
              "final per-fold table, out-of-fold predictions and summary.")
    )

    parser.add_argument(
        "--nested_full_refresh_thresholds",
        action='store_true',
        default=False,
        help=("Re-derive the operating-point thresholds of the stored fold partials of "
              "this experiment (plain run, ksweep_k* and sizesweep_f* subdirectories) "
              "from the winning configuration's scores on the inner validation folds, "
              "without repeating the inner search: each stored winner is refitted, must "
              "reproduce its stored test predictions, and gets 1 + n_inner fits. Use the "
              "data arguments of the original run. Combine with --nested_full_aggregate "
              "(and --nested_full_ksweep / --nested_full_sizesweep where applicable) to "
              "rebuild the tables in the same call.")
    )

    parser.add_argument(
        "--nested_full_ksweep",
        action='store_true',
        default=False,
        help=("Run one complete nested selection per k instead of one selection over the "
              "whole k grid. Feature, ranking function and hyperparameters are still "
              "chosen inside each fold; only k is fixed. Gives a k trend whose every "
              "point is leakage-free. Combine with --nested_full_only_fold, then with "
              "--nested_full_aggregate to write nested_full_ksweep.csv plus ksweep_tuned/ "
              "(per fold the k with the best inner score = the fully nested k-as-"
              "hyperparameter arm, derived without a second run). Works on any region "
              "set: the comparator sweeps use --no_gene_selection --features_dir (RNA) "
              "or --predefined_feature_matrix (ATAC) with a k grid up to that set's width.")
    )
    parser.add_argument(
        "--nested_full_sizesweep",
        action='store_true',
        help=("Sample-size robustness sweep: one complete fully nested selection per "
              "sample fraction, each on a deterministic nested stratified subset of the "
              "cohort (per-stratum seeded shuffle, prefix per fraction). The outer fold "
              "count per fraction is min(10, smaller binary class count). Combine with "
              "--nested_full_only_fold (the fraction loop runs inside each task), then "
              "with --nested_full_aggregate to write nested_full_sizesweep.csv.")
    )
    parser.add_argument(
        "--nested_full_sizesweep_fractions",
        nargs='+',
        type=float,
        default=None,
        help=("Sample fractions for --nested_full_sizesweep "
              "(default: 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1). Subsets depend only on "
              "the seed and cohort, so fractions may be split across submissions.")
    )
    parser.add_argument(
        "--nested_full_subsample_seed",
        type=int,
        default=42,
        help="Seed for the per-stratum shuffle defining the sizesweep subsets."
    )
    parser.add_argument(
        "--nested_full_permute_labels",
        type=int,
        default=None,
        help="Shuffle the diagnosis labels with this seed before the fully nested procedure "
             "(label-permutation null with the complete model selection repeated)."
    )

    parser.add_argument(
        "--run_cross_depth_eval",
        action='store_true',
        help=("Cross-depth evaluation of a completed fully nested run: refit each "
              "fold's stored winning configuration on its full-depth training "
              "partition (deterministic; verified against the stored predictions), "
              "then score the SAME outer test folds on downsampled feature matrices. "
              "Assesses whether FUTURE samples must be sequenced deep. Requires "
              "--cross_depth_source_dir and --cross_depth_eval_dirs; --features_dir "
              "must be the source run's (full-depth) feature directory.")
    )
    parser.add_argument(
        "--cross_depth_source_dir",
        type=str,
        default=None,
        help="Completed nested_full result directory (holds nested_full_per_fold.pkl)."
    )
    parser.add_argument(
        "--cross_depth_eval_dirs",
        nargs='+',
        default=[],
        metavar="NAME=DIR",
        help=("Downsampled feature directories to evaluate the test folds on, e.g. "
              "cov1x=extracted_features/downsampled/cov1x_seed42/biomart_10kb")
    )

    parser.add_argument(
        "--no_gene_selection",
        action='store_true',
        default=False,
        help="Run nested CV without gene selection - uses all features from features_dir. "
             "Skips blacklist removal and uses custom feature directory."
    )
    
    parser.add_argument(
        "--run_baseline_aggregate",
        action='store_true',
        default=False,
        help="Run baseline aggregate study: select top genes, aggregate from raw files, "
             "and evaluate with ROC directly (no ML classifier)."
    )
    
    parser.add_argument(
        "--validate_external",
        action='store_true',
        default=False,
        help="Validate best model from one dataset on a different external dataset. "
             "Use --external_model_dir to specify where the trained models are."
    )
    
    # External validation specific arguments
    parser.add_argument(
        "--external_model_dir",
        type=str,
        default=None,
        help="Directory containing best_model_GridSearchCV_*.pkl files from training dataset "
             "(for --validate_external mode)."
    )
    
    parser.add_argument(
        "--external_metadata_file",
        type=str,
        default=None,
        help="Metadata CSV file for the external validation dataset. "
             "Required when using --validate_external. This determines which samples "
             "are healthy vs. cancer in the external dataset."
    )
    
    parser.add_argument(
        "--comprehensive_external_validation",
        action="store_true",
        help="Run comprehensive external validation analyzing generalization across "
             "all feature types and gene counts (requires --validate_external)."
    )
    
    parser.add_argument(
        "--filter_feature",
        type=str,
        default=None,
        help="Filter validation to only use models trained on this feature "
             "(e.g., --filter_feature rcov). Works with both --validate_external "
             "and --comprehensive_external_validation."
    )
    
    parser.add_argument(
        "--external_validation_mode",
        type=str,
        choices=['pipeline', 'clinical'],
        default='pipeline',
        help="Validation mode: 'pipeline' (default) uses training StandardScaler statistics; "
             "'clinical' uses only gene selection + fresh StandardScaler, simulating deployment "
             "where training statistics are unavailable. Helps assess model robustness."
    )
    
    # Baseline aggregate study specific arguments
    parser.add_argument(
        "--baseline_feature",
        type=str,
        default="fslr",
        help="Feature type for baseline aggregate study (e.g., 'fslr', 'fslr_coverage', 'griffin_diff')."
    )
    
    parser.add_argument(
        "--baseline_selection",
        type=str,
        choices=['p_value', 'fold_change', 'weighted', 'mean_diff', 'random'],
        default="fold_change",
        help="Feature selection method for baseline aggregate study."
    )
    
    # Within-dataset generalization arguments
    parser.add_argument(
        "--run_within_dataset_generalization_train",
        action='store_true',
        default=False,
        help="Train models on subset of cancer types with 20%% holdout for generalization testing. "
             "By default, runs all three feature types (DD, RNA, ATAC) automatically."
    )
    
    parser.add_argument(
        "--run_within_dataset_generalization_eval",
        action='store_true',
        default=False,
        help="Evaluate generalization on withheld samples + novel cancer types. "
             "By default, evaluates all three feature types (DD, RNA, ATAC) automatically."
    )
    
    parser.add_argument(
        "--within_gen_feature_types",
        nargs='+',
        type=str,
        default=['dd', 'rna', 'atac'],
        choices=['dd', 'rna', 'atac'],
        help="Feature types to use for within-dataset generalization (default: all three)."
    )
    
    parser.add_argument(
        "--skip_training",
        action='store_true',
        default=False,
        help="Skip training step and only run evaluation (requires existing trained models)."
    )
    
    parser.add_argument(
        "--within_gen_dd_features_dir",
        type=str,
        default="extracted_features/biomart_10kb",
        help="Features directory for DD (biomarker) type."
    )
    
    parser.add_argument(
        "--within_gen_rna_features_dir",
        type=str,
        default="extracted_features/zhu_all_after_lift",
        help="Features directory for RNA type."
    )
    
    parser.add_argument(
        "--within_gen_atac_matrix",
        type=str,
        default="extracted_features/ulz_atac_tfbs_features.npy",
        help="Predefined matrix path for ATAC type."
    )
    
    parser.add_argument(
        "--within_gen_atac_metadata",
        type=str,
        default="extracted_features/ulz_atac_tfbs_features.metadata.csv",
        help="Metadata CSV for ATAC predefined matrix."
    )
    
    parser.add_argument(
        "--generalization_training_types",
        nargs='+',
        type=str,
        default=None,
        help="Cancer types to use for training (e.g., 'Breast cancer' 'Colorectal cancer'). "
             "Other cancer types in PANCAN will be used for generalization testing."
    )
    
    parser.add_argument(
        "--generalization_withhold_fraction",
        type=float,
        default=0.2,
        help="Fraction of training data to withhold for evaluation (default: 0.2 for 20%%)."
    )

    parser.add_argument(
        "--within_gen_n_splits",
        type=int,
        default=5,
        help="Number of repeated stratified holdout splits for within-dataset generalization (default: 5)."
    )

    parser.add_argument(
        "--within_gen_split_seed",
        type=int,
        default=42,
        help="Base random seed for repeated holdout splits (default: 42)."
    )
    
    parser.add_argument(
        "--generalization_training_dir",
        type=str,
        default=None,
        help="Directory containing training results from --run_within_dataset_generalization_train "
             "(for --run_within_dataset_generalization_eval mode)."
    )

    parser.add_argument(
        "--feature_subset",
        nargs='+',
        type=str,
        default=None,
        help="Limit feature loading to specific feature names (e.g., --feature_subset rcov)."
    )
    
    parser.add_argument(
        "--comprehensive_within_dataset_generalization",
        action='store_true',
        default=False,
        help="Run comprehensive within-dataset generalization analysis (all features & gene counts). "
             "Use with --run_within_dataset_generalization_eval."
    )

    parser.add_argument(
        "--run_within_dataset_generalization_compare",
        action='store_true',
        default=False,
        help="Compare DD/RNA/ATAC within-dataset generalization results in a single plot."
    )

    parser.add_argument(
        "--generalization_compare_dirs",
        nargs='+',
        type=str,
        default=None,
        help="Directories containing comprehensive_generalization_summary.pkl for DD/RNA/ATAC."
    )
    
    parser.add_argument(
        "-c", "--cores", 
        type=int, 
        default=64, 
        help="Number of CPU cores to use for parallel execution."
    )
    parser.add_argument(
        "--features_dir", 
        type=str, 
        default="./extracted_features/biomart_10kb", 
        help="Path to the directory containing the extracted feature CSV/TSV files."
    )
    parser.add_argument(
        "--metadata_file", 
        type=str, 
        default="accessory_files/cris_metadata.csv", 
        help="Path to the sample metadata CSV file."
    )
    parser.add_argument(
        "--blacklist_indices", 
        type=str, 
        default="accessory_files/final_tss_indices_to_remove_hg38.npy", 
        help="Path to the .npy file with indices of regions to be removed."
    )
    parser.add_argument(
        "-o", "--output_dir", 
        type=str,
        default="nested_cv_results", 
        help="Directory to save the output .pkl files."
    )
    parser.add_argument(
        "-e", "--experiment_name", 
        type=str, 
        default="nested_cv_results", 
        help="Name of the experiment (used in output filenames)."
    )

    parser.add_argument(
        "-s", "--selection_type", 
        type=str, 
        choices=['p_value', 'fold_change', 'weighted', 'mean_diff', 'random'], 
        default='mean_diff',
        help="Feature selection method to use."
    )

    parser.add_argument(
        "--predefined_feature_matrix",
        type=str,
        default=None,
        help="Path to a predefined feature matrix (.npy) with shape (n_samples, n_features)."
    )

    parser.add_argument(
        "--predefined_metadata_csv",
        type=str,
        default=None,
        help="CSV file describing rows of the predefined matrix (must contain sample IDs in column 'sample_id' or 'ID' and optionally 'disease')."
    )

    parser.add_argument(
        "--predefined_no_row_standardize",
        action="store_true",
        default=False,
        help="Skip the row-wise z-score standardization of the predefined matrix. Required for "
             "matrices of heterogeneous scalars (e.g. the scalar-panel baseline), where "
             "standardizing across features within a sample mixes units and destroys the "
             "per-feature information; per-feature scaling still happens inside the CV pipeline's "
             "StandardScaler, fitted on training folds only. The matrix must be NaN-free."
    )

    parser.add_argument(
        "--store_fitted_models",
        action="store_true",
        default=False,
        help="Per-classifier nested CV without gene selection (predefined matrices, --no_gene_selection): "
             "also pickle the ten fitted fold pipelines to models_<classifier>_<feature>_all_features.pkl "
             "next to the results pickle (covariate-panel deep dive: partial dependence, decision maps). "
             "Small matrices only - a 500-tree forest on 10 covariates is a few MB per fold."
    )

    parser.add_argument(
        "--pon_n",
        type=int,
        default=52,
        help="Number of initial healthy samples to remove as Panel-of-Normals (when using CRIS predefined matrices)."
    )

    parser.add_argument(
        "-t", "--cancer_types",
        nargs='+',
        #default=['breast', 'colorectal'],
        default=['PANCAN'],
        help=f"List of cancer types. Use short names (e.g., breast, colorectal). "
             f"Provide 'PANCAN' for all types. Available: {', '.join(CANCER_MAP.keys())}"
    )
    parser.add_argument(
        "--control_groups",
        nargs='+',
        default=None,
        help=("Restrict class 0 to these `disease` labels of the metadata file (must be a "
              "subset of the dataset's default control labels). Jiang default: "
              "'Healthy' 'Cirrhosis' 'Hepatitis B' (the 135 non-HCC comparators); pass "
              "--control_groups Healthy for the HCC-vs-healthy-only task.")
    )
    parser.add_argument(
        "--exclude_samples",
        default=None,
        metavar="FILE",
        help=("Text file with sample IDs (one per line, '#' comments) to drop from BOTH "
              "classes before anything else happens - fold generation, feature loading, "
              "predefined-matrix rows alike. The run is otherwise identical to one without "
              "the flag, so pass a distinct --experiment_name. Used for the sensitivity "
              "runs without the 30 deep-sequenced healthy samples "
              "(accessory_files/cris_exclude_deep_healthy.txt; see rebuttal Fig S10).")
    )

    args = parser.parse_args()
    
    # Use the number of cores specified, but not more than available
    args.cores = min(args.cores, multiprocessing.cpu_count())

    main(args)
