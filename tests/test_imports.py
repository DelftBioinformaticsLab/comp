"""Every pipeline script imports, the way it is run.

The analysis is a set of scripts, not an installed package (README: step-by-step module calls):
- features.py and the feature modules it uses are run from the repository root and import `src.comp.*`;
- the other scripts in src/comp are run as `python src/comp/<script>.py` and import each other by file name.
A failing import here is what reviewer 4 hit with the former `comp.main` entry point.
"""

import importlib
import sys
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
COMP = ROOT / "src" / "comp"

# run from the repository root (import src.comp.*)
FROM_ROOT = ["features", "src.comp.bin", "src.comp.em", "src.comp.wps", "src.comp.gc", "src.comp.util", "src.comp.compute_mean_coverage"]
# run as scripts inside src/comp (import each other by file name)
SCRIPTS = [
    "build_cris_manifest", "build_jiang_manifest", "build_deep_healthy_exclusion", "build_loco_splits", "build_scalar_panel_matrix",
    "build_tss_set_matrices", "derive_tss_filter", "baseline_aggregate", "extract_aggregate_features",
    "extract_ulz_atac_features", "util", "compute_mean_coverage", "compute_sex", "summarize_coverage", "collect_ichorcna_tf",
    "fill_manifest_oof", "model_hpc",
]


def _import(name: str, path: Path):
    sys.path.insert(0, str(path))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")      # third-party deprecation noise is not an import failure
            return importlib.import_module(name)
    finally:
        sys.path.remove(str(path))


@pytest.mark.parametrize("name", FROM_ROOT)
def test_import_from_root(name):
    assert _import(name, ROOT) is not None


@pytest.mark.parametrize("name", SCRIPTS)
def test_import_script(name):
    assert _import(name, COMP) is not None
