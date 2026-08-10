"""Ranking-model validation package.

Fits and honestly validates a LightGBM ranking model for insider cluster
buys, replacing the hand-tuned flags in insider_cluster_buys.DEFAULT_WEIGHTS.
See research/model.py for the purged walk-forward CV harness, the model
fits, the leakage controls, and the ten-percent-owner interaction test.

Public entry points are re-exported here so callers can write
`import research` then `research.fit_and_validate(df)` without reaching
into the submodule.
"""

from __future__ import annotations

from .model import (
    Fold,
    ValidationResult,
    fit_and_validate,
    load_research_dataset,
    make_purged_expanding_folds,
    save_model_bundle,
    load_model_bundle,
    write_markdown_summary,
    find_leaky_features,
    assert_no_leaky_features,
)

__all__ = [
    "Fold",
    "ValidationResult",
    "fit_and_validate",
    "load_research_dataset",
    "make_purged_expanding_folds",
    "save_model_bundle",
    "load_model_bundle",
    "write_markdown_summary",
    "find_leaky_features",
    "assert_no_leaky_features",
]
