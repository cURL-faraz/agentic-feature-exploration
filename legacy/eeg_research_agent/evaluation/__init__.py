"""Predictive evaluation and leakage control."""

from .classification import (
    NON_CAUSAL_FEATURES,
    ClassifierSpec,
    EvaluationResult,
    ablation_feature_sets,
    available_classifiers,
    feature_families,
    run_predictive_evaluation,
)
from .cross_validation import CrossValidationResult, cross_validate, subject_grouped_folds
from .leakage_checks import run_leakage_audit
from .metrics import classification_report, cohens_kappa

__all__ = [
    "ClassifierSpec",
    "EvaluationResult",
    "NON_CAUSAL_FEATURES",
    "ablation_feature_sets",
    "available_classifiers",
    "feature_families",
    "run_predictive_evaluation",
    "cross_validate",
    "subject_grouped_folds",
    "CrossValidationResult",
    "classification_report",
    "cohens_kappa",
    "run_leakage_audit",
]
