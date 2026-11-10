"""Subject-grouped cross-validation.

``KFold`` with ``shuffle=True`` on epoch rows is the single most common way to
inflate sleep-staging results, because epochs from one night are nearly
duplicates of each other.  Every split here groups by subject, and the fold
assignments are deterministic given the seed and are returned so they can be
audited and re-used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .classification import ClassifierSpec, build_pipeline
from .metrics import classification_report

__all__ = ["FoldResult", "CrossValidationResult", "subject_grouped_folds", "cross_validate"]


@dataclass
class FoldResult:
    """One cross-validation fold."""

    fold: int
    train_subjects: list[str]
    test_subjects: list[str]
    n_train: int
    n_test: int
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold": self.fold,
            "train_subjects": sorted(self.train_subjects),
            "test_subjects": sorted(self.test_subjects),
            "n_train": self.n_train,
            "n_test": self.n_test,
            "accuracy": self.metrics.get("accuracy"),
            "balanced_accuracy": self.metrics.get("balanced_accuracy"),
            "macro_f1": self.metrics.get("macro_f1"),
            "cohens_kappa": self.metrics.get("cohens_kappa"),
            "per_class_f1": {
                label: values.get("f1")
                for label, values in (self.metrics.get("per_class") or {}).items()
            },
        }


@dataclass
class CrossValidationResult:
    """Aggregated cross-validation output."""

    folds: list[FoldResult] = field(default_factory=list)
    classifier: str = ""
    n_features: int = 0
    seed: int = 0
    errors: list[str] = field(default_factory=list)

    def aggregate(self) -> dict[str, Any]:
        """Mean / SD / min / max of the primary metrics across folds."""
        keys = ("accuracy", "balanced_accuracy", "macro_f1", "cohens_kappa")
        summary: dict[str, Any] = {"n_folds": len(self.folds)}
        for key in keys:
            values = [
                fold.metrics.get(key)
                for fold in self.folds
                if fold.metrics.get(key) is not None and np.isfinite(fold.metrics.get(key, float("nan")))
            ]
            if not values:
                summary[key] = None
                continue
            summary[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        per_stage: dict[str, list[float]] = {}
        for fold in self.folds:
            for label, values in (fold.metrics.get("per_class") or {}).items():
                if values.get("support", 0) > 0 and np.isfinite(values.get("f1", float("nan"))):
                    per_stage.setdefault(label, []).append(values["f1"])
        summary["per_stage_f1_mean"] = {
            label: float(np.mean(values)) for label, values in sorted(per_stage.items())
        }
        summary["per_stage_f1_std"] = {
            label: (float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)
            for label, values in sorted(per_stage.items())
        }
        return summary

    def to_dict(self) -> dict[str, Any]:
        return {
            "classifier": self.classifier,
            "n_features": self.n_features,
            "seed": self.seed,
            "folds": [fold.to_dict() for fold in self.folds],
            "aggregate": self.aggregate(),
            "errors": self.errors,
        }


def subject_grouped_folds(
    subjects: Sequence[str], n_splits: int = 5, seed: int = 20240517
) -> list[tuple[list[str], list[str]]]:
    """Deterministic subject-level fold assignment.

    Subjects are shuffled once with ``seed`` and then dealt round-robin into
    ``n_splits`` folds, which keeps fold sizes within one subject of each other.
    No subject ever appears in both sides of a fold.
    """
    unique = sorted(set(map(str, subjects)))
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    if len(unique) < n_splits:
        raise ValueError(
            f"cannot build {n_splits} folds from {len(unique)} subjects "
            "(folds are made of whole subjects)"
        )
    rng = np.random.default_rng(seed)
    shuffled = list(unique)
    rng.shuffle(shuffled)
    buckets: list[list[str]] = [[] for _ in range(n_splits)]
    for index, subject in enumerate(shuffled):
        buckets[index % n_splits].append(subject)
    folds: list[tuple[list[str], list[str]]] = []
    for index, bucket in enumerate(buckets):
        test_subjects = sorted(bucket)
        train_subjects = sorted(
            subject for position, other in enumerate(buckets) if position != index for subject in other
        )
        folds.append((train_subjects, test_subjects))
    return folds


def cross_validate(
    table: pd.DataFrame,
    features: Sequence[str],
    spec: ClassifierSpec,
    config: Any,
    n_splits: int | None = None,
    seed: int | None = None,
    label_column: str = "stage",
    select_k: int | None = None,
) -> CrossValidationResult:
    """Run subject-grouped cross-validation for one classifier.

    Returns per-fold metrics plus the fold membership, so a reader can confirm
    that no subject is shared between the training and test part of a fold.
    """
    from ..statistics.feature_selection import mutual_information_ranking, select_top_k

    seed = int(config.seed if seed is None else seed)
    n_splits = int(config.analysis.predictive.cv_folds if n_splits is None else n_splits)
    columns = [c for c in features if c in table.columns]
    result = CrossValidationResult(classifier=spec.name, n_features=len(columns), seed=seed)
    if len(columns) == 0:
        result.errors.append("no usable feature columns")
        return result
    subjects = sorted(set(table["subject_id"].astype(str)))
    try:
        folds = subject_grouped_folds(subjects, n_splits=n_splits, seed=seed)
    except ValueError as exc:
        result.errors.append(str(exc))
        return result

    for index, (train_subjects, test_subjects) in enumerate(folds):
        train = table.loc[table["subject_id"].astype(str).isin(train_subjects)]
        test = table.loc[table["subject_id"].astype(str).isin(test_subjects)]
        if len(train) < 10 or train[label_column].nunique() < 2 or len(test) == 0:
            result.errors.append(f"fold {index} unusable (train={len(train)}, test={len(test)})")
            continue
        selected_columns = columns
        if select_k is not None and spec.supports_feature_selection and len(columns) > select_k:
            ranking = mutual_information_ranking(train.loc[:, columns], train[label_column], seed=seed)
            selected_columns = select_top_k(ranking, int(select_k))
        pipeline = build_pipeline(spec, seed=seed)
        pipeline.fit(train.loc[:, selected_columns], train[label_column].astype(str).to_numpy())
        predictions = np.asarray(pipeline.predict(test.loc[:, selected_columns]))
        y_true = test[label_column].astype(str).to_numpy()
        labels = sorted(set(train[label_column].astype(str)) | set(y_true))
        metrics = classification_report(y_true, predictions, labels)
        result.folds.append(
            FoldResult(
                fold=index,
                train_subjects=train_subjects,
                test_subjects=test_subjects,
                n_train=int(len(train)),
                n_test=int(len(test)),
                metrics=metrics,
            )
        )
    return result
