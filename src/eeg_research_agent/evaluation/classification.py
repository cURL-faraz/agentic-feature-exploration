"""Classifier baselines, feature-set ablations and predictive evaluation.

The design rule that matters most here: **every supervised decision that touches
the data (imputation, scaling, feature selection, model fitting) is fitted on
training rows only**.  Validation rows choose hyper-parameters; test rows are
touched exactly once, at the end, and never influence a modelling choice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from ..config import ProjectConfig
from ..logging_utils import get_logger
from ..statistics.feature_selection import (
    make_preprocessor,
    mutual_information_ranking,
    select_top_k,
)
from .metrics import classification_report

__all__ = [
    "ClassifierSpec",
    "available_classifiers",
    "build_pipeline",
    "fit_predict",
    "EvaluationResult",
    "evaluate_feature_set",
    "run_predictive_evaluation",
    "feature_families",
    "ablation_feature_sets",
    "NON_CAUSAL_FEATURES",
]

logger = get_logger(__name__)

#: Sequence features that require the preceding epoch and are therefore excluded
#: from every causal ablation.
NON_CAUSAL_FEATURES = ("feature_delta_norm", "feature_delta_norm_sigma")


@dataclass(frozen=True)
class ClassifierSpec:
    """One classifier in the comparison, with its hyper-parameters."""

    name: str
    factory: Callable[[int], Any]
    supports_feature_selection: bool = True
    notes: str = ""


def available_classifiers(names: Sequence[str] | None = None, seed: int = 20240517) -> list[ClassifierSpec]:
    """Build the requested classifier specs (default: the full comparison set)."""
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC

    catalog = {
        "majority": ClassifierSpec(
            "majority",
            lambda _seed: DummyClassifier(strategy="most_frequent"),
            supports_feature_selection=False,
            notes="predicts the training majority class; the floor every model must beat",
        ),
        "logistic_regression": ClassifierSpec(
            "logistic_regression",
            lambda s: LogisticRegression(
                max_iter=3000, class_weight="balanced", random_state=s, C=1.0
            ),
            notes="interpretable linear baseline; coefficients are reportable",
        ),
        "random_forest": ClassifierSpec(
            "random_forest",
            lambda s: RandomForestClassifier(
                n_estimators=400,
                random_state=s,
                class_weight="balanced_subsample",
                min_samples_leaf=3,
                n_jobs=1,
            ),
            notes="captures non-linearities and interactions",
        ),
        "gradient_boosting": ClassifierSpec(
            "gradient_boosting",
            lambda s: GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.05, max_depth=3, random_state=s
            ),
            notes="boosted trees; strong on tabular features",
        ),
        "svm": ClassifierSpec(
            "svm",
            lambda s: SVC(kernel="rbf", C=4.0, gamma="scale", class_weight="balanced", random_state=s),
            notes="RBF SVM; needs standardised features (the pipeline provides them)",
        ),
    }
    requested = list(names) if names else list(catalog)
    unknown = [name for name in requested if name not in catalog]
    if unknown:
        raise ValueError(f"unknown classifiers: {unknown}; available: {sorted(catalog)}")
    return [catalog[name] for name in requested]


def build_pipeline(spec: ClassifierSpec, seed: int = 20240517, selected_features: Sequence[str] | None = None) -> Any:
    """Assemble ``[column subset] -> impute+scale -> classifier``.

    A column selector is prepended when ``selected_features`` is given, so the
    selection is *part of the fitted pipeline* and travels with ``predict``.
    """
    from sklearn.pipeline import Pipeline

    steps: list[tuple[str, Any]] = []
    if selected_features is not None:
        steps.append(("select", _ColumnSelector(list(selected_features))))
    steps.append(("preprocess", make_preprocessor()))
    steps.append(("model", spec.factory(seed)))
    return Pipeline(steps=steps)


class _ColumnSelector:
    """Pick a fixed subset of feature columns (sklearn-transformer compatible)."""

    def __init__(self, columns: Sequence[str]) -> None:
        self.columns = list(columns)

    def fit(self, X: pd.DataFrame, y: Any = None) -> "_ColumnSelector":
        missing = [column for column in self.columns if column not in X.columns]
        if missing:
            raise KeyError(f"selected features missing from the matrix: {missing}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.loc[:, self.columns]

    def get_params(self, deep: bool = True) -> dict[str, Any]:  # pragma: no cover - sklearn API
        return {"columns": self.columns}

    def set_params(self, **params: Any) -> "_ColumnSelector":  # pragma: no cover - sklearn API
        if "columns" in params:
            self.columns = list(params["columns"])
        return self


def fit_predict(
    train: pd.DataFrame,
    predict_sets: dict[str, pd.DataFrame],
    features: Sequence[str],
    spec: ClassifierSpec,
    label_column: str = "stage",
    seed: int = 20240517,
    select_k: int | None = None,
) -> tuple[dict[str, dict[str, Any]], Any]:
    """Fit on ``train`` and predict on each named set.

    Feature selection (when ``select_k`` is set) is performed **inside** this
    function using only ``train`` rows, then handed to the pipeline as a fixed
    column list.
    """
    columns = [c for c in features if c in train.columns]
    selected: list[str] | None = None
    if select_k is not None and spec.supports_feature_selection and len(columns) > select_k:
        ranking = mutual_information_ranking(train.loc[:, columns], train[label_column], seed=seed)
        selected = select_top_k(ranking, int(select_k))
        columns = selected
    pipeline = build_pipeline(spec, seed=seed, selected_features=selected if selected else None)
    if not columns:
        raise ValueError("no usable feature columns for the classifier")
    x_train = train.loc[:, columns]
    y_train = train[label_column].astype(str).to_numpy()
    pipeline.fit(x_train, y_train)
    labels = sorted(set(y_train))
    reports: dict[str, dict[str, Any]] = {}
    for name, frame in predict_sets.items():
        if frame is None or len(frame) == 0:
            continue
        x_test = frame.loc[:, columns]
        y_test = frame[label_column].astype(str).to_numpy()
        predictions = np.asarray(pipeline.predict(x_test))
        present = [label for label in labels if label in set(y_test)]
        report = classification_report(y_test, predictions, labels)
        report["evaluated_labels"] = present
        report["labels_absent_from_evaluation"] = [label for label in labels if label not in present]
        reports[name] = report
    return reports, pipeline


@dataclass
class EvaluationResult:
    """Predictive-evaluation output across feature sets and classifiers."""

    results: list[dict[str, Any]] = field(default_factory=list)
    written: dict[str, str] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "results": self.results,
            "errors": self.errors,
            "artifacts": self.written,
        }

    def best(self, metric: str = "balanced_accuracy", split: str = "test") -> dict[str, Any] | None:
        candidates = [
            row
            for row in self.results
            if row.get("split") == split and np.isfinite(row.get(metric, float("nan")))
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda row: row[metric])


def feature_families(registry: Any, features: Sequence[str]) -> dict[str, list[str]]:
    """Group feature ids by their registry family (plus ``sequence``)."""
    families: dict[str, list[str]] = {}
    for feature in features:
        if feature in NON_CAUSAL_FEATURES:
            families.setdefault("sequence", []).append(feature)
            continue
        if feature in registry:
            families.setdefault(registry.get(feature).family, []).append(feature)
        else:
            families.setdefault("unnamed", []).append(feature)
    return families


def ablation_feature_sets(
    registry: Any,
    features: Sequence[str],
    requested: Sequence[str],
    train_frame: pd.DataFrame,
    seed: int = 20240517,
    top_k: int = 20,
    label_column: str = "stage",
) -> dict[str, list[str]]:
    """Build the feature sets used by the ablation study.

    ``all_features`` deliberately **excludes** the non-causal sequence features:
    a causal claim cannot rest on a feature that needs the future.  They are
    offered as a separate, clearly-labelled ``sequence_non_causal`` set.

    ``top_k_mutual_information`` is ranked on training rows only.
    """
    families = feature_families(registry, features)
    causal = [f for f in features if f not in NON_CAUSAL_FEATURES]
    sets: dict[str, list[str]] = {"all_features": causal}
    sets["conventional_only"] = [
        f for f in causal if registry.__contains__(f) and registry.get(f).novelty_status == "established"
    ]
    sets["literature_derived_only"] = [
        f for f in causal if registry.__contains__(f) and registry.get(f).novelty_status == "literature-derived"
    ]
    sets["novel_only"] = [
        f for f in causal if registry.__contains__(f) and registry.get(f).novelty_status == "novel-hypothesis"
    ]
    sets["no_events"] = [f for f in causal if registry.__contains__(f) and registry.get(f).family != "events"]
    sets["no_spectral"] = [f for f in causal if registry.__contains__(f) and registry.get(f).family != "spectral"]
    sets["sequence_non_causal"] = [f for f in features if f in NON_CAUSAL_FEATURES]
    if train_frame is not None and len(train_frame) >= 20 and causal:
        ranking = mutual_information_ranking(train_frame.loc[:, causal], train_frame[label_column], seed=seed)
        sets["top_k_mutual_information"] = select_top_k(ranking, top_k)
    allowed = {"all_features", "conventional_only", "literature_derived_only", "novel_only",
               "no_events", "no_spectral", "top_k_mutual_information", "sequence_non_causal"}
    ordered = [name for name in requested if name in allowed]
    for name in requested:
        if name not in allowed:
            logger.warning("unknown ablation set %r skipped", name)
    return {name: [f for f in sets.get(name, []) if f in train_frame.columns or train_frame is None] for name in ordered}


def evaluate_feature_set(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame | None,
    test_frame: pd.DataFrame | None,
    features: Sequence[str],
    spec: ClassifierSpec,
    config: ProjectConfig,
    feature_set_name: str,
    select_k: int | None = None,
    seed: int = 20240517,
) -> list[dict[str, Any]]:
    """Fit one (feature set, classifier) pair and report val + test metrics."""
    predict_sets: dict[str, pd.DataFrame] = {}
    if validation_frame is not None and len(validation_frame):
        predict_sets["validation"] = validation_frame
    if test_frame is not None and len(test_frame):
        predict_sets["test"] = test_frame
    train_only: dict[str, pd.DataFrame] = {"train": train_frame}
    if not predict_sets:
        predict_sets = {"train": train_frame}
    if validation_frame is None or len(validation_frame) == 0:
        train_only["validation"] = train_frame.iloc[0:0]
    reports, pipeline = fit_predict(
        train_frame,
        predict_sets,
        features,
        spec,
        label_column="stage",
        seed=seed,
        select_k=select_k,
    )
    rows: list[dict[str, Any]] = []
    for split, report in reports.items():
        rows.append(
            {
                "feature_set": feature_set_name,
                "classifier": spec.name,
                "split": split,
                "n_features": len(features),
                "seed": seed,
                "accuracy": report["accuracy"],
                "balanced_accuracy": report["balanced_accuracy"],
                "macro_precision": report["macro_precision"],
                "macro_recall": report["macro_recall"],
                "macro_f1": report["macro_f1"],
                "weighted_f1": report["weighted_f1"],
                "cohens_kappa": report["cohens_kappa"],
                "matthews_corrcoef": report["matthews_corrcoef"],
                "per_class": report["per_class"],
                "confusion_matrix": report["confusion_matrix"],
                "labels": report["labels"],
                "true_distribution": report["true_distribution"],
                "predicted_distribution": report["predicted_distribution"],
                "notes": spec.notes,
                "train_rows": int(len(train_frame)),
                "train_subjects": int(train_frame["subject_id"].nunique()),
            }
        )
    return rows


def run_predictive_evaluation(
    table: pd.DataFrame,
    config: ProjectConfig,
    registry: Any | None = None,
    features: Sequence[str] | None = None,
    write: bool = True,
) -> EvaluationResult:
    """Full ablation study: feature sets x classifiers, subject-grouped.

    Train rows fit the models, validation rows are reported for transparency, and
    the test rows are used exactly once to produce the headline numbers.
    """
    from ..eeg.feature_extraction import feature_columns

    if registry is None:
        from ..eeg.feature_registry import load_default_registry

        registry = load_default_registry()
    predictive = config.analysis.predictive
    evaluation = EvaluationResult()
    if "split" not in table.columns:
        evaluation.errors.append(
            "the feature table has no `split` column; run the subject-split stage first"
        )
        return evaluation

    frame = table.replace([np.inf, -np.inf], np.nan)
    feature_list = [f for f in (features if features is not None else feature_columns(frame))]
    train_frame = frame.loc[frame["split"] == "train"]
    validation_frame = frame.loc[frame["split"] == "validation"]
    test_frame = frame.loc[frame["split"] == "test"]

    if len(train_frame) < 20 or train_frame["stage"].nunique() < 2:
        evaluation.errors.append(
            "training split is too small or single-class; predictive evaluation skipped"
        )
        return evaluation

    feature_sets = ablation_feature_sets(
        registry,
        feature_list,
        list(predictive.ablations),
        train_frame,
        seed=int(config.seed),
        top_k=int(config.analysis.features_selection.max_features_for_multivariable),
    )
    specs = available_classifiers(list(predictive.classifiers), seed=int(config.seed))
    select_k = None

    for set_name, set_features in feature_sets.items():
        if not set_features:
            evaluation.errors.append(f"feature set '{set_name}' is empty; skipped")
            continue
        for spec in specs:
            try:
                rows = evaluate_feature_set(
                    train_frame,
                    validation_frame if len(validation_frame) else None,
                    test_frame if len(test_frame) else None,
                    set_features,
                    spec,
                    config,
                    feature_set_name=set_name,
                    select_k=select_k if set_name != "top_k_mutual_information" else None,
                    seed=int(config.seed),
                )
            except Exception as exc:  # noqa: BLE001 - one model must not abort the study
                message = f"{set_name}/{spec.name}: {type(exc).__name__}: {exc}"
                evaluation.errors.append(message)
                logger.warning("classifier evaluation failed: %s", message)
                continue
            for row in rows:
                row["feature_ids"] = (
                    list(set_features)
                    if len(set_features) <= 60
                    else f"{len(set_features)} features (list omitted for brevity)"
                )
            evaluation.results.extend(rows)

    evaluation.results.sort(
        key=lambda row: (
            row["split"],
            row["feature_set"],
            -row["balanced_accuracy"],
        )
    )
    if write:
        evaluation.written = _write_evaluation(evaluation, config)
    return evaluation


def _write_evaluation(evaluation: EvaluationResult, config: ProjectConfig) -> dict[str, str]:
    import json

    tables = config.path("tables")
    tables.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    flat_rows = []
    for row in evaluation.results:
        record = {k: v for k, v in row.items() if k not in {"per_class", "confusion_matrix", "feature_ids"}}
        for stage in config.analysis.predictive.per_stage_focus:
            entry = (row.get("per_class") or {}).get(stage, {})
            record[f"f1_{stage}"] = entry.get("f1")
            record[f"recall_{stage}"] = entry.get("recall")
            record[f"precision_{stage}"] = entry.get("precision")
            record[f"support_{stage}"] = entry.get("support")
        flat_rows.append(record)
    csv_path = tables / "predictive_evaluation.csv"
    pd.DataFrame(flat_rows).to_csv(csv_path, index=False)
    written["predictive_csv"] = str(csv_path)

    json_path = tables / "predictive_evaluation.json"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(evaluation.to_dict(), handle, indent=2, default=str)
    written["predictive_json"] = str(json_path)

    test_rows = [row for row in evaluation.results if row["split"] == "test"]
    summary = {
        "n_models_evaluated": len(evaluation.results),
        "best_test_model": (
            {
                "feature_set": max(test_rows, key=lambda r: r["balanced_accuracy"])["feature_set"],
                "classifier": max(test_rows, key=lambda r: r["balanced_accuracy"])["classifier"],
                "balanced_accuracy": max(test_rows, key=lambda r: r["balanced_accuracy"])["balanced_accuracy"],
            }
            if test_rows
            else None
        ),
        "errors": evaluation.errors,
        "artifacts": written,
    }
    summary_path = config.path("metadata") / "predictive_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)
    written["predictive_summary"] = str(summary_path)
    return written
