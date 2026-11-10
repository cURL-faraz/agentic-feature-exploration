"""Feature ranking, redundancy control and bounded interaction search.

Every function in this module takes a **training-only** matrix.  Nothing here is
allowed to look at validation or test rows; the evaluation module enforces that
by fitting preprocessing and selection inside the training fold only.

Redundancy is handled *before* multivariate modelling: EEG band powers are
strongly collinear, and a model fed 30 near-duplicate features has unstable
coefficients and meaningless importances.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

__all__ = [
    "RedundancyReport",
    "make_preprocessor",
    "variance_inflation_factors",
    "redundancy_clusters",
    "prune_redundant",
    "mutual_information_ranking",
    "anova_f_ranking",
    "l1_ranking",
    "random_forest_importance",
    "permutation_importance_scores",
    "recursive_feature_elimination",
    "select_top_k",
    "interaction_candidates",
    "evaluate_interactions",
]


@dataclass
class RedundancyReport:
    """Result of the collinearity screen."""

    threshold: float
    clusters: list[list[str]] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    kept: list[str] = field(default_factory=list)
    high_vif: dict[str, float] = field(default_factory=dict)
    max_vif: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "n_clusters": len(self.clusters),
            "clusters": self.clusters,
            "dropped": self.dropped,
            "kept": self.kept,
            "high_vif": self.high_vif,
            "max_vif": self.max_vif,
        }


def _make_pipeline(preprocessor: Any, model: Any) -> Any:
    from sklearn.pipeline import Pipeline

    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def make_preprocessor() -> Any:
    """Median imputation + standardisation, as an sklearn transformer.

    Deliberately a *pipeline* so the statistics are re-learned on each training
    fold and can never leak test-set information.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]
    )


def _impute(data: np.ndarray) -> np.ndarray:
    """Median-impute non-finite entries in place-safe fashion (returns a copy)."""
    data = np.array(data, dtype=float, copy=True)
    if data.ndim != 2:
        raise ValueError("expected a 2-D array")
    medians = np.nanmedian(data, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    bad_rows, bad_columns = np.where(~np.isfinite(data))
    data[bad_rows, bad_columns] = medians[bad_columns]
    return data


def variance_inflation_factors(
    frame: pd.DataFrame, features: Sequence[str] | None = None
) -> dict[str, float]:
    """VIF of each column, computed by regressing it on all the others.

    Implemented with least squares rather than an external library so the
    definition stays explicit: ``VIF_j = 1 / (1 - R_j^2)`` where ``R_j^2`` comes
    from regressing feature *j* on every other feature.  A column with
    (near-)zero variance or a perfect linear dependence gets ``inf``.
    """
    columns = [c for c in (features or list(frame.columns)) if c in frame.columns]
    if len(columns) < 2:
        return {c: 1.0 for c in columns}
    data = _impute(frame.loc[:, columns].astype(float).to_numpy())
    result: dict[str, float] = {}
    for index, column in enumerate(columns):
        others = np.delete(data, index, axis=1)
        target = data[:, index]
        if float(np.var(target)) == 0.0:
            result[column] = float("inf")
            continue
        design = np.column_stack([np.ones(len(others)), others])
        try:
            coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
        except np.linalg.LinAlgError:  # pragma: no cover - defensive
            result[column] = float("inf")
            continue
        residual = target - design @ coefficients
        ss_res = float(np.sum(residual**2))
        ss_tot = float(np.sum((target - target.mean()) ** 2))
        if ss_tot <= 0.0:
            result[column] = float("inf")
            continue
        r_squared = 1.0 - ss_res / ss_tot
        result[column] = float("inf") if r_squared >= 1.0 - 1e-12 else float(1.0 / (1.0 - r_squared))
    return result


def redundancy_clusters(
    frame: pd.DataFrame, features: Sequence[str], threshold: float = 0.9
) -> list[list[str]]:
    """Hierarchical clustering of features by ``1 - |corr|`` distance.

    Returns clusters whose members are more correlated than ``threshold`` and
    therefore carry largely redundant information.
    """
    columns = [c for c in features if c in frame.columns]
    if len(columns) < 2:
        return [[c] for c in columns]
    correlation = frame.loc[:, columns].astype(float).corr(method="spearman", min_periods=8)
    correlation = correlation.fillna(0.0).to_numpy()
    distance = 1.0 - np.abs(correlation)
    np.fill_diagonal(distance, 0.0)
    distance = (distance + distance.T) / 2.0
    condensed = squareform(distance, checks=False)
    linkage = hierarchy.linkage(condensed, method="average")
    labels = hierarchy.fcluster(linkage, t=1.0 - float(threshold), criterion="distance")
    clusters: dict[int, list[str]] = {}
    for column, label in zip(columns, labels):
        clusters.setdefault(int(label), []).append(column)
    return sorted(clusters.values(), key=lambda group: (-len(group), group[0]))


def prune_redundant(
    frame: pd.DataFrame,
    features: Sequence[str],
    threshold: float = 0.9,
    importance: dict[str, float] | None = None,
    vif_threshold: float = 10.0,
) -> RedundancyReport:
    """Drop redundant features, keeping the most important member of each cluster.

    ``importance`` (e.g. mutual information with the stage label) decides which
    member of a cluster survives.  A surviving feature with
    ``VIF > vif_threshold`` is dropped as well.
    """
    report = RedundancyReport(threshold=threshold)
    report.clusters = redundancy_clusters(frame, features, threshold)
    keep: list[str] = []
    for cluster in report.clusters:
        if len(cluster) == 1:
            keep.append(cluster[0])
        elif importance:
            keep.append(max(cluster, key=lambda name: importance.get(name, 0.0)))
        else:
            keep.append(cluster[0])
    dropped = [c for c in features if c not in keep]

    vifs = variance_inflation_factors(frame, keep or list(features))
    finite_vifs = [v for v in vifs.values() if np.isfinite(v)]
    report.max_vif = max(finite_vifs) if finite_vifs else None
    report.high_vif = {
        column: (value if np.isfinite(value) else float("inf"))
        for column, value in vifs.items()
        if value > vif_threshold
    }
    final_keep = [c for c in keep if vifs.get(c, 0.0) <= vif_threshold]
    dropped += [c for c in keep if c not in final_keep]
    report.kept = final_keep
    report.dropped = sorted(set(dropped))
    return report


def mutual_information_ranking(
    frame: pd.DataFrame, labels: Sequence[str], seed: int = 20240517, n_neighbors: int = 5
) -> dict[str, float]:
    """Mutual information of every column with the stage label (in nats)."""
    from sklearn.feature_selection import mutual_info_classif

    if frame.shape[1] == 0:
        return {}
    data = _impute(frame.astype(float).to_numpy())
    values = mutual_info_classif(
        data, np.asarray(labels), n_neighbors=n_neighbors, random_state=seed, discrete_features=False
    )
    return {
        column: float(value) for column, value in zip(frame.columns, np.nan_to_num(values))
    }


def anova_f_ranking(frame: pd.DataFrame, labels: Sequence[str]) -> dict[str, float]:
    """One-way ANOVA F statistic of every column against the stage label."""
    from sklearn.feature_selection import f_classif

    if frame.shape[1] == 0:
        return {}
    data = _impute(frame.astype(float).to_numpy())
    statistics, _ = f_classif(data, np.asarray(labels))
    return {
        column: float(value)
        for column, value in zip(frame.columns, np.nan_to_num(statistics, nan=0.0))
    }


def l1_ranking(
    frame: pd.DataFrame,
    labels: Sequence[str],
    C: float = 0.5,
    seed: int = 20240517,
    max_iter: int = 4000,
) -> dict[str, float]:
    """Absolute L1 (lasso) logistic-regression coefficients as a ranking."""
    from sklearn.linear_model import LogisticRegression

    if frame.shape[1] == 0:
        return {}
    binary = len(np.unique(labels)) <= 2
    model = LogisticRegression(
        penalty="l1",
        solver="liblinear" if binary else "saga",
        C=C,
        max_iter=max_iter,
        random_state=seed,
        class_weight="balanced",
    )
    pipeline = _make_pipeline(make_preprocessor(), model)
    pipeline.fit(frame, np.asarray(labels))
    coefficients = np.asarray(pipeline.named_steps["model"].coef_)
    magnitude = np.abs(coefficients).max(axis=0)
    return {column: float(value) for column, value in zip(frame.columns, magnitude)}


def random_forest_importance(
    frame: pd.DataFrame,
    labels: Sequence[str],
    n_estimators: int = 300,
    seed: int = 20240517,
) -> dict[str, float]:
    """Impurity-based random-forest importances (Gini importance)."""
    from sklearn.ensemble import RandomForestClassifier

    if frame.shape[1] == 0:
        return {}
    pipeline = _make_pipeline(
        make_preprocessor(),
        RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=seed,
            class_weight="balanced_subsample",
            n_jobs=1,
        ),
    )
    pipeline.fit(frame, np.asarray(labels))
    values = pipeline.named_steps["model"].feature_importances_
    return {column: float(value) for column, value in zip(frame.columns, values)}


def permutation_importance_scores(
    frame: pd.DataFrame,
    labels: Sequence[str],
    n_repeats: int = 8,
    seed: int = 20240517,
    scoring: str = "balanced_accuracy",
) -> dict[str, float]:
    """Permutation importance (model-agnostic, computed on the supplied rows)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance

    if frame.shape[1] == 0:
        return {}
    pipeline = _make_pipeline(
        make_preprocessor(),
        RandomForestClassifier(
            n_estimators=200, random_state=seed, class_weight="balanced_subsample", n_jobs=1
        ),
    )
    pipeline.fit(frame, np.asarray(labels))
    result = permutation_importance(
        pipeline,
        frame,
        np.asarray(labels),
        n_repeats=n_repeats,
        random_state=seed,
        scoring=scoring,
        n_jobs=1,
    )
    return {
        column: float(value) for column, value in zip(frame.columns, result.importances_mean)
    }


def recursive_feature_elimination(
    frame: pd.DataFrame,
    labels: Sequence[str],
    n_features: int = 20,
    step: float = 0.1,
    seed: int = 20240517,
) -> list[str]:
    """Recursive feature elimination with a logistic-regression estimator."""
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LogisticRegression

    if frame.shape[1] == 0:
        return []
    n_features = max(1, min(int(n_features), frame.shape[1]))
    estimator = LogisticRegression(max_iter=3000, random_state=seed, class_weight="balanced")
    pipeline = _make_pipeline(
        make_preprocessor(), RFE(estimator, n_features_to_select=n_features, step=step)
    )
    pipeline.fit(frame, np.asarray(labels))
    selector = pipeline.named_steps["model"]
    return [column for column, keep in zip(frame.columns, selector.support_) if keep]


def select_top_k(ranking: dict[str, float], k: int, exclude: Sequence[str] = ()) -> list[str]:
    """Take the ``k`` highest-ranked features, skipping excluded names."""
    banned = set(exclude)
    ordered = sorted(
        (name for name in ranking if name not in banned),
        key=lambda name: (
            -(ranking[name] if np.isfinite(ranking[name]) else -np.inf),
            name,
        ),
    )
    return ordered[: max(0, int(k))]


def _zscore(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values)
    center = float(np.nanmedian(values))
    scale = float(np.nanpercentile(values[finite], 75) - np.nanpercentile(values[finite], 25))
    if not np.isfinite(scale) or scale == 0.0:
        scale = float(np.nanstd(values)) or 1.0
    return (values - center) / scale


def interaction_candidates(
    frame: pd.DataFrame,
    seed_features: Sequence[str],
    max_pairs: int = 12,
    epsilon: float = 1e-9,
) -> tuple[pd.DataFrame, list[str]]:
    """Build a *bounded* candidate interaction space.

    Only pairs drawn from ``seed_features`` (already the top main-effect
    features) and only two operators (standardised product, standardised ratio)
    are generated.  The search space is deliberately small so that the
    multiple-comparison correction across interactions stays meaningful: the
    brief explicitly forbids generating millions of combinations and reporting
    only the winners.
    """
    columns = [c for c in seed_features if c in frame.columns]
    pairs: list[tuple[str, str]] = []
    for index, first in enumerate(columns):
        for second in columns[index + 1 :]:
            pairs.append((first, second))
    pairs = pairs[: max(0, int(max_pairs))]
    data: dict[str, np.ndarray] = {}
    names: list[str] = []
    for first, second in pairs:
        a_std = _zscore(frame[first].astype(float).to_numpy())
        b_std = _zscore(frame[second].astype(float).to_numpy())
        product_name = f"interaction::{first}*{second}"
        ratio_name = f"interaction::{first}/{second}"
        data[product_name] = a_std * b_std
        data[ratio_name] = a_std / (np.abs(b_std) + epsilon) * np.sign(b_std)
        names.extend([product_name, ratio_name])
    return pd.DataFrame(data, index=frame.index), names


def _mi_of(frame: pd.DataFrame, labels: np.ndarray, columns: Sequence[str]) -> dict[str, float]:
    from sklearn.feature_selection import mutual_info_classif

    usable = [c for c in columns if c in frame.columns]
    if not usable:
        return {}
    data = _impute(frame.loc[:, usable].astype(float).to_numpy())
    if np.unique(labels).size < 2:
        return dict.fromkeys(usable, 0.0)
    values = mutual_info_classif(
        data, labels, random_state=20240517, discrete_features=False
    )
    return {
        column: float(value) for column, value in zip(usable, np.nan_to_num(values))
    }


def evaluate_interactions(
    frame: pd.DataFrame,
    labels: Sequence[str],
    interaction_names: Sequence[str],
    baseline: dict[str, float],
    seed: int = 20240517,
    alpha: float = 0.05,
    method: str = "fdr_bh",
) -> list[dict[str, Any]]:
    """Score candidate interactions by cross-validated *gain* in mutual information.

    For each candidate the score is the mean over subject-grouped folds of
    ``MI(interaction) - max(MI(parent_1), MI(parent_2))``.  Candidates are
    labelled **exploratory**; the only significance machinery attached is a
    monotone heuristic score transform, which is explicitly *not* presented as a
    sampling p-value.
    """
    from sklearn.model_selection import GroupKFold

    from .multiple_testing import correct_pvalues

    if not interaction_names:
        return []
    labels_array = np.asarray(labels)
    records: list[dict[str, Any]] = []
    for name in interaction_names:
        body = name.replace("interaction::", "")
        parents = [part for part in body.replace("*", "/").split("/") if part]
        parent_scores = [baseline.get(parent, 0.0) for parent in parents]
        gains: list[float] = []
        n_rows = len(frame)
        if n_rows >= 20:
            n_splits = int(min(5, max(2, n_rows // 20)))
            splitter = GroupKFold(n_splits=n_splits)
            groups = np.arange(n_rows)
            for train_index, _ in splitter.split(frame, labels_array, groups):
                train = frame.iloc[train_index]
                independent = [c for c in parents if c in train.columns]
                scores = _mi_of(train, labels_array[train_index], [*independent, name])
                if name not in scores:
                    continue
                reference = max([scores.get(parent, 0.0) for parent in independent] or [0.0])
                gains.append(float(scores[name] - reference))
        gain = float(np.mean(gains)) if gains else float("nan")
        records.append(
            {
                "interaction": name,
                "parents": parents,
                "parent_mi_max": max(parent_scores) if parent_scores else 0.0,
                "cv_mi_gain": gain,
                "n_folds": len(gains),
                "role": "exploratory",
            }
        )
    usable = [index for index, record in enumerate(records) if np.isfinite(record["cv_mi_gain"])]
    if usable:
        heuristic = [
            max(1e-12, float(2.0 ** (-max(0.0, records[i]["cv_mi_gain"]) * 10.0))) for i in usable
        ]
        correction = correct_pvalues(
            heuristic, [records[i]["interaction"] for i in usable], method=method, alpha=alpha
        )
        for position, index in enumerate(usable):
            records[index]["heuristic_score"] = correction.p_raw[position]
            records[index]["heuristic_adjusted"] = correction.p_adjusted[position]
            records[index]["passes_heuristic_screen"] = bool(correction.reject[position])
            records[index]["heuristic_note"] = (
                "monotone transform of the CV MI gain used only to rank candidates; "
                "this is NOT a sampling p-value and MUST NOT be reported as one"
            )
    records.sort(
        key=lambda record: -(
            record["cv_mi_gain"] if np.isfinite(record["cv_mi_gain"]) else -np.inf
        )
    )
    return records
