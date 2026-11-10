"""Feature <-> sleep-stage association measures.

Interpretation matters more than the numbers here.  Sleep stages are a *nominal*
multiclass variable: encoding them as 0..4 and running Pearson correlation would
impose a meaningless ordering (N1 and N2 would appear "closer" than N1 and REM
for no biological reason).  The measures used are therefore:

* ``kruskal_h`` / ``epsilon_squared`` / ``eta_squared_h`` - rank-based, order-free
  measures of how much a feature separates the stages.
* ``mutual_information`` - information-theoretic, order-free, captures
  non-monotonic relationships that correlation misses.
* ``point_biserial`` - only for genuinely *binary* contrasts (e.g. N2 vs REM),
  where it equals Pearson correlation with the 0/1 indicator.
* ``spearman`` / ``pearson`` - only reported for a binary contrast or for
  continuous reference variables, never for the 5-way stage label.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .effect_sizes import epsilon_squared_from_h, eta_squared_from_h

__all__ = [
    "AssociationResult",
    "point_biserial",
    "pearson",
    "spearman",
    "mutual_information",
    "multiclass_association",
    "feature_stage_associations",
    "correlation_matrix",
]


@dataclass
class AssociationResult:
    """One association between a feature and a target variable."""

    feature: str
    target: str
    measure: str
    value: float
    p_value: float | None = None
    n: int = 0
    order_imposed: bool = False
    interpretation: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "target": self.target,
            "measure": self.measure,
            "value": None if not np.isfinite(self.value) else float(self.value),
            "p_value": None if self.p_value is None or not np.isfinite(self.p_value) else float(self.p_value),
            "n": int(self.n),
            "order_imposed": self.order_imposed,
            "interpretation": self.interpretation,
            "notes": self.notes,
        }


def _finite_pairs(x: Sequence[float], y: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(x, dtype=float).ravel()
    b = np.asarray(y, dtype=float).ravel()
    if a.size != b.size:
        raise ValueError("x and y must have the same length")
    mask = np.isfinite(a) & np.isfinite(b)
    return a[mask], b[mask]


def point_biserial(feature: Sequence[float], binary: Sequence[int]) -> AssociationResult:
    """Point-biserial correlation between a continuous feature and a 0/1 label.

    Only valid for a genuine two-group contrast; the 0/1 coding is arbitrary but
    innocuous, which is *not* true for the 5-way sleep-stage label.
    """
    x, y = _finite_pairs(feature, binary)
    if x.size < 3 or np.unique(y).size != 2:
        return AssociationResult("", "binary", "point_biserial", float("nan"), n=int(x.size))
    result = stats.pointbiserialr(y, x)
    return AssociationResult(
        feature="",
        target="binary",
        measure="point_biserial",
        value=float(result.statistic),
        p_value=float(result.pvalue),
        n=int(x.size),
        order_imposed=False,
        interpretation="r_pb = Pearson r with the 0/1 indicator",
    )


def pearson(x: Sequence[float], y: Sequence[float]) -> AssociationResult:
    """Pearson product-moment correlation (linear, interval-scaled targets only)."""
    a, b = _finite_pairs(x, y)
    if a.size < 3 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return AssociationResult("", "", "pearson", float("nan"), n=int(a.size))
    result = stats.pearsonr(a, b)
    return AssociationResult(
        "", "", "pearson", float(result.statistic), float(result.pvalue), int(a.size)
    )


def spearman(x: Sequence[float], y: Sequence[float]) -> AssociationResult:
    """Spearman rank correlation (monotonic, robust to outliers)."""
    a, b = _finite_pairs(x, y)
    if a.size < 3 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return AssociationResult("", "", "spearman", float("nan"), n=int(a.size))
    result = stats.spearmanr(a, b)
    return AssociationResult(
        "", "", "spearman", float(result.statistic), float(result.pvalue), int(a.size)
    )


def mutual_information(
    feature: Sequence[float],
    labels: Sequence[str],
    n_neighbors: int = 5,
    seed: int = 20240517,
) -> AssociationResult:
    """Mutual information between a feature and the (nominal) stage label.

    Mutual information is expressed in **nats** and is not normalised, so it is
    comparable across features only because every feature is compared against the
    same label vector.  The estimator (sklearn's k-NN estimator) is biased upward
    for small samples, so it is used for *ranking*, never as a p-value.
    """
    from sklearn.feature_selection import mutual_info_classif

    x = np.asarray(feature, dtype=float).ravel()
    y = np.asarray(labels, dtype=object).ravel()
    mask = np.isfinite(x)
    x, y = x[mask], y[mask]
    if x.size < n_neighbors + 2 or np.unique(y).size < 2 or np.allclose(x, x[0]):
        return AssociationResult("", "stage", "mutual_information", float("nan"), n=int(x.size))
    value = mutual_info_classif(
        x.reshape(-1, 1), y, n_neighbors=n_neighbors, random_state=seed, discrete_features=False
    )
    return AssociationResult(
        "",
        "stage",
        "mutual_information",
        float(value[0]),
        n=int(x.size),
        interpretation="nats; no p-value (estimator is upward-biased for small n)",
    )


def multiclass_association(
    feature: Sequence[float], stages: Sequence[str], min_group_size: int = 10
) -> AssociationResult:
    """Rank-based, order-free association for the nominal 5-way stage label.

    Uses the Kruskal-Wallis H statistic as the effect-size backbone and reports
    epsilon-squared (rank-based) alongside it.  **No ordering is imposed on the
    stages.**
    """
    x = np.asarray(feature, dtype=float).ravel()
    y = np.asarray(stages, dtype=object).ravel()
    mask = np.isfinite(x)
    x, y = x[mask], y[mask]
    groups = [x[y == stage] for stage in sorted(set(y))]
    groups = [g for g in groups if g.size >= min_group_size]
    if len(groups) < 2:
        return AssociationResult("", "stage", "epsilon_squared", float("nan"), n=int(x.size))
    statistic, p_value = stats.kruskal(*groups)
    n_total = int(sum(g.size for g in groups))
    epsilon = epsilon_squared_from_h(float(statistic), n_total)
    eta = eta_squared_from_h(float(statistic), n_total, len(groups))
    return AssociationResult(
        "",
        "stage",
        "epsilon_squared",
        float(epsilon),
        float(p_value),
        n=n_total,
        order_imposed=False,
        interpretation="Kruskal-Wallis; epsilon^2 (rank-based, order-free)",
        notes=f"H={float(statistic):.4f}, eta^2_H={eta:.4f}, k={len(groups)}",
    )


def feature_stage_associations(
    table: pd.DataFrame,
    features: Iterable[str],
    min_group_size: int = 10,
    n_neighbors: int = 5,
    seed: int = 20240517,
    aggregate_by_subject: bool = True,
) -> list[dict[str, Any]]:
    """Associate every feature with the sleep-stage label, subject-aware.

    When ``aggregate_by_subject`` is set, the association is computed on
    subject x stage means: with a handful of subjects the epoch-level p-values
    from ~10^4 epochs would be meaningless.
    """
    rows: list[dict[str, Any]] = []
    for feature in features:
        if aggregate_by_subject:
            from .descriptive import subject_stage_means

            aggregates = subject_stage_means(table, feature, min_epochs=1)
            if aggregates.empty:
                continue
            values = aggregates["mean"].to_numpy(dtype=float)
            labels = aggregates["stage"].to_numpy(dtype=object)
            n_subjects = int(aggregates["subject_id"].nunique())
        else:
            values = table[feature].to_numpy(dtype=float)
            labels = table["stage"].to_numpy(dtype=object)
            n_subjects = int(table["subject_id"].nunique())
        rank = multiclass_association(values, labels, min_group_size=min_group_size)
        information = mutual_information(values, labels, n_neighbors=n_neighbors, seed=seed)
        rows.append(
            {
                "feature": feature,
                "unit": "subject_stage_mean" if aggregate_by_subject else "epoch",
                "n_subjects": n_subjects,
                "kruskal_epsilon_squared": rank.to_dict(),
                "mutual_information_nats": information.to_dict(),
            }
        )
    rows.sort(
        key=lambda row: (
            -(row["kruskal_epsilon_squared"]["value"] or 0.0),
            -(row["mutual_information_nats"]["value"] or 0.0),
        )
    )
    return rows


def correlation_matrix(
    table: pd.DataFrame, features: Sequence[str], method: str = "spearman"
) -> pd.DataFrame:
    """Pairwise correlation matrix of the feature columns (NaN-aware, pairwise)."""
    frame = table.loc[:, [f for f in features if f in table.columns]].astype(float)
    if method not in {"spearman", "pearson", "kendall"}:
        raise ValueError(f"unsupported correlation method: {method}")
    return frame.corr(method=method, min_periods=8)
