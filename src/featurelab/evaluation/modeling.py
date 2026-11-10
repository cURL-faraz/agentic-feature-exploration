"""Design matrices, the fixed classifier and subject-grouped cross-validation.

"Epoch context" (a.k.a. neighbouring epochs): a human scorer never stages a 30 s
epoch in isolation - they look at what came before and after.  N1 vs REM is
often ambiguous inside one epoch but obvious from the sequence.  So the model
sees, for every feature, its value in the previous and next epoch of the same
recording.  Baseline and candidate get exactly the same treatment, so a
candidate is credited only for information the context does not already give.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def add_context(frame: pd.DataFrame, recording: np.ndarray, k: int) -> pd.DataFrame:
    """Append previous/next-epoch copies of every column (within a recording)."""
    if k <= 0:
        return frame
    values = frame.to_numpy(dtype=np.float32)
    rec = np.asarray(recording)
    n = len(rec)
    parts, names = [values], list(frame.columns)
    for lag in range(1, k + 1):
        for direction, shift in (("prev", lag), ("next", -lag)):
            source = np.arange(n) - shift                       # row we copy from
            valid = (source >= 0) & (source < n)
            valid[valid] &= rec[source[valid]] == rec[valid]     # same recording only
            source = np.where(valid, source, np.arange(n))       # recording edge: repeat own value
            parts.append(values[source])
            names += [f"{c}__{direction}{lag}" for c in frame.columns]
    return pd.DataFrame(np.concatenate(parts, axis=1), columns=names)


def design(base: pd.DataFrame, recording, k: int, candidate: np.ndarray | None = None,
           name: str = "candidate") -> np.ndarray:
    frame = base.copy()
    if candidate is not None:
        frame[name] = np.asarray(candidate, dtype=np.float32)
    return add_context(frame, recording, k).to_numpy(dtype=np.float32)


def make_model(cfg: dict, seed: int):
    if cfg.get("kind", "hgb") == "logreg":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(max_iter=2000, class_weight="balanced"))
    return HistGradientBoostingClassifier(
        max_iter=int(cfg.get("max_iter", 120)), learning_rate=float(cfg.get("learning_rate", 0.1)),
        class_weight="balanced", early_stopping=False, random_state=seed,
    )


def folds(subject: np.ndarray, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Deterministic subject-grouped folds: a subject is never in train and test at once."""
    n_splits = min(n_splits, len(np.unique(subject)))
    return list(GroupKFold(n_splits=n_splits).split(np.zeros(len(subject)), groups=subject))


def cross_val_predict(X: np.ndarray, y: np.ndarray, subject: np.ndarray, cfg: dict) -> np.ndarray:
    model_cfg, seed = cfg["model"], int(cfg["seed"])
    pred = np.empty_like(y)
    for train, test in folds(subject, int(cfg["cv_folds"])):
        model = make_model(model_cfg, seed)
        model.fit(X[train], y[train])
        pred[test] = model.predict(X[test])
    return pred


def fit_predict(X_train, y_train, X_test, cfg: dict) -> np.ndarray:
    model = make_model(cfg["model"], int(cfg["seed"]))
    model.fit(X_train, y_train)
    return model.predict(X_test)
