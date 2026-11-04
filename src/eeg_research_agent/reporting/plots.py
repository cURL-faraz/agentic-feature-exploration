"""Figure generation.

Every figure is written to ``artifacts/figures`` with a stable filename and is
derived exclusively from saved intermediate results (the feature table and the
statistical/evaluation JSON artifacts), so the report can be regenerated without
re-running the science.

Figures are optional: if matplotlib/seaborn fail for any reason the report stage
records the failure and continues, because a missing plot must never invalidate a
numeric result.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ..logging_utils import get_logger

__all__ = [
    "plot_stage_distribution",
    "plot_feature_by_stage",
    "plot_feature_distributions",
    "plot_subject_level",
    "plot_correlation_heatmap",
    "plot_confusion_matrix",
    "plot_model_comparison",
    "generate_figures",
]

logger = get_logger(__name__)

STAGE_ORDER = ("W", "N1", "N2", "N3", "REM")


def _setup():
    os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "mplcache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="notebook")
    return plt, sns


def _stages(frame: pd.DataFrame, minimum: int = 2) -> list[str]:
    counts = frame["stage"].value_counts()
    present = [stage for stage in STAGE_ORDER if stage in counts.index]
    present += [str(s) for s in counts.index if str(s) not in present]
    return [stage for stage in present if counts[stage] >= minimum]


def _save(fig: Any, path: Path, plt: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=int(os.environ.get("EEG_AGENT_FIGURE_DPI", "120")), bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_stage_distribution(frame: pd.DataFrame, figures_dir: Path) -> str:
    plt, sns = _setup()
    stages = _stages(frame, minimum=1)
    counts = frame["stage"].value_counts().reindex(stages)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.barplot(x=counts.index, y=counts.to_numpy(), ax=axes[0], palette="viridis", hue=counts.index, legend=False)
    axes[0].set_title("Epochs per sleep stage")
    axes[0].set_xlabel("stage")
    axes[0].set_ylabel("epochs")
    if "subject_id" in frame.columns:
        per_subject = (
            frame.groupby(["subject_id", "stage"]).size().unstack(fill_value=0).reindex(columns=stages, fill_value=0)
        )
        per_subject.plot(kind="bar", stacked=True, ax=axes[1], colormap="viridis")
        axes[1].set_title("Stage composition per subject")
        axes[1].set_ylabel("epochs")
        axes[1].legend(title="stage", fontsize=7, ncol=len(stages))
        axes[1].tick_params(axis="x", labelsize=7)
    return _save(fig, figures_dir / "stage_distribution.png", plt)


def plot_feature_by_stage(frame: pd.DataFrame, feature: str, figures_dir: Path) -> str:
    plt, sns = _setup()
    stages = _stages(frame)
    data = frame.loc[frame["stage"].isin(stages), ["stage", feature]].dropna()
    if data.empty or data[feature].notna().sum() < 4:
        raise ValueError(f"not enough finite observations of {feature} to plot")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.boxplot(data=data, x="stage", y=feature, order=stages, ax=axes[0], palette="mako", hue="stage", legend=False)
    axes[0].set_title(f"{feature} by stage (box)")
    sns.violinplot(data=data, x="stage", y=feature, order=stages, ax=axes[1], cut=0, palette="mako", hue="stage", legend=False)
    axes[1].set_title(f"{feature} by stage (density)")
    for axis in axes:
        axis.set_xlabel("stage")
    return _save(fig, figures_dir / f"feature_{feature}.png", plt)


def plot_feature_distributions(frame: pd.DataFrame, features: Sequence[str], figures_dir: Path) -> str:
    plt, sns = _setup()
    features = [feature for feature in features if feature in frame.columns][:12]
    if not features:
        raise ValueError("no plottable features supplied")
    ncols = 3
    nrows = int(np.ceil(len(features) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.8 * nrows))
    axes = np.atleast_1d(axes).ravel()
    stages = _stages(frame)
    for axis, feature in zip(axes, features):
        for stage in stages:
            values = frame.loc[frame["stage"] == stage, feature].dropna()
            if len(values) >= 3:
                sns.kdeplot(values, ax=axis, label=stage, fill=False, warn_singular=False)
        axis.set_title(feature, fontsize=8)
        axis.tick_params(labelsize=6)
    for axis in axes[len(features) :]:
        axis.axis("off")
    axes[0].legend(fontsize=6)
    return _save(fig, figures_dir / "feature_distributions.png", plt)


def plot_subject_level(frame: pd.DataFrame, feature: str, figures_dir: Path) -> str:
    plt, sns = _setup()
    data = frame[["subject_id", "stage", feature]].dropna()
    if data.empty:
        raise ValueError(f"no finite observations of {feature}")
    fig, axis = plt.subplots(figsize=(max(6, 0.9 * data["subject_id"].nunique()), 4))
    sns.pointplot(data=data, x="subject_id", y=feature, hue="stage", ax=axis, dodge=True, errorbar="se")
    axis.set_title(f"{feature}: subject-level stage means (SE)")
    axis.tick_params(axis="x", rotation=45, labelsize=7)
    axis.legend(fontsize=6, ncol=2)
    return _save(fig, figures_dir / f"subject_level_{feature}.png", plt)


def plot_correlation_heatmap(frame: pd.DataFrame, features: Sequence[str], figures_dir: Path) -> str:
    plt, sns = _setup()
    features = [feature for feature in features if feature in frame.columns][:20]
    if len(features) < 2:
        raise ValueError("need at least two features for a correlation heatmap")
    corr = frame.loc[:, features].astype(float).corr()
    fig, axis = plt.subplots(figsize=(0.42 * len(features) + 3, 0.42 * len(features) + 2))
    sns.heatmap(corr, ax=axis, cmap="coolwarm", center=0.0, vmin=-1, vmax=1, square=True, cbar_kws={"shrink": 0.7})
    axis.set_title("Feature correlation (Pearson, marginal distributions ignored)")
    axis.tick_params(labelsize=6)
    return _save(fig, figures_dir / "feature_correlation.png", plt)


def plot_confusion_matrix(row: Mapping[str, Any], figures_dir: Path, tag: str) -> str:
    plt, sns = _setup()
    matrix = np.asarray(row.get("confusion_matrix") or [])
    labels = [str(label) for label in (row.get("labels") or [])]
    if matrix.size == 0 or not labels:
        raise ValueError("evaluation row has no confusion matrix")
    fig, axis = plt.subplots(figsize=(1.0 + 0.55 * len(labels), 1.0 + 0.5 * len(labels)))
    sns.heatmap(
        matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1),
        annot=matrix,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=axis,
        cbar=False,
    )
    axis.set_xlabel("predicted")
    axis.set_ylabel("true")
    axis.set_title(f"{tag}\nrow-normalised, counts annotated", fontsize=8)
    axis.tick_params(labelsize=7)
    return _save(fig, figures_dir / f"confusion_{tag}.png", plt)


def plot_model_comparison(evaluation: Mapping[str, Any], figures_dir: Path, split: str = "test") -> str:
    plt, sns = _setup()
    rows = [
        {"model": f"{row['classifier']}\n[{row['feature_set']}]", "balanced_accuracy": row.get("balanced_accuracy"), "macro_f1": row.get("macro_f1")}
        for row in evaluation.get("results", [])
        if row.get("split") == split
    ]
    if not rows:
        raise ValueError(f"no {split} rows in the evaluation artifact")
    frame = pd.DataFrame(rows).dropna(subset=["balanced_accuracy"])
    if frame.empty:
        raise ValueError("no finite metrics to plot")
    melted = frame.melt(id_vars="model", var_name="metric", value_name="value")
    fig, axis = plt.subplots(figsize=(max(8, 0.5 * len(frame)), 5))
    sns.barplot(data=melted, y="model", x="value", hue="metric", ax=axis)
    axis.set_title(f"{split} performance by model and feature set")
    axis.tick_params(labelsize=6)
    axis.set_xlim(0, 1)
    return _save(fig, figures_dir / f"model_comparison_{split}.png", plt)


def generate_figures(
    frame: pd.DataFrame,
    config: Any,
    statistical_results: Mapping[str, Any] | None = None,
    evaluation: Mapping[str, Any] | None = None,
    features: Sequence[str] | None = None,
) -> dict[str, str]:
    """Generate every figure that can be produced from the saved results."""
    artifacts_cfg = config.analysis.artifacts
    if not artifacts_cfg.save_figures:
        return {}
    figures_dir = config.path("figures")
    written: dict[str, str] = {}
    failures: dict[str, str] = {}

    def attempt(name: str, function: Any, *args: Any, **kwargs: Any) -> None:
        try:
            written[name] = function(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - a figure must never break the report
            failures[name] = f"{type(exc).__name__}: {exc}"
            logger.warning("figure %s failed: %s", name, exc)

    features = list(features or [])
    attempt("stage_distribution", plot_stage_distribution, frame, figures_dir)

    top = features[: int(artifacts_cfg.top_feature_plots)]
    for feature in top:
        attempt(f"feature_by_stage:{feature}", plot_feature_by_stage, frame, feature, figures_dir)
    attempt("feature_distributions", plot_feature_distributions, frame, top, figures_dir)
    if top:
        attempt(f"subject_level:{top[0]}", plot_subject_level, frame, top[0], figures_dir)
    attempt("feature_correlation", plot_correlation_heatmap, frame, top, figures_dir)

    if evaluation:
        for row in evaluation.get("results", []):
            if row.get("split") != "test":
                continue
            tag = f"{row.get('feature_set')}_{row.get('classifier')}"
            attempt(f"confusion:{tag}", plot_confusion_matrix, row, figures_dir, tag)
        attempt("model_comparison", plot_model_comparison, evaluation, figures_dir)
    if failures:
        import json

        failures_path = config.path("tables") / "figure_failures.json"
        failures_path.parent.mkdir(parents=True, exist_ok=True)
        with open(failures_path, "w", encoding="utf-8") as handle:
            json.dump(failures, handle, indent=2)
        written["figure_failures"] = str(failures_path)
    return written
