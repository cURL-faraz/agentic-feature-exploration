"""Annotation handling: canonical stage mapping and hypnogram construction.

The mapping is driven entirely by ``config/dataset.yaml``.  Unknown annotation
strings are never silently coerced into a sleep stage: they are reported and
counted so that the dataset summary can state exactly what was dropped.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable

from .recording import CANONICAL_STAGES, UNKNOWN_STAGE, StageAnnotation

__all__ = ["StageMapper", "AnnotationReport", "summarise_annotations"]


@dataclass
class AnnotationReport:
    """What the annotation mapper saw and how it was interpreted.

    ``mapped_counts`` counts annotations that became a *canonical sleep stage*
    (the labels the analysis can actually use).  ``unmapped_counts`` counts every
    annotation that did **not**: unrecognised strings, and descriptions the
    configuration deliberately declares to be non-stage (``"Movement time"`` ->
    ``UNKNOWN``).  Keeping them apart from ``mapped_counts`` means the dataset
    summary can state exactly how much recording time was usable.
    """

    mapped_counts: Counter = field(default_factory=Counter)
    unmapped_counts: Counter = field(default_factory=Counter)
    total_duration_s: float = 0.0
    unknown_duration_s: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mapped_counts": dict(self.mapped_counts),
            "unmapped_counts": dict(self.unmapped_counts),
            "total_duration_s": round(self.total_duration_s, 3),
            "unknown_duration_s": round(self.unknown_duration_s, 3),
            "unknown_fraction": (
                round(self.unknown_duration_s / self.total_duration_s, 6)
                if self.total_duration_s
                else None
            ),
            "notes": list(self.notes),
        }


class StageMapper:
    """Map dataset-specific annotation strings to the canonical sleep stages."""

    def __init__(self, annotation_map: dict[str, str], supported: Iterable[str]) -> None:
        self.annotation_map = dict(annotation_map)
        self.supported = tuple(supported)
        self._normalised = {self._normalise_key(k): v for k, v in self.annotation_map.items()}

    @staticmethod
    def _normalise_key(value: str) -> str:
        return " ".join(str(value).strip().lower().replace("_", " ").split())

    def map_label(self, description: str) -> str | None:
        """Return a canonical stage, ``UNKNOWN`` or ``None`` if unrecognised."""
        key = self._normalise_key(description)
        if key in self._normalised:
            return self._normalised[key]
        # tolerate punctuation/case variants without inventing new semantics
        for pattern, stage in self._normalised.items():
            if pattern and pattern in key:
                return stage
        return None

    def map_annotations(
        self, raw: Iterable[tuple[float, float, str]]
    ) -> tuple[list[StageAnnotation], AnnotationReport]:
        report = AnnotationReport()
        annotations: list[StageAnnotation] = []
        for onset, duration, description in raw:
            stage = self.map_label(description)
            report.total_duration_s += float(duration)
            if stage is None:
                report.unmapped_counts[str(description).strip()] += 1
                report.unknown_duration_s += float(duration)
                annotations.append(
                    StageAnnotation(float(onset), float(duration), UNKNOWN_STAGE, str(description))
                )
                continue
            if stage == UNKNOWN_STAGE:
                # Declared non-stage by configuration (movement time, unscored
                # epochs).  Recorded as unmapped so it is visible in the summary.
                report.unmapped_counts[str(description).strip()] += 1
                report.unknown_duration_s += float(duration)
                report.notes.append(f"{description!r} is declared non-stage in the annotation map")
                annotations.append(
                    StageAnnotation(float(onset), float(duration), stage, str(description))
                )
                continue
            if stage not in self.supported and stage != UNKNOWN_STAGE:
                report.notes.append(f"unsupported stage {stage!r} from {description!r}")
                report.unmapped_counts[str(description).strip()] += 1
                report.unknown_duration_s += float(duration)
                annotations.append(
                    StageAnnotation(float(onset), float(duration), UNKNOWN_STAGE, str(description))
                )
                continue
            report.mapped_counts[stage] += 1
            annotations.append(StageAnnotation(float(onset), float(duration), stage, str(description)))
        annotations.sort(key=lambda a: a.onset)
        return annotations, report


def summarise_annotations(annotations: list[StageAnnotation]) -> dict[str, Any]:
    """Duration-weighted summary of a hypnogram."""
    counts = Counter(ann.stage for ann in annotations)
    durations: dict[str, float] = {}
    for ann in annotations:
        durations[ann.stage] = durations.get(ann.stage, 0.0) + ann.duration
    total = sum(durations.values()) or 1.0
    return {
        "n_annotations": len(annotations),
        "counts": dict(counts),
        "durations_s": {k: round(v, 3) for k, v in durations.items()},
        "fraction": {k: round(v / total, 6) for k, v in durations.items()},
        "stages_present": [s for s in CANONICAL_STAGES if s in counts],
        "stages_missing": [s for s in CANONICAL_STAGES if s not in counts],
    }
