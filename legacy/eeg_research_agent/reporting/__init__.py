"""Reproducible report generation.

``markdown_utils``   dependency-free markdown rendering helpers
``plots``            every figure, derived from saved results
``context``          loads the artifacts a report may quote + the claim register
``literature_report````literature_review.md``
``feature_catalog``  ``feature_catalog.md``
``statistical_report``  ``statistical_analysis.md``
``validation_report``   ``validation_report.md``
``final_report``     ``final_research_report.md`` and :func:`generate_all_reports`

Reports are regenerable from saved intermediate results alone: no stage of report
generation re-runs signal processing, statistics or model fitting.
"""

from .context import ReportContext, load_report_context
from .feature_catalog import write_feature_catalog
from .final_report import generate_all_reports, write_final_report
from .literature_report import write_literature_review
from .plots import generate_figures
from .statistical_report import write_statistical_report
from .validation_report import write_validation_report

__all__ = [
    "ReportContext",
    "load_report_context",
    "generate_all_reports",
    "generate_figures",
    "write_literature_review",
    "write_feature_catalog",
    "write_statistical_report",
    "write_validation_report",
    "write_final_report",
]
