"""``reports/validation_report.md``.

The verifier's own report.  It is written from
``artifacts/tables/verification_snapshot.json`` (a copy of the verification stage
output) so that the document can never disagree with the recorded checks.

The sections map one-to-one onto the adversarial questions the brief asks:
are the equations right, does signal processing respond as expected, are the
statistics calibrated, is the machine-learning evaluation leak-free, and is the
evidence traceable.
"""

from __future__ import annotations

from typing import Any

from .context import ReportContext
from .markdown_utils import heading, render_key_values, render_list, render_table

__all__ = ["write_validation_report"]

SOURCE = "tables/verification_snapshot.json"

SECTION_TITLES = {
    "synthetic_signals": "Synthetic-signal checks (signal processing)",
    "feature_mathematics": "Feature mathematics",
    "statistics": "Statistical validation",
    "machine_learning": "Machine-learning validation",
    "leakage": "Data-leakage audit",
    "evidence": "Evidence and citation integrity",
}


def _check_rows(section: dict[str, Any], limit: int = 40) -> list[dict[str, Any]]:
    rows = []
    for check in (section.get("checks") or [])[:limit]:
        detail = check.get("detail")
        if isinstance(detail, dict):
            detail = "; ".join(f"{key}={value}" for key, value in detail.items() if key != "ok")
        rows.append(
            {
                "check": check.get("name"),
                "passed": check.get("passed"),
                "detail": detail,
            }
        )
    return rows


def write_validation_report(context: ReportContext) -> str:
    verification = context.verification
    sections = verification.get("sections") or {}
    ok = verification.get("ok")

    if verification.get("n_checks") is not None:
        context.add_claim(
            "verification.ok",
            "overall verification verdict (true means every section passed)",
            bool(ok),
            "ok",
            SOURCE,
        )
        context.add_claim(
            "verification.n_checks",
            "number of individual verification checks executed",
            verification.get("n_checks"),
            "n_checks",
            SOURCE,
            unit="checks",
        )
        context.add_claim(
            "verification.n_failed_checks",
            "number of verification checks that failed",
            verification.get("n_failed_checks"),
            "n_failed_checks",
            SOURCE,
            unit="checks",
        )

    lines: list[str] = [
        heading(1, "Verification and Validation Report"),
        context.origin_caveat(),
        heading(2, "Verdict"),
        render_key_values(
            {
                "overall": "PASS" if ok else "FAIL",
                "checks executed": verification.get("n_checks"),
                "checks failed": verification.get("n_failed_checks"),
                "failed sections": verification.get("failed_checks") or [],
                "data origin": verification.get("data_origin"),
            }
        ),
        (
            "A failing check is not hidden: it is recorded here, the affected conclusion is "
            "downgraded in the final report, and nothing that failed verification is presented "
            "as a validated finding."
        ),
    ]
    if not sections:
        lines.append(
            "_No verification snapshot was available when this report was generated. "
            "Run `python -m eeg_research_agent.cli verify` first._"
        )
        return "\n".join(lines).strip() + "\n"

    for key, title in SECTION_TITLES.items():
        section = sections.get(key)
        if not section:
            continue
        lines += [
            heading(2, title),
            render_key_values(
                {
                    "status": "PASS" if section.get("ok") else "FAIL",
                    "checks": len(section.get("checks") or []),
                    "failed": section.get("failed_checks") or [],
                }
            ),
        ]
        if key == "evidence":
            checks = section.get("checks") or []
            for check in checks:
                detail = check.get("detail")
                if not isinstance(detail, dict):
                    continue
                lines.append(heading(3, str(check.get("name"))))
                lines.append(
                    render_key_values(
                        {
                            k: ("PASS" if v is True else "FAIL" if v is False else v)
                            for k, v in detail.items()
                            if k
                            not in {
                                "checked",
                                "problems",
                                "failures",
                                "incomplete",
                                "incomplete_unverified",
                                "unverified_offline_titles",
                            }
                        }
                    )
                )
                failures = (
                    detail.get("failures")
                    or detail.get("problems")
                    or detail.get("incomplete")
                    or detail.get("incomplete_unverified")
                )
                if failures:
                    lines.append(render_table(failures[:20]))
        else:
            rows = _check_rows(section)
            if rows:
                lines.append(render_table(rows, columns=["check", "passed", "detail"]))
        failed = section.get("failed_checks") or []
        if failed:
            lines.append(f"Failed checks: {', '.join(str(name) for name in failed)}")

    lines += [
        heading(2, "What these checks do not establish"),
        render_list(
            [
                "Passing synthetic-signal checks proves the implementation behaves as specified, "
                "not that the feature is a sleep biomarker.",
                "Passing the leakage audit proves the recorded split and selection scopes are sound; "
                "it cannot prove that unrecorded preprocessing was also leak-free.",
                "Passing the claim check proves the report matches the saved artifacts, not that the "
                "underlying measurements are correct.",
                "Citation validation checks structure and traceability. When retrieval was offline, "
                "no source was verified against a publisher and the records say so.",
            ]
        ),
    ]
    return "\n".join(lines).strip() + "\n"
