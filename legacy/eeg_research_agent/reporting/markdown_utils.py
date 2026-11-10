"""Small markdown rendering helpers.

Deliberately dependency-free (``tabulate`` is not available in the offline
environment) and deterministic: identical inputs always render byte-identical
markdown, which makes the reports diffable and cacheable.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "format_value",
    "render_table",
    "render_key_values",
    "render_list",
    "heading",
    "render_records",
    "escape_pipes",
    "normalise_markdown",
]


def escape_pipes(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def format_value(value: Any) -> str:
    """Format a value for a markdown table without losing precision."""
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if value != value:  # NaN
            return "n/a"
        if value != 0 and (abs(value) < 1e-3 or abs(value) >= 1e5):
            return f"{value:.3e}"
        return f"{value:.4f}"
    if isinstance(value, (list, tuple)):
        return ", ".join(f"{format_value(item)}" for item in value) if value else "none"
    if isinstance(value, Mapping):
        return ", ".join(f"{key}={format_value(item)}" for key, item in value.items())
    return escape_pipes(value)


def render_table(rows: Iterable[Mapping[str, Any]], columns: Sequence[str] | None = None) -> str:
    """Render a list of mappings as a GitHub-flavoured markdown table."""
    rows = [dict(row) for row in rows]
    if not rows:
        return "_No rows._"
    if columns is None:
        columns = list(rows[0].keys())
    header = "| " + " | ".join(str(column) for column in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(format_value(row.get(column)) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def render_key_values(values: Mapping[str, Any], key_header: str = "Field") -> str:
    rows = [{key_header: key, "Value": value} for key, value in values.items()]
    return render_table(rows, columns=[key_header, "Value"])


def render_list(items: Iterable[Any], bullet: str = "-") -> str:
    items = list(items)
    if not items:
        return "_None._"
    return "\n".join(f"{bullet} {format_value(item)}" for item in items)


def heading(level: int, text: str) -> str:
    return f"{'#' * max(1, min(6, level))} {text}"


def render_records(records: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> str:
    """Render each record as a small heading plus a key/value table."""
    blocks: list[str] = []
    for index, record in enumerate(records, start=1):
        title = record.get("title") or record.get("name") or f"Record {index}"
        blocks.append(heading(4, f"{index}. {escape_pipes(title)}"))
        blocks.append(render_key_values({field: record.get(field) for field in fields}))
    return "\n\n".join(blocks) if blocks else "_No records._"


def normalise_markdown(text: str) -> str:
    """Make block-level spacing explicit.

    GitHub-flavoured markdown needs a blank line around a table; the report
    writers assemble content as line lists, so the spacing is normalised once,
    deterministically, instead of being hand-tuned at every call site.  Headings
    and blockquotes also get a preceding blank line.  Bullet lists are
    unaffected because only those block types and blank-line runs change.
    """
    out: list[str] = []
    for line in text.split("\n"):
        is_table = line.startswith("|")
        previous = out[-1] if out else ""
        starts_block = line.startswith("#") or line.startswith(">")
        if starts_block and previous.strip():
            out.append("")
        elif is_table and previous.strip() and not previous.startswith("|"):
            out.append("")
        elif not is_table and previous.startswith("|") and line.strip():
            out.append("")
        out.append(line.rstrip())
    collapsed: list[str] = []
    for line in out:
        if not line and collapsed and not collapsed[-1]:
            continue
        collapsed.append(line)
    return "\n".join(collapsed).strip() + "\n"
