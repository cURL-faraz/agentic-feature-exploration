"""Filesystem layout helpers.

All paths used by the project are derived from a single project root so that the
pipeline behaves identically whether it is launched from the repository root, a
notebook or the test suite.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["project_root", "resolve_path", "ensure_dir"]


def project_root(start: Path | None = None) -> Path:
    """Return the repository root.

    The root is the first ancestor containing ``pyproject.toml``.  If none is
    found (for example when the package is installed into site-packages), the
    ``EEG_AGENT_PROJECT_ROOT`` environment variable is honoured, and finally the
    current working directory is used.
    """
    env_root = os.environ.get("EEG_AGENT_PROJECT_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    here = (start or Path(__file__)).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return Path.cwd().resolve()


def resolve_path(path: str | os.PathLike[str], root: Path | None = None) -> Path:
    """Resolve ``path`` against the project root unless it is already absolute."""
    path = Path(path)
    if path.is_absolute():
        return path
    return (root or project_root()) / path


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    """Create ``path`` (and parents) and return it as a :class:`Path`."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
