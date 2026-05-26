"""Path helpers for local task runs."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


def project_root() -> Path:
    """Return the repository root."""

    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "README.md").exists() and (parent / "serves.json").exists():
            return parent
    return current.parents[6]


def resolve_repo_path(relative_path: str | Path) -> Path:
    """Resolve a repository-relative path."""

    path = Path(relative_path)
    if path.is_absolute():
        return path
    return (project_root() / path).resolve()


def _normalize_fragment(value: str, fallback: str) -> str:
    fragment = value.strip().replace(" ", "_").replace("/", "-").replace("\\", "-")
    return fragment or fallback


def prepare_run_dir(task_id: str, run_name: str = "") -> Path:
    """Create a stable run directory under runs/tasks/."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    task_fragment = _normalize_fragment(task_id, "task")
    run_fragment = _normalize_fragment(run_name, "")
    leaf = f"{timestamp}_{run_fragment}" if run_fragment else timestamp
    run_dir = project_root() / "runs" / "tasks" / task_fragment / leaf
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
