"""Benchmark dataset: repos with known migration targets.

Each BenchmarkRepo specifies either a remote URL to clone or a local path.
Call repo.resolve_path(project_root) to get an absolute path suitable for
passing to POST /migrations as repo_path.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class BenchmarkRepo:
    name: str
    migration_type: str
    url: str | None = None
    # Absolute path or path relative to the project root.
    local_path: str | None = None
    # If set, use this subdirectory within the cloned/local repo as repo_path.
    subdir: str | None = None

    def resolve_path(self, project_root: Path) -> str | None:
        """Return the absolute path to the repo (or None for remote-only repos)."""
        if self.local_path is None:
            return None
        p = Path(self.local_path)
        resolved = p if p.is_absolute() else (project_root / p).resolve()
        if self.subdir:
            resolved = resolved / self.subdir
        return str(resolved)


# ---------------------------------------------------------------------------
# Benchmark dataset — 5 repos
# ---------------------------------------------------------------------------

REPOS: list[BenchmarkRepo] = [
    # 1. Flask → FastAPI (remote)
    BenchmarkRepo(
        name="flask-realworld",
        url="https://github.com/mjhea0/flask-realworld-example-app",
        migration_type="flask_to_fastapi",
    ),
    # 2. Python 2 → 3 (remote, subdirectory)
    BenchmarkRepo(
        name="realpython-py2-scripts",
        url="https://github.com/realpython/materials",
        migration_type="python2_to_python3",
        subdir="python2-scripts",
    ),
    # 3. Django FBV → CBV (remote)
    BenchmarkRepo(
        name="simple-django-login",
        url="https://github.com/sibtc/simple-django-login-and-register",
        migration_type="django_fbv_to_cbv",
    ),
    # 4. Microblog — local Flask app (already cloned)
    BenchmarkRepo(
        name="microblog",
        local_path="/Users/suraj/Desktop/microblog",
        migration_type="flask_to_fastapi",
    ),
    # 5. sample_flask_app fixture — known-good baseline
    BenchmarkRepo(
        name="sample-flask-app",
        local_path="tests/fixtures/sample_flask_app",
        migration_type="flask_to_fastapi",
    ),
]
