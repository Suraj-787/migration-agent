import os
from collections.abc import Iterator
from pathlib import Path

import pathspec
from loguru import logger

_SKIP_DIRS: frozenset[str] = frozenset(
    {
        "node_modules",
        ".venv",
        "venv",
        "env",
        "__pycache__",
        "dist",
        "build",
        ".git",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
    }
)

_SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".py", ".java", ".cs", ".ts", ".tsx", ".js", ".jsx", ".go"}
)


def _load_gitignore_spec(root: Path) -> pathspec.PathSpec | None:
    gitignore = root / ".gitignore"
    if gitignore.exists():
        lines = gitignore.read_text(encoding="utf-8", errors="replace").splitlines()
        return pathspec.PathSpec.from_lines("gitwildmatch", lines)
    return None


def walk_repo(repo_root: str) -> Iterator[str]:
    """Yield absolute paths of all supported source files under repo_root.

    Respects .gitignore and skips common noise directories.
    """
    root = Path(repo_root).resolve()
    if not root.is_dir():
        logger.warning("walk_repo: {} is not a directory", repo_root)
        return

    spec = _load_gitignore_spec(root)

    for dirpath, dirnames, filenames in os.walk(root):
        # Prune noise dirs in-place so os.walk skips their subtrees.
        dirnames[:] = [
            d for d in dirnames if d not in _SKIP_DIRS and not d.startswith(".")
        ]

        for filename in sorted(filenames):
            full = Path(dirpath) / filename
            if full.suffix.lower() not in _SUPPORTED_EXTENSIONS:
                continue
            rel = full.relative_to(root)
            if spec and spec.match_file(str(rel)):
                continue
            yield str(full)
