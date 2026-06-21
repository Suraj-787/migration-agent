"""Tests for CodeChunker and walk_repo using the sample_flask_app fixture."""

from pathlib import Path

import pytest

from rag.chunker import CodeChunker
from rag.models import CodeChunk
from rag.repo_walker import walk_repo

FIXTURE_DIR = Path(__file__).parent.parent / "tests" / "fixtures" / "sample_flask_app"


@pytest.fixture(scope="module")
def chunker() -> CodeChunker:
    return CodeChunker()


@pytest.fixture(scope="module")
def all_chunks(chunker: CodeChunker) -> list[CodeChunk]:
    chunks: list[CodeChunk] = []
    for source_file in walk_repo(str(FIXTURE_DIR)):
        chunks.extend(chunker.chunk_file(source_file))
    return chunks


# ── basic volume ──────────────────────────────────────────────────────────────

def test_total_chunks_exceeds_five(all_chunks: list[CodeChunk]) -> None:
    assert len(all_chunks) > 5, f"Expected >5 chunks, got {len(all_chunks)}"


# ── parent context is always populated ───────────────────────────────────────

def test_every_chunk_has_non_empty_parent_context(all_chunks: list[CodeChunk]) -> None:
    for chunk in all_chunks:
        assert chunk.parent_context, (
            f"Empty parent_context on chunk {chunk.file_path}:{chunk.start_line}"
        )


# ── size constraint ───────────────────────────────────────────────────────────

def test_chunks_fit_under_1500_chars(all_chunks: list[CodeChunk]) -> None:
    # ASTChunk measures size in non-whitespace chars; verify that metric
    for chunk in all_chunks:
        nws = len(chunk.content.replace(" ", "").replace("\n", "").replace("\t", ""))
        assert nws <= 1500, (
            f"Chunk too large ({nws} non-ws chars) in "
            f"{chunk.file_path}:{chunk.start_line}-{chunk.end_line}"
        )


# ── required fields ───────────────────────────────────────────────────────────

def test_chunks_have_required_fields(all_chunks: list[CodeChunk]) -> None:
    for chunk in all_chunks:
        assert chunk.file_path
        assert chunk.language == "python"
        assert chunk.node_type in {"function", "class", "import", "code"}
        assert chunk.start_line >= 0
        assert chunk.end_line >= chunk.start_line


# ── per-file chunking ─────────────────────────────────────────────────────────

def test_chunk_file_produces_results(chunker: CodeChunker) -> None:
    app_py = FIXTURE_DIR / "app.py"
    chunks = chunker.chunk_file(str(app_py))
    assert len(chunks) > 0


def test_models_file_extracts_docstrings(chunker: CodeChunker) -> None:
    models_py = FIXTURE_DIR / "models.py"
    chunks = chunker.chunk_file(str(models_py))
    docstrings = [c.docstring for c in chunks if c.docstring]
    assert docstrings, "Expected at least one chunk with a docstring in models.py"


def test_config_file_detects_classes(chunker: CodeChunker) -> None:
    config_py = FIXTURE_DIR / "config.py"
    chunks = chunker.chunk_file(str(config_py))
    class_chunks = [c for c in chunks if c.node_type == "class"]
    assert class_chunks, "Expected class-type chunks from config.py"


# ── walk_repo integration ─────────────────────────────────────────────────────

def test_walk_repo_finds_all_python_files() -> None:
    found = list(walk_repo(str(FIXTURE_DIR)))
    py_files = [f for f in found if f.endswith(".py")]
    assert len(py_files) >= 3, f"Expected >=3 .py files, got {py_files}"


def test_walk_repo_skips_noise_dirs(tmp_path: Path) -> None:
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "skip_me.py").write_text("x = 1")
    (tmp_path / "real.py").write_text("x = 1")
    found = list(walk_repo(str(tmp_path)))
    assert all("__pycache__" not in f for f in found)
    assert any("real.py" in f for f in found)


def test_unsupported_extension_returns_empty(chunker: CodeChunker, tmp_path: Path) -> None:
    go_file = tmp_path / "main.go"
    go_file.write_text('package main\nfunc main() {}\n')
    assert chunker.chunk_file(str(go_file)) == []
