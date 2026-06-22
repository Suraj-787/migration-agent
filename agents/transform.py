"""Transform agent: migrates a single source file to the target framework.

Entry point: run_transform(task, spec, run_id, repo_path)
  1. Reads the source file (chunking per AST node if > 500 lines).
  2. Retrieves top-3 migration patterns + top-6 doc chunks + top-3 code chunks.
  3. Constructs a prompt (system rules + few-shot patterns + context + file).
  4. Calls Qwen3 Coder via OpenRouter (262K context window).
  5. Extracts the migrated code from a fenced code block in the response.
  6. Writes output to a new git branch  migration/{run_id}/{module}  via pygit2.
  7. Returns TaskResult with status="transformed".

Large files (>= 500 lines) are split into AST-based segments.  Each segment
is migrated independently with a concurrency cap of 3 (asyncio.Semaphore).
"""
from __future__ import annotations

import asyncio
import difflib
import re
from pathlib import Path
from typing import Any, cast

import pygit2
from langchain_core.messages import HumanMessage, SystemMessage
from langfuse import observe
from loguru import logger

from agents.llm_router import get_router
from rag.chunker import CodeChunker
from rag.models import CodeChunk
from rag.pattern_store import MigrationPattern, PatternStore
from rag.retriever import HybridRetriever, SearchHit
from workflows.state import MigrationSpec, MigrationTask, TaskResult

_CHUNK_LINE_THRESHOLD = 500
_RAG_PATTERN_K = 3
_RAG_DOC_K = 6
_RAG_CODE_K = 3
_MAX_DIFF_LINES = 60

# Limits concurrent Qwen3 Coder calls across all parallel fan-out invocations.
_transform_sem: asyncio.Semaphore | None = None


def _get_sem() -> asyncio.Semaphore:
    """Lazy-init semaphore so it binds to the running event loop."""
    global _transform_sem
    if _transform_sem is None:
        _transform_sem = asyncio.Semaphore(3)
    return _transform_sem


# ---------------------------------------------------------------------------
# Code-block extraction
# ---------------------------------------------------------------------------

_CODE_BLOCK_RE = re.compile(
    r"```[ \t]*(?:python|py)?[ \t]*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)
_BRANCH_UNSAFE_RE = re.compile(r"[^\w\-./]")


def _extract_code_block(response: str) -> str:
    """Return the content of the first fenced code block in *response*.

    Handles both ``\\`\\`\\`python`` and bare ``\\`\\`\\`` variants, plus optional
    trailing spaces after the language tag.

    Raises ``ValueError`` with a descriptive message if no block is found —
    never silently returns the raw response.
    """
    match = _CODE_BLOCK_RE.search(response)
    if match is None:
        raise ValueError(
            f"LLM response contained no fenced code block. "
            f"First 300 chars: {response[:300]!r}"
        )
    return match.group(1)


# ---------------------------------------------------------------------------
# System / human prompt construction
# ---------------------------------------------------------------------------

_WHOLE_FILE_SYSTEM = (
    "You are an expert migration engineer converting {source} code to {target}.\n\n"
    "Migration rules:\n{rules}\n\n"
    "IMPORTANT: output ONLY the complete migrated file content in a single fenced "
    "code block — no explanation, no commentary.\n\n"
    "```python\n# migrated code here\n```"
)

_PREAMBLE_SYSTEM = (
    "You are migrating module-level imports and constants from {source} to {target}.\n"
    "Output ONLY the migrated imports/constants in a fenced code block. "
    "No function or class bodies.\n\n"
    "```python\n# migrated imports and constants\n```"
)

_FUNCTION_SYSTEM = (
    "You are migrating a single function or class from {source} to {target}.\n"
    "Output ONLY the migrated function/class in a fenced code block. "
    "Do NOT include import statements — those are handled separately.\n\n"
    "```python\n# migrated function or class\n```"
)


def _format_rules(spec: MigrationSpec) -> str:
    return (
        f"- Replace all {spec.source_framework} imports with "
        f"{spec.target_framework} equivalents\n"
        f"- Convert synchronous I/O to async where {spec.target_framework} requires it\n"
        "- Adapt decorators, request/response objects, and routing patterns\n"
        "- Preserve all business logic exactly — only change framework-specific code\n"
        "- Keep all docstrings, comments, and existing type hints\n"
        f"- Add return type annotations if missing ({spec.target_framework} convention)\n"
    )


def _fmt_patterns(patterns: list[MigrationPattern]) -> str:
    if not patterns:
        return "(no patterns available)"
    parts = []
    for i, p in enumerate(patterns):
        parts.append(
            f"[Pattern {i + 1}: {p.migration_type}]\n"
            f"Before:\n```python\n{p.before_code}\n```\n"
            f"After:\n```python\n{p.after_code}\n```"
        )
    return "\n\n".join(parts)


def _fmt_hits(hits: list[SearchHit], label: str) -> str:
    if not hits:
        return f"({label}: none retrieved)"
    return "\n\n".join(
        f"[{label} {i + 1}]\n{hit.payload.get('content', '')[:400]}"
        for i, hit in enumerate(hits)
    )


def _whole_file_messages(
    task: MigrationTask,
    spec: MigrationSpec,
    patterns: list[MigrationPattern],
    doc_hits: list[SearchHit],
    code_hits: list[SearchHit],
    content: str,
) -> list[Any]:
    source = f"{spec.source_framework} {spec.source_version}"
    target = f"{spec.target_framework} {spec.target_version}"
    system = _WHOLE_FILE_SYSTEM.format(
        source=source, target=target, rules=_format_rules(spec)
    )
    predicted = "\n".join(f"- {c}" for c in task.predicted_changes)
    human = (
        f"## Task\n{task.description}\n\n"
        f"## Predicted changes\n{predicted}\n\n"
        f"## Migration patterns (few-shot examples)\n{_fmt_patterns(patterns)}\n\n"
        f"## Migration guide excerpts\n{_fmt_hits(doc_hits, 'Doc')}\n\n"
        f"## Similar code examples\n{_fmt_hits(code_hits, 'Code')}\n\n"
        f"## File to migrate: {task.module_path}\n"
        f"```python\n{content}\n```\n\n"
        "Migrate this file. Preserve all behavior. "
        "Update imports, decorators, sync→async if needed."
    )
    return [SystemMessage(content=system), HumanMessage(content=human)]


def _segment_messages(
    seg_type: str,
    seg_content: str,
    task: MigrationTask,
    spec: MigrationSpec,
    patterns: list[MigrationPattern],
    doc_hits: list[SearchHit],
) -> list[Any]:
    source = f"{spec.source_framework} {spec.source_version}"
    target = f"{spec.target_framework} {spec.target_version}"
    if seg_type == "preamble":
        system = _PREAMBLE_SYSTEM.format(source=source, target=target)
    else:
        system = _FUNCTION_SYSTEM.format(source=source, target=target)
    human = (
        f"## Migration: {source} → {target}\n\n"
        f"## Migration patterns\n{_fmt_patterns(patterns)}\n\n"
        f"## Guide excerpts\n{_fmt_hits(doc_hits, 'Doc')}\n\n"
        f"## Code to migrate\n"
        f"```python\n{seg_content}\n```"
    )
    return [SystemMessage(content=system), HumanMessage(content=human)]


# ---------------------------------------------------------------------------
# RAG retrieval for transform context
# ---------------------------------------------------------------------------


async def _retrieve_transform_context(
    task: MigrationTask,
    spec: MigrationSpec,
    content: str,
) -> tuple[list[MigrationPattern], list[SearchHit], list[SearchHit]]:
    """Fetch patterns, doc chunks, and code chunks for the transform prompt.

    All three are empty lists if any retrieval service is unavailable —
    the transform degrades gracefully (like the planner does).
    """
    migration_type = f"{spec.source_framework}→{spec.target_framework}"
    doc_query = (
        f"{spec.source_framework} {spec.source_version} to "
        f"{spec.target_framework} {spec.target_version} migration"
    )
    code_query = (
        f"{spec.source_framework} {spec.target_framework} migration\n"
        + content[:400]
    )
    try:
        pattern_store = PatternStore()
        doc_retriever = HybridRetriever("doc_chunks")
        code_retriever = HybridRetriever("code_chunks")
        patterns, doc_hits, code_hits = await asyncio.gather(
            pattern_store.find_similar(content[:500], migration_type, k=_RAG_PATTERN_K),
            doc_retriever.search(doc_query, top_k=_RAG_DOC_K),
            code_retriever.search(code_query, top_k=_RAG_CODE_K),
        )
        return patterns, doc_hits, code_hits
    except Exception as exc:
        logger.warning(
            "RAG unavailable for transform module={} ({}): {}",
            task.module_path,
            type(exc).__name__,
            exc,
        )
        return [], [], []


# ---------------------------------------------------------------------------
# LLM call (rate-limited by semaphore)
# ---------------------------------------------------------------------------


async def _call_llm(
    messages: list[Any],
    run_id: str,
    session_id: str | None,
    label: str,
) -> tuple[str, int]:
    """Invoke the transform LLM under the concurrency semaphore.

    Returns ``(raw_response_text, total_tokens_used)``.
    """
    router = get_router()
    async with _get_sem():
        client, callbacks = router.get_client(
            "transform",
            session_id=session_id,
            run_id=run_id,
            tags=["transform", f"label:{label}"],
        )
        try:
            response = await client.ainvoke(messages, config={"callbacks": callbacks})
            router.record_success("transform")
        except Exception:
            router.record_rate_limit_error("transform")
            raise

    raw = str(response.content) if hasattr(response, "content") else str(response)
    tokens_used = 0
    if hasattr(response, "response_metadata"):
        usage = response.response_metadata.get("token_usage") or {}
        tokens_used = int(usage.get("total_tokens", 0))
    return raw, tokens_used


# ---------------------------------------------------------------------------
# Whole-file migration
# ---------------------------------------------------------------------------


async def _migrate_whole_file(
    content: str,
    task: MigrationTask,
    spec: MigrationSpec,
    patterns: list[MigrationPattern],
    doc_hits: list[SearchHit],
    code_hits: list[SearchHit],
    run_id: str,
    session_id: str | None,
) -> tuple[str, int]:
    messages = _whole_file_messages(task, spec, patterns, doc_hits, code_hits, content)
    raw, tokens = await _call_llm(
        messages, run_id, session_id, Path(task.module_path).name
    )
    migrated = _extract_code_block(raw)
    return migrated, tokens


# ---------------------------------------------------------------------------
# Chunked migration (files >= 500 lines)
# ---------------------------------------------------------------------------


def _build_segments(
    file_lines: list[str],
    chunks: list[CodeChunk],
) -> list[tuple[str, str]]:
    """Split file lines into (seg_type, content) pairs based on AST chunks.

    - ``"preamble"`` — module-level code before the first function/class.
    - ``"section"``  — gap code immediately before a chunk, plus the chunk body.
    - ``"epilogue"`` — code after the last chunk (if any).

    This groups decorator lines (which sit between functions in the AST gap)
    with the function that follows them, so each section migrates cleanly.
    """
    if not chunks:
        return [("preamble", "".join(file_lines))]

    sorted_chunks = sorted(chunks, key=lambda c: c.start_line)
    segments: list[tuple[str, str]] = []

    # Lines before the first chunk (0-indexed: lines[:first_start])
    first_start = sorted_chunks[0].start_line - 1
    if first_start > 0:
        preamble = "".join(file_lines[:first_start])
        if preamble.strip():
            segments.append(("preamble", preamble))

    prev_end = first_start
    for chunk in sorted_chunks:
        chunk_start = chunk.start_line - 1   # 0-indexed inclusive
        chunk_end = chunk.end_line            # exclusive upper bound

        # Gap between previous chunk's end and this chunk's start (decorators, blanks)
        gap = "".join(file_lines[prev_end:chunk_start])
        body = "".join(file_lines[chunk_start:chunk_end])
        section = gap + body
        if section.strip():
            segments.append(("section", section))
        prev_end = chunk_end

    # Lines after the last chunk
    if prev_end < len(file_lines):
        epilogue = "".join(file_lines[prev_end:])
        if epilogue.strip():
            segments.append(("epilogue", epilogue))

    return segments


async def _migrate_chunked(
    file_path: str,
    content: str,
    task: MigrationTask,
    spec: MigrationSpec,
    patterns: list[MigrationPattern],
    doc_hits: list[SearchHit],
    code_hits: list[SearchHit],
    run_id: str,
    session_id: str | None,
) -> tuple[str, int]:
    """Migrate a large file by processing AST-based segments concurrently.

    Falls back to whole-file migration if the chunker returns nothing.
    """
    chunker = CodeChunker()
    ast_chunks = await asyncio.to_thread(chunker.chunk_file, file_path)
    if not ast_chunks:
        logger.warning(
            "Chunker returned no chunks for {}, falling back to whole-file migration",
            file_path,
        )
        return await _migrate_whole_file(
            content, task, spec, patterns, doc_hits, code_hits, run_id, session_id
        )

    file_lines = content.splitlines(keepends=True)
    segments = _build_segments(file_lines, ast_chunks)
    label = Path(file_path).name
    logger.info(
        "[transform] Chunked migration: file={} segments={}", label, len(segments)
    )

    async def _migrate_one(seg_type: str, seg_content: str, idx: int) -> tuple[str, int]:
        msgs = _segment_messages(seg_type, seg_content, task, spec, patterns, doc_hits)
        raw, tokens = await _call_llm(msgs, run_id, session_id, f"{label}:seg{idx}")
        return _extract_code_block(raw), tokens

    results = await asyncio.gather(
        *[_migrate_one(st, sc, i) for i, (st, sc) in enumerate(segments)]
    )

    migrated_parts = [r[0].rstrip() for r in results]
    total_tokens = sum(r[1] for r in results)
    migrated_content = "\n\n".join(migrated_parts) + "\n"
    return migrated_content, total_tokens


# ---------------------------------------------------------------------------
# Git branch creation via pygit2
# ---------------------------------------------------------------------------


def _insert_into_tree(
    repo: pygit2.Repository,
    tree: pygit2.Tree | None,
    path_parts: list[str],
    blob_id: pygit2.Oid,
) -> pygit2.Oid:
    """Recursively insert a blob at *path_parts* into *tree*.

    Handles arbitrarily nested paths by rebuilding subtrees bottom-up.
    Returns the OID of the new root tree.
    """
    if len(path_parts) == 1:
        tb = repo.TreeBuilder(tree) if tree is not None else repo.TreeBuilder()
        tb.insert(path_parts[0], blob_id, pygit2.GIT_FILEMODE_BLOB)
        return tb.write()

    name, rest = path_parts[0], path_parts[1:]
    subtree: pygit2.Tree | None = None
    if tree is not None:
        try:
            entry = tree[name]
            subtree = cast(pygit2.Tree | None, repo.get(entry.id))
        except KeyError:
            pass

    new_subtree_id = _insert_into_tree(repo, subtree, rest, blob_id)
    tb = repo.TreeBuilder(tree) if tree is not None else repo.TreeBuilder()
    tb.insert(name, new_subtree_id, pygit2.GIT_FILEMODE_TREE)
    return tb.write()


def _write_branch(
    repo_path: str,
    rel_path: str,
    migrated_content: str,
    branch_name: str,
) -> None:
    """Create a new git branch containing the migrated file.

    Does NOT touch HEAD or the working tree — the migration branch is created
    as a lightweight ref pointing at a new commit on top of the current HEAD.
    The original file on main remains untouched.
    """
    repo = pygit2.Repository(repo_path)
    head_commit = repo.head.peel(pygit2.Commit)

    blob_id = repo.create_blob(migrated_content.encode("utf-8"))
    path_parts = list(Path(rel_path).parts)
    new_tree_id = _insert_into_tree(repo, head_commit.tree, path_parts, blob_id)

    sig = pygit2.Signature("Migration Agent", "agent@migration.local")
    repo.create_commit(
        f"refs/heads/{branch_name}",
        sig,
        sig,
        f"[migration] Migrate {rel_path}",
        new_tree_id,
        [head_commit.id],
    )
    logger.info("Created branch '{}' with migrated '{}'", branch_name, rel_path)


def _make_branch_name(run_id: str, rel_path: str) -> str:
    module = rel_path.replace("\\", "/").removesuffix(".py").replace("/", ".")
    safe = _BRANCH_UNSAFE_RE.sub("-", module)
    return f"migration/{run_id}/{safe}"


# ---------------------------------------------------------------------------
# Diff summary
# ---------------------------------------------------------------------------


def _diff_summary(original: str, migrated: str, name: str) -> str:
    lines = list(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            migrated.splitlines(keepends=True),
            fromfile=f"a/{name}",
            tofile=f"b/{name}",
            n=2,
        )
    )
    if not lines:
        return "(no diff)"
    if len(lines) > _MAX_DIFF_LINES:
        lines = lines[:_MAX_DIFF_LINES] + [
            f"... ({len(lines) - _MAX_DIFF_LINES} more lines)\n"
        ]
    return "".join(lines)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

# Backoff schedule for transform LLM 429s. One initial attempt + these delays.
_TRANSFORM_RETRY_DELAYS = (10.0, 30.0, 60.0)


def _is_rate_limit_error(exc: Exception) -> bool:
    """True if *exc* looks like a provider 429 / rate-limit, not a generic failure."""
    msg = str(exc).lower()
    return (
        "429" in msg
        or "rate limit" in msg
        or "rate-limited" in msg
        or "too many requests" in msg
        or "quota" in msg
        or "resource_exhausted" in msg
    )


async def _migrate_with_backoff(
    *,
    chunked: bool,
    module_path: str,
    content: str,
    task: MigrationTask,
    spec: MigrationSpec,
    patterns: Any,
    doc_hits: Any,
    code_hits: Any,
    run_id: str,
    session_id: str | None,
) -> tuple[str, int]:
    """Run the migrate LLM call with exponential backoff on 429s.

    Retries only on rate-limit errors (10s, 30s, 60s); re-raises any other
    exception immediately so the caller can fail fast. Raises the last 429 if
    all retries are exhausted.
    """
    last_exc: Exception | None = None
    for attempt in range(len(_TRANSFORM_RETRY_DELAYS) + 1):
        try:
            if chunked:
                return await _migrate_chunked(
                    module_path, content, task, spec,
                    patterns, doc_hits, code_hits, run_id, session_id,
                )
            return await _migrate_whole_file(
                content, task, spec, patterns, doc_hits, code_hits, run_id, session_id,
            )
        except Exception as exc:
            if not _is_rate_limit_error(exc):
                raise  # non-429 — do not retry
            last_exc = exc
            if attempt < len(_TRANSFORM_RETRY_DELAYS):
                delay = _TRANSFORM_RETRY_DELAYS[attempt]
                logger.warning(
                    "[transform] rate-limited (attempt {}/{}), retrying in {:.0f}s: "
                    "module={} {}",
                    attempt + 1,
                    len(_TRANSFORM_RETRY_DELAYS) + 1,
                    delay,
                    Path(module_path).name,
                    exc,
                )
                await asyncio.sleep(delay)
    raise last_exc  # type: ignore[misc]  # exhausted retries


@observe(name="run_transform")
async def run_transform(
    task: MigrationTask,
    spec: MigrationSpec,
    run_id: str,
    repo_path: str,
    *,
    session_id: str | None = None,
) -> TaskResult:
    """Migrate one source file and write the result to a new git branch.

    Args:
        task:       Which file to migrate (path, complexity, predicted changes).
        spec:       Source/target framework and version.
        run_id:     UUID of the current migration run — used in the branch name.
        repo_path:  Absolute path to the repository root.
        session_id: Langfuse session ID for grouping traces.

    Returns:
        ``TaskResult`` with ``status="transformed"`` on success.
        On failure, ``status="failed"`` with the exception message in ``error``.
    """
    logger.info(
        "[transform] Starting: module={} complexity={}",
        Path(task.module_path).name,
        task.complexity,
    )
    try:
        content = await asyncio.to_thread(
            Path(task.module_path).read_text, encoding="utf-8", errors="replace"
        )
        rel_path = str(Path(task.module_path).relative_to(repo_path))
        branch_name = _make_branch_name(run_id, rel_path)

        patterns, doc_hits, code_hits = await _retrieve_transform_context(
            task, spec, content
        )

        line_count = content.count("\n") + 1
        chunked = line_count >= _CHUNK_LINE_THRESHOLD
        if chunked:
            logger.info(
                "[transform] {} lines ≥ {} → chunked migration",
                line_count,
                _CHUNK_LINE_THRESHOLD,
            )
        try:
            migrated, tokens_used = await _migrate_with_backoff(
                chunked=chunked,
                module_path=task.module_path,
                content=content,
                task=task,
                spec=spec,
                patterns=patterns,
                doc_hits=doc_hits,
                code_hits=code_hits,
                run_id=run_id,
                session_id=session_id,
            )
        except Exception as exc:
            if _is_rate_limit_error(exc):
                logger.error(
                    "[transform] Failed: module={} rate_limit_exhausted after {} attempts",
                    task.module_path,
                    len(_TRANSFORM_RETRY_DELAYS) + 1,
                )
                return TaskResult(
                    module_path=task.module_path,
                    status="failed",
                    error="rate_limit_exhausted",
                )
            raise  # non-429 — handled by the outer except below

        await asyncio.to_thread(
            _write_branch, repo_path, rel_path, migrated, branch_name
        )

        diff = _diff_summary(content, migrated, Path(task.module_path).name)
        logger.info(
            "[transform] Done: branch={} tokens={}",
            branch_name,
            tokens_used,
        )
        return TaskResult(
            module_path=task.module_path,
            status="transformed",
            diff=diff,
            branch_name=branch_name,
            tokens_used=tokens_used,
        )

    except Exception as exc:
        logger.error("[transform] Failed: module={} error={}", task.module_path, exc)
        return TaskResult(
            module_path=task.module_path,
            status="failed",
            error=str(exc),
        )
