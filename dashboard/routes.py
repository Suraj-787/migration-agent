"""Dashboard routes: list runs, per-run module table, diff view, approve, rerun."""
from __future__ import annotations

import asyncio
import base64
import difflib
import os
from pathlib import Path
from typing import Any

import asyncpg
import pygit2
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from loguru import logger
from starlette.responses import Response

router = APIRouter(prefix="/dashboard", tags=["dashboard"])

_TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dsn() -> str:
    return (
        f"postgresql://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}"
        f"@{os.environ.get('POSTGRES_HOST', 'localhost')}"
        f":{os.environ.get('POSTGRES_PORT', '5432')}"
        f"/{os.environ['POSTGRES_DB']}"
    )


def _encode_slug(module_path: str) -> str:
    """URL-safe base64-encode an absolute module path to a URL segment."""
    return base64.urlsafe_b64encode(module_path.encode()).decode().rstrip("=")


def _decode_slug(slug: str) -> str:
    """Decode a slug back to the original module path."""
    padding = 4 - len(slug) % 4
    if padding != 4:
        slug += "=" * padding
    return base64.urlsafe_b64decode(slug.encode()).decode()


async def _fetch_runs() -> list[dict[str, Any]]:
    """Query migration_reports JOIN migration_runs, ordered newest-first."""
    try:
        conn: asyncpg.Connection = await asyncpg.connect(_dsn(), timeout=5.0)
        try:
            rows = await conn.fetch(
                """
                SELECT
                    mr.thread_id,
                    mr.final_status,
                    mr.succeeded,
                    mr.total_tasks,
                    mr.estimated_cost_usd,
                    runs.started_at
                FROM migration_reports mr
                LEFT JOIN migration_runs runs ON mr.thread_id = runs.thread_id
                ORDER BY runs.started_at DESC NULLS LAST
                """
            )
            return [dict(r) for r in rows]
        finally:
            await conn.close()
    except Exception as exc:
        logger.warning("[dashboard] Failed to fetch runs: {}", exc)
        return []


def _build_task_rows(channel_values: dict[str, Any]) -> list[dict[str, Any]]:
    """Build per-module rows by cross-referencing task_queue with task_results."""
    task_queue = channel_values.get("task_queue") or []
    task_results = channel_values.get("task_results") or []
    passed_paths: set[str] = set(channel_values.get("passed_paths") or [])

    result_by_path: dict[str, Any] = {}
    for r in task_results:
        if isinstance(r, dict):
            path = r.get("module_path", "")
            r_status = r.get("status", "")
            tokens = r.get("tokens_used", 0)
            branch = r.get("branch_name")
        else:
            path = getattr(r, "module_path", "")
            r_status = getattr(r, "status", "")
            tokens = getattr(r, "tokens_used", 0)
            branch = getattr(r, "branch_name", None)
        result_by_path[path] = {"status": r_status, "tokens_used": tokens, "branch_name": branch}

    rows = []
    for task in task_queue:
        if isinstance(task, dict):
            task_id = task.get("task_id", "")
            module_path = task.get("module_path", "")
        else:
            task_id = getattr(task, "task_id", "")
            module_path = getattr(task, "module_path", "")

        result = result_by_path.get(module_path)
        if result is None:
            status = "pending"
            tokens_used = 0
            branch_name: str | None = None
        else:
            r_status = result["status"]
            if module_path in passed_paths:
                status = "done"
            elif r_status == "failed":
                status = "failed"
            elif r_status in ("transformed", "skipped"):
                status = "in_progress"
            else:
                status = "pending"
            tokens_used = result["tokens_used"]
            branch_name = result["branch_name"]

        rows.append(
            {
                "task_id": task_id,
                "module_path": module_path,
                "module_name": Path(module_path).name,
                "module_slug": _encode_slug(module_path),
                "status": status,
                "tokens_used": tokens_used,
                "branch_name": branch_name or "—",
                "can_diff": status == "done" and branch_name is not None,
            }
        )
    return rows


async def _load_checkpoint(
    thread_id: str, request: Request
) -> dict[str, Any]:
    """Load channel_values from the LangGraph checkpointer, raise 404/502 on error."""
    checkpointer = request.app.state.checkpointer
    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
    try:
        cp = await checkpointer.aget_tuple(config)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Checkpointer error: {exc}") from exc
    if cp is None:
        raise HTTPException(
            status_code=404, detail=f"No migration found for thread_id={thread_id}"
        )
    return cp.checkpoint.get("channel_values", {})


# ---------------------------------------------------------------------------
# GET /dashboard/
# ---------------------------------------------------------------------------


@router.get("/", response_class=HTMLResponse)
async def runs_list(request: Request) -> Response:
    runs = await _fetch_runs()
    for run in runs:
        run["thread_id_short"] = (run["thread_id"] or "")[:8]
        total = run["total_tasks"] or 0
        succeeded = run["succeeded"] or 0
        run["completion_pct"] = int(succeeded / total * 100) if total > 0 else 0
        started = run.get("started_at")
        run["started_at_str"] = (
            started.strftime("%Y-%m-%d %H:%M UTC") if started else "—"
        )
    return templates.TemplateResponse(request, "runs_list.html", {"runs": runs})


# ---------------------------------------------------------------------------
# GET /dashboard/runs/{thread_id}
# ---------------------------------------------------------------------------


@router.get("/runs/{thread_id}", response_class=HTMLResponse)
async def run_detail(
    thread_id: str, request: Request, flash: str | None = None
) -> Response:
    channel_values = await _load_checkpoint(thread_id, request)
    tasks = _build_task_rows(channel_values)
    has_failed = any(t["status"] == "failed" for t in tasks)
    return templates.TemplateResponse(
        request,
        "run_detail.html",
        {
            "thread_id": thread_id,
            "thread_id_short": thread_id[:8],
            "tasks": tasks,
            "flash": flash,
            "has_failed": has_failed,
            "repo_path": channel_values.get("repo_path", ""),
            "final_status": channel_values.get("final_status", "pending"),
        },
    )


# ---------------------------------------------------------------------------
# GET /dashboard/runs/{thread_id}/stream  — SSE event log page
# ---------------------------------------------------------------------------


@router.get("/runs/{thread_id}/stream", response_class=HTMLResponse)
async def run_stream_page(thread_id: str, request: Request) -> Response:
    """Renders an event-log page that connects to the migration SSE stream."""
    return templates.TemplateResponse(
        request,
        "run_detail.html",
        {
            "thread_id": thread_id,
            "thread_id_short": thread_id[:8],
            "tasks": [],
            "flash": None,
            "has_failed": False,
            "repo_path": "",
            "final_status": "pending",
            "show_stream": True,
        },
    )


# ---------------------------------------------------------------------------
# GET /dashboard/runs/{thread_id}/diff/{module_slug}
# ---------------------------------------------------------------------------


@router.get("/runs/{thread_id}/diff/{module_slug}", response_class=HTMLResponse)
async def diff_view(
    thread_id: str, module_slug: str, request: Request
) -> Response:
    try:
        module_path = _decode_slug(module_slug)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid module_slug: {exc}") from exc

    channel_values = await _load_checkpoint(thread_id, request)
    repo_path: str = channel_values.get("repo_path", "")

    # Locate branch_name for this module in task_results
    task_results = channel_values.get("task_results") or []
    branch_name: str | None = None
    for r in task_results:
        if isinstance(r, dict):
            path = r.get("module_path", "")
            bn = r.get("branch_name")
        else:
            path = getattr(r, "module_path", "")
            bn = getattr(r, "branch_name", None)
        if path == module_path:
            branch_name = bn
            break

    if not branch_name:
        raise HTTPException(
            status_code=404,
            detail=f"No migration branch found for {module_path}",
        )

    # Read original file from disk
    try:
        original_content = Path(module_path).read_text(encoding="utf-8")
    except OSError as exc:
        raise HTTPException(
            status_code=404, detail=f"Original file not found: {exc}"
        ) from exc

    # Read migrated file via pygit2
    try:
        repo = pygit2.Repository(repo_path)
        source_ref = repo.references[f"refs/heads/{branch_name}"]
        source_commit = source_ref.peel(pygit2.Commit)
        rel_path = str(Path(module_path).relative_to(repo_path))
        blob: pygit2.Blob = source_commit.tree / rel_path  # type: ignore[assignment]
        migrated_content = blob.data.decode("utf-8")
    except KeyError:
        raise HTTPException(
            status_code=404, detail=f"Branch '{branch_name}' not found in repo"
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed to read migrated file: {exc}"
        ) from exc

    diff_table = difflib.HtmlDiff(wrapcolumn=80).make_table(
        original_content.splitlines(),
        migrated_content.splitlines(),
        fromdesc=f"original/{Path(module_path).name}",
        todesc=f"{branch_name}",
    )

    return templates.TemplateResponse(
        request,
        "diff_view.html",
        {
            "thread_id": thread_id,
            "module_path": module_path,
            "module_name": Path(module_path).name,
            "branch_name": branch_name,
            "module_slug": module_slug,
            "diff_table": diff_table,
        },
    )


# ---------------------------------------------------------------------------
# POST /dashboard/runs/{thread_id}/approve/{module_slug}
# ---------------------------------------------------------------------------


@router.post("/runs/{thread_id}/approve/{module_slug}")
async def approve_module(
    thread_id: str, module_slug: str, request: Request
) -> RedirectResponse:
    """Merge the module's migration branch into migration-staging via pygit2."""
    try:
        module_path = _decode_slug(module_slug)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid module_slug: {exc}") from exc

    channel_values = await _load_checkpoint(thread_id, request)
    repo_path: str = channel_values.get("repo_path", "")

    # Locate branch_name for this module
    task_results = channel_values.get("task_results") or []
    branch_name: str | None = None
    for r in task_results:
        if isinstance(r, dict):
            path, bn = r.get("module_path", ""), r.get("branch_name")
        else:
            path, bn = getattr(r, "module_path", ""), getattr(r, "branch_name", None)
        if path == module_path:
            branch_name = bn
            break

    if not branch_name:
        raise HTTPException(
            status_code=404, detail=f"No migration branch for {module_path}"
        )

    try:
        repo = pygit2.Repository(repo_path)

        # Ensure migration-staging exists; seed from HEAD if not
        staging_ref_name = "refs/heads/migration-staging"
        if staging_ref_name not in repo.references:
            head_commit = repo.head.peel(pygit2.Commit)
            repo.create_branch("migration-staging", head_commit, False)

        staging_ref = repo.references[staging_ref_name]
        staging_commit = staging_ref.peel(pygit2.Commit)

        try:
            source_ref = repo.references[f"refs/heads/{branch_name}"]
        except KeyError:
            raise HTTPException(
                status_code=404, detail=f"Branch '{branch_name}' not found"
            )
        source_commit = source_ref.peel(pygit2.Commit)

        # Determine merge strategy using merge base
        try:
            merge_base_oid = repo.merge_base(source_commit.id, staging_commit.id)
        except Exception:
            merge_base_oid = None

        if merge_base_oid == source_commit.id:
            flash_msg = "Module+already+merged+into+staging"
        elif merge_base_oid == staging_commit.id:
            # Fast-forward
            staging_ref.set_target(source_commit.id)
            logger.info(
                "[dashboard] Fast-forward '{}' → migration-staging", branch_name
            )
            flash_msg = "Module+approved+and+merged+to+staging"
        else:
            # Three-way merge commit
            merge_index = repo.merge_commits(staging_commit, source_commit)
            if merge_index.conflicts:
                raise HTTPException(
                    status_code=409,
                    detail="Merge conflicts — resolve manually before approving",
                )
            tree_id = merge_index.write_tree(repo)
            sig = pygit2.Signature("Migration Agent", "agent@migration-agent.local")
            repo.create_commit(
                staging_ref_name,
                sig,
                sig,
                f"Merge '{branch_name}' into migration-staging",
                tree_id,
                [staging_commit.id, source_commit.id],
            )
            logger.info(
                "[dashboard] Merge commit created '{}' → migration-staging", branch_name
            )
            flash_msg = "Module+approved+and+merged+to+staging"

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("[dashboard] Approve failed: {}", exc)
        raise HTTPException(status_code=500, detail=f"Merge failed: {exc}") from exc

    return RedirectResponse(
        url=f"/dashboard/runs/{thread_id}?flash={flash_msg}", status_code=303
    )


# ---------------------------------------------------------------------------
# POST /dashboard/runs/{thread_id}/rerun
# ---------------------------------------------------------------------------


@router.post("/runs/{thread_id}/rerun")
async def rerun_failed(thread_id: str, request: Request) -> RedirectResponse:
    """Reset failed tasks to pending, re-push to queue, re-invoke the graph."""
    channel_values = await _load_checkpoint(thread_id, request)

    task_queue = channel_values.get("task_queue") or []
    task_results = channel_values.get("task_results") or []

    failed_paths = {
        (r.get("module_path") if isinstance(r, dict) else getattr(r, "module_path", ""))
        for r in task_results
        if (r.get("status") if isinstance(r, dict) else getattr(r, "status", "")) == "failed"
    }

    if not failed_paths:
        return RedirectResponse(
            url=f"/dashboard/runs/{thread_id}?flash=No+failed+modules+to+rerun",
            status_code=303,
        )

    from workflows.state import MigrationTask

    reset_tasks: list[MigrationTask] = [
        MigrationTask.model_validate(task) if isinstance(task, dict) else task
        for task in task_queue
        if (task.get("module_path") if isinstance(task, dict) else getattr(task, "module_path", ""))
        in failed_paths
    ]

    # Re-push to Redis TaskQueue
    run_id = str(channel_values.get("dep_graph_id", ""))
    if run_id:
        from workflows.queue import TaskQueue as TQ

        q = TQ(run_id)
        for task in reset_tasks:
            await q.push(task)
        logger.info(
            "[dashboard] Re-pushed {} tasks to queue run_id={}", len(reset_tasks), run_id
        )

    # Fire re-invocation in background — resumes from checkpointed state
    graph = request.app.state.graph
    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}

    async def _rerun() -> None:
        try:
            await graph.ainvoke({"task_queue": reset_tasks}, config=config)
            logger.info("[dashboard] Rerun complete thread_id={}", thread_id)
        except Exception as exc:
            logger.error("[dashboard] Rerun failed thread_id={}: {}", thread_id, exc)

    bg_task: asyncio.Task[None] = asyncio.create_task(_rerun())
    active_tasks: set[asyncio.Task[None]] = getattr(
        request.app.state, "active_tasks", set()
    )
    active_tasks.add(bg_task)
    bg_task.add_done_callback(active_tasks.discard)

    return RedirectResponse(
        url=(
            f"/dashboard/runs/{thread_id}"
            f"?flash=Rerun+triggered+for+{len(reset_tasks)}+failed+module(s)"
        ),
        status_code=303,
    )
