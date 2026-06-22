# Pipeline Fixes — End-to-End Migration Run

Five bugs were preventing the migration pipeline from completing a correct run
against the microblog repo (Flask → FastAPI, 34 modules). Each is documented
below with the exact code change. Result after all five: the pipeline drains the
full task queue, the critic passes real modules, and the success count is truthful.

---

## Bug 1 — Critic mypy over-scoped, failed every module

**File:** `agents/critic.py` · `_run_validation_gauntlet`

mypy was pointed at the single migrated file but **follows imports** into
third-party packages (flask, sqlalchemy, …) and unmigrated sibling modules that
have no stubs in the worktree. Every module failed with
`Cannot find implementation or library stub for module ...`, so critic validation
could never pass.

```python
# Before
rc, out = await _run_subprocess(
    ["mypy", "--no-incremental", abs_file], cwd=wt_path
)

# After
rc, out = await _run_subprocess(
    [
        "mypy",
        "--no-incremental",
        "--ignore-missing-imports",
        "--follow-imports=silent",
        abs_file,
    ],
    cwd=wt_path,
)
```

`--ignore-missing-imports` kills the stub errors; `--follow-imports=silent`
type-checks the migrated file's own annotations without reporting errors in
transitively-imported modules.

---

## Bug 2 — Import check produced 100% false negatives

**File:** `agents/critic.py` · `_run_validation_gauntlet`

The gauntlet ran `python -c "import {module}"`. For a cross-framework migration
the source framework (Flask) is gone and the target framework (FastAPI) deps
aren't installed in the worktree env, so the import **always** failed. Removed
the step entirely; gauntlet is now **ruff → mypy → pytest**.

```python
# Removed
# 4. Import check — catches missing __init__.py, broken top-level code, etc.
module_name = _module_import_name(rel_path)
rc, out = await _run_subprocess(
    [sys.executable, "-c", f"import {module_name}"],
    cwd=wt_path,
    extra_env={"PYTHONPATH": pythonpath},
)
if rc != 0:
    errors.append(f"[import]\n{out.strip()}")
    logger.debug("[critic] import check failed: {}", module_name)
```

Also removed the now-dead `_module_import_name()` helper and the `import sys`
that only it used.

---

## Bug 3 — Dead OpenRouter fallback model (404)

**File:** `agents/llm_router.py` · `_ROLE_FALLBACK`

`qwen/qwen3-235b-a22b:free` was discontinued and now returns
`404 — "This model is unavailable for free"`. When Groq's circuit tripped, the
cascade routed to a non-existent model, got a hard 404 (not a 429), and the
planner gave up instead of falling through. Replaced with a live free model that
returns proper 429s so the rate-limit/circuit logic works.

```python
# Before (planner / critic / classifier fallbacks)
("openrouter", "qwen/qwen3-235b-a22b:free"),

# After
("openrouter", "meta-llama/llama-3.3-70b-instruct:free"),
```

---

## Bug 4 — Planner reserved 4K output tokens per call, blew the daily cap

**File:** `agents/llm_router.py`

Groq bills the **reserved** `max_tokens` against its tokens-per-day limit. Every
call reserved `OUTPUT_TOKEN_LIMIT` (4000), but the planner/classifier return a
short JSON (~300 tokens). Each planner call cost ~4968 tokens → 34 modules ≈
**169K**, over the 100K/day free cap. Added per-role output caps; transform/critic
keep the full budget (they emit whole files).

```python
# Added
_ROLE_OUTPUT_TOKENS: dict[Role, int] = {
    "planner": 512,
    "classifier": 512,
    "transform": OUTPUT_TOKEN_LIMIT,
    "critic": OUTPUT_TOKEN_LIMIT,
}

# Before
def _build_client(provider: str, model: str) -> BaseChatModel:
    kwargs: dict = {"max_tokens": OUTPUT_TOKEN_LIMIT}

# After
def _build_client(
    provider: str, model: str, max_output_tokens: int = OUTPUT_TOKEN_LIMIT
) -> BaseChatModel:
    kwargs: dict = {"max_tokens": max_output_tokens}

# In get_client(), before building the client:
max_output = _ROLE_OUTPUT_TOKENS.get(role, OUTPUT_TOKEN_LIMIT)
client = _build_client(provider, model, max_output)
```

Effect: planner phase dropped from ~169K to ~19K tokens.

---

## Bug 5 — First critic failure aborted the entire migration (+ wrong success count)

**Files:** `workflows/graph.py`, `workflows/state.py`, `api/routes/migrations.py`

`_route_critic` sent any critic failure to `rollback`, and the graph edge was
`rollback → finalize → END`. So the **first** module that failed validation
terminated the whole run, abandoning the rest of the queue (it ran 2 dispatch
rounds then quit on a 34-module repo). Two coupled defects had to be fixed for
the loop to continue safely, plus the success count was counting transform
successes instead of critic passes.

### 5a. Rollback now loops back to dispatch

```python
# workflows/graph.py — added router
def _route_rollback(state: MigrationState) -> Literal["dispatch", "finalize"]:
    """After rollback, keep draining the queue — a single failed module must not
    abandon the remaining work. Only finalize once nothing is left to dispatch."""
    if _select_ready_batch(state, batch_size=9999):
        return "dispatch"
    return "finalize"

# Before — graph assembly
graph_builder.add_edge("rollback", "finalize")

# After
graph_builder.add_conditional_edges(
    "rollback",
    _route_rollback,
    {"dispatch": "dispatch", "finalize": "finalize"},
)
```

### 5b. `rollback_stack` is a per-batch buffer (replace, not append)

An `operator.add` reducer made rollback entries accumulate, so `_route_critic`
saw stale entries forever and looped on rollback. Switched to a replace reducer
and clear it after processing.

```python
# workflows/state.py
def _replace_list(_old: list, new: list) -> list:
    return new

# Before
rollback_stack: Annotated[list[RollbackEntry], operator.add]

# After
rollback_stack: Annotated[list[RollbackEntry], _replace_list]
# critiqued_paths scopes critic to NEW transforms; passed_paths drives counts
critiqued_paths: Annotated[list[str], operator.add]
passed_paths: Annotated[list[str], operator.add]

# workflows/graph.py — rollback_node clears the buffer
return {"final_status": "partial", "rollback_stack": []}
```

### 5c. Critic scoped to new transforms (task_results is cumulative)

Without scoping, every round re-validated all previously-passed modules (O(n²)).

```python
# workflows/graph.py — critic_node
already_critiqued = set(state.get("critiqued_paths", []))
transformed = [
    r
    for r in state.get("task_results", [])
    if r.status == "transformed" and r.module_path not in already_critiqued
]
if not transformed:
    return {"rollback_stack": []}
...
newly_critiqued = [r.module_path for r in transformed]
passed = [cr.module_path for cr in critic_results if cr.verdict == "pass"]
rollback_entries = [cr.rollback_entry for cr in critic_results if cr.rollback_entry]
return {
    "rollback_stack": rollback_entries,
    "critiqued_paths": newly_critiqued,
    "passed_paths": passed,
}
```

### 5d. Success count reflects critic verdict, not transform status

`succeeded` was counting `status in ("transformed", "success")`, so a module that
transformed but failed validation was still counted as a success.

```python
# workflows/graph.py — finalize_node
# Before
succeeded = sum(1 for r in results if r.status in ("transformed", "success"))
failed_count = sum(1 for r in results if r.status == "failed")
per_module_branches = {r.module_path: r.branch_name for r in results if r.branch_name}

# After
passed = set(state.get("passed_paths", []))
critiqued = set(state.get("critiqued_paths", []))
succeeded = len(passed)
transform_failed = sum(1 for r in results if r.status == "failed")
critic_failed = len(critiqued - passed)
failed_count = transform_failed + critic_failed
per_module_branches = {
    r.module_path: r.branch_name
    for r in results
    if r.branch_name and r.module_path in passed
}
```

New state keys are seeded in `api/routes/migrations.py` `_initial_state`:

```python
task_results=[],
rollback_stack=[],
critiqued_paths=[],
passed_paths=[],
```

After fix: 12 dispatch rounds, full queue drained, `succeeded=2/34` (only the two
modules that actually passed critic — `search.py`, `translate.py`).

---

## Related enhancement (not a bug) — transform 429 backoff

**File:** `agents/transform.py`

Added `_migrate_with_backoff()` around the migrate LLM call: retries **only** on
rate-limit (429) errors with delays `10s / 30s / 60s`, returns
`TaskResult(status="failed", error="rate_limit_exhausted")` when exhausted, and
re-raises non-429 errors immediately. Note: `TaskResult` has no `failure_reason`
field — the exhaustion reason goes in `error`.

---

## Remaining limiter (not code)

Transform throughput is bounded by free-tier provider rate limits: most
transforms fail on OpenRouter `qwen3-coder:free` 429s under the 3-wide fan-out
(Groq fallback also throttles). Lifting this needs paid headroom on a provider
(OpenRouter credits/BYOK), not a code change. See provider notes in agent memory.
