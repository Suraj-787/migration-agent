# Autonomous Codebase Migration Agent — Complete Build Plan

A production-grade, 4-week build plan optimized for Claude Code workflows and zero-budget free tiers (no credit card required for the entire core stack).

---

## Section 1 — Tech Stack & Why Each Choice (Free-Tier Optimized, June 2026)

Every choice below is verified for free-tier viability in 2026. Where a choice has tradeoffs, alternatives are listed.

### Core stack

| Layer | Choice | Why this & not the alternative |
|---|---|---|
| Agent orchestration | **LangGraph** (OSS, MIT) | Largest production footprint in 2026, native checkpointing (acts as your "durable workflow" engine without needing Temporal), stateful graph model fits planner→worker→critic patterns. CrewAI is easier but trails on observability and recovery. |
| LLM for agents (primary) | **Google Gemini 2.5 Flash** via AI Studio | 1,500 requests/day + 1M TPM free, no card required. Worth ~$20–40/month in equivalent paid usage. |
| LLM for agents (fallback / coding) | **Qwen3 Coder 480B** via OpenRouter `:free` variant | Strongest free coding model with 262K context. Use for the Transform agent specifically. |
| LLM for agents (speed lane) | **Groq** with Llama 3.3 70B | 30 RPM, sub-200ms TTFT. Use for query router and short classification calls. |
| Code embeddings | **voyage-code-3** | 200M tokens free per account, beats OpenAI text-embedding-3-large by 13.8% on code retrieval. Matryoshka dimensions (256–2048) reduce vector DB cost. |
| AST code chunker | **ASTChunk** (from CMU `cAST` paper) | Recursive split-then-merge on tree-sitter ASTs. +4.3 points Recall@5 on RepoEval vs naive chunking. Supports Python, Java, C#, TypeScript out of the box. |
| Vector database | **Qdrant** self-hosted (Apache 2.0) | Native sparse + dense hybrid, prefilter correctness (Chroma post-filters which breaks at scale), runs in a single Docker container. |
| Reranker | **BAAI/bge-reranker-v2-m3** via FastEmbed | Free, runs on CPU, multilingual, comparable to Cohere Rerank quality. Wrap with the `rerankers` library for swap-friendly API. |
| Durable workflows | **LangGraph checkpointer + Postgres** | Skip Temporal initially — LangGraph's built-in PostgresCheckpointer gives you workflow persistence, pause/resume, and full state history without running a separate Temporal cluster. |
| Backend API | **FastAPI** + **uvicorn** | Native async, OpenAPI docs for free, fits the LangGraph async model. |
| Relational DB | **PostgreSQL 16** | Stores run metadata, task state, audit logs, dependency graphs, eval results. Also hosts the LangGraph checkpointer. |
| Cache / locks | **Redis 7** | Module-level locks during parallel agent execution, task queue if needed. |
| LLM observability | **Langfuse** self-hosted (MIT) | Single Docker container, OpenTelemetry-based so framework-agnostic. LangSmith free tier is only 5K traces/month and self-hosting needs Enterprise license. |
| Frontend dashboard | **FastAPI + HTMX + Tailwind via CDN** | Zero build step, server-rendered, real-time updates via SSE. Skip React for an internal tool. |
| Containerization | **Docker Compose** | Everything runs locally first; the same compose file deploys to a Hetzner VPS when ready. |

### Free-tier accounts you'll create (all no-card-required for the dev path)

1. **Google AI Studio** → `GEMINI_API_KEY` (1,500 RPD)
2. **OpenRouter** → `OPENROUTER_API_KEY` (free models, 20 RPM)
3. **Groq Cloud** → `GROQ_API_KEY` (30 RPM Llama 3.3 70B)
4. **Voyage AI** → `VOYAGE_API_KEY` (200M code-embedding tokens free)
5. **GitHub** → for cloning legacy repos to migrate (you already have)

### Optional paid path (only when you're ready to deploy publicly)

- **Hetzner Cloud CX23**: €3.49/month for the deployed VPS
- **Northflank free tier**: real free tier with no spin-down if you prefer managed
- Skip Railway/Render — they require cards and the free tier expires

### What you're NOT using (and why)

- **Temporal.io** — operationally heavy for a solo 4-week project; LangGraph's checkpointer covers 90% of what you need
- **LangSmith** — closed-source, 5K traces/month free, self-host needs Enterprise license
- **Cohere Rerank** — paid; BGE reranker matches it for free
- **Pinecone / Weaviate Cloud** — paid at scale; Qdrant self-hosted beats both on price
- **OpenAI embeddings** — voyage-code-3 is both free (200M tokens) and better for code

---

## Section 2 — Project Setup (Day 0, ~1 hour)

Do this all yourself in the terminal before you touch Claude Code.

### 2.1 Create the repo

```bash
mkdir migration-agent && cd migration-agent
git init
mkdir -p agents rag workflows api dashboard eval benchmarks tests docker
touch CLAUDE.md README.md .env.example .gitignore docker-compose.yml pyproject.toml
```

### 2.2 The root `CLAUDE.md` (most important file in the project)

This is the briefing document Claude Code reads at the start of every session. Treat it like a production prompt. Paste exactly this content:

```markdown
# Migration Agent — Project Brief

## What this is
A multi-agent system that autonomously migrates legacy codebases between framework versions
(e.g., Flask → FastAPI, Python 2 → 3, React 16 → 18). Specialist agents handle transform,
dependency, tests, and type-checking. A critic agent validates and rolls back failures.

## Stack (do not deviate without asking)
- Python 3.11+, async/await throughout
- FastAPI for the API layer
- LangGraph for agent orchestration (with PostgresCheckpointer for durability)
- Qdrant for vector store (hybrid dense + sparse)
- voyage-code-3 for code embeddings, bge-reranker-v2-m3 for reranking
- Postgres 16 + Redis 7 + Docker Compose for infra
- Langfuse for tracing
- LLM router: Gemini Flash (default) | Qwen3 Coder via OpenRouter (transform agent) | Groq Llama (classification)

## Conventions
- Type hints on every function signature, return types included
- Pydantic v2 models for all data passed between layers
- Use `loguru` not `logging`
- Tests live alongside code: `agents/planner.py` → `agents/test_planner.py`
- Use `uv` for package management, not pip/poetry
- All LLM calls go through `agents/llm_router.py` — never instantiate clients elsewhere

## Critical rules
- NEVER write to the main git branch during migration. Always use a temp branch per module.
- Every transform must be paired with a critic validation pass.
- Token budget per single LLM call: 8K input, 4K output. Hard fail anything larger.
- All file paths absolute, never relative.
- Run `ruff check` + `mypy` before declaring any task done.

## Commands you'll need
- Run app: `uv run uvicorn api.main:app --reload`
- Run tests: `uv run pytest -xvs`
- Lint: `uv run ruff check . && uv run mypy .`
- Bring up infra: `docker compose up -d`
- Bring down infra: `docker compose down`

## What weird looks like
- If you see a function over 50 lines, suggest splitting it
- If you see an LLM call outside `llm_router.py`, that's a bug
- If you see synchronous I/O in an async function, that's a bug
- If you see hardcoded model names, use the router

## Out of scope (don't suggest)
- Adding Temporal.io
- Adding LangSmith
- Adding a React frontend
- Adding Celery
```

### 2.3 The `.env.example`

```bash
GEMINI_API_KEY=
OPENROUTER_API_KEY=
GROQ_API_KEY=
VOYAGE_API_KEY=

POSTGRES_USER=migration
POSTGRES_PASSWORD=migration_dev_pw
POSTGRES_DB=migration_agent
POSTGRES_HOST=localhost
POSTGRES_PORT=5432

REDIS_URL=redis://localhost:6379/0
QDRANT_URL=http://localhost:6333

LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=http://localhost:3000
```

Copy `.env.example` to `.env` and fill in API keys from the four free-tier accounts.

### 2.4 The `docker-compose.yml`

```yaml
services:
  postgres:
    image: postgres:16
    environment:
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: ${POSTGRES_DB}
    ports: ["5432:5432"]
    volumes: [pgdata:/var/lib/postgresql/data]

  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]

  qdrant:
    image: qdrant/qdrant:latest
    ports: ["6333:6333", "6334:6334"]
    volumes: [qdrant_storage:/qdrant/storage]

  langfuse:
    image: langfuse/langfuse:latest
    depends_on: [postgres]
    environment:
      DATABASE_URL: postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}
      NEXTAUTH_SECRET: dev-secret-change-me
      NEXTAUTH_URL: http://localhost:3000
      SALT: dev-salt
    ports: ["3000:3000"]

volumes:
  pgdata:
  qdrant_storage:
```

Run `docker compose up -d` and check that all four services are up:
- Postgres: `psql -h localhost -U migration -d migration_agent` (password `migration_dev_pw`)
- Redis: `redis-cli ping` → `PONG`
- Qdrant: `curl http://localhost:6333/healthz` → `healthz check passed`
- Langfuse: open http://localhost:3000, sign up, create a project, copy keys into `.env`

### 2.5 The `pyproject.toml`

```toml
[project]
name = "migration-agent"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "fastapi[standard]>=0.115",
    "uvicorn[standard]>=0.32",
    "langgraph>=0.2.50",
    "langgraph-checkpoint-postgres>=2.0",
    "langchain-google-genai>=2.0",
    "langchain-openai>=0.2",  # for OpenRouter (OpenAI-compatible)
    "langchain-groq>=0.2",
    "voyageai>=0.3",
    "qdrant-client[fastembed]>=1.12",
    "tree-sitter>=0.23",
    "tree-sitter-languages>=1.10",  # bundles parsers for 100+ languages
    "astchunk @ git+https://github.com/yilinjz/astchunk.git",
    "rerankers[transformers]>=0.6",
    "networkx>=3.4",
    "pydantic>=2.9",
    "sqlalchemy[asyncio]>=2.0",
    "asyncpg>=0.30",
    "redis>=5.2",
    "loguru>=0.7",
    "langfuse>=2.55",
    "httpx>=0.27",
    "rank-bm25>=0.2",
    "jinja2>=3.1",  # for HTMX dashboard
    "pygit2>=1.16",  # programmatic git operations
]

[tool.uv]
dev-dependencies = [
    "pytest>=8.3",
    "pytest-asyncio>=0.24",
    "ruff>=0.7",
    "mypy>=1.13",
]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.pytest.ini_options]
asyncio_mode = "auto"
```

Then: `uv sync` to install everything.

---

## Section 3 — Week 1: Foundation & Ingestion (Days 1–5)

**Goal:** by Friday you can clone a repo, parse it into AST chunks, embed them with voyage-code-3, store in Qdrant, and retrieve relevant chunks for a code question.

For each day below: the **Claude Code prompt** is what you literally paste into your Claude Code session. Keep sessions narrow — one feature per session, then commit and start fresh.

### Day 1 — Scaffold the project skeleton

**Subdir CLAUDE.md to create:**

```bash
echo "# Agents directory
All LangGraph nodes and agent definitions live here.
Each agent: own file, async functions only, returns Pydantic models.
LLM clients are imported from llm_router.py — never instantiated here." > agents/CLAUDE.md

echo "# RAG layer
Code chunking, embedding, retrieval, reranking.
Qdrant collections are defined in collections.py.
Embedding via voyage-code-3 in embedder.py — batch up to 128 chunks per call." > rag/CLAUDE.md

echo "# Workflows
LangGraph state machine definitions.
State schemas in state.py as TypedDict.
Graph assembly in graph.py.
Checkpointer config in checkpointer.py." > workflows/CLAUDE.md
```

**Claude Code prompt:**

> Read CLAUDE.md and pyproject.toml. Set up the initial Python package structure with `__init__.py` files in agents/, rag/, workflows/, api/, dashboard/. Create a minimal `api/main.py` with a FastAPI app that has `/healthz` returning `{"status": "ok", "services": {...}}` where services checks Postgres, Redis, Qdrant, and Voyage AI connectivity. Use httpx for the Qdrant and Voyage checks. Add a basic `loguru` config in `api/logging_config.py`. Verify everything works by running `uv run uvicorn api.main:app --reload` and curling /healthz.

**Verification:** `curl localhost:8000/healthz` returns all services as `"ok"`.

**Common pitfall:** Voyage rejects connectivity checks with empty bodies — send an actual minimal embedding request (one token) and catch auth errors specifically.

### Day 2 — AST-aware code chunker

**Claude Code prompt:**

> Build `rag/chunker.py` that wraps ASTChunk for code chunking. The class `CodeChunker` should:
> 1. Accept a file path, auto-detect language from extension (Python, JS, TS, Java, Go, C#)
> 2. Use ASTChunk with max_chunk_size=1200 non-whitespace characters
> 3. Enrich each chunk with metadata: file_path, language, node_type (function/class/import), parent_class (if nested), start_line, end_line, docstring (if present)
> 4. Return Pydantic models defined in `rag/models.py` as `CodeChunk`
>
> Also build `rag/repo_walker.py` that walks a repo directory using pathspec (respecting .gitignore), yields all supported source files, and skips common noise dirs: node_modules, .venv, __pycache__, dist, build, .git.
>
> Write pytest tests using a fixture repo at `tests/fixtures/sample_flask_app/` (create a 3-file Flask app as the fixture). Assert: total chunks > 5, every chunk has non-empty parent context, chunks fit under 1500 chars.

**Verification:** `uv run pytest tests/test_chunker.py -xvs` passes. Manually inspect 3 chunks to confirm they're semantically coherent.

### Day 3 — Qdrant schema + Voyage embedding pipeline

**Claude Code prompt:**

> Build the embedding and storage layer:
>
> 1. `rag/collections.py` — Pydantic-modeled definitions for three Qdrant collections: `code_chunks` (dense 1024-dim voyage-code-3 + sparse BM25 named vector), `doc_chunks` (same shape, for migration guides), `migration_patterns` (same shape, for learned patterns). Use Qdrant's hybrid search via named vectors. Bootstrap function that idempotently creates all three collections if missing.
>
> 2. `rag/embedder.py` — async wrapper around the voyageai Python SDK. Method `embed_chunks(chunks: list[CodeChunk]) -> list[list[float]]` that batches in groups of 128 (Voyage's max), retries with exponential backoff on rate limits, uses `output_dimension=1024` and `input_type="document"`. Separate `embed_query(text: str)` with `input_type="query"`.
>
> 3. `rag/ingest.py` — orchestration: walk repo → chunk → embed (parallel batches via asyncio.gather, max concurrency 4) → upsert to Qdrant with both dense and sparse vectors. Sparse vectors via Qdrant's fastembed BM25.
>
> 4. CLI entry point `python -m rag.ingest --repo PATH --collection code_chunks` using typer.
>
> Test by ingesting `tests/fixtures/sample_flask_app/` and querying Qdrant directly to confirm vectors exist.

**Verification:** `curl http://localhost:6333/collections/code_chunks` shows non-zero point count.

**Common pitfall:** Voyage's free tier has a 3 RPM limit on `voyage-code-3` — your batching code must respect this. Add a token bucket limiter at 3 RPM.

### Day 4 — Dependency graph builder

**Claude Code prompt:**

> Build `rag/dep_graph.py` that produces a module-level dependency graph for a Python repo:
> 1. Use the standard library `ast` module (not tree-sitter — we need actual imports resolved). Parse every .py file, extract Import and ImportFrom nodes.
> 2. Build a NetworkX DiGraph where nodes are module paths (dotted form, e.g. `app.models.user`) and edges are import dependencies.
> 3. Compute and store: topological order (for migration ordering), cyclic dependencies (warn but don't fail), leaf modules (no outgoing edges), entry points (no incoming edges).
> 4. Persist the graph to Postgres via SQLAlchemy in a `dependency_graphs` table with columns: id (uuid), repo_path, graph_json (JSONB), created_at.
> 5. Method `get_migration_order(graph) -> list[list[str]]` returning batches of modules that can be migrated in parallel (each batch is one topological level).
>
> Also add JS/TS support using tree-sitter to extract `import` and `require()` calls. Keep the Python and JS extractors as separate strategy classes implementing a `DepExtractor` protocol.
>
> Write tests asserting: leaf modules detected correctly, parallel batches are valid (no intra-batch deps), graph is JSON-serializable.

**Verification:** Run on a real open-source Flask app from GitHub. Inspect `get_migration_order()` output — leaf modules (like `utils.py`) should appear first.

### Day 5 — Doc ingestion + basic retrieval endpoint

**Claude Code prompt:**

> Two tasks today:
>
> 1. Doc ingestion: build `rag/doc_ingest.py` that takes a directory of Markdown files (migration guides like the Flask→FastAPI guide), chunks them with simple recursive markdown splitting (1000 chars, 100 overlap), embeds with voyage-code-3 (yes, code-3 works fine for code-related docs), and stores in `doc_chunks`. CLI: `python -m rag.doc_ingest --dir PATH`.
>
> 2. Build `api/routes/search.py`: POST /search endpoint accepting `{query: str, collection: str, top_k: int = 10}`. Performs dense + sparse hybrid retrieval via Qdrant's named vector search with RRF fusion server-side. Returns top results with all metadata.
>
> Download the official FastAPI migration guide from https://fastapi.tiangolo.com (use 3 relevant pages) and the Flask docs, save as markdown in `data/migration_guides/`. Ingest both.
>
> Verify with pytest: query "how to replace flask render_template" against doc_chunks, assert a chunk mentioning Jinja2Templates appears in top 5.

**Verification:** `curl -X POST localhost:8000/search -d '{"query":"replace flask render_template","collection":"doc_chunks","top_k":5}'` returns ranked results.

**End of Week 1 deliverable:** You can ingest a real repo and a doc corpus, then retrieve semantically-relevant chunks for any code question. Commit this with tag `v0.1-foundation`.

---

## Section 4 — Week 2: RAG Pipeline (Days 6–10)

**Goal:** Production-grade retrieval — reranking, query routing, pattern store, evaluation harness with measurable accuracy targets.

### Day 6 — RRF hybrid retrieval, tuned

**Claude Code prompt:**

> Refactor `rag/retriever.py` into a clean class `HybridRetriever`:
> 1. Issue parallel dense and sparse queries to Qdrant.
> 2. Apply Reciprocal Rank Fusion with k=60 (the standard RRF constant).
> 3. Support an `alpha` parameter (0.0=pure sparse, 1.0=pure dense, 0.5=balanced) that scales the contribution before RRF.
> 4. Metadata filtering: accept a `MetadataFilter` Pydantic model supporting language, file_path patterns, node_type — convert to Qdrant Filter.
> 5. Add a structured logging hook that emits the dense top-5 and sparse top-5 separately for debugging.
>
> Write a benchmark script `eval/retrieval_bench.py` that runs 20 hand-written (query, expected_file) pairs against four configs: alpha=0.0, 0.5, 0.7, 1.0. Reports Recall@5 and MRR for each. Save results to Postgres `eval_runs` table.

**Verification:** Run the benchmark. Alpha=0.7 (slight dense bias) typically wins for code retrieval. Recall@5 should be > 0.70 at this stage (no reranker yet).

### Day 7 — BGE reranker integration

**Claude Code prompt:**

> Add a reranking layer between hybrid retrieval and final results:
> 1. `rag/reranker.py` — use the `rerankers` library with `BAAI/bge-reranker-v2-m3` model. Async wrapper that takes a query + list of CodeChunks, returns reranked list with relevance scores.
> 2. Run the model on CPU (it's small enough). Cache the model instance as a module-level singleton — don't reload between calls.
> 3. Pipeline: hybrid retrieve top-20 → rerank → return top-6.
> 4. Add a `RetrievalConfig` Pydantic model: `{hybrid_alpha, retrieve_k, rerank_k, use_reranker}` and thread it through the API.
> 5. Re-run the Day 6 benchmark with reranker on/off. Expected: Recall@5 should jump 10–20 points.
>
> Also log every retrieval to Langfuse as a span with: query, retrieved_ids, reranked_ids, latency_ms.

**Verification:** Recall@5 > 0.82 after reranker. Latency under 500ms per query on a modest laptop.

### Day 8 — Query router agent

**Claude Code prompt:**

> Build `agents/llm_router.py` first — this is the central LLM client factory:
> 1. Class `LLMRouter` with method `get_client(role: Literal["planner", "transform", "critic", "classifier", "embed"])`.
> 2. Mapping (configurable via env): planner→Gemini 2.5 Flash, transform→Qwen3 Coder 480B via OpenRouter, critic→Gemini 2.5 Flash, classifier→Groq Llama 3.3 70B, embed→voyage-code-3.
> 3. Wrap each in LangChain's respective `ChatModel` class. Add token usage tracking via Langfuse callbacks.
> 4. Implement a circuit breaker: on 3 consecutive rate-limit errors from a provider, fall back to OpenRouter free Qwen for 5 minutes.
>
> Then build `agents/query_router.py`:
> 1. LangGraph node that takes a query, calls the classifier LLM with a few-shot prompt to classify into: `code_pattern | migration_guide | api_reference | combined`.
> 2. Returns a structured `RouterDecision` Pydantic model with which Qdrant collections to query and which metadata filter to apply.
> 3. Output via Pydantic's `with_structured_output` on the LangChain client.
>
> Write tests with 10 example queries, assert correct classification.

**Verification:** All 10 test queries route correctly. Latency under 800ms per routing call (Groq is fast).

### Day 9 — Eval harness

**Claude Code prompt:**

> Build a proper RAG evaluation pipeline at `eval/`:
> 1. `eval/dataset.py` — schema for an eval dataset: `(query, ground_truth_chunk_ids, query_type, difficulty)`. Load from a JSONL file at `eval/datasets/code_retrieval_v1.jsonl`.
> 2. Manually create 50 entries against the sample Flask repo. Mix of: exact symbol lookup, semantic intent ("how does auth work"), multi-hop ("which routes use the database"), API migration ("flask to fastapi equivalent").
> 3. `eval/metrics.py` — Recall@k, MRR, NDCG, and a custom "context precision" metric that LLM-judges whether retrieved chunks are actually relevant to the query.
> 4. `eval/run.py` CLI: `python -m eval.run --config CONFIG_NAME` runs the full pipeline against the dataset and writes results to Postgres + a summary report to `eval/reports/`.
> 5. Track every run in Langfuse as a Dataset Run.
>
> Set the bar: any retrieval config change must clear Recall@5 > 0.82 OR be explicitly justified in the commit message.

**Verification:** `python -m eval.run --config default` outputs a report. Reproducibility check — run twice, same numbers (within ±0.01).

### Day 10 — Pattern store + feedback loop

**Claude Code prompt:**

> Build the pattern store — this is the "learns from itself" part:
> 1. Schema: a `MigrationPattern` Pydantic model with fields: before_code, after_code, migration_type, source_framework, target_framework, success_count, embedded_signature.
> 2. The signature is `embed(before_code + " -> " + migration_type)` stored in the `migration_patterns` Qdrant collection.
> 3. `rag/pattern_store.py` with methods: `record_success(pattern)`, `find_similar(query_code, migration_type, k=3)`. Increment success_count on duplicates.
> 4. Seed the store with 10 hand-written canonical patterns (e.g., `@app.route('/x')` → `@router.get('/x')`, `flask.request.json` → `await request.json()`).
> 5. Build a feedback endpoint `POST /feedback/pattern` that any agent can call after a successful migration to record the (before, after) pair.
>
> Add a unit test that retrieves a similar pattern given a slightly different input.

**Verification:** Query the store with a Flask route, get back the FastAPI equivalent in top-1.

**End of Week 2 deliverable:** A retrieval system that benchmarks at Recall@5 > 0.82, with query routing, reranking, and a self-improving pattern store. Commit with tag `v0.2-rag`.

---

## Section 5 — Week 3: Agent Graph & Orchestration (Days 11–15)

**Goal:** Full migration pipeline running end-to-end. Planner decomposes work, specialist agents execute, critic validates, rollback works.

### Day 11 — LangGraph state machine scaffold

**Claude Code prompt:**

> Build the orchestration graph in `workflows/`:
>
> 1. `workflows/state.py` — define `MigrationState` as a TypedDict with: `repo_path: str`, `target_spec: MigrationSpec`, `dep_graph_id: UUID`, `task_queue: list[MigrationTask]`, `task_results: list[TaskResult]`, `rollback_stack: list[RollbackEntry]`, `current_batch: list[str]`, `attempt_count: dict[str, int]`, `final_status: Literal["pending", "success", "partial", "failed"]`.
>
> 2. `workflows/graph.py` — assemble the LangGraph StateGraph:
>    - Nodes: `plan`, `dispatch`, `transform`, `test`, `critic`, `rollback`, `finalize`
>    - Edges: plan → dispatch → transform → test → critic → (dispatch | rollback | finalize)
>    - Conditional edges based on critic outcome
>
> 3. `workflows/checkpointer.py` — set up PostgresSaver from `langgraph-checkpoint-postgres`. Tables are auto-created.
>
> 4. Stub the node functions for now (return state unchanged with a log statement). Just verify the graph compiles and a dummy run hits every node.
>
> Add an API endpoint `POST /migrations` that starts a run and returns a `thread_id`. `GET /migrations/{thread_id}` returns current state from checkpointer.

**Verification:** Start a run, see the thread_id, get state — see it's been through every node.

### Day 12 — Planner agent

**Claude Code prompt:**

> Implement `agents/planner.py`:
> 1. Input: dep_graph_id + MigrationSpec (target framework, version, custom rules).
> 2. Load dep graph from Postgres. Compute migration batches via `get_migration_order()`.
> 3. For each module in topological order, call Gemini Flash with a structured prompt:
>    - Retrieve top-3 most similar files via RAG
>    - Retrieve top-3 relevant migration guide chunks
>    - Ask LLM to classify the module's migration complexity (trivial / standard / complex) and predict required transformations.
> 4. Output: `list[MigrationTask]` where each task has: module_path, priority, predicted_changes, retrieved_context_ids, depends_on (other task IDs).
> 5. Store all tasks in Redis with status="pending". Use a Redis hash per task for atomic status updates.
>
> Constraint: cap planner LLM calls at 1 per module. If repo has 50 modules, that's 50 calls — within free tier.
>
> Test against the sample Flask repo. Assert planner generates one task per migrating module, batches respect dep order.

**Verification:** A real planning run on a 10-module repo produces a structured task list in Redis. Check via `redis-cli HGETALL task:<id>`.

### Day 13 — Transform agent + RAG context injection

**Claude Code prompt:**

> Build `agents/transform.py` — the workhorse:
> 1. Input: a single MigrationTask.
> 2. Use the OpenRouter Qwen3 Coder client (262K context — plenty of room).
> 3. Construct the prompt:
>    - System: migration rules from MigrationSpec, hard constraint "output only the migrated file content in a fenced code block"
>    - Few-shot: top-3 migration patterns from pattern_store
>    - Context: top-6 retrieved doc chunks
>    - The original file content
>    - User instruction: "Migrate this file. Preserve all behavior. Update imports, decorators, sync→async if needed."
> 4. Write output to a new git branch named `migration/{run_id}/{module}` via pygit2. Original file untouched on main.
> 5. Return `TaskResult` with: status="transformed", branch_name, diff_summary, tokens_used.
>
> Implement per-function granularity: if a file is over 500 lines, use the chunker to split it, migrate each function separately, and stitch the file back together. This is critical — full-file migration on large files fails consistently.
>
> Add LangGraph Send API support so multiple transform calls can fan out in parallel (capped at concurrency=3 to respect rate limits).

**Verification:** Run transform on one Flask route file. See a new git branch with the migrated FastAPI version.

### Day 14 — Critic agent + rollback

**Claude Code prompt:**

> Build `agents/critic.py`:
> 1. Input: a TaskResult with a migrated branch.
> 2. Checkout the branch in a subprocess (use a separate working tree via `git worktree add`).
> 3. Run validation gauntlet:
>    - `ruff check` for syntax/imports
>    - `mypy --no-incremental` for type errors
>    - `pytest` on tests targeting that module
>    - For Python imports — actually try `python -c "import migrated_module"` in a subprocess to catch ImportError
> 4. If all pass: mark task done, optionally update pattern_store with the (before, after) as a learned pattern.
> 5. If fails: parse the error output. Call Gemini Flash with the diff + error and ask for a targeted patch. Apply patch via str_replace-like surgical edit. Re-run validation. Max 2 retry loops.
> 6. After 2 failed retries: git checkout main + delete the branch (rollback). Push task back to Redis with status="failed", store failure_reason.
>
> Build `workflows/rollback.py` with the rollback semantics. Make sure no partial state leaks — if a module fails, none of its incomplete commits should remain.

**Verification:** Deliberately introduce a broken pattern in your sample repo. Run the agent. Confirm: 2 retry attempts logged, rollback executed, no dangling branches.

### Day 15 — Wire it all together

**Claude Code prompt:**

> Connect every piece into a working pipeline:
> 1. Update `workflows/graph.py` to use the real planner, transform, test, critic, rollback nodes from Days 12–14.
> 2. Implement parallel batch execution: dispatch node reads current_batch from state, fans out transform calls via LangGraph Send API.
> 3. Use Redis locks (`SET module_path NX EX 600`) to prevent two agents touching the same file. The lock is per-module-path.
> 4. Implement `finalize` node: produce a `MigrationReport` Pydantic model — total tasks, success rate, total tokens used, total cost estimate, per-module branch names.
> 5. The API endpoint `POST /migrations` now actually triggers the real pipeline.
> 6. Add a streaming endpoint `GET /migrations/{thread_id}/stream` that uses Server-Sent Events to emit state updates as the graph progresses.
>
> Test end-to-end: pick a small real Flask app on GitHub (e.g., a TodoMVC clone), clone it, point the migration agent at it with spec "migrate to FastAPI", watch it run. Goal: 7/10 modules should auto-migrate successfully.

**Verification:** End-to-end run on a real Flask app produces majority-passing migrated branches. Open the Langfuse dashboard at localhost:3000 and inspect the full trace.

**End of Week 3 deliverable:** A working autonomous migration that you can demo. Commit with tag `v0.3-agents`.

---

## Section 6 — Week 4: Hardening, Observability, Dashboard (Days 16–20)

**Goal:** Ready-to-publish. Production-grade observability, parallel safety, a dashboard for review, and a published benchmark.

### Day 16 — Parallel execution hardening

**Claude Code prompt:**

> Stress-test and harden the parallel execution path:
> 1. Add a global concurrency limiter via `asyncio.Semaphore(5)` — never more than 5 LLM calls in flight at once (free tier rate limits).
> 2. Implement exponential backoff with jitter on all LLM client wrappers in `llm_router.py`. Max 5 retries.
> 3. Add a TaskQueue abstraction in `workflows/queue.py` backed by Redis sorted sets. Priority = topological depth. Workers pull, execute, push results.
> 4. Implement graceful shutdown: SIGTERM → finish in-flight tasks, persist state to checkpointer, exit. Test by Ctrl+C-ing mid-run and resuming.
> 5. Add `pytest -k stress` tests that run 20 module migrations in parallel against a fixture repo, assert no race conditions, no duplicate writes.

**Verification:** Kill the worker mid-run with `kill -SIGTERM`. Restart. State resumes from last checkpoint, no work lost.

### Day 17 — Cost tracking + alerts

**Claude Code prompt:**

> Wire up full cost observability:
> 1. Every LLM call in `llm_router.py` emits a Langfuse generation event with input_tokens, output_tokens, model.
> 2. Compute estimated cost using a pricing table in `agents/pricing.py` (Gemini Flash, Qwen via OpenRouter, Groq, Voyage — all free in our case, but log as if they were paid for benchmarking purposes).
> 3. Per-migration-run cost aggregation stored in Postgres `migration_runs` table.
> 4. Hard ceiling: if a run's estimated cost exceeds $5 in equivalent paid tokens, pause the workflow and require manual resume via the dashboard.
> 5. Daily cost summary CLI: `python -m api.tools.cost_report --days 7` outputs a table.
>
> Bonus: surface cost in the SSE stream so the dashboard shows live spend.

**Verification:** Run a migration, check Langfuse — every LLM call appears with cost data. Cost report CLI renders correctly.

### Day 18 — HTMX dashboard

**Claude Code prompt:**

> Build the diff-review dashboard in `dashboard/`:
> 1. FastAPI routes serving Jinja2 templates. Tailwind via CDN. HTMX for interactivity.
> 2. Pages:
>    - `/` — list of migration runs with status, started_at, completion %, cost
>    - `/runs/{id}` — per-module table: status, tokens, branch name, diff preview button
>    - `/runs/{id}/diff/{module}` — side-by-side unified diff (use `difflib.HtmlDiff` or pygments). Approve button posts to `/runs/{id}/approve/{module}`.
>    - `/runs/{id}/stream` — live event log via SSE
> 3. Approve action: merges the migration branch into a `migration-staging` branch (NOT main — never main).
> 4. Add a "rerun failed" action that requeues all failed tasks with status reset.
>
> Use the `htmx` ext-sse extension for the live event log.

**Verification:** Open localhost:8000/, see all runs, drill into one, view diffs, approve a module, see staging branch updated in git.

### Day 19 — Public benchmark

**Claude Code prompt:**

> Build `benchmarks/migration_bench.py`:
> 1. Define a benchmark dataset: 5 small open-source repos with known migrations available as public PRs. Examples:
>    - A Flask app → FastAPI conversion PR
>    - A Python 2 → 3 modernization PR (e.g., from one of the many `2to3` projects on GitHub)
>    - A React class component → hooks PR
>    - A jQuery → vanilla JS PR
>    - A Django function-based views → class-based views PR
> 2. For each: clone at the pre-migration commit, run the agent, compare the agent's output diff to the human PR's diff.
> 3. Scoring metrics:
>    - **module_pass_rate**: % of modules where the agent's migrated code passes the original test suite
>    - **diff_similarity**: token-level similarity between agent diff and human diff
>    - **avg_attempts**: how many critic retries needed
>    - **total_cost** and **total_time**
> 4. Output: a markdown report at `benchmarks/results/REPORT_<date>.md` with a table per repo + an aggregate.
>
> Target: module_pass_rate > 0.70 averaged across the 5 repos.

**Verification:** Generate the report. Numbers are real and reproducible.

### Day 20 — Human-in-the-loop gate + docs

**Claude Code prompt:**

> Two final pieces:
>
> 1. Human-in-the-loop approval for destructive changes:
>    - Define a `RiskAssessment` step after planning. Categorize each task as low/medium/high risk based on heuristics: touches DB schema, touches auth code, changes public API surface, > 200 lines changed.
>    - For high-risk tasks: LangGraph `interrupt()` pauses the workflow. State persists via checkpointer.
>    - Dashboard shows pending approvals. Approving via the dashboard issues a `Command(resume=...)` to the graph.
>    - The graph waits indefinitely — Postgres checkpoint means it survives restart.
>
> 2. README + demo recording:
>    - Comprehensive README.md with: project overview, architecture diagram (mermaid), setup instructions, the benchmark numbers, screenshots of the dashboard, known limitations.
>    - A 3-minute Loom/asciinema recording of: start a migration, watch it run, review diffs, approve.

**Verification:** Pause a high-risk migration, resume it from the dashboard a day later. Workflow continues correctly.

**End of Week 4 deliverable:** A complete, demoable, benchmarkable system with full observability. Tag `v1.0` and write the blog post.

---

## Section 7 — Working with Claude Code: Tactical Tips

### Session hygiene

- **One feature per session.** When a session has produced a working feature, commit, then `/clear` and start fresh. Long contexts degrade output quality on complex tasks.
- **Always start with: "Read CLAUDE.md and the relevant subdir CLAUDE.md."** Even when you think the context is loaded, force the re-read.
- **Have Claude Code plan before coding.** Prefix complex tasks with: "First, draft a plan as a numbered list. Wait for me to approve before writing code."
- **Run tests inside the session.** Don't context-switch to a terminal. Claude Code runs `uv run pytest` and sees the failures directly.

### When Claude Code goes off-track

- If it suggests `npm install`, `pip install`, or `poetry add` → reply "Use uv add."
- If it tries to write to main → reply "Branch first. Never main."
- If it stubs a function with `pass # TODO` → reply "Implement fully. No TODOs in this PR."
- If it adds a logging library → reply "Use loguru, per CLAUDE.md."

### Parallel sessions

Anthropic recommends git worktrees for parallel Claude Code sessions:
```bash
git worktree add ../migration-agent-rag-tuning rag-tuning-branch
git worktree add ../migration-agent-dashboard dashboard-branch
```
Then open two Claude Code sessions, one per worktree. Useful if you want Claude tuning RAG params while you're hand-editing dashboard CSS.

### What to never let Claude Code touch

Per the CLAUDE.md "out of scope" section. If a session keeps drifting toward Temporal, LangSmith, etc, copy the out-of-scope line back into the prompt.

---

## Section 8 — Free-Tier Budget Calculation

Run the entire 4-week build on free tiers? Yes, if you plan calls right.

| Provider | Free limit | Your expected usage |
|---|---|---|
| Gemini 2.5 Flash | 1,500 RPD, 1M TPM | Planner + Critic ~ 200 calls/day during dev |
| OpenRouter (Qwen3 Coder free) | 50–1000 RPD | Transform agent ~ 50 calls/day during dev |
| Groq (Llama 3.3 70B) | 30 RPM, 1,000 RPD | Query router ~ 100 calls/day |
| Voyage AI (voyage-code-3) | 200M tokens lifetime | Ingestion: ~5M tokens for a medium repo + docs |
| Qdrant self-hosted | Unlimited | Local Docker |
| Langfuse self-hosted | Unlimited | Local Docker |
| Hetzner CX23 (optional) | €3.49/month | Only when you deploy publicly |

**Total estimated cost for the 4-week build: €0 if you stay local, ~€3.50 if you deploy at the end.**

The one risk: if you re-embed your repo 50 times during dev, you'll burn through Voyage tokens. Solution — cache embeddings to disk in `/data/embeddings_cache/<hash>.json` and only re-embed on actual chunk content change.

---

## Section 9 — Troubleshooting Quick-Reference

| Symptom | Likely cause | Fix |
|---|---|---|
| Voyage 429s | Free tier RPM limit (3 RPM on code-3) | Add token-bucket limiter, batch size ≤ 128 |
| Qdrant slow on large repo | No HNSW index tuning | Set `hnsw_config.m=16, ef_construct=200` |
| Langfuse not capturing traces | Wrong env vars | Verify `LANGFUSE_HOST` doesn't have trailing slash |
| LangGraph checkpointer errors | Postgres tables missing | Call `PostgresSaver.setup()` once at app startup |
| `mypy` failures everywhere | Missing stubs | Add `types-redis`, `types-requests` to dev deps |
| Tree-sitter "no language found" | Missing parser | Use `tree-sitter-languages` not raw `tree-sitter` |
| Gemini quota exhausted mid-day | Hit 1500 RPD | Circuit breaker auto-falls-back to OpenRouter |
| Critic loops forever | No max_retries enforced | Hard cap at 2 retries in critic.py, then force rollback |

---

## Section 10 — What "Done" Looks Like

By end of Week 4, your portfolio piece should include:

1. ✅ A GitHub repo with clean commits tagged v0.1 through v1.0
2. ✅ A README with architecture diagram, setup steps, and benchmark numbers
3. ✅ A 3-minute demo video
4. ✅ A published benchmark on 5 real open-source repos showing > 70% module pass rate
5. ✅ Langfuse traces that prove the system actually works end-to-end
6. ✅ A blog post titled something like "Building an Autonomous Codebase Migrator with LangGraph and voyage-code-3"

That last point matters more than the code. Recruiters and senior engineers can read a thoughtful technical blog post in 10 minutes. Writing it forces you to articulate the hard decisions, which is the difference between "I built an agent" and "I understand agent systems."

Good luck — go build it.
