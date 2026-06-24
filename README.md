# Migration Agent

An autonomous multi-agent system that migrates legacy Python codebases between framework versions. Specialist agents handle code transformation, dependency analysis, test validation, and type checking. A critic agent validates every change and rolls back failures. High-risk changes (auth, schema, API surface) pause for human review before execution.

The system is built for real constraints: free-tier LLM quotas, bare-git writes that never touch the working tree, and durable checkpoints that survive server restarts.

## Architecture

```mermaid
graph TD
    A[REST API<br/>POST /migrations] --> B[LangGraph Orchestrator<br/>AsyncPostgresSaver]

    B --> C[Planner Agent<br/>Gemini Flash]
    C --> D{Risk Gate<br/>interrupt()}
    D -->|low / medium| E[Dispatch<br/>Send API fan-out]
    D -->|high risk| F[Dashboard<br/>Human Approval]
    F -->|Command resume| E

    E --> G[Transform Agent<br/>Qwen3 Coder / OpenRouter]
    G --> H[Critic Agent<br/>Gemini Flash]
    H -->|pass| I[Staging Branch<br/>pygit2 bare write]
    H -->|fail| J[Rollback<br/>branch delete]

    G -->|RAG context| K[Qdrant<br/>Hybrid dense + sparse]
    K -->|voyage-code-3| L[Code Chunks]
    K -->|bge-reranker-v2-m3| M[Doc Chunks]

    B --> N[Postgres 16<br/>checkpoints + reports]
    B --> O[Redis 7<br/>locks + cost counters]
    B --> P[Langfuse<br/>traces + cost]
```

**Five layers:**

| Layer | Component | Purpose |
|---|---|---|
| Input | FastAPI (`api/`) | REST + SSE, submit migrations, stream events |
| Orchestrator | LangGraph (`workflows/`) | Durable state machine, interrupt/resume, rollback |
| Agents | `agents/` | Planner, transform, critic — each an async LangGraph node |
| RAG | Qdrant + `rag/` | Hybrid retrieval for code patterns + migration guides |
| Backend | Postgres + Redis + Docker | Checkpoints, cost counters, module-level write locks |

## Setup

### Prerequisites

- Python 3.11+
- Docker Desktop
- [`uv`](https://docs.astral.sh/uv/) (`pip install uv`)
- API keys: Google AI (Gemini), OpenRouter (Qwen3), Groq (Llama), Voyage AI

### 1. Clone and install

```bash
git clone <repo-url>
cd migration-agent
uv sync
```

### 2. Configure environment

Copy and fill in `.env`:

```bash
# LLM providers
GOOGLE_API_KEY=...
OPENROUTER_API_KEY=...
GROQ_API_KEY=...
VOYAGE_API_KEY=...

# Postgres
POSTGRES_USER=migration
POSTGRES_PASSWORD=migration
POSTGRES_DB=migration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432

# Redis
REDIS_URL=redis://localhost:6379/0

# Langfuse (optional — traces degrade gracefully without it)
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_HOST=http://localhost:3000
```

### 3. Start infrastructure

```bash
docker compose up -d
```

This starts Postgres 16, Redis 7, and Langfuse (tracing UI at `http://localhost:3000`).

### 4. Run the API

```bash
uv run uvicorn api.main:app --reload
```

Dashboard: `http://localhost:8000/dashboard/`

### 5. Run your first migration

```bash
curl -X POST http://localhost:8000/migrations \
  -H "Content-Type: application/json" \
  -d '{
    "repo_path": "/absolute/path/to/your/flask-app",
    "source_framework": "flask",
    "source_version": "2.x",
    "target_framework": "fastapi",
    "target_version": "0.100"
  }'
```

The response includes a `thread_id`. Watch progress at `/dashboard/runs/{thread_id}` or stream events at `/migrations/{thread_id}/stream`.

**High-risk modules** (auth, schema, API surface, files > 200 lines) will pause with an "Awaiting approval" badge in the dashboard. Click Approve or Reject to continue.

## Benchmark Results

Tested on free-tier LLMs — Groq (100K tokens/day hard limit), OpenRouter free tier (heavily throttled):

| Repository | Modules | Migrated | Notes |
|---|---|---|---|
| `microblog` (Flask app) | 34 | 2 | Groq TPD exhausted after ~2 modules; remaining tasks quota-blocked |
| `sample-flask-app` (fixture) | 3 | 3 | Full migration — all modules transformed and critic-passed |

**Honest framing:** On free-tier quotas the bottleneck is token budget, not migration quality. With paid API access (Gemini Flash at $0.075/1M tokens, OpenRouter Qwen3) the same pipeline migrated the 3-module fixture end-to-end with zero manual intervention. The 2/34 number on microblog is a quota exhaustion result, not a correctness failure — the 2 modules that completed were critic-validated.

**With paid access the bottleneck shifts from quota to LLM quality** — specifically, Qwen3 Coder occasionally generates syntactically valid but semantically incorrect transforms on complex architectural rewrites. The critic catches these and triggers rollback.

## Known Limitations

- **Free-tier rate limits:** Groq 100K tokens/day per org; OpenRouter `:free` models throttle to ~10 RPM. Production use requires paid keys.
- **No JS/TS support:** The planner, transform, and RAG pipeline are Python-only. TypeScript migration is out of scope.
- **`avg_attempts` not populated:** The benchmark report shows `null` for avg_attempts — the field exists in the schema but the transform agent does not write it back to the runs table.
- **Pre-existing mypy errors:** Several repos in the wild have mypy errors before migration. The critic's mypy gate reports these as failures even when the migration itself is correct. The `--ignore-errors` flag is intentionally not set so real regressions surface.
- **Single-interrupt-at-a-time:** The human-in-the-loop gate processes high-risk tasks sequentially. If 5 modules need approval, the dashboard shows one at a time; you approve each before the next appears.

## Dashboard Screenshots

### Run list
[screenshot — runs list with status badges and cost]

### Run detail with approval gate
[screenshot — run detail showing orange "Awaiting approval" badge with Approve/Reject buttons for auth/ module]

### Diff view
[screenshot — side-by-side diff of original vs migrated Flask route handler]

## Development

```bash
# Tests
uv run pytest -xvs

# Lint + type check (required before any PR)
uv run ruff check . && uv run mypy .

# Infra
docker compose up -d    # start
docker compose down     # stop

# Index a repo into Qdrant for RAG
uv run python -m rag.indexer --repo /path/to/repo
```

## Project Structure

```
agents/          # LangGraph nodes: planner, transform, critic, llm_router
api/             # FastAPI routes: submit migration, SSE stream
benchmarks/      # CLI benchmark harness against real repos
dashboard/       # HTMX dashboard: run list, diff view, approval gate
rag/             # Qdrant hybrid retrieval, dep graph, indexer
workflows/       # LangGraph state, graph, checkpointer, cost, rollback
tests/           # Integration tests (run against live infra)
docker/          # Dockerfiles for infra services
```
