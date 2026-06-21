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