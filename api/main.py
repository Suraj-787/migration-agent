import asyncio
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator, Literal

import asyncpg
import httpx
import redis.asyncio as aioredis
from dotenv import load_dotenv
from fastapi import FastAPI
from loguru import logger
from pydantic import BaseModel

from api.logging_config import configure_logging
from api.routes.search import router as search_router

load_dotenv()
configure_logging(level=os.environ.get("LOG_LEVEL", "INFO"))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("Migration Agent API starting up")
    yield
    logger.info("Migration Agent API shutting down")
    from agents.tracing import flush
    flush()


app = FastAPI(title="Migration Agent API", version="0.1.0", lifespan=lifespan)
app.include_router(search_router)


class ServiceStatuses(BaseModel):
    postgres: str
    redis: str
    qdrant: str
    voyage: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    services: ServiceStatuses


async def _check_postgres() -> str:
    dsn = (
        f"postgresql://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}"
        f"@{os.environ.get('POSTGRES_HOST', 'localhost')}"
        f":{os.environ.get('POSTGRES_PORT', '5432')}"
        f"/{os.environ['POSTGRES_DB']}"
    )
    try:
        conn: asyncpg.Connection = await asyncpg.connect(dsn, timeout=3.0)
        await conn.fetchval("SELECT 1")
        await conn.close()
        return "ok"
    except Exception as exc:
        logger.warning("Postgres health check failed: {}", exc)
        return f"error: {exc}"


async def _check_redis() -> str:
    url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    try:
        client: aioredis.Redis = aioredis.from_url(url, socket_timeout=3.0)
        await client.ping()
        await client.aclose()
        return "ok"
    except Exception as exc:
        logger.warning("Redis health check failed: {}", exc)
        return f"error: {exc}"


async def _check_qdrant() -> str:
    url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{url}/healthz")
            resp.raise_for_status()
        return "ok"
    except Exception as exc:
        logger.warning("Qdrant health check failed: {}", exc)
        return f"error: {exc}"


async def _check_voyage() -> str:
    api_key = os.environ.get("VOYAGE_API_KEY", "")
    if not api_key:
        return "error: VOYAGE_API_KEY not set"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                "https://api.voyageai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"input": ["health"], "model": "voyage-code-3"},
            )
            resp.raise_for_status()
        return "ok"
    except Exception as exc:
        logger.warning("Voyage AI health check failed: {}", exc)
        return f"error: {exc}"


@app.get("/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse:
    postgres, redis_, qdrant, voyage = await asyncio.gather(
        _check_postgres(),
        _check_redis(),
        _check_qdrant(),
        _check_voyage(),
    )
    services = ServiceStatuses(postgres=postgres, redis=redis_, qdrant=qdrant, voyage=voyage)
    all_ok = all(v == "ok" for v in services.model_dump().values())
    return HealthResponse(status="ok" if all_ok else "degraded", services=services)
