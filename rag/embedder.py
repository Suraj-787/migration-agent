from __future__ import annotations

import asyncio
import os
import time

# voyageai.AsyncClient uses aiohttp which fails SSL verification on macOS Python.
# voyageai.Client uses requests + certifi and works correctly — wrap with asyncio.to_thread.
import voyageai
from loguru import logger

from rag.models import CodeChunk

_BATCH_SIZE = 128
_RPM_LIMIT = 3
_RATE_PER_SEC = _RPM_LIMIT / 60.0  # 0.05 tokens/sec — Voyage free-tier hard cap


class _TokenBucket:
    def __init__(self, rate: float, capacity: float) -> None:
        self._rate = rate
        self._capacity = capacity
        self._tokens = capacity
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
            self._last_refill = now
            if self._tokens < 1.0:
                wait = (1.0 - self._tokens) / self._rate
                logger.debug("Rate limit: waiting {:.1f}s before next Voyage request", wait)
                await asyncio.sleep(wait)
                self._tokens = 0.0
            else:
                self._tokens -= 1.0


class VoyageEmbedder:
    """Async wrapper around the Voyage AI SDK with 3-RPM rate limiting and exponential backoff."""

    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or os.environ.get("VOYAGE_API_KEY")
        if not key:
            raise ValueError("VOYAGE_API_KEY not set")
        self._client = voyageai.Client(api_key=key)
        self._limiter = _TokenBucket(rate=_RATE_PER_SEC, capacity=float(_RPM_LIMIT))

    async def _embed_raw(self, texts: list[str], input_type: str) -> list[list[float]]:
        await self._limiter.acquire()
        for attempt in range(5):
            try:
                result = await asyncio.to_thread(
                    self._client.embed,
                    texts,
                    model="voyage-code-3",
                    input_type=input_type,
                    output_dimension=1024,
                )
                return result.embeddings  # type: ignore[return-value]
            except Exception as exc:
                msg = str(exc).lower()
                is_rate_limit = "rate" in msg or "429" in msg or "too many" in msg
                if is_rate_limit and attempt < 4:
                    wait = float(2**attempt) + 1.0
                    logger.warning(
                        "Voyage rate limit hit (attempt {}/5), retrying in {:.1f}s: {}",
                        attempt + 1,
                        wait,
                        exc,
                    )
                    await asyncio.sleep(wait)
                else:
                    raise
        raise RuntimeError("Unreachable")

    async def embed_chunks(self, chunks: list[CodeChunk]) -> list[list[float]]:
        """Embed a list of code chunks in batches of ≤128, returning 1024-dim dense vectors."""
        texts = [chunk.content for chunk in chunks]
        all_embeddings: list[list[float]] = []
        total_batches = (len(texts) + _BATCH_SIZE - 1) // _BATCH_SIZE
        for i in range(0, len(texts), _BATCH_SIZE):
            batch = texts[i : i + _BATCH_SIZE]
            batch_num = i // _BATCH_SIZE + 1
            logger.debug("Embedding batch {}/{} ({} items)", batch_num, total_batches, len(batch))
            embeddings = await self._embed_raw(batch, input_type="document")
            all_embeddings.extend(embeddings)
        return all_embeddings

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of plain text strings as documents (for doc ingestion)."""
        all_embeddings: list[list[float]] = []
        total_batches = (len(texts) + _BATCH_SIZE - 1) // _BATCH_SIZE
        for i in range(0, len(texts), _BATCH_SIZE):
            batch = texts[i : i + _BATCH_SIZE]
            batch_num = i // _BATCH_SIZE + 1
            logger.debug("Embedding batch {}/{} ({} items)", batch_num, total_batches, len(batch))
            embeddings = await self._embed_raw(batch, input_type="document")
            all_embeddings.extend(embeddings)
        return all_embeddings

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query string with query input_type."""
        embeddings = await self._embed_raw([text], input_type="query")
        return embeddings[0]
