from __future__ import annotations

import asyncio
import os
import time

# voyageai.AsyncClient uses aiohttp which fails SSL verification on macOS Python.
# voyageai.Client uses requests + certifi and works correctly — wrap with asyncio.to_thread.
import voyageai
from loguru import logger

from rag.models import CodeChunk

_BATCH_SIZE = 8
_INTER_BATCH_SLEEP = 22.0  # seconds — keeps us under 3 RPM on Voyage free tier (10K TPM)


class VoyageEmbedder:
    """Async wrapper around the Voyage AI SDK with fixed inter-batch sleep for 3-RPM free tier."""

    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or os.environ.get("VOYAGE_API_KEY")
        if not key:
            raise ValueError("VOYAGE_API_KEY not set")
        self._client = voyageai.Client(api_key=key)

    async def _embed_raw(self, texts: list[str], input_type: str) -> list[list[float]]:
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

    async def _embed_batched(
        self, texts: list[str], input_type: str, label: str = "batch"
    ) -> list[list[float]]:
        all_embeddings: list[list[float]] = []
        total_batches = (len(texts) + _BATCH_SIZE - 1) // _BATCH_SIZE
        t_start = time.monotonic()
        for i in range(0, len(texts), _BATCH_SIZE):
            batch = texts[i : i + _BATCH_SIZE]
            batch_num = i // _BATCH_SIZE + 1
            batches_done = batch_num - 1
            if batches_done > 0:
                elapsed = time.monotonic() - t_start
                secs_per_batch = elapsed / batches_done
                remaining_secs = secs_per_batch * (total_batches - batches_done)
                mins, secs = divmod(int(remaining_secs), 60)
                eta = f"{mins}m{secs:02d}s" if mins else f"{secs}s"
            else:
                eta = "calculating…"
            logger.info(
                "Embedding {} {}/{} ({} items) — ETA {}",
                label, batch_num, total_batches, len(batch), eta,
            )
            embeddings = await self._embed_raw(batch, input_type=input_type)
            all_embeddings.extend(embeddings)
            if batch_num < total_batches:
                logger.debug("Sleeping {}s to stay under 3 RPM", _INTER_BATCH_SLEEP)
                await asyncio.sleep(_INTER_BATCH_SLEEP)
        return all_embeddings

    async def embed_chunks(self, chunks: list[CodeChunk]) -> list[list[float]]:
        """Embed code chunks in batches of ≤8, returning 1024-dim dense vectors."""
        texts = [chunk.content for chunk in chunks]
        return await self._embed_batched(texts, input_type="document", label="chunk batch")

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed plain text strings as documents (for doc ingestion)."""
        return await self._embed_batched(texts, input_type="document", label="text batch")

    async def embed_queries(self, texts: list[str]) -> list[list[float]]:
        """Embed query strings in batches of ≤8, using query input_type."""
        return await self._embed_batched(texts, input_type="query", label="query batch")

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query string with query input_type."""
        embeddings = await self._embed_raw([text], input_type="query")
        return embeddings[0]
