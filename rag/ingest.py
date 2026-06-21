"""Orchestrates repo ingestion: walk → chunk → embed → upsert to Qdrant.

CLI: python -m rag.ingest --repo PATH --collection code_chunks
"""
from __future__ import annotations

import asyncio
import os
import uuid

import typer
from dotenv import load_dotenv
from fastembed import SparseTextEmbedding
from loguru import logger
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct, SparseVector

from rag.chunker import CodeChunker
from rag.collections import (
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    ensure_collection,
    get_qdrant_client,
)
from rag.embedder import VoyageEmbedder
from rag.models import CodeChunk
from rag.repo_walker import walk_repo

_EMBED_BATCH_SIZE = 128
_UPSERT_BATCH_SIZE = 100
_MAX_EMBED_CONCURRENCY = 4

_BM25_MODEL: SparseTextEmbedding | None = None


def _get_bm25() -> SparseTextEmbedding:
    global _BM25_MODEL
    if _BM25_MODEL is None:
        logger.info("Loading BM25 model (first use)")
        _BM25_MODEL = SparseTextEmbedding(model_name="Qdrant/bm25")
    return _BM25_MODEL


def _chunk_id(chunk: CodeChunk) -> str:
    """Deterministic UUID5 so re-ingestion is idempotent."""
    key = f"{chunk.file_path}:{chunk.start_line}:{chunk.end_line}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def _chunk_payload(chunk: CodeChunk) -> dict[str, object]:
    return {
        "file_path": chunk.file_path,
        "language": chunk.language,
        "node_type": chunk.node_type,
        "parent_class": chunk.parent_class,
        "parent_context": chunk.parent_context,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "docstring": chunk.docstring,
        "content": chunk.content,
    }


async def ingest_repo(
    repo_path: str,
    collection_name: str,
    qdrant_url: str | None = None,
) -> int:
    """Ingest all source files from *repo_path* into the named Qdrant collection.

    Returns the number of points upserted.
    """
    client: AsyncQdrantClient = get_qdrant_client(qdrant_url)
    await ensure_collection(client, collection_name)

    chunker = CodeChunker()
    embedder = VoyageEmbedder()

    # --- Walk and chunk ---
    all_chunks: list[CodeChunk] = []
    for file_path in walk_repo(repo_path):
        chunks = chunker.chunk_file(file_path)
        if chunks:
            logger.debug("Chunked {} → {} chunks", file_path, len(chunks))
        all_chunks.extend(chunks)

    if not all_chunks:
        logger.warning("No chunks produced from {}", repo_path)
        return 0

    logger.info("Total chunks to embed: {}", len(all_chunks))

    # --- Dense embeddings (parallel batches, max concurrency 4) ---
    semaphore = asyncio.Semaphore(_MAX_EMBED_CONCURRENCY)
    batches = [
        all_chunks[i : i + _EMBED_BATCH_SIZE]
        for i in range(0, len(all_chunks), _EMBED_BATCH_SIZE)
    ]

    async def embed_batch(batch: list[CodeChunk]) -> list[list[float]]:
        async with semaphore:
            return await embedder.embed_chunks(batch)

    dense_batches: list[list[list[float]]] = await asyncio.gather(
        *[embed_batch(b) for b in batches]
    )
    all_dense: list[list[float]] = [vec for batch in dense_batches for vec in batch]

    # --- Sparse BM25 vectors (CPU-bound, run synchronously via fastembed) ---
    logger.info("Computing BM25 sparse vectors")
    texts = [c.content for c in all_chunks]
    bm25 = _get_bm25()
    sparse_results = list(bm25.embed(texts))

    # --- Build and upsert points ---
    points: list[PointStruct] = [
        PointStruct(
            id=_chunk_id(chunk),
            vector={
                DENSE_VECTOR_NAME: dense_vec,
                SPARSE_VECTOR_NAME: SparseVector(
                    indices=sparse.indices.tolist(),
                    values=sparse.values.tolist(),
                ),
            },
            payload=_chunk_payload(chunk),
        )
        for chunk, dense_vec, sparse in zip(all_chunks, all_dense, sparse_results)
    ]

    total = len(points)
    for i in range(0, total, _UPSERT_BATCH_SIZE):
        batch = points[i : i + _UPSERT_BATCH_SIZE]
        await client.upsert(collection_name=collection_name, points=batch)
        logger.info(
            "Upserted {}/{} points to '{}'",
            min(i + _UPSERT_BATCH_SIZE, total),
            total,
            collection_name,
        )

    logger.info("Ingestion complete: {} points in '{}'", total, collection_name)
    return total


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

app = typer.Typer(add_completion=False)


@app.command()
def main(
    repo: str = typer.Option(..., help="Absolute path to the repository to ingest"),
    collection: str = typer.Option("code_chunks", help="Target Qdrant collection name"),
    qdrant_url: str = typer.Option(
        None, envvar="QDRANT_URL", help="Qdrant base URL (default: http://localhost:6333)"
    ),
) -> None:
    """Ingest a repository's source files into a Qdrant collection."""
    load_dotenv()
    count = asyncio.run(
        ingest_repo(
            repo_path=repo,
            collection_name=collection,
            qdrant_url=qdrant_url or os.environ.get("QDRANT_URL"),
        )
    )
    typer.echo(f"Done — {count} chunks ingested into '{collection}'")


if __name__ == "__main__":
    app()
