"""Ingest Markdown migration guides into the doc_chunks Qdrant collection.

CLI: python -m rag.doc_ingest --dir PATH
"""
from __future__ import annotations

import asyncio
import os
import re
import uuid
from pathlib import Path

import typer
from dotenv import load_dotenv
from fastembed import SparseTextEmbedding
from loguru import logger
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct, SparseVector

from rag.collections import (
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    ensure_collection,
    get_qdrant_client,
)
from rag.embedder import VoyageEmbedder
from rag.models import DocChunk

_CHUNK_SIZE = 1000
_CHUNK_OVERLAP = 100
_UPSERT_BATCH_SIZE = 100
_DOC_COLLECTION = "doc_chunks"

_MD_SEPARATORS = ["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " "]
_HEADING_RE = re.compile(r"^#{1,6}\s+(.+)$", re.MULTILINE)

_BM25_MODEL: SparseTextEmbedding | None = None


def _get_bm25() -> SparseTextEmbedding:
    global _BM25_MODEL
    if _BM25_MODEL is None:
        logger.info("Loading BM25 model (first use)")
        _BM25_MODEL = SparseTextEmbedding(model_name="Qdrant/bm25")
    return _BM25_MODEL


def _merge_chunks(pieces: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if not piece.strip():
            continue
        candidate = (current + "\n" + piece).strip() if current else piece.strip()
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
                overlap = current[-chunk_overlap:] if len(current) > chunk_overlap else current
                candidate = (overlap + "\n" + piece).strip()
                current = candidate if len(candidate) <= chunk_size else piece.strip()
            else:
                current = piece.strip()
    if current:
        chunks.append(current)
    return chunks


def _recursive_split(
    text: str,
    separators: list[str],
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []

    chosen_sep: str | None = None
    for sep in separators:
        if sep in text:
            chosen_sep = sep
            break

    if chosen_sep is None:
        step = max(1, chunk_size - chunk_overlap)
        return [text[i : i + chunk_size].strip() for i in range(0, len(text), step) if text[i : i + chunk_size].strip()]

    raw = text.split(chosen_sep)
    splits = [raw[0]] + [chosen_sep + p for p in raw[1:]]
    remaining = separators[separators.index(chosen_sep) + 1 :]

    fine: list[str] = []
    for piece in splits:
        piece = piece.strip()
        if not piece:
            continue
        if len(piece) <= chunk_size:
            fine.append(piece)
        elif remaining:
            fine.extend(_recursive_split(piece, remaining, chunk_size, chunk_overlap))
        else:
            step = max(1, chunk_size - chunk_overlap)
            fine.extend(piece[i : i + chunk_size].strip() for i in range(0, len(piece), step) if piece[i : i + chunk_size].strip())

    return _merge_chunks(fine, chunk_size, chunk_overlap)


def split_markdown(text: str, chunk_size: int = _CHUNK_SIZE, chunk_overlap: int = _CHUNK_OVERLAP) -> list[str]:
    """Split a markdown document into chunks of ≤chunk_size chars with chunk_overlap overlap."""
    return _recursive_split(text, list(_MD_SEPARATORS), chunk_size, chunk_overlap)


def _extract_title(chunk: str) -> str:
    m = _HEADING_RE.search(chunk)
    if m:
        return m.group(1).strip()
    first_line = chunk.split("\n")[0].strip().lstrip("#").strip()
    return first_line[:80] or "untitled"


def chunk_markdown_file(file_path: str) -> list[DocChunk]:
    """Read a markdown file and split it into DocChunk objects."""
    path = Path(file_path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Cannot read {}: {}", file_path, exc)
        return []

    if not text.strip():
        return []

    source = path.stem
    raw_chunks = split_markdown(text)
    return [
        DocChunk(
            file_path=str(path.resolve()),
            title=_extract_title(chunk),
            content=chunk,
            chunk_index=i,
            source=source,
        )
        for i, chunk in enumerate(raw_chunks)
        if chunk.strip()
    ]


def _doc_chunk_id(file_path: str, chunk_index: int) -> str:
    key = f"{file_path}:{chunk_index}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def _doc_chunk_payload(chunk: DocChunk) -> dict[str, object]:
    return {
        "file_path": chunk.file_path,
        "title": chunk.title,
        "content": chunk.content,
        "chunk_index": chunk.chunk_index,
        "source": chunk.source,
    }


async def ingest_docs(
    docs_dir: str,
    collection_name: str = _DOC_COLLECTION,
    qdrant_url: str | None = None,
) -> int:
    """Ingest all .md files from docs_dir into the named Qdrant collection.

    Returns the number of points upserted.
    """
    client: AsyncQdrantClient = get_qdrant_client(qdrant_url)
    await ensure_collection(client, collection_name)

    embedder = VoyageEmbedder()

    md_files = sorted(Path(docs_dir).glob("**/*.md"))
    if not md_files:
        logger.warning("No .md files found in {}", docs_dir)
        return 0

    all_chunks: list[DocChunk] = []
    for md_file in md_files:
        chunks = chunk_markdown_file(str(md_file))
        logger.debug("Chunked {} → {} chunks", md_file.name, len(chunks))
        all_chunks.extend(chunks)

    if not all_chunks:
        logger.warning("No chunks produced from {}", docs_dir)
        return 0

    logger.info("Total doc chunks to embed: {}", len(all_chunks))

    # Dense embeddings
    texts = [c.content for c in all_chunks]
    all_dense = await embedder.embed_texts(texts)

    # Sparse BM25 vectors
    logger.info("Computing BM25 sparse vectors for {} doc chunks", len(all_chunks))
    bm25 = _get_bm25()
    sparse_results = list(bm25.embed(texts))

    # Build and upsert points
    points: list[PointStruct] = [
        PointStruct(
            id=_doc_chunk_id(chunk.file_path, chunk.chunk_index),
            vector={
                DENSE_VECTOR_NAME: dense_vec,
                SPARSE_VECTOR_NAME: SparseVector(
                    indices=sparse.indices.tolist(),
                    values=sparse.values.tolist(),
                ),
            },
            payload=_doc_chunk_payload(chunk),
        )
        for chunk, dense_vec, sparse in zip(all_chunks, all_dense, sparse_results)
    ]

    total = len(points)
    for i in range(0, total, _UPSERT_BATCH_SIZE):
        batch = points[i : i + _UPSERT_BATCH_SIZE]
        await client.upsert(collection_name=collection_name, points=batch)
        logger.info(
            "Upserted {}/{} doc points to '{}'",
            min(i + _UPSERT_BATCH_SIZE, total),
            total,
            collection_name,
        )

    logger.info("Doc ingestion complete: {} points in '{}'", total, collection_name)
    return total


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

app = typer.Typer(add_completion=False)


@app.command()
def main(
    dir: str = typer.Option(..., help="Absolute path to directory of Markdown migration guides"),
    collection: str = typer.Option(_DOC_COLLECTION, help="Target Qdrant collection name"),
    qdrant_url: str = typer.Option(
        None, envvar="QDRANT_URL", help="Qdrant base URL (default: http://localhost:6333)"
    ),
) -> None:
    """Ingest Markdown migration guides into a Qdrant collection."""
    load_dotenv()
    count = asyncio.run(
        ingest_docs(
            docs_dir=dir,
            collection_name=collection,
            qdrant_url=qdrant_url or os.environ.get("QDRANT_URL"),
        )
    )
    typer.echo(f"Done — {count} doc chunks ingested into '{collection}'")


if __name__ == "__main__":
    app()
