from __future__ import annotations

import os
from typing import Literal

from loguru import logger
from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)

DENSE_DIM = 1024
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"

CollectionName = Literal["code_chunks", "doc_chunks", "migration_patterns"]

ALL_COLLECTIONS: list[str] = ["code_chunks", "doc_chunks", "migration_patterns"]


class CollectionConfig(BaseModel):
    name: str
    description: str
    dense_dim: int = DENSE_DIM


COLLECTION_CONFIGS: dict[str, CollectionConfig] = {
    "code_chunks": CollectionConfig(
        name="code_chunks",
        description="Source code chunks from repositories being migrated",
    ),
    "doc_chunks": CollectionConfig(
        name="doc_chunks",
        description="Migration guide documentation chunks",
    ),
    "migration_patterns": CollectionConfig(
        name="migration_patterns",
        description="Learned before/after migration patterns",
    ),
}


def get_qdrant_client(url: str | None = None) -> AsyncQdrantClient:
    return AsyncQdrantClient(url=url or os.environ.get("QDRANT_URL", "http://localhost:6333"))


async def ensure_collection(
    client: AsyncQdrantClient,
    name: str,
    dense_dim: int = DENSE_DIM,
) -> None:
    """Create a single named collection with the standard vector config if it doesn't exist."""
    if await client.collection_exists(name):
        logger.debug("Collection '{}' already exists, skipping", name)
        return

    await client.create_collection(
        collection_name=name,
        vectors_config={
            DENSE_VECTOR_NAME: VectorParams(
                size=dense_dim,
                distance=Distance.COSINE,
                hnsw_config=HnswConfigDiff(m=16, ef_construct=200),
            )
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: SparseVectorParams(
                index=SparseIndexParams(on_disk=False)
            )
        },
    )
    logger.info("Created Qdrant collection '{}'", name)


async def bootstrap_collections(client: AsyncQdrantClient | None = None) -> None:
    """Idempotently create all three standard Qdrant collections if they don't already exist."""
    if client is None:
        client = get_qdrant_client()

    for name in COLLECTION_CONFIGS:
        await ensure_collection(client, name)
