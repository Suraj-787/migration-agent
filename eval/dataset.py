"""EvalEntry schema and JSONL loader for the RAG evaluation pipeline."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

QueryType = Literal["exact_symbol", "semantic_intent", "multi_hop", "api_migration"]
Difficulty = Literal["easy", "medium", "hard"]


class EvalEntry(BaseModel):
    query: str
    ground_truth_chunk_ids: list[str]
    query_type: QueryType
    difficulty: Difficulty


class EvalDataset(BaseModel):
    name: str
    entries: list[EvalEntry]

    @classmethod
    def from_jsonl(cls, path: str | Path, *, name: str | None = None) -> "EvalDataset":
        p = Path(path)
        entries = [
            EvalEntry.model_validate(json.loads(line))
            for line in p.read_text().splitlines()
            if line.strip()
        ]
        return cls(name=name or p.stem, entries=entries)

    def item_id(self, entry: EvalEntry) -> str:
        """Deterministic ID for idempotent Langfuse dataset item upserts."""
        return hashlib.sha1(f"{self.name}:{entry.query}".encode()).hexdigest()[:32]
