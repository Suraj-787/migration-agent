"""Retrieval evaluation metrics: Recall@k, MRR, NDCG@k, and LLM-judge context precision."""
from __future__ import annotations

import math

from loguru import logger

from rag.models import SearchHit

_JUDGE_SYSTEM = """\
You are evaluating code retrieval quality. Given a search query and retrieved code chunks, \
judge whether each chunk is RELEVANT (true) or NOT RELEVANT (false) to answering the query.

A chunk is relevant if it contains code, class definitions, function implementations, \
or context that would directly help a developer address the query — even if other chunks \
provide a better or more complete answer.

Return a JSON object with field "judgments" containing a list of exactly N booleans \
(one per chunk, in the same order they were presented).\
"""


def recall_at_k(retrieved_ids: list[str], ground_truth: set[str], k: int) -> float:
    """Proportion of ground-truth IDs found in the top-k results (true Recall@k)."""
    if not ground_truth:
        return 0.0
    top = set(retrieved_ids[:k])
    return len(top & ground_truth) / len(ground_truth)


def mrr(retrieved_ids: list[str], ground_truth: set[str]) -> float:
    """Mean Reciprocal Rank: reciprocal of the rank of the first relevant hit."""
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in ground_truth:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], ground_truth: set[str], k: int) -> float:
    """Normalised Discounted Cumulative Gain at k (binary relevance)."""
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, rid in enumerate(retrieved_ids[:k], start=1)
        if rid in ground_truth
    )
    ideal_hits = min(len(ground_truth), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0.0 else 0.0


async def context_precision(
    query: str,
    hits: list[SearchHit],
    *,
    run_id: str | None = None,
) -> float | None:
    """LLM-judge context precision via Groq.

    Judges each hit as relevant/not-relevant to the query in a single call.
    Returns the fraction of retrieved chunks judged relevant, or None on failure.

    Uses the groq SDK directly (not LangChain) to avoid Langfuse's global
    LangChain callback registration conflicting with the missing langchain package.
    """
    if not hits:
        return None

    chunks_text = "\n\n".join(
        f"Chunk {i + 1} (file: {h.payload.get('file_path', '?')}):\n"
        + h.payload.get("content", "")[:400]
        for i, h in enumerate(hits)
    )
    prompt = (
        f"Query: {query}\n\n"
        f"{chunks_text}\n\n"
        f"Return judgments for all {len(hits)} chunks."
    )

    try:
        import json
        import os

        from groq import AsyncGroq

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            logger.warning("GROQ_API_KEY not set — skipping LLM judge")
            return None

        groq_client = AsyncGroq(api_key=api_key)
        response = await groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            max_tokens=500,
            temperature=0,
        )
        content = response.choices[0].message.content
        if not content:
            return None
        data = json.loads(content)
        raw = data.get("judgments", [])
        judgments = [bool(j) for j in raw[: len(hits)]]
        while len(judgments) < len(hits):
            judgments.append(False)
        return sum(judgments) / len(judgments)
    except Exception as exc:
        logger.warning("LLM context-precision judge failed (non-fatal): {}", exc)
        return None
