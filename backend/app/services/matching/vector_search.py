"""
Nearest-neighbour candidate generation over the Milvus material embeddings.

This is an ADDITIVE candidate source. The rule-based blocking in
app/services/blocking/service.py remains the primary generator; vector search
surfaces pairs that share no block key but are semantically close -- the
"written differently, same item" case that blocking by construction cannot
reach.

Returned ids are Postgres/store material ids: the Milvus primary key mirrors
materials.id (see milvus_client.insert_material_embeddings), which is what
lets a hit be joined straight back to the real material record.
"""

from __future__ import annotations

import logging

from app.core.config import settings
from app.services.matching.embeddings import generate_embedding
from app.services.matching.milvus_client import (
    MilvusUnavailable,
    ensure_loaded,
    milvus_available,
)

logger = logging.getLogger(__name__)

SEARCH_PARAMS = {"metric_type": "COSINE", "params": {"nprobe": 16}}


def _escape_expr_literal(value: str) -> str:
    """Escapes a value for use inside a Milvus boolean expression string literal."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


def search_similar_materials(
    description: str,
    category: str | None = None,
    top_k: int | None = None,
    exclude_cpse: str | None = None,
    embedding_cache: Any = None,
) -> list[int]:
    """
    Returns material ids ordered most-similar-first, or [] if unavailable.

    category      -- when given, restricts the search to the same category,
                     which is both cheaper and more precise than searching
                     the whole collection.
    exclude_cpse  -- when given, filters out same-CPSE hits inside Milvus
                     rather than in Python, so the top_k budget is spent
                     entirely on cross-CPSE candidates (batch harmonization
                     is cross-CPSE only).

    Never raises: a Milvus outage degrades this to "no extra candidates",
    leaving rule-based matching results unaffected.
    """
    if not settings.milvus_enabled or not milvus_available():
        return []

    description = (description or "").strip()
    if not description:
        return []

    if embedding_cache is not None:
        emb = embedding_cache.get_or_encode(description)
        if emb is None:
            return []
        query_embedding = emb.tolist() if hasattr(emb, "tolist") else emb
    else:
        query_embedding = generate_embedding(description)
    if not query_embedding:
        return []

    clauses: list[str] = []
    if category:
        clauses.append(f'category == "{_escape_expr_literal(category)}"')
    if exclude_cpse:
        clauses.append(f'cpse != "{_escape_expr_literal(exclude_cpse)}"')
    expr = " and ".join(clauses) if clauses else None

    try:
        collection = ensure_loaded()
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=SEARCH_PARAMS,
            limit=top_k or settings.milvus_top_k,
            expr=expr,
            output_fields=["id"],
        )
    except MilvusUnavailable as exc:
        logger.debug("Vector search skipped: %s", exc)
        return []
    except Exception as exc:
        logger.warning("Vector search failed: %s", exc)
        return []

    if not results:
        return []

    return [int(hit.id) for hit in results[0]]