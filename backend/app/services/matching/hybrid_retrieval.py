"""
Hybrid candidate retrieval.

MiniLM is used only for fast candidate retrieval.
MIRA.ai remains responsible for final semantic scoring and
technical validation.

This module does not use Milvus, so MiniLM's 384-dimensional
vectors never get sent to the 1024-dimensional MIRA.ai collection.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from app.services.matching.embeddings import EmbeddingCache


MINILM_MODEL = "base-minilm"
MINILM_BATCH_SIZE = 64


def _text(material: dict[str, Any]) -> str:
    return str(
        material.get("normalized_description")
        or material.get("description")
        or ""
    ).strip()


def _category_matches(
    source: dict[str, Any],
    target: dict[str, Any],
) -> bool:
    source_category = str(
        source.get("category") or ""
    ).strip().upper()

    target_category = str(
        target.get("category") or ""
    ).strip().upper()

    if not source_category or not target_category:
        return True

    return source_category == target_category


def retrieve_candidates(
    source: dict[str, Any],
    targets: Iterable[dict[str, Any]],
    embedding_cache: EmbeddingCache,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """
    Return the strongest target materials using MiniLM.

    MiniLM embeddings are generated in batches and are used only
    to rank candidates. Final decisions must still use MIRA.ai.
    """
    target_list = [
        target
        for target in targets
        if target.get("id") != source.get("id")
        and _category_matches(source, target)
        and _text(target)
    ]

    if not target_list or not _text(source):
        return []

    source_text = _text(source)
    target_texts = [
        _text(target)
        for target in target_list
    ]

    embedding_cache.precompute(
        [source_text, *target_texts],
        batch_size=MINILM_BATCH_SIZE,
        model_name=MINILM_MODEL,
    )

    source_vector = embedding_cache.get_or_encode(
        source_text,
        model_name=MINILM_MODEL,
    )

    if source_vector is None:
        return []

    target_vectors = []

    for text in target_texts:
        vector = embedding_cache.get_or_encode(
            text,
            model_name=MINILM_MODEL,
        )

        if vector is None:
            target_vectors.append(
                np.zeros_like(source_vector)
            )
        else:
            target_vectors.append(vector)

    matrix = np.asarray(target_vectors)

    scores = matrix @ source_vector

    ranked_indexes = np.argsort(
        -scores
    )[:top_k]

    return [
        target_list[int(index)]
        for index in ranked_indexes
    ]