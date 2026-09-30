"""
Batch embedding utilities for MIRA.ai.

The model is loaded once and multiple descriptions are encoded together.
This is faster than calling model.encode() once for every material.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from app.services.matching.embeddings import (
    get_embedding_model,
    resolve_model_name,
)


DEFAULT_BATCH_SIZE = 8


def generate_embeddings(
    texts: Iterable[str],
    batch_size: int = DEFAULT_BATCH_SIZE,
    model_name: str | None = None,
) -> dict[str, list[float]]:
    """
    Generate normalized embeddings for unique non-empty descriptions.

    Duplicate descriptions are embedded only once.
    """
    unique_texts = list(
        dict.fromkeys(
            text.strip()
            for text in texts
            if text and text.strip()
        )
    )

    if not unique_texts:
        return {}

    resolved_model = resolve_model_name(model_name)
    model = get_embedding_model(resolved_model)

    embeddings = model.encode(
        unique_texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    vectors = (
        embeddings
        if isinstance(embeddings, np.ndarray)
        else np.asarray(embeddings)
    )

    return {
        text: vector.astype(float).tolist()
        for text, vector in zip(unique_texts, vectors)
    }