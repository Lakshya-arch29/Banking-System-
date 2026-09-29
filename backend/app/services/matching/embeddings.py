import os
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# This is the trained MIRA model available in the user's Hugging Face cache.
# The cache contains:
#   AshIndian/Mira.ai
#   Qwen/Qwen3-Embedding-0.6B
#   sentence-transformers/all-MiniLM-L6-v2
#
# MIRA.ai is now the production default.
DEFAULT_MODEL_NAME = "AshIndian/Mira.ai"

# The expected dimension for the MIRA/Qwen embedding architecture.
EMBEDDING_DIM = 1024

PROJECT_ROOT = Path(__file__).resolve().parents[4]

PROJECT_MODEL_PATHS = [
    PROJECT_ROOT / "models" / "Mira.ai",
    PROJECT_ROOT / "models" / "trained" / "Mira.ai",
    PROJECT_ROOT / "models" / "qwen_quantized",
    PROJECT_ROOT / "models" / "trained" / "minilm_cpse_v1",
]

HF_CACHE_ROOT = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
)


# ============================================================================
# MODEL PATH DISCOVERY
# ============================================================================


def _find_huggingface_snapshot(
    repository_folder_name: str,
) -> Path | None:
    """
    Find the latest local Hugging Face snapshot for a cached model.

    Example:
        models--AshIndian--Mira.ai
        models--Qwen--Qwen3-Embedding-0.6B
    """
    model_root = (
        HF_CACHE_ROOT
        / repository_folder_name
        / "snapshots"
    )

    if not model_root.exists():
        return None

    snapshots = [
        path
        for path in model_root.iterdir()
        if path.is_dir()
    ]

    if not snapshots:
        return None

    # The latest modified snapshot is the safest choice when more than
    # one cached revision exists.
    return max(
        snapshots,
        key=lambda path: path.stat().st_mtime,
    )


def _mira_model_target() -> str:
    """
    Resolve the trained MIRA model.

    Priority:
    1. A project-local model directory.
    2. A cached Hugging Face AshIndian/Mira.ai snapshot.
    3. The Hugging Face model ID, allowing a normal cached/downloaded load.
    """
    for path in PROJECT_MODEL_PATHS:
        if (
            path.exists()
            and (
                (path / "config.json").exists()
                or (path / "modules.json").exists()
            )
        ):
            return str(path)

    cached_snapshot = _find_huggingface_snapshot(
        "models--AshIndian--Mira.ai"
    )

    if cached_snapshot is not None:
        return str(cached_snapshot)

    return DEFAULT_MODEL_NAME


def _qwen_model_target() -> str:
    """
    Resolve the base Qwen model.

    Priority:
    1. A cached Qwen snapshot.
    2. The Hugging Face model ID.
    """
    cached_snapshot = _find_huggingface_snapshot(
        "models--Qwen--Qwen3-Embedding-0.6B"
    )

    if cached_snapshot is not None:
        return str(cached_snapshot)

    return "Qwen/Qwen3-Embedding-0.6B"


def _minilm_model_target() -> str:
    """
    Resolve the optional trained MiniLM model, falling back to the
    standard MiniLM model if the trained checkpoint is unavailable.
    """
    trained_minilm = (
        PROJECT_ROOT
        / "models"
        / "trained"
        / "minilm_cpse_v1"
    )

    if (
        trained_minilm.exists()
        and (trained_minilm / "config.json").exists()
    ):
        return str(trained_minilm)

    cached_snapshot = _find_huggingface_snapshot(
        "models--sentence-transformers--all-MiniLM-L6-v2"
    )

    if cached_snapshot is not None:
        return str(cached_snapshot)

    return "all-MiniLM-L6-v2"


def resolve_model_name(
    name_or_alias: str | None = None,
) -> str:
    """
    Resolve the active embedding model.

    Supported aliases:

        mira
        mira.ai
        qwen
        minilm
        base-minilm
        base_minilm

    An explicit path or Hugging Face model ID is also accepted.

    Environment variable:

        MIRA_EMBEDDING_MODEL

    Examples:

        MIRA_EMBEDDING_MODEL=mira
        MIRA_EMBEDDING_MODEL=qwen
        MIRA_EMBEDDING_MODEL=C:\\models\\Mira.ai
    """
    target = (
        name_or_alias
        if name_or_alias is not None
        else os.getenv("MIRA_EMBEDDING_MODEL", "")
    )

    target = target.strip()

    # No override means use the trained MIRA model.
    if not target:
        return _mira_model_target()

    target_lower = target.lower()

    if target_lower in {
        "mira",
        "mira.ai",
        "ashindian/mira.ai",
    }:
        return _mira_model_target()

    if target_lower == "qwen":
        return _qwen_model_target()

    if target_lower == "minilm":
        return _minilm_model_target()

    if target_lower in {
        "base-minilm",
        "base_minilm",
    }:
        return "all-MiniLM-L6-v2"

    # An explicit local path or Hugging Face model ID.
    return target


def get_embedding_model_name() -> str:
    """Return the resolved model target used by the application."""
    return resolve_model_name()


@lru_cache(maxsize=4)
def _load_model(target: str) -> SentenceTransformer:
    """
    Load and cache one SentenceTransformer instance per model target.
    """
    return SentenceTransformer(
        target,
        device="cpu",
    )


def get_embedding_model(
    model_name: str | None = None,
) -> SentenceTransformer:
    """Return the active embedding model."""
    resolved = resolve_model_name(model_name)
    return _load_model(resolved)


# ============================================================================
# BATCH EMBEDDING CACHE
# ============================================================================


class EmbeddingCache:
    """
    Scoped in-memory embedding cache for one matching batch.

    Each cache key includes the resolved model target so embeddings from
    different models can never be mixed accidentally.
    """

    def __init__(
        self,
        initial_embeddings: (
            dict[str, np.ndarray] | None
        ) = None,
        model_name: str | None = None,
    ):
        self.model_name = resolve_model_name(model_name)
        self._cache: dict[
            tuple[str, str],
            np.ndarray,
        ] = {}

        if initial_embeddings:
            for text, embedding in initial_embeddings.items():
                self._cache[
                    (self.model_name, text)
                ] = embedding

    def _resolve_model(
        self,
        model_name: str | None = None,
    ) -> str:
        return (
            resolve_model_name(model_name)
            if model_name is not None
            else self.model_name
        )

    def get(
        self,
        text: str,
        model_name: str | None = None,
    ) -> np.ndarray | None:
        resolved = self._resolve_model(model_name)
        return self._cache.get(
            (resolved, text)
        )

    def set(
        self,
        text: str,
        embedding: np.ndarray,
        model_name: str | None = None,
    ) -> None:
        resolved = self._resolve_model(model_name)
        self._cache[
            (resolved, text)
        ] = embedding

    def precompute(
        self,
        texts: Iterable[str],
        batch_size: int = 64,
        model_name: str | None = None,
    ) -> None:
        resolved = self._resolve_model(model_name)

        unique_texts = {
            text
            for text in texts
            if text
        }

        missing = [
            text
            for text in unique_texts
            if (
                resolved,
                text,
            ) not in self._cache
        ]

        if not missing:
            return

        model = get_embedding_model(resolved)

        embeddings = model.encode(
            missing,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        for text, embedding in zip(
            missing,
            embeddings,
        ):
            self._cache[
                (resolved, text)
            ] = embedding

    def get_or_encode(
        self,
        text: str,
        model_name: str | None = None,
    ) -> np.ndarray | None:
        if not text:
            return None

        resolved = self._resolve_model(model_name)
        key = (resolved, text)

        if key not in self._cache:
            model = get_embedding_model(resolved)

            self._cache[key] = model.encode(
                text,
                normalize_embeddings=True,
            )

        return self._cache[key]

    def similarity(
        self,
        left: str,
        right: str,
        model_name: str | None = None,
    ) -> float:
        if not left or not right:
            return 0.0

        left_vector = self.get_or_encode(
            left,
            model_name=model_name,
        )
        right_vector = self.get_or_encode(
            right,
            model_name=model_name,
        )

        if (
            left_vector is None
            or right_vector is None
        ):
            return 0.0

        similarity = float(
            left_vector @ right_vector
        )

        return max(
            0.0,
            min(1.0, similarity),
        )

    def clear(self) -> None:
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, text: str) -> bool:
        return (
            self.model_name,
            text,
        ) in self._cache


def precompute_embeddings(
    texts: Iterable[str],
    batch_size: int = 64,
    model_name: str | None = None,
) -> EmbeddingCache:
    """Precompute a group of embeddings into a new cache."""
    cache = EmbeddingCache(
        model_name=model_name
    )

    cache.precompute(
        texts,
        batch_size=batch_size,
        model_name=model_name,
    )

    return cache


# ============================================================================
# SINGLE EMBEDDING HELPERS
# ============================================================================


def generate_embedding(
    text: str,
    model_name: str | None = None,
) -> list[float]:
    """Generate one normalized embedding."""
    if not text:
        return []

    model = get_embedding_model(model_name)

    embedding = model.encode(
        text,
        normalize_embeddings=True,
    )

    return embedding.tolist()


def semantic_similarity(
    left: str,
    right: str,
    embedding_cache: EmbeddingCache | None = None,
    model_name: str | None = None,
) -> float:
    """Calculate cosine similarity between two descriptions."""
    if not left or not right:
        return 0.0

    if embedding_cache is not None:
        return embedding_cache.similarity(
            left,
            right,
            model_name=model_name,
        )

    model = get_embedding_model(model_name)

    embeddings = model.encode(
        [left, right],
        normalize_embeddings=True,
    )

    similarity = float(
        embeddings[0] @ embeddings[1]
    )

    return max(
        0.0,
        min(1.0, similarity),
    )