"""
Shared Milvus connection and batch write helpers.

Milvus is an optional candidate-recall layer.
PostgreSQL remains the source of truth.

MIRA.ai embeddings are generated in batches before being upserted.
"""

from __future__ import annotations

import logging
import socket
import time
from typing import Any, Iterable, Sequence

from app.core.config import settings
from app.services.matching.batch_embeddings import (
    generate_embeddings,
)

logger = logging.getLogger(__name__)


COLLECTION_NAME = settings.milvus_collection
MILVUS_HOST = settings.milvus_host
MILVUS_PORT = settings.milvus_port

EMBEDDING_BATCH_SIZE = 8

_connected = False
_collection: Any = None
_loaded = False


class MilvusUnavailable(RuntimeError):
    """Milvus cannot currently be reached."""


_PROBE_TIMEOUT_S = 2.0
_PROBE_TTL_UP_S = 60.0
_PROBE_TTL_DOWN_S = 15.0

_probe: dict[str, Any] = {
    "ok": False,
    "until": 0.0,
}


def _mark_down() -> None:
    _probe.update(
        ok=False,
        until=time.monotonic() + _PROBE_TTL_DOWN_S,
    )


def milvus_available(force: bool = False) -> bool:
    """Return whether Milvus is reachable without raising."""
    if not settings.milvus_enabled:
        return False

    now = time.monotonic()

    if not force and now < _probe["until"]:
        return bool(_probe["ok"])

    try:
        with socket.create_connection(
            (
                MILVUS_HOST,
                int(MILVUS_PORT),
            ),
            timeout=_PROBE_TIMEOUT_S,
        ):
            available = True
    except (OSError, ValueError):
        available = False

    _probe.update(
        ok=available,
        until=now + (
            _PROBE_TTL_UP_S
            if available
            else _PROBE_TTL_DOWN_S
        ),
    )

    if not available:
        logger.warning(
            "Milvus is unavailable at %s:%s.",
            MILVUS_HOST,
            MILVUS_PORT,
        )

    return available


def _ensure_connected() -> None:
    global _connected

    if _connected:
        return

    if not settings.milvus_enabled:
        raise MilvusUnavailable(
            "Milvus is disabled in settings."
        )

    try:
        from pymilvus import connections
    except ImportError as exc:
        raise MilvusUnavailable(
            "pymilvus is not installed."
        ) from exc

    connections.connect(
        alias="default",
        host=MILVUS_HOST,
        port=MILVUS_PORT,
    )

    _connected = True


def get_collection() -> Any:
    """Return the cached Milvus collection handle."""
    global _collection

    if _collection is not None:
        return _collection

    if not milvus_available():
        raise MilvusUnavailable(
            f"Milvus unavailable at "
            f"{MILVUS_HOST}:{MILVUS_PORT}."
        )

    try:
        _ensure_connected()

        from pymilvus import Collection, utility

        if not utility.has_collection(COLLECTION_NAME):
            raise MilvusUnavailable(
                f"Collection '{COLLECTION_NAME}' does not exist."
            )

        _collection = Collection(COLLECTION_NAME)

    except MilvusUnavailable:
        raise

    except Exception as exc:
        _mark_down()
        raise MilvusUnavailable(
            f"Milvus call failed: {exc}"
        ) from exc

    return _collection


def ensure_loaded() -> Any:
    """Load the collection into memory once per process."""
    global _loaded

    collection = get_collection()

    if not _loaded:
        collection.load()
        _loaded = True

    return collection


def reset_client_state() -> None:
    """Reset cached connection and collection state."""
    global _connected
    global _collection
    global _loaded

    _connected = False
    _collection = None
    _loaded = False
    _probe.update(
        ok=False,
        until=0.0,
    )


def insert_material_embedding(
    material_id: int,
    description: str,
    cpse: str,
    category: str,
) -> bool:
    """Batch-compatible helper for one material."""
    inserted = insert_material_embeddings(
        [
            {
                "id": material_id,
                "description": description,
                "cpse": cpse,
                "category": category,
            }
        ]
    )

    return inserted > 0


def insert_material_embeddings(
    records: Iterable[dict[str, Any]],
) -> int:
    """
    Generate embeddings in one batch and upsert them into Milvus.

    Duplicate descriptions are embedded only once.
    """
    rows = [
        record
        for record in records
        if str(
            record.get("description") or ""
        ).strip()
    ]

    if not rows:
        return 0

    if not settings.milvus_enabled:
        return 0

    try:
        collection = get_collection()
    except Exception as exc:
        logger.warning(
            "Skipping Milvus write: %s",
            exc,
        )
        return 0

    descriptions = [
        str(record["description"]).strip()
        for record in rows
    ]

    try:
        embedding_map = generate_embeddings(
            descriptions,
            batch_size=EMBEDDING_BATCH_SIZE,
        )
    except Exception as exc:
        logger.warning(
            "Batch embedding generation failed: %s",
            exc,
        )
        return 0

    ids: list[int] = []
    vectors: list[Sequence[float]] = []
    cpses: list[str] = []
    categories: list[str] = []

    for record in rows:
        description = str(
            record["description"]
        ).strip()

        vector = embedding_map.get(description)

        if not vector:
            continue

        ids.append(int(record["id"]))
        vectors.append(vector)
        cpses.append(
            str(record.get("cpse") or "")[:100]
        )
        categories.append(
            str(record.get("category") or "")[:100]
        )

    if not ids:
        return 0

    try:
        collection.upsert(
            [
                ids,
                vectors,
                cpses,
                categories,
            ]
        )
        collection.flush()

    except Exception as exc:
        logger.warning(
            "Milvus batch upsert failed for %d rows: %s",
            len(ids),
            exc,
        )
        _mark_down()
        return 0

    logger.info(
        "Milvus batch upsert complete: %d vectors.",
        len(ids),
    )

    return len(ids)


def delete_material_embeddings(
    material_ids: Sequence[int],
) -> int:
    """Delete vectors by material ID without raising."""
    if not material_ids:
        return 0

    if not settings.milvus_enabled:
        return 0

    try:
        collection = get_collection()
        id_list = ", ".join(
            str(int(material_id))
            for material_id in material_ids
        )

        collection.delete(
            f"id in [{id_list}]"
        )
        collection.flush()

    except Exception as exc:
        logger.warning(
            "Milvus delete failed: %s",
            exc,
        )
        return 0

    return len(material_ids)