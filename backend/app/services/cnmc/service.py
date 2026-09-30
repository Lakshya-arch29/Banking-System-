from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select

from app import store
from app.core.database import engine
from app.db_adapter import cnmc_table, next_global_id, next_id
from app.services.cnmc.canonicalization import (
    build_canonical_identity_string,
    compute_identity_hash,
)
from app.services.cnmc.taxonomy import resolve_type_and_category


def generate_or_get_cnmc(
    materials: list[dict[str, Any]],
    cmr: dict[str, Any],
    approved_by: int | None = None,
) -> dict[str, Any]:
    """
    Given an approved Common Material Record and source materials:
    1. Resolve TYPE and CATEGORY codes deterministically.
    2. Build the canonical identity representation and SHA-256 fingerprint.
    3. Look up in CNMC registry.
       - If existing identity found -> reuse existing CNMC.
       - If new identity -> allocate next GLOBAL_ID and register MIRA-<TYPE>-<CATEGORY>-<GLOBAL_ID>.
    """
    category = cmr.get("category")
    canonical_description = cmr.get("canonical_description")

    type_code, category_code = resolve_type_and_category(
        category=category,
        canonical_description=canonical_description,
        materials=materials,
    )

    canonical_identity_str = build_canonical_identity_string(
        cmr=cmr,
        type_code=type_code,
        category_code=category_code,
    )
    identity_hash = compute_identity_hash(canonical_identity_str)

    # 1. Check registry for existing canonical identity
    with engine.begin() as conn:
        existing_row = conn.execute(
            select(cnmc_table).where(cnmc_table.c.identity_hash == identity_hash)
        ).mappings().first()

    if existing_row:
        return {
            "id": existing_row["id"],
            "cnmc_code": existing_row["cnmc_code"],
            "identity_hash": existing_row["identity_hash"],
            "category": existing_row["category"],
            "canonical_material_record": existing_row["canonical_material_record"],
            "status": existing_row["status"],
            "is_new": False,
        }

    # 2. Allocate next global sequence number across all categories
    global_id = next_global_id()
    cnmc_code = f"MIRA-{type_code}-{category_code}-{global_id}"

    # Determine description for registry
    standardized_description = (
        canonical_description
        if canonical_description and canonical_description != "UNKNOWN"
        else (materials[0].get("description") if materials else "Standard Material")
    )

    now = datetime.now(timezone.utc)
    new_record = {
        "cnmc_code": cnmc_code,
        "identity_hash": identity_hash,
        "standardized_description": standardized_description,
        "category": category if category and category != "UNKNOWN" else "General",
        "unspsc_code": None,
        "canonical_material_record": cmr,
        "status": "ACTIVE",
        "created_at": now.isoformat(),
        "approved_by": approved_by,
    }

    # Persist in CNMC registry with concurrency safety
    try:
        store.CNMC_REGISTRY.append(new_record)
    except Exception:
        with engine.begin() as conn:
            existing_row = conn.execute(
                select(cnmc_table).where(cnmc_table.c.identity_hash == identity_hash)
            ).mappings().first()
        if existing_row:
            return {
                "id": existing_row["id"],
                "cnmc_code": existing_row["cnmc_code"],
                "identity_hash": existing_row["identity_hash"],
                "category": existing_row["category"],
                "canonical_material_record": existing_row["canonical_material_record"],
                "status": existing_row["status"],
                "is_new": False,
            }
        raise

    with engine.begin() as conn:
        inserted_row = conn.execute(
            select(cnmc_table).where(cnmc_table.c.identity_hash == identity_hash)
        ).mappings().first()

    return {
        "id": inserted_row["id"] if inserted_row else global_id,
        "cnmc_code": cnmc_code,
        "identity_hash": identity_hash,
        "category": new_record["category"],
        "canonical_material_record": cmr,
        "status": "ACTIVE",
        "is_new": True,
    }



