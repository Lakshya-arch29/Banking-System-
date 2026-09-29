import uuid
from typing import Any

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Query,
    UploadFile,
)

from app import store
from app.core.rbac import require_permission
from app.core.security import get_current_active_user
from app.db_adapter import materials_table, next_id
from app.models.user import User
from app.services.ingestion.provenance import resolve_provenance
from app.services.ingestion.service import parse_legacy_file
from app.services.matching.milvus_client import (
    insert_material_embeddings,
)
from app.services.normalization.service import (
    normalize_material_description,
)
from app.services.parsing.service import parse_specifications


router = APIRouter(
    prefix="/materials",
    tags=["Materials"],
)


@router.post("/upload")
async def upload_materials_file(
    file: UploadFile = File(...),
    current_user: User = Depends(
        require_permission("upload_data")
    ),
):
    """
    Upload material master records.

    Supported formats are handled by parse_legacy_file().
    PostgreSQL is written first. Milvus is an enhancement and must
    not make the material upload fail if it is temporarily unavailable.
    """
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename must be provided",
        )

    content = await file.read()

    try:
        raw_rows = parse_legacy_file(
            content,
            file.filename,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    start_id = next_id(materials_table)
    new_records: list[dict[str, Any]] = []

    for row in raw_rows:
        sheet_name = row.get("_sheet_name")

        provenance = resolve_provenance(
            row,
            sheet_name=sheet_name,
            filename=file.filename,
        )

        cpse = (
            provenance.cpse
            if (
                provenance.cpse
                and provenance.cpse != "UNKNOWN"
            )
            else "CPSE_GENERIC"
        )

        material_code = (
            row.get("material_code")
            or row.get("source_material_code")
            or row.get("item_code")
            or f"MAT-{uuid.uuid4().hex[:8].upper()}"
        )

        raw_description = (
            row.get("description")
            or row.get("material_description")
            or ""
        ).strip()

        if not raw_description:
            continue

        normalized_description = (
            normalize_material_description(
                raw_description
            )
        )

        parsed = parse_specifications(
            raw_description
        )

        parsed_specifications = {
            key: value
            for key, value in parsed.items()
            if value is not None
        }

        dimensions = (
            row.get("dimensions")
            or parsed.get("dimensions")
        )

        specifications = (
            row.get("specifications")
            or {}
        )

        other_attributes = (
            row.get("other_attributes")
            or {}
        )

        record: dict[str, Any] = {
            "id": start_id + len(new_records),

            "cpse": cpse,
            "cpse_id": None,

            "material_code": material_code,
            "description": raw_description,
            "normalized_description": normalized_description,

            "category": (
                row.get("category")
                or "General"
            ),
            "unit": row.get("unit"),
            "manufacturer": row.get("manufacturer"),
            "manufacturer_part_number": row.get(
                "manufacturer_part_number"
            ),
            "material_grade": (
                row.get("material_grade")
                or parsed.get("material_grade")
            ),

            "dimensions": dimensions,
            "specifications": specifications,
            "parsed_specifications": parsed_specifications,
            "other_attributes": other_attributes,

            "upload_batch_id": None,
            "last_purchase_price": row.get(
                "last_purchase_price"
            ),
            "avg_annual_quantity": row.get(
                "avg_annual_quantity"
            ),
            "data_quality_score": None,
            "status": "active",

            "provenance_level": provenance.level.value,
            "provenance_confidence": (
                provenance.confidence
            ),
            "provenance_source": provenance.source,
            "provenance_conflict": (
                provenance.conflict_detected
            ),
            "requires_review": (
                provenance.requires_review
            ),
            "provenance_details": {
                "all_evidence": provenance.all_evidence,
                "conflicts": provenance.conflicting_evidence,
            },
        }

        new_records.append(record)

    if not new_records:
        raise HTTPException(
            status_code=400,
            detail=(
                "No valid material records were found "
                "in the uploaded file."
            ),
        )

    # PostgreSQL is the authoritative store.
    store.MATERIALS.extend(new_records)

    milvus_status = "not_attempted"
    milvus_error = None
    embeddings_requested = len(new_records)

    try:
        insert_material_embeddings(
            [
                {
                    "id": record["id"],
                    "description": (
                        record["normalized_description"]
                    ),
                    "cpse": record["cpse"],
                    "category": record.get(
                        "category"
                    ),
                }
                for record in new_records
            ]
        )

        milvus_status = "success"

    except Exception as exc:
        # The upload remains successful because PostgreSQL already
        # contains the materials. The matching route can still use
        # rule-based candidates while Milvus is unavailable.
        milvus_status = "unavailable"
        milvus_error = (
            f"{type(exc).__name__}: {exc}"
        )

    return {
        "status": "success",
        "records_ingested": len(new_records),
        "total_materials": len(store.MATERIALS),
        "milvus": {
            "status": milvus_status,
            "embeddings_requested": embeddings_requested,
            "error": milvus_error,
        },
        "sample": new_records[:10],
    }


@router.get("")
def list_materials(
    query: str | None = Query(
        None,
        description="Search description or material code",
    ),
    cpse: str | None = Query(
        None,
        description="Filter by CPSE",
    ),
    category: str | None = Query(
        None,
        description="Filter by category",
    ),
    skip: int = 0,
    limit: int = 50,
    current_user: User = Depends(
        get_current_active_user
    ),
):
    """List ingested materials with optional filters."""
    filtered = list(store.MATERIALS)

    if cpse:
        expected_cpse = cpse.lower()
        filtered = [
            material
            for material in filtered
            if str(
                material.get("cpse", "")
            ).lower()
            == expected_cpse
        ]

    if category:
        expected_category = category.lower()
        filtered = [
            material
            for material in filtered
            if str(
                material.get("category", "")
            ).lower()
            == expected_category
        ]

    if query:
        search_text = query.lower()
        filtered = [
            material
            for material in filtered
            if (
                search_text
                in str(
                    material.get(
                        "description",
                        "",
                    )
                ).lower()
                or search_text
                in str(
                    material.get(
                        "material_code",
                        "",
                    )
                ).lower()
            )
        ]

    total = len(filtered)
    paginated = filtered[
        skip : skip + limit
    ]

    return {
        "total": total,
        "skip": skip,
        "limit": limit,
        "materials": paginated,
    }


@router.get("/stats")
def materials_stats(
    current_user: User = Depends(
        get_current_active_user
    ),
):
    """Return material counts for dashboard panels."""
    materials = list(store.MATERIALS)

    cpse_set = {
        str(
            material.get("cpse")
            or "CPSE_GENERIC"
        )
        for material in materials
    }

    category_counts: dict[str, int] = {}

    for material in materials:
        category = (
            material.get("category")
            or "Uncategorised"
        )

        category_counts[category] = (
            category_counts.get(category, 0)
            + 1
        )

    return {
        "total_materials": len(materials),
        "cpse_count": len(cpse_set),
        "cpse_list": sorted(cpse_set),
        "category_distribution": category_counts,
    }


@router.get("/{material_id}")
def get_material(
    material_id: int,
    current_user: User = Depends(
        get_current_active_user
    ),
):
    """Retrieve one material by ID."""
    for material in store.MATERIALS:
        if material["id"] == material_id:
            return material

    raise HTTPException(
        status_code=404,
        detail=f"Material {material_id} not found",
    )