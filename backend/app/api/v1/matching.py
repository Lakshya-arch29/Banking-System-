from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app import store
from app.core.rbac import require_permission
from app.core.security import get_current_active_user
from app.models.user import User
from app.services.blocking.service import (
    MaterialForBlocking,
    generate_block_keys,
    generate_candidates as blocking_generate_candidates,
)
from app.services.matching.classifier import classify_match
from app.services.matching.cnmc_matcher import (
    compute_cnmc_candidate_margin,
    find_cnmc_candidates_for_material,
    match_new_materials_against_cnmcs,
)
from app.services.matching.embeddings import EmbeddingCache
from app.services.matching.vector_search import (
    search_similar_materials,
)
from app.services.normalization.service import (
    normalize_material_description,
)
from app.services.parsing.service import parse_specifications


router = APIRouter(
    prefix="/matching",
    tags=["Matching"],
)


class MaterialInput(BaseModel):
    id: int | None = None
    cpse: str = Field(min_length=1)
    material_code: str = Field(min_length=1)
    description: str = Field(min_length=1)
    category: str | None = None
    unit: str | None = None
    manufacturer: str | None = None
    manufacturer_part_number: str | None = None
    material_grade: str | None = None
    dimensions: dict[str, Any] | None = None
    specifications: dict[str, Any] | None = None
    parsed_specifications: dict[str, Any] | None = None
    other_attributes: dict[str, Any] | None = None
    normalized_description: str | None = None


class CompareRequest(BaseModel):
    source: MaterialInput
    target: MaterialInput


class BatchRunRequest(BaseModel):
    max_candidates_per_material: int = Field(
        default=15,
        ge=1,
        le=500,
    )
    overwrite: bool = False
    source_cpse: str | None = None
    target_cpse: str | None = None


class CnmcMatchRequest(BaseModel):
    material: MaterialInput
    max_candidates: int = Field(default=10, ge=1, le=50)
    min_score: float = Field(default=0.50, ge=0.0, le=1.0)


class CnmcBatchRunRequest(BaseModel):
    max_candidates_per_material: int = Field(
        default=5,
        ge=1,
        le=50,
    )
    min_score: float = Field(default=0.65, ge=0.0, le=1.0)
    create_review_candidates: bool = True


def _prepare_material(material: MaterialInput) -> dict[str, Any]:
    data = material.model_dump()

    data["normalized_description"] = (
        data.get("normalized_description")
        or normalize_material_description(
            data["description"]
        )
    )

    data["parsed_specifications"] = (
        data.get("parsed_specifications")
        or parse_specifications(data["description"])
    )

    if not data.get("dimensions"):
        data["dimensions"] = data[
            "parsed_specifications"
        ].get("dimensions")

    if data.get("other_attributes") is None:
        data["other_attributes"] = {}

    return data


def _blocker(material: dict[str, Any]) -> MaterialForBlocking:
    return MaterialForBlocking(
        id=material["id"],
        category=material.get("category"),
        normalized_description=(
            material.get("normalized_description")
            or material["description"]
        ),
        material_grade=material.get("material_grade"),
        manufacturer_part_number=material.get(
            "manufacturer_part_number"
        ),
    )


def _pair_key(first: int, second: int) -> tuple[int, int]:
    return min(first, second), max(first, second)


def _gate_status(checks: list[dict[str, Any]]) -> str:
    statuses = {
        str(item.get("status", "UNKNOWN")).upper()
        for item in checks
    }

    if "CONFLICT" in statuses:
        return "CONFLICT"

    if "UNKNOWN" in statuses:
        return "UNKNOWN"

    return "PASS"


def _explanation(
    decision: str,
    gate_status: str,
    scores: dict[str, Any],
    checks: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "decision": decision,
        "gate_status": gate_status,
        "scores": scores,
        "critical_checks": checks,
        "summary": (
            "All applicable technical gates passed."
            if gate_status == "PASS"
            else "Human review is required."
        ),
    }


def _candidate(
    source: dict[str, Any],
    target: dict[str, Any],
    result: dict[str, Any],
    candidate_id: int,
    created_at: str,
) -> dict[str, Any]:
    scores = result["scores"]
    checks = result["critical_checks"]
    decision = str(result["decision"]).upper()
    gate_status = _gate_status(checks)

    return {
        "id": candidate_id,
        "source_material_id": source["id"],
        "target_material_id": target["id"],
        "source_cpse": source.get("cpse"),
        "target_cpse": target.get("cpse"),
        "source_code": source.get("material_code"),
        "target_code": target.get("material_code"),
        "source_description": source.get("description"),
        "target_description": target.get("description"),
        "text_similarity": scores.get(
            "text_similarity",
            0.0,
        ),
        "semantic_similarity": scores.get(
            "semantic_similarity",
            0.0,
        ),
        "specification_similarity": scores.get(
            "specification_similarity",
            0.0,
        ),
        "material_grade_similarity": scores.get(
            "material_grade_similarity",
            0.0,
        ),
        "other_attributes_similarity": scores.get(
            "other_attributes_similarity",
            0.0,
        ),
        "final_score": scores.get(
            "final_score",
            0.0,
        ),
        "gate_status": gate_status,
        "ai_decision": decision,
        "review_status": "PENDING",
        "critical_checks": checks,
        "explanation": _explanation(
            decision,
            gate_status,
            scores,
            checks,
        ),
        "reviewer_id": None,
        "reviewer_comments": None,
        "reviewed_at": None,
        "created_at": created_at,
    }


@router.post("/compare")
def compare_materials(
    request: CompareRequest,
    current_user: User = Depends(
        get_current_active_user
    ),
):
    source = _prepare_material(request.source)
    target = _prepare_material(request.target)
    result = classify_match(source, target)

    scores = result["scores"]
    checks = result["critical_checks"]
    decision = str(result["decision"]).upper()
    gate_status = _gate_status(checks)

    return {
        "source_material_id": source.get("id"),
        "target_material_id": target.get("id"),
        "text_similarity": scores.get(
            "text_similarity",
            0.0,
        ),
        "semantic_similarity": scores.get(
            "semantic_similarity",
            0.0,
        ),
        "specification_similarity": scores.get(
            "specification_similarity",
            0.0,
        ),
        "material_grade_similarity": scores.get(
            "material_grade_similarity",
            0.0,
        ),
        "other_attributes_similarity": scores.get(
            "other_attributes_similarity",
            0.0,
        ),
        "final_score": scores.get(
            "final_score",
            0.0,
        ),
        "gate_status": gate_status,
        "ai_decision": decision,
        "critical_checks": checks,
        "explanation": _explanation(
            decision,
            gate_status,
            scores,
            checks,
        ),
    }


@router.post("/run-batch")
def run_batch_matching(
    request: BatchRunRequest,
    current_user: User = Depends(
        require_permission("run_matching")
    ),
):
    if not store.MATERIALS:
        raise HTTPException(
            status_code=400,
            detail="No materials ingested.",
        )

    if request.overwrite:
        store.CANDIDATES.clear()

    started = datetime.now(timezone.utc)
    created_at = started.isoformat()
    materials = list(store.MATERIALS)

    blockers = [_blocker(item) for item in materials]
    material_index = {
        item["id"]: item
        for item in materials
    }

    cpse_by_id = {
        item["id"]: str(
            item.get("cpse") or "CPSE_GENERIC"
        ).strip().upper()
        for item in materials
    }

    source_blockers = blockers

    if request.source_cpse:
        source_cpse = request.source_cpse.upper().strip()
        source_blockers = [
            item
            for item in blockers
            if cpse_by_id.get(item.id) == source_cpse
        ]

    target_blockers = blockers

    if request.target_cpse:
        target_cpse = request.target_cpse.upper().strip()
        target_blockers = [
            item
            for item in blockers
            if cpse_by_id.get(item.id) == target_cpse
        ]

    target_map = {
        item.id: item
        for item in target_blockers
    }

    target_order = {
        item.id: index
        for index, item in enumerate(target_blockers)
    }

    block_index: dict[str, list[int]] = {}

    for item in target_blockers:
        for key in generate_block_keys(item):
            block_index.setdefault(key, []).append(item.id)

    seen = {
        _pair_key(
            item["source_material_id"],
            item["target_material_id"],
        )
        for item in store.CANDIDATES
    }

    cache = EmbeddingCache()
    new_candidates: list[dict[str, Any]] = []
    evaluated = 0
    vector_hits = 0
    vector_failures = 0

    for source_blocker in source_blockers:
        source = material_index[source_blocker.id]

        candidates = blocking_generate_candidates(
            source_blocker,
            block_index=block_index,
            target_map=target_map,
            target_order=target_order,
            source_keys=generate_block_keys(
                source_blocker
            ),
        )

        candidates = candidates[
            :request.max_candidates_per_material
        ]

        for target_blocker in candidates:
            if source_blocker.id == target_blocker.id:
                continue

            if (
                cpse_by_id[source_blocker.id]
                == cpse_by_id[target_blocker.id]
            ):
                continue

            pair = _pair_key(
                source_blocker.id,
                target_blocker.id,
            )

            if pair in seen:
                continue

            seen.add(pair)
            evaluated += 1

            target = material_index[target_blocker.id]
            result = classify_match(
                source,
                target,
                embedding_cache=cache,
            )

            new_candidates.append(
                _candidate(
                    source,
                    target,
                    result,
                    store.next_candidate_id(),
                    created_at,
                )
            )

    for source_blocker in source_blockers:
        source = material_index[source_blocker.id]

        try:
            vector_ids = search_similar_materials(
                description=(
                    source.get("normalized_description")
                    or source.get("description")
                    or ""
                ),
                category=source.get("category"),
                top_k=100,
                exclude_cpse=(
                    cpse_by_id[source_blocker.id]
                ),
            )
            vector_hits += len(vector_ids)
        except Exception:
            vector_failures += 1
            continue

        for target_id in vector_ids:
            if target_id not in material_index:
                continue

            pair = _pair_key(
                source_blocker.id,
                target_id,
            )

            if pair in seen:
                continue

            seen.add(pair)
            evaluated += 1

            target = material_index[target_id]
            result = classify_match(
                source,
                target,
                embedding_cache=cache,
            )

            new_candidates.append(
                _candidate(
                    source,
                    target,
                    result,
                    store.next_candidate_id(),
                    created_at,
                )
            )

    store.CANDIDATES.extend(new_candidates)

    decisions: dict[str, int] = {}
    gates: dict[str, int] = {}

    for item in new_candidates:
        decisions[item["ai_decision"]] = (
            decisions.get(item["ai_decision"], 0) + 1
        )
        gates[item["gate_status"]] = (
            gates.get(item["gate_status"], 0) + 1
        )

    elapsed = datetime.now(timezone.utc) - started

    return {
        "status": "complete",
        "materials_processed": len(materials),
        "candidate_pairs_evaluated": evaluated,
        "new_candidates_stored": len(new_candidates),
        "total_candidates": len(store.CANDIDATES),
        "decision_breakdown": decisions,
        "gate_breakdown": gates,
        "vector_search_hits": vector_hits,
        "vector_search_failures": vector_failures,
        "elapsed_ms": int(
            elapsed.total_seconds() * 1000
        ),
    }


@router.get("/candidates")
def list_candidates(
    decision: str | None = Query(None),
    review_status: str | None = Query(None),
    gate_status: str | None = Query(None),
    cpse: str | None = Query(None),
    min_score: float | None = Query(
        None,
        ge=0.0,
        le=1.0,
    ),
    skip: int = 0,
    limit: int = 50,
    current_user: User = Depends(
        get_current_active_user
    ),
):
    items = list(store.CANDIDATES)

    if decision:
        expected = decision.upper()
        items = [
            item for item in items
            if item.get("ai_decision") == expected
        ]

    if review_status:
        expected = review_status.upper()
        items = [
            item for item in items
            if item.get("review_status") == expected
        ]

    if gate_status:
        expected = gate_status.upper()
        items = [
            item for item in items
            if item.get("gate_status") == expected
        ]

    if cpse:
        expected = cpse.upper()
        items = [
            item for item in items
            if str(
                item.get("source_cpse") or ""
            ).upper() == expected
            or str(
                item.get("target_cpse") or ""
            ).upper() == expected
        ]

    if min_score is not None:
        items = [
            item for item in items
            if float(
                item.get("final_score") or 0.0
            ) >= min_score
        ]

    return {
        "total": len(items),
        "skip": skip,
        "limit": limit,
        "candidates": items[
            skip : skip + limit
        ],
    }


@router.get("/candidates/{candidate_id}")
def get_candidate(
    candidate_id: int,
    current_user: User = Depends(
        get_current_active_user
    ),
):
    for item in store.CANDIDATES:
        if item["id"] == candidate_id:
            return item

    raise HTTPException(
        status_code=404,
        detail=f"Candidate {candidate_id} not found",
    )


@router.get("/stats")
def matching_stats(
    current_user: User = Depends(
        get_current_active_user
    ),
):
    items = list(store.CANDIDATES)

    decisions: dict[str, int] = {}
    gates: dict[str, int] = {}
    reviews: dict[str, int] = {}
    scores: list[float] = []

    for item in items:
        decision = item.get("ai_decision")
        gate = item.get("gate_status")
        review = item.get("review_status")

        decisions[decision] = decisions.get(decision, 0) + 1
        gates[gate] = gates.get(gate, 0) + 1
        reviews[review] = reviews.get(review, 0) + 1
        scores.append(
            float(item.get("final_score") or 0.0)
        )

    scores.sort()
    count = len(scores)

    def percentile(ratio: float) -> float:
        if not scores:
            return 0.0
        index = min(
            int(count * ratio),
            count - 1,
        )
        return round(scores[index], 4)

    high_confidence = decisions.get(
        "HIGH_CONFIDENCE",
        0,
    )
    review = decisions.get("REVIEW", 0)
    actionable = high_confidence + review

    return {
        "total_candidates": len(items),
        "decision_breakdown": decisions,
        "gate_breakdown": gates,
        "review_status_breakdown": reviews,
        "automation_rate": (
            round(high_confidence / actionable, 4)
            if actionable
            else None
        ),
        "score_percentiles": {
            "p50": percentile(0.50),
            "p90": percentile(0.90),
            "p95": percentile(0.95),
        },
    }


@router.post("/cnmc/candidates")
def find_cnmc_proposals(
    request: CnmcMatchRequest,
    current_user: User = Depends(
        get_current_active_user
    ),
):
    material = _prepare_material(request.material)

    proposals = find_cnmc_candidates_for_material(
        material,
        max_candidates=request.max_candidates,
        min_score=request.min_score,
    )

    margin = compute_cnmc_candidate_margin(proposals)

    return {
        "material": material,
        "total_candidates": len(proposals),
        "best_score": margin["best_score"],
        "second_best_score": margin[
            "second_best_score"
        ],
        "score_margin": margin["score_margin"],
        "candidates": proposals,
    }


@router.post("/cnmc/run-batch")
def run_cnmc_batch_matching(
    request: CnmcBatchRunRequest,
    current_user: User = Depends(
        require_permission("run_matching")
    ),
):
    if not store.MATERIALS:
        raise HTTPException(
            status_code=400,
            detail="No materials ingested.",
        )

    if not store.CNMC_REGISTRY:
        return {
            "status": "no_existing_cnmc",
            "materials_evaluated": len(store.MATERIALS),
            "proposals_generated": 0,
            "proposals": [],
        }

    return match_new_materials_against_cnmcs(
        list(store.MATERIALS),
        max_candidates_per_material=(
            request.max_candidates_per_material
        ),
        min_score=request.min_score,
        create_review_candidates=(
            request.create_review_candidates
        ),
        current_user=current_user,
    )