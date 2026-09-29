from fastapi import APIRouter, Depends

from app import store
from app.core.security import get_current_active_user
from app.models.user import User


router = APIRouter(
    prefix="/analytics",
    tags=["Analytics"],
)


def _materials():
    return list(store.MATERIALS)


def _candidates():
    return list(store.CANDIDATES)


@router.get("/overview")
def analytics_overview(
    current_user: User = Depends(get_current_active_user),
):
    materials = _materials()
    candidates = _candidates()

    high_confidence = sum(
        c.get("ai_decision") == "HIGH_CONFIDENCE"
        for c in candidates
    )

    review_recommendations = sum(
        c.get("ai_decision") == "REVIEW"
        for c in candidates
    )

    pending = sum(
        c.get("review_status") == "PENDING"
        and c.get("ai_decision") in {
            "HIGH_CONFIDENCE",
            "REVIEW",
        }
        for c in candidates
    )

    approved = sum(
        c.get("review_status") == "APPROVED"
        for c in candidates
    )

    rejected = sum(
        c.get("review_status") == "REJECTED"
        for c in candidates
    )

    gate_pass = sum(
        c.get("gate_status") == "PASS"
        for c in candidates
    )

    gate_unknown = sum(
        c.get("gate_status") == "UNKNOWN"
        for c in candidates
    )

    gate_conflict = sum(
        c.get("gate_status") == "CONFLICT"
        for c in candidates
    )

    actionable = (
        high_confidence + review_recommendations
    )

    automation_rate = (
        round(high_confidence / actionable, 4)
        if actionable
        else None
    )

    cpse_count = len(
        {
            str(m.get("cpse") or "CPSE_GENERIC")
            for m in materials
        }
    )

    return {
        "total_materials": len(materials),
        "cpse_count": cpse_count,
        "total_candidate_pairs": len(candidates),
        "high_confidence": high_confidence,
        "review_recommendations": review_recommendations,
        "review_pending": pending,
        "approved": approved,
        "rejected": rejected,
        "automation_rate": automation_rate,
        "gate_breakdown": {
            "pass": gate_pass,
            "unknown": gate_unknown,
            "conflict": gate_conflict,
        },
    }


@router.get("/by-cpse")
def analytics_by_cpse(
    current_user: User = Depends(get_current_active_user),
):
    material_counts: dict[str, int] = {}
    candidate_counts: dict[str, int] = {}

    for material in _materials():
        cpse = material.get("cpse") or "CPSE_GENERIC"
        material_counts[cpse] = (
            material_counts.get(cpse, 0) + 1
        )

    for candidate in _candidates():
        source_cpse = (
            candidate.get("source_cpse")
            or "CPSE_GENERIC"
        )
        target_cpse = (
            candidate.get("target_cpse")
            or "CPSE_GENERIC"
        )

        candidate_counts[source_cpse] = (
            candidate_counts.get(source_cpse, 0) + 1
        )

        candidate_counts[target_cpse] = (
            candidate_counts.get(target_cpse, 0) + 1
        )

    rows = []

    for cpse in sorted(material_counts):
        rows.append(
            {
                "cpse": cpse,
                "material_count": material_counts[cpse],
                "candidate_pair_involvements": (
                    candidate_counts.get(cpse, 0)
                ),
            }
        )

    return {"cpse_breakdown": rows}


@router.get("/categories")
def analytics_categories(
    current_user: User = Depends(get_current_active_user),
):
    counts: dict[str, int] = {}

    for material in _materials():
        category = (
            material.get("category")
            or "Uncategorised"
        )

        counts[category] = (
            counts.get(category, 0) + 1
        )

    rows = sorted(
        counts.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    return {
        "category_distribution": [
            {
                "category": category,
                "count": count,
            }
            for category, count in rows
        ]
    }


@router.get("/scores")
def analytics_scores(
    current_user: User = Depends(get_current_active_user),
):
    buckets = {
        f"{i / 10:.1f}-{(i + 1) / 10:.1f}": 0
        for i in range(10)
    }

    for candidate in _candidates():
        try:
            score = float(
                candidate.get("final_score") or 0.0
            )
        except (TypeError, ValueError):
            score = 0.0

        score = max(0.0, min(1.0, score))
        index = min(int(score * 10), 9)
        key = f"{index / 10:.1f}-{(index + 1) / 10:.1f}"
        buckets[key] += 1

    return {
        "total_candidates": len(_candidates()),
        "score_histogram": [
            {
                "bucket": bucket,
                "count": count,
            }
            for bucket, count in buckets.items()
        ],
    }


@router.get("/data-quality")
def analytics_data_quality(
    current_user: User = Depends(get_current_active_user),
):
    materials = _materials()
    total = len(materials)

    if total == 0:
        return {
            "total_materials": 0,
            "with_parsed_specs": 0,
            "parsed_specs_rate": 0.0,
            "missing_description": 0,
            "missing_category": 0,
            "missing_material_grade": 0,
            "missing_dimensions": 0,
            "missing_pressure_rating": 0,
            "parsing_failures": 0,
            "completeness_score": 0.0,
            "by_cpse_quality": [],
        }

    missing_description = 0
    missing_category = 0
    missing_grade = 0
    missing_dimensions = 0
    missing_pressure = 0
    parsed_count = 0

    cpse_stats: dict[str, dict[str, int]] = {}

    for material in materials:
        cpse = material.get("cpse") or "CPSE_GENERIC"

        stats = cpse_stats.setdefault(
            cpse,
            {
                "total": 0,
                "with_parsed_specs": 0,
                "missing_grade": 0,
                "missing_dimensions": 0,
                "missing_pressure": 0,
            },
        )

        stats["total"] += 1

        if not str(
            material.get("description") or ""
        ).strip():
            missing_description += 1

        category = str(
            material.get("category") or ""
        ).strip().upper()

        if category in {
            "",
            "GENERAL",
            "UNKNOWN",
            "UNCATEGORISED",
        }:
            missing_category += 1

        parsed = (
            material.get("parsed_specifications")
            or {}
        )

        has_parsed = bool(parsed) and any(
            value is not None
            for value in parsed.values()
        )

        if has_parsed:
            parsed_count += 1
            stats["with_parsed_specs"] += 1

        grade = (
            material.get("material_grade")
            or parsed.get("material_grade")
        )

        if not grade:
            missing_grade += 1
            stats["missing_grade"] += 1

        dimensions = (
            material.get("dimensions")
            or parsed.get("dimensions")
            or parsed.get("nominal_bore")
            or parsed.get("dimension_tokens")
        )

        if not dimensions:
            missing_dimensions += 1
            stats["missing_dimensions"] += 1

        if not parsed.get("pressure_rating"):
            missing_pressure += 1
            stats["missing_pressure"] += 1

    description_rate = (
        total - missing_description
    ) / total

    category_rate = (
        total - missing_category
    ) / total

    parsed_rate = parsed_count / total

    grade_rate = (
        total - missing_grade
    ) / total

    completeness_score = round(
        description_rate * 0.30
        + category_rate * 0.20
        + parsed_rate * 0.30
        + grade_rate * 0.20,
        4,
    )

    by_cpse_quality = []

    for cpse in sorted(cpse_stats):
        stats = cpse_stats[cpse]
        cpse_total = stats["total"]

        by_cpse_quality.append(
            {
                "cpse": cpse,
                "total_materials": cpse_total,
                "with_parsed_specs": (
                    stats["with_parsed_specs"]
                ),
                "parsed_specs_rate": round(
                    stats["with_parsed_specs"]
                    / cpse_total,
                    4,
                ),
                "missing_grade": stats["missing_grade"],
                "missing_dimensions": (
                    stats["missing_dimensions"]
                ),
                "missing_pressure": (
                    stats["missing_pressure"]
                ),
            }
        )

    return {
        "total_materials": total,
        "with_parsed_specs": parsed_count,
        "parsed_specs_rate": round(
            parsed_rate,
            4,
        ),
        "missing_description": missing_description,
        "missing_category": missing_category,
        "missing_material_grade": missing_grade,
        "missing_dimensions": missing_dimensions,
        "missing_pressure_rating": missing_pressure,
        "parsing_failures": total - parsed_count,
        "completeness_score": completeness_score,
        "by_cpse_quality": by_cpse_quality,
    }