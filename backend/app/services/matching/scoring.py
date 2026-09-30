from typing import Any

from app.services.matching.similarity import (
    semantic_similarity,
    specification_similarity,
    text_similarity,
    value_similarity,
)


TEXT_WEIGHT = 0.20
SEMANTIC_WEIGHT = 0.20
SPECIFICATION_WEIGHT = 0.35
GRADE_WEIGHT = 0.15
OTHER_ATTRIBUTES_WEIGHT = 0.10


def calculate_match_score(
    source: dict[str, Any],
    target: dict[str, Any],
    embedding_cache: Any = None,
) -> dict[str, float]:
    """
    Calculate the complete material-pair score.

    The embedding cache is passed to semantic similarity so MIRA.ai
    does not encode the same description repeatedly.
    """
    text_score = text_similarity(
        source.get(
            "normalized_description",
            "",
        ),
        target.get(
            "normalized_description",
            "",
        ),
    )

    semantic_score = semantic_similarity(
        source.get(
            "normalized_description",
            "",
        ),
        target.get(
            "normalized_description",
            "",
        ),
        embedding_cache=embedding_cache,
    )

    specification_score = specification_similarity(
        source.get("parsed_specifications"),
        target.get("parsed_specifications"),
        source.get("category"),
    )

    grade_score = value_similarity(
        source.get("material_grade"),
        target.get("material_grade"),
    )

    left_other = source.get("other_attributes")
    right_other = target.get("other_attributes")

    if not left_other and not right_other:
        other_score = 1.0
    else:
        other_score = specification_similarity(
            left_other,
            right_other,
        )

    final_score = (
        TEXT_WEIGHT * text_score
        + SEMANTIC_WEIGHT * semantic_score
        + SPECIFICATION_WEIGHT * specification_score
        + GRADE_WEIGHT * grade_score
        + OTHER_ATTRIBUTES_WEIGHT * other_score
    )

    return {
        "text_similarity": round(
            text_score,
            4,
        ),
        "semantic_similarity": round(
            semantic_score,
            4,
        ),
        "specification_similarity": round(
            specification_score,
            4,
        ),
        "material_grade_similarity": round(
            grade_score,
            4,
        ),
        "other_attributes_similarity": round(
            other_score,
            4,
        ),
        "final_score": round(
            final_score,
            4,
        ),
    }