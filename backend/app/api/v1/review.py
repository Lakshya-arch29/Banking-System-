"""
Human review queue routes.

GET  /api/v1/review/queue
GET  /api/v1/review/queue/{id}
POST /api/v1/review/queue/{id}/action
GET  /api/v1/review/summary

Similarity finds candidates.
Technical gates identify risk.
Human reviewers control approval.
"""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app import store
from app.core.rbac import require_permission
from app.core.security import get_current_active_user
from app.models.user import User


router = APIRouter(
    prefix="/review",
    tags=["Review Queue"],
)


class ReviewActionRequest(BaseModel):
    action: str
    reviewer_comments: str | None = None
    user_id: str | None = None


def _is_reviewable(candidate: dict) -> bool:
    """
    Both REVIEW and HIGH_CONFIDENCE recommendations remain reviewable.

    HIGH_CONFIDENCE describes the engine recommendation. It does not mean
    the human approval has already happened.
    """
    return (
        (candidate.get("ai_decision") or candidate.get("engine_decision")) in {
            "HIGH_CONFIDENCE",
            "REVIEW",
        }
        and candidate.get("review_status") == "PENDING"
    )


@router.get("/queue")
def get_review_queue(
    skip: int = 0,
    limit: int = 50,
    current_user: User = Depends(get_current_active_user),
):
    """
    Return pending candidates that require human validation.
    """
    pending = [
        candidate
        for candidate in store.CANDIDATES
        if _is_reviewable(candidate)
    ]

    total = len(pending)
    paginated = pending[skip : skip + limit]

    return {
        "total_pending": total,
        "skip": skip,
        "limit": limit,
        "queue": paginated,
    }


@router.get("/queue/{candidate_id}")
def get_review_item(
    candidate_id: int,
    current_user: User = Depends(get_current_active_user),
):
    """
    Return one candidate with scores, gates and explanation.
    """
    for candidate in store.CANDIDATES:
        if candidate["id"] == candidate_id:
            return candidate

    raise HTTPException(
        status_code=404,
        detail=f"Candidate {candidate_id} not found",
    )


@router.post("/queue/{candidate_id}/action")
def submit_review_action(
    candidate_id: int,
    request: ReviewActionRequest,
    current_user: User = Depends(
        require_permission("review_match")
    ),
):
    """
    Approve or reject one candidate.

    The engine recommendation remains in ai_decision.
    The human decision is stored separately in review_status.
    """
    action = request.action.upper().strip()

    if action not in {"APPROVE", "REJECT"}:
        raise HTTPException(
            status_code=400,
            detail="action must be APPROVE or REJECT",
        )

    candidate = None

    for item in store.CANDIDATES:
        if item["id"] == candidate_id:
            candidate = item
            break

    if candidate is None:
        raise HTTPException(
            status_code=404,
            detail=f"Candidate {candidate_id} not found",
        )

    if candidate.get("review_status") not in {
        "PENDING",
        "REVIEW",
    }:
        raise HTTPException(
            status_code=409,
            detail=(
                "Candidate already has review status "
                f"'{candidate.get('review_status')}'"
            ),
        )

    actor_identity = (
        current_user.email
        if current_user and current_user.email
        else request.user_id or "reviewer"
    )

    now = datetime.now(timezone.utc).isoformat()

    candidate["review_status"] = (
        "APPROVED"
        if action == "APPROVE"
        else "REJECTED"
    )
    candidate["reviewer_id"] = actor_identity
    candidate["reviewer_comments"] = request.reviewer_comments
    candidate["reviewed_at"] = now

    # Keep the audit import local to avoid a circular import.
    from app.api.v1.audit import AUDIT_EVENTS

    event_type = (
        "MATCH_APPROVED"
        if action == "APPROVE"
        else "MATCH_REJECTED"
    )

    AUDIT_EVENTS.append(
        {
            "event_type": event_type,
            "candidate_id": candidate_id,
            "source_code": candidate.get("source_code"),
            "target_code": candidate.get("target_code"),
            "source_cpse": candidate.get("source_cpse"),
            "target_cpse": candidate.get("target_cpse"),
            "actor": actor_identity,
            "comments": request.reviewer_comments,
            "final_score": candidate.get("final_score"),
            "created_at": now,
        }
    )

    return {
        "status": "success",
        "candidate_id": candidate_id,
        "action_applied": action,
        "candidate": candidate,
    }


@router.get("/summary")
def review_summary(
    current_user: User = Depends(get_current_active_user),
):
    """
    Return review statistics.

    The engine recommendation and human review status are counted
    separately.
    """
    candidates = list(store.CANDIDATES)

    review_candidates = [
        candidate
        for candidate in candidates
        if (candidate.get("ai_decision") or candidate.get("engine_decision")) == "REVIEW"
    ]

    high_confidence_candidates = [
        candidate
        for candidate in candidates
        if (candidate.get("ai_decision") or candidate.get("engine_decision")) == "HIGH_CONFIDENCE"
    ]

    reviewable_candidates = [
        candidate
        for candidate in candidates
        if (candidate.get("ai_decision") or candidate.get("engine_decision")) in {
            "HIGH_CONFIDENCE",
            "REVIEW",
        }
    ]

    pending = sum(
        1
        for candidate in reviewable_candidates
        if candidate.get("review_status") == "PENDING"
    )

    approved = sum(
        1
        for candidate in reviewable_candidates
        if candidate.get("review_status") == "APPROVED"
    )

    rejected = sum(
        1
        for candidate in reviewable_candidates
        if candidate.get("review_status") == "REJECTED"
    )

    return {
        "total_review_queue": len(review_candidates),
        "total_reviewable": len(reviewable_candidates),
        "pending": pending,
        "approved": approved,
        "rejected": rejected,
        "high_confidence": len(high_confidence_candidates),
        "review_recommendations": len(review_candidates),
    }

