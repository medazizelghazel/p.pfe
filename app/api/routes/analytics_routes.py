from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from app import models
from app.core.dependencies import get_current_user
from app.database import get_db
from app.services.analytics_service import AnalyticsService


router = APIRouter(prefix="/api/analytics", tags=["Analytics"])


def _is_admin(user: models.User) -> bool:
    return user.role == "admin"


def _can_access_analysis(user: models.User, analysis: models.Analysis) -> bool:
    return _is_admin(user) or analysis.trainer_id == user.id


def _course_name(analysis: models.Analysis) -> str:
    if analysis.video and analysis.video.course_name:
        return analysis.video.course_name

    if analysis.course_title:
        return analysis.course_title

    return "Cours sans titre"


def _trainer_name(analysis: models.Analysis) -> str | None:
    if analysis.trainer:
        return analysis.trainer.full_name

    return None


def _safe_round(value: Any, digits: int = 2) -> float:
    try:
        if value is None:
            return 0.0
        return round(float(value), digits)
    except Exception:
        return 0.0


# ============================================================
# COURSE BADGES
# ============================================================

@router.get("/course/{analysis_id}/badges")
def get_course_badges(
    analysis_id: str,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    analysis = (
        db.query(models.Analysis)
        .filter(models.Analysis.analysis_id == analysis_id)
        .first()
    )

    if analysis is None:
        raise HTTPException(status_code=404, detail="Analysis not found.")

    if not _can_access_analysis(current_user, analysis):
        raise HTTPException(status_code=403, detail="Access denied.")

    badges = (
        db.query(models.CourseBadge)
        .filter(models.CourseBadge.analysis_id == analysis.id)
        .order_by(models.CourseBadge.created_at.desc())
        .all()
    )

    return [
        {
            "id": badge.id,
            "badge_key": badge.badge_key,
            "badge_label": badge.badge_label,
            "badge_icon": badge.badge_icon,
            "badge_reason": badge.badge_reason,
            "score_value": badge.score_value,
            "threshold_value": badge.threshold_value,
            "created_at": badge.created_at,
        }
        for badge in badges
    ]


# ============================================================
# COURSE INSIGHTS
# ============================================================

@router.get("/course/{analysis_id}/insights")
def get_course_insights(
    analysis_id: str,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    analysis = (
        db.query(models.Analysis)
        .filter(models.Analysis.analysis_id == analysis_id)
        .first()
    )

    if analysis is None:
        raise HTTPException(status_code=404, detail="Analysis not found.")

    if not _can_access_analysis(current_user, analysis):
        raise HTTPException(status_code=403, detail="Access denied.")

    insights = (
        db.query(models.AnalysisInsight)
        .filter(models.AnalysisInsight.analysis_id == analysis.id)
        .order_by(models.AnalysisInsight.created_at.desc())
        .all()
    )

    return [
        {
            "id": insight.id,
            "insight_type": insight.insight_type,
            "title": insight.title,
            "description": insight.description,
            "start_sec": insight.start_sec,
            "end_sec": insight.end_sec,
            "severity": insight.severity,
            "score": insight.score,
            "metadata": insight.metadata_json,
            "created_at": insight.created_at,
        }
        for insight in insights
    ]


# ============================================================
# COURSE PERCENTILE
# ============================================================

@router.get("/course/{analysis_id}/percentile")
def get_course_percentile(
    analysis_id: str,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    analysis = (
        db.query(models.Analysis)
        .filter(models.Analysis.analysis_id == analysis_id)
        .first()
    )

    if analysis is None:
        raise HTTPException(status_code=404, detail="Analysis not found.")

    if not _can_access_analysis(current_user, analysis):
        raise HTTPException(status_code=403, detail="Access denied.")

    percentile = AnalyticsService().calculate_course_percentile(
        db=db,
        analysis=analysis,
    )

    return {
        "analysis_id": analysis.analysis_id,
        "course_name": _course_name(analysis),
        "global_score": analysis.global_score,
        "percentile": percentile,
        "message": f"Ce cours est dans le top {round(100 - percentile, 2)}% de la plateforme."
        if percentile < 100
        else "Ce cours est parmi les meilleurs cours de la plateforme.",
    }


# ============================================================
# COURSE LEADERBOARD
# ============================================================

@router.get("/leaderboard/courses")
def get_courses_leaderboard(
    category: str = "global",
    limit: int = 10,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    if limit < 1:
        limit = 10

    if limit > 50:
        limit = 50

    score_column = models.Analysis.global_score

    if category == "clarity":
        score_column = models.Analysis.clarity_score
    elif category == "engagement":
        score_column = models.Analysis.engagement_score
    elif category == "interaction":
        score_column = models.Analysis.interaction_component
    elif category == "participation":
        score_column = models.Analysis.learner_participation_component
    elif category == "vocal":
        score_column = models.Analysis.trainer_vocal_component
    elif category == "global":
        score_column = models.Analysis.global_score
    else:
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid category. Use one of: "
                "global, clarity, engagement, interaction, participation, vocal."
            ),
        )

    query = (
        db.query(models.Analysis)
        .filter(models.Analysis.status == "completed")
        .order_by(desc(score_column))
    )

    # Trainer sees only his own courses.
    # Admin sees all platform courses.
    if not _is_admin(current_user):
        query = query.filter(models.Analysis.trainer_id == current_user.id)

    analyses = query.limit(limit).all()

    result = []

    for index, analysis in enumerate(analyses, start=1):
        score_value = getattr(analysis, score_column.key, 0)

        result.append(
            {
                "rank": index,
                "analysis_id": analysis.analysis_id,
                "course_name": _course_name(analysis),
                "trainer_id": analysis.trainer_id,
                "trainer_name": _trainer_name(analysis),
                "category": category,
                "score": _safe_round(score_value),
                "global_score": _safe_round(analysis.global_score),
                "clarity_score": _safe_round(analysis.clarity_score),
                "engagement_score": _safe_round(analysis.engagement_score),
                "interaction_component": _safe_round(analysis.interaction_component),
                "learner_participation_component": _safe_round(
                    analysis.learner_participation_component
                ),
                "trainer_vocal_component": _safe_round(
                    analysis.trainer_vocal_component
                ),
                "trainer_dominant_emotion": analysis.trainer_dominant_emotion,
                "learner_dominant_emotion": analysis.learner_dominant_emotion,
                "completed_at": analysis.completed_at,
                "created_at": analysis.created_at,
            }
        )

    return {
        "category": category,
        "limit": limit,
        "items": result,
    }


# ============================================================
# TRAINER LEADERBOARD
# ============================================================

@router.get("/leaderboard/trainers")
def get_trainers_leaderboard(
    limit: int = 10,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    if not _is_admin(current_user):
        raise HTTPException(status_code=403, detail="Admin access required.")

    if limit < 1:
        limit = 10

    if limit > 50:
        limit = 50

    rows = (
        db.query(
            models.User.id.label("trainer_id"),
            models.User.full_name.label("trainer_name"),
            models.User.email.label("trainer_email"),
            func.count(models.Analysis.id).label("completed_courses"),
            func.avg(models.Analysis.global_score).label("avg_global_score"),
            func.avg(models.Analysis.clarity_score).label("avg_clarity_score"),
            func.avg(models.Analysis.engagement_score).label("avg_engagement_score"),
            func.avg(models.Analysis.interaction_component).label("avg_interaction"),
        )
        .join(models.Analysis, models.Analysis.trainer_id == models.User.id)
        .filter(models.User.role == "trainer")
        .filter(models.Analysis.status == "completed")
        .group_by(models.User.id, models.User.full_name, models.User.email)
        .order_by(desc(func.avg(models.Analysis.global_score)))
        .limit(limit)
        .all()
    )

    return [
        {
            "rank": index,
            "trainer_id": row.trainer_id,
            "trainer_name": row.trainer_name,
            "trainer_email": row.trainer_email,
            "completed_courses": row.completed_courses,
            "avg_global_score": _safe_round(row.avg_global_score),
            "avg_clarity_score": _safe_round(row.avg_clarity_score),
            "avg_engagement_score": _safe_round(row.avg_engagement_score),
            "avg_interaction": _safe_round(row.avg_interaction),
        }
        for index, row in enumerate(rows, start=1)
    ]


# ============================================================
# HALL OF FAME
# ============================================================

@router.get("/hall-of-fame")
def get_hall_of_fame(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    query = db.query(models.Analysis).filter(models.Analysis.status == "completed")

    if not _is_admin(current_user):
        query = query.filter(models.Analysis.trainer_id == current_user.id)

    best_global = query.order_by(desc(models.Analysis.global_score)).first()
    best_clarity = query.order_by(desc(models.Analysis.clarity_score)).first()
    best_engagement = query.order_by(desc(models.Analysis.engagement_score)).first()
    best_interaction = query.order_by(desc(models.Analysis.interaction_component)).first()

    def serialize_item(analysis: models.Analysis | None, category: str):
        if analysis is None:
            return None

        score = analysis.global_score

        if category == "clarity":
            score = analysis.clarity_score
        elif category == "engagement":
            score = analysis.engagement_score
        elif category == "interaction":
            score = analysis.interaction_component

        return {
            "analysis_id": analysis.analysis_id,
            "course_name": _course_name(analysis),
            "trainer_id": analysis.trainer_id,
            "trainer_name": _trainer_name(analysis),
            "score": _safe_round(score),
            "global_score": _safe_round(analysis.global_score),
            "completed_at": analysis.completed_at,
        }

    return {
        "best_global_course": serialize_item(best_global, "global"),
        "clearest_course": serialize_item(best_clarity, "clarity"),
        "most_engaging_course": serialize_item(best_engagement, "engagement"),
        "most_interactive_course": serialize_item(best_interaction, "interaction"),
    }


# ============================================================
# NOTIFICATIONS
# ============================================================

@router.get("/notifications")
def get_my_notifications(
    unread_only: bool = False,
    limit: int = 20,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    if limit < 1:
        limit = 20

    if limit > 100:
        limit = 100

    query = (
        db.query(models.Notification)
        .filter(models.Notification.user_id == current_user.id)
    )

    if unread_only:
        query = query.filter(models.Notification.is_read == False)

    notifications = (
        query
        .order_by(models.Notification.created_at.desc())
        .limit(limit)
        .all()
    )

    return [
        {
            "id": notification.id,
            "notification_type": notification.notification_type,
            "title": notification.title,
            "message": notification.message,
            "priority": notification.priority,
            "is_read": notification.is_read,
            "analysis_id": notification.analysis.analysis_id
            if notification.analysis
            else None,
            "metadata": notification.metadata_json,
            "created_at": notification.created_at,
            "read_at": notification.read_at,
        }
        for notification in notifications
    ]


@router.patch("/notifications/{notification_id}/read")
def mark_notification_as_read(
    notification_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    notification = (
        db.query(models.Notification)
        .filter(models.Notification.id == notification_id)
        .filter(models.Notification.user_id == current_user.id)
        .first()
    )

    if notification is None:
        raise HTTPException(status_code=404, detail="Notification not found.")

    notification.is_read = True
    notification.read_at = datetime.utcnow()

    db.commit()
    db.refresh(notification)

    return {
        "id": notification.id,
        "is_read": notification.is_read,
        "read_at": notification.read_at,
    }


@router.patch("/notifications/read-all")
def mark_all_notifications_as_read(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    notifications = (
        db.query(models.Notification)
        .filter(models.Notification.user_id == current_user.id)
        .filter(models.Notification.is_read == False)
        .all()
    )

    now = datetime.utcnow()

    for notification in notifications:
        notification.is_read = True
        notification.read_at = now

    db.commit()

    return {
        "updated_count": len(notifications),
        "message": "All notifications marked as read.",
    }