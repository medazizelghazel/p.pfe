from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app import models
from app.core.dependencies import get_current_user
from app.database import get_db
from app.services.platform_analytics_service import PlatformAnalyticsService

router = APIRouter(
    prefix="/api/platform-analytics",
    tags=["Platform Analytics"],
)

service = PlatformAnalyticsService()


def _require_admin(current_user: models.User) -> None:
    if current_user.role != "admin":
        raise HTTPException(
            status_code=403,
            detail="Only administrators can access platform analytics.",
        )


@router.get("/overview")
def get_platform_overview(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    _require_admin(current_user)
    return service.overview(db)


@router.get("/longest-courses")
def get_longest_courses(
    limit: int = Query(default=10, ge=1, le=50),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    _require_admin(current_user)
    return service.longest_courses(db, limit=limit)


@router.get("/stress-alerts")
def get_stress_alerts(
    limit: int = Query(default=10, ge=1, le=50),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    _require_admin(current_user)
    return service.stress_alerts(db, limit=limit)


@router.get("/trainer-monthly-progress")
def get_trainer_monthly_progress(
    trainer_id: int | None = Query(default=None),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    _require_admin(current_user)
    return service.trainer_monthly_progress(db, trainer_id=trainer_id)


@router.get("/emotion-trends")
def get_emotion_trends(
    period: str = Query(default="month", pattern="^(month|week)$"),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    _require_admin(current_user)
    return service.emotion_trends(db, period=period)