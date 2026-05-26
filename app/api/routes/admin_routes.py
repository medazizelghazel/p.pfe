from __future__ import annotations

from pydantic import BaseModel, EmailStr, Field
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app import models
from app.core.dependencies import get_current_user
from app.core.security import hash_password
from app.database import get_db
from app.services.email_service import EmailService
from app.utils.password_utils import generate_temporary_password


router = APIRouter(prefix="/api/admin", tags=["Admin"])


class TrainerCreateRequest(BaseModel):
    full_name: str = Field(min_length=2, max_length=150)
    email: EmailStr


class TrainerUpdateRequest(BaseModel):
    full_name: str = Field(min_length=2, max_length=150)
    email: EmailStr
    password: str | None = Field(default=None, min_length=6)


class UserRoleUpdateRequest(BaseModel):
    role: str = Field(pattern="^(admin|trainer)$")


def require_admin(
    current_user: models.User = Depends(get_current_user),
) -> models.User:
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required.")

    return current_user


@router.get("/stats")
def get_admin_stats(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_admin),
):
    trainers_count = (
        db.query(models.User)
        .filter(models.User.role == "trainer")
        .count()
    )

    admins_count = (
        db.query(models.User)
        .filter(models.User.role == "admin")
        .count()
    )

    videos_count = db.query(models.Video).count()
    analyses_count = db.query(models.Analysis).count()

    completed_count = (
        db.query(models.Analysis)
        .filter(models.Analysis.status == "completed")
        .count()
    )

    processing_count = (
        db.query(models.Analysis)
        .filter(models.Analysis.status.in_(["uploaded", "processing"]))
        .count()
    )

    failed_count = (
        db.query(models.Analysis)
        .filter(models.Analysis.status == "failed")
        .count()
    )

    completed_analyses = (
        db.query(models.Analysis)
        .filter(models.Analysis.status == "completed")
        .all()
    )

    if completed_analyses:
        avg_global_score = round(
            sum(a.global_score or 0 for a in completed_analyses)
            / len(completed_analyses),
            2,
        )

        avg_clarity_score = round(
            sum(a.clarity_score or 0 for a in completed_analyses)
            / len(completed_analyses),
            2,
        )

        avg_engagement_score = round(
            sum(a.engagement_score or 0 for a in completed_analyses)
            / len(completed_analyses),
            2,
        )
    else:
        avg_global_score = 0
        avg_clarity_score = 0
        avg_engagement_score = 0

    return {
        "trainers_count": trainers_count,
        "admins_count": admins_count,
        "videos_count": videos_count,
        "analyses_count": analyses_count,
        "completed_count": completed_count,
        "processing_count": processing_count,
        "failed_count": failed_count,
        "avg_global_score": avg_global_score,
        "avg_clarity_score": avg_clarity_score,
        "avg_engagement_score": avg_engagement_score,
    }


@router.get("/trainers")
def list_trainers(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_admin),
):
    users = (
        db.query(models.User)
        .filter(models.User.id != current_user.id)
        .order_by(models.User.created_at.desc())
        .all()
    )

    result = []

    for user in users:
        analyses = (
            db.query(models.Analysis)
            .filter(models.Analysis.trainer_id == user.id)
            .all()
        )

        completed = [a for a in analyses if a.status == "completed"]

        avg_score = 0
        if completed:
            avg_score = round(
                sum(a.global_score or 0 for a in completed) / len(completed),
                2,
            )

        result.append(
            {
                "id": user.id,
                "full_name": user.full_name,
                "email": user.email,
                "role": user.role,
                "created_at": user.created_at,
                "courses_count": len(analyses),
                "completed_analyses_count": len(completed),
                "average_global_score": avg_score,
            }
        )

    return result


@router.post("/trainers")
def create_trainer(
    payload: TrainerCreateRequest,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_admin),
):
    existing_user = (
        db.query(models.User)
        .filter(models.User.email == payload.email)
        .first()
    )

    if existing_user:
        raise HTTPException(status_code=400, detail="Email already exists.")

    temporary_password = generate_temporary_password()

    trainer = models.User(
        full_name=payload.full_name,
        email=payload.email,
        password_hash=hash_password(temporary_password),
        role="trainer",
    )

    db.add(trainer)
    db.commit()
    db.refresh(trainer)

    email_sent = False
    email_error = None

    try:
        EmailService().send_trainer_credentials(
            to_email=trainer.email,
            full_name=trainer.full_name,
            password=temporary_password,
        )
        email_sent = True
    except Exception as e:
        email_error = str(e)

    return {
        "id": trainer.id,
        "full_name": trainer.full_name,
        "email": trainer.email,
        "role": trainer.role,
        "created_at": trainer.created_at,
        "email_sent": email_sent,
        "email_error": email_error,
    }


@router.put("/trainers/{trainer_id}")
def update_trainer(
    trainer_id: int,
    payload: TrainerUpdateRequest,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_admin),
):
    user = (
        db.query(models.User)
        .filter(models.User.id == trainer_id)
        .first()
    )

    if user is None:
        raise HTTPException(status_code=404, detail="User not found.")

    if user.id == current_user.id:
        raise HTTPException(
            status_code=400,
            detail="You cannot update your own account from this page.",
        )

    existing_email = (
        db.query(models.User)
        .filter(
            models.User.email == payload.email,
            models.User.id != trainer_id,
        )
        .first()
    )

    if existing_email:
        raise HTTPException(status_code=400, detail="Email already used.")

    user.full_name = payload.full_name
    user.email = payload.email

    if payload.password:
        user.password_hash = hash_password(payload.password)

    db.commit()
    db.refresh(user)

    return {
        "id": user.id,
        "full_name": user.full_name,
        "email": user.email,
        "role": user.role,
        "created_at": user.created_at,
    }


@router.delete("/trainers/{trainer_id}")
def delete_trainer(
    trainer_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_admin),
):
    user = (
        db.query(models.User)
        .filter(models.User.id == trainer_id)
        .first()
    )

    if user is None:
        raise HTTPException(status_code=404, detail="User not found.")

    if user.id == current_user.id:
        raise HTTPException(
            status_code=400,
            detail="You cannot delete your own account.",
        )

    db.delete(user)
    db.commit()

    return {
        "message": "User deleted successfully.",
        "user_id": trainer_id,
    }


@router.put("/users/{user_id}/role")
def update_user_role(
    user_id: int,
    payload: UserRoleUpdateRequest,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_admin),
):
    user = (
        db.query(models.User)
        .filter(models.User.id == user_id)
        .first()
    )

    if user is None:
        raise HTTPException(status_code=404, detail="User not found.")

    if user.id == current_user.id:
        raise HTTPException(
            status_code=400,
            detail="You cannot change your own role.",
        )

    if payload.role not in {"admin", "trainer"}:
        raise HTTPException(status_code=400, detail="Invalid role.")

    user.role = payload.role

    db.commit()
    db.refresh(user)

    return {
        "id": user.id,
        "full_name": user.full_name,
        "email": user.email,
        "role": user.role,
        "created_at": user.created_at,
    }


@router.get("/courses")
def list_all_courses(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_admin),
):
    analyses = (
        db.query(models.Analysis)
        .order_by(models.Analysis.created_at.desc())
        .all()
    )

    return [
        {
            "analysis_id": analysis.analysis_id,
            "status": analysis.status,
            "progress": analysis.progress,
            "trainer_id": analysis.trainer_id,
            "trainer_name": analysis.trainer.full_name if analysis.trainer else None,
            "trainer_email": analysis.trainer.email if analysis.trainer else None,
            "video_id": analysis.video_id,
            "course_name": analysis.video.course_name if analysis.video else None,
            "course_date": analysis.video.course_date if analysis.video else None,
            "learner_count": analysis.video.learner_count if analysis.video else None,
            "description": analysis.video.description if analysis.video else None,
            "clarity_score": analysis.clarity_score,
            "engagement_score": analysis.engagement_score,
            "global_score": analysis.global_score,
            "trainer_dominant_emotion": analysis.trainer_dominant_emotion,
            "learner_dominant_emotion": analysis.learner_dominant_emotion,
            "course_title": analysis.course_title,
            "created_at": analysis.created_at,
            "completed_at": analysis.completed_at,
        }
        for analysis in analyses
    ]