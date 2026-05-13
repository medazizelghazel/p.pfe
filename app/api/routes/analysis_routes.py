from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app import models
from app.config import INPUT_VIDEOS_DIR, REPORTS_DIR, ensure_directories
from app.core.dependencies import get_current_user
from app.database import get_db
from app.services.analysis_job_service import AnalysisJobService
from app.services.db_analysis_service import DBAnalysisService

router = APIRouter(prefix="/api/analysis", tags=["Analysis"])

job_service = AnalysisJobService()
db_service = DBAnalysisService()


def _safe_filename(filename: str) -> str:
    return Path(filename).name.replace(" ", "_")


def _is_admin(user: models.User) -> bool:
    return user.role == "admin"


def _can_access_analysis(user: models.User, analysis: models.Analysis) -> bool:
    return _is_admin(user) or analysis.trainer_id == user.id


@router.post("/upload")
def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    ensure_directories()

    if current_user.role not in {"admin", "trainer"}:
        raise HTTPException(status_code=403, detail="Not allowed.")

    allowed_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    original_name = _safe_filename(file.filename or "uploaded_video.mp4")
    extension = Path(original_name).suffix.lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format: {extension}",
        )

    analysis_id = str(uuid.uuid4())
    stored_filename = f"{analysis_id}_{original_name}"
    saved_path = INPUT_VIDEOS_DIR / stored_filename

    with saved_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    file_size_bytes = saved_path.stat().st_size if saved_path.exists() else None

    db_service.create_video_and_analysis(
        db=db,
        analysis_id=analysis_id,
        original_filename=original_name,
        stored_filename=stored_filename,
        video_path=str(saved_path),
        file_size_bytes=file_size_bytes,
        mime_type=file.content_type,
        trainer_id=current_user.id,
    )

    background_tasks.add_task(
        job_service.run_analysis_job,
        analysis_id,
        str(saved_path),
    )

    return {
        "analysis_id": analysis_id,
        "status": "uploaded",
        "message": "Video uploaded. Analysis started in background.",
        "video_path": str(saved_path),
    }


@router.get("")
def list_analyses(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    query = db.query(models.Analysis)

    if not _is_admin(current_user):
        query = query.filter(models.Analysis.trainer_id == current_user.id)

    analyses = query.order_by(models.Analysis.created_at.desc()).all()

    return [
        {
            "analysis_id": analysis.analysis_id,
            "status": analysis.status,
            "progress": analysis.progress,
            "trainer_id": analysis.trainer_id,
            "video_id": analysis.video_id,
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


@router.get("/{analysis_id}/status")
def get_analysis_status(
    analysis_id: str,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    analysis = db_service.get_analysis_by_analysis_id(db, analysis_id)

    if analysis is None:
        raise HTTPException(status_code=404, detail="Analysis not found.")

    if not _can_access_analysis(current_user, analysis):
        raise HTTPException(status_code=403, detail="Access denied.")

    return {
        "analysis_id": analysis.analysis_id,
        "status": analysis.status,
        "progress": analysis.progress,
        "message": analysis.message,
        "error": analysis.error,
        "created_at": analysis.created_at,
        "updated_at": analysis.updated_at,
        "started_at": analysis.started_at,
        "completed_at": analysis.completed_at,
    }


@router.get("/{analysis_id}/result")
def get_analysis_result(
    analysis_id: str,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    analysis = db_service.get_analysis_by_analysis_id(db, analysis_id)

    if analysis is None:
        raise HTTPException(status_code=404, detail="Analysis not found.")

    if not _can_access_analysis(current_user, analysis):
        raise HTTPException(status_code=403, detail="Access denied.")

    if analysis.status != "completed":
        return {
            "analysis_id": analysis.analysis_id,
            "status": analysis.status,
            "progress": analysis.progress,
            "message": analysis.message,
            "error": analysis.error,
        }

    return {
        "analysis_id": analysis.analysis_id,
        "status": analysis.status,
        "progress": analysis.progress,
        "result": analysis.full_result,
    }


@router.get("/{analysis_id}/dashboard")
def get_analysis_dashboard(
    analysis_id: str,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    analysis = db_service.get_analysis_by_analysis_id(db, analysis_id)

    if analysis is None:
        raise HTTPException(status_code=404, detail="Analysis not found.")

    if not _can_access_analysis(current_user, analysis):
        raise HTTPException(status_code=403, detail="Access denied.")

    if analysis.status != "completed":
        raise HTTPException(
            status_code=409,
            detail=f"Analysis is not completed yet. Current status: {analysis.status}",
        )

    return {
        "analysis_id": analysis.analysis_id,
        "dashboard": analysis.dashboard,
        "recommendations": analysis.recommendations,
        "interpretation": analysis.interpretation,
    }


@router.get("/{analysis_id}/report")
def download_analysis_report(
    analysis_id: str,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    analysis = db_service.get_analysis_by_analysis_id(db, analysis_id)

    if analysis is None:
        raise HTTPException(status_code=404, detail="Analysis not found.")

    if not _can_access_analysis(current_user, analysis):
        raise HTTPException(status_code=403, detail="Access denied.")

    if analysis.report_pdf_path and Path(analysis.report_pdf_path).exists():
        report_path = Path(analysis.report_pdf_path)
    else:
        if not REPORTS_DIR.exists():
            raise HTTPException(status_code=404, detail="Reports directory not found.")

        pdf_files = sorted(
            REPORTS_DIR.glob("*.pdf"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        if not pdf_files:
            raise HTTPException(status_code=404, detail="No PDF report found.")

        matched = [p for p in pdf_files if analysis_id in p.name]
        report_path = matched[0] if matched else pdf_files[0]

    return FileResponse(
        path=str(report_path),
        media_type="application/pdf",
        filename=report_path.name,
    )