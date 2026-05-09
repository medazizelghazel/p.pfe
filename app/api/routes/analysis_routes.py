from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app import models
from app.config import INPUT_VIDEOS_DIR, REPORTS_DIR, ensure_directories
from app.database import get_db
from app.services.analysis_job_service import AnalysisJobService
from app.services.db_analysis_service import DBAnalysisService

router = APIRouter(prefix="/api/analysis", tags=["Analysis"])

job_service = AnalysisJobService()
db_service = DBAnalysisService()


def _safe_filename(filename: str) -> str:
    return Path(filename).name.replace(" ", "_")


@router.post("/upload")
def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    ensure_directories()

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

    db_service.create_video_and_job(
        db=db,
        analysis_id=analysis_id,
        original_filename=original_name,
        stored_filename=stored_filename,
        video_path=str(saved_path),
        file_size_bytes=file_size_bytes,
        mime_type=file.content_type,
        user_id=None,
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


@router.get("/{analysis_id}/status")
def get_analysis_status(
    analysis_id: str,
    db: Session = Depends(get_db),
):
    job = db_service.get_job_by_analysis_id(db, analysis_id)

    if job is None:
        raise HTTPException(status_code=404, detail="Analysis job not found.")

    return {
        "analysis_id": job.analysis_id,
        "status": job.status,
        "progress": job.progress,
        "message": job.message,
        "error": job.error,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "started_at": job.started_at,
        "completed_at": job.completed_at,
    }


@router.get("/{analysis_id}/result")
def get_analysis_result(
    analysis_id: str,
    db: Session = Depends(get_db),
):
    job = (
        db.query(models.AnalysisJob)
        .filter(models.AnalysisJob.analysis_id == analysis_id)
        .first()
    )

    if job is None:
        raise HTTPException(status_code=404, detail="Analysis job not found.")

    if job.status != "completed":
        return {
            "analysis_id": job.analysis_id,
            "status": job.status,
            "progress": job.progress,
            "message": job.message,
            "error": job.error,
        }

    result = job.result

    if result is None:
        raise HTTPException(status_code=404, detail="Analysis result not found.")

    return {
        "analysis_id": job.analysis_id,
        "status": job.status,
        "progress": job.progress,
        "result": {
            "id": result.id,
            "clarity_score": result.clarity_score,
            "engagement_score": result.engagement_score,
            "global_score": result.global_score,
            "dominant_emotion": result.dominant_emotion,
            "emotion_confidence": result.emotion_confidence,
            "emotion_num_segments": result.emotion_num_segments,
            "engagement_label": result.engagement_label,
            "engagement_method": result.engagement_method,
            "interpretation": result.interpretation,
            "processed_audio_path": result.processed_audio_path,
            "scoring_audio_source": result.scoring_audio_source,
            "duration_seconds": result.duration_seconds,
        },
    }


@router.get("/{analysis_id}/dashboard")
def get_analysis_dashboard(
    analysis_id: str,
    db: Session = Depends(get_db),
):
    job = (
        db.query(models.AnalysisJob)
        .filter(models.AnalysisJob.analysis_id == analysis_id)
        .first()
    )

    if job is None:
        raise HTTPException(status_code=404, detail="Analysis job not found.")

    if job.status != "completed":
        raise HTTPException(
            status_code=409,
            detail=f"Analysis is not completed yet. Current status: {job.status}",
        )

    result = job.result

    if result is None:
        raise HTTPException(status_code=404, detail="Analysis result not found.")

    emotion = result.emotion
    engagement = result.engagement
    diarization = result.diarization
    transcription = result.transcription
    course_summary = result.course_summary

    return {
        "analysis_id": job.analysis_id,
        "scores": {
            "clarity_score": result.clarity_score,
            "engagement_score": result.engagement_score,
            "global_score": result.global_score,
            "engagement_label": result.engagement_label,
            "engagement_method": result.engagement_method,
        },
        "emotion": {
            "dominant_emotion": result.dominant_emotion,
            "confidence": result.emotion_confidence,
            "num_segments": result.emotion_num_segments,
            "distribution": {
                "neutral_calm": emotion.neutral_calm if emotion else 0,
                "energetic_engaged": emotion.energetic_engaged if emotion else 0,
                "low_energy": emotion.low_energy if emotion else 0,
                "tense_stressed": emotion.tense_stressed if emotion else 0,
            },
        },
        "engagement": {
            "trainer_vocal_component": engagement.trainer_vocal_component if engagement else 0,
            "learner_participation_component": engagement.learner_participation_component if engagement else 0,
            "interaction_component": engagement.interaction_component if engagement else 0,
            "learner_talk_ratio": engagement.learner_talk_ratio if engagement else 0,
            "trainer_talk_ratio": engagement.trainer_talk_ratio if engagement else 0,
            "learner_turn_count": engagement.learner_turn_count if engagement else 0,
            "active_learner_speaker_count": engagement.active_learner_speaker_count if engagement else 0,
        },
        "diarization": {
            "enabled": diarization.enabled if diarization else False,
            "trainer_speaker_id": diarization.trainer_speaker_id if diarization else None,
            "trainer_detection_confidence": diarization.trainer_detection_confidence if diarization else None,
        },
        "transcription": {
            "enabled": transcription.enabled if transcription else False,
            "language": transcription.language if transcription else None,
            "language_confidence": transcription.language_confidence if transcription else 0,
            "segments_count": transcription.segments_count if transcription else 0,
            "transcript_txt_path": transcription.transcript_txt_path if transcription else None,
        },
        "summary": {
            "enabled": course_summary.enabled if course_summary else False,
            "course_title": course_summary.course_title if course_summary else None,
            "course_language": course_summary.course_language if course_summary else None,
            "sections_count": course_summary.sections_count if course_summary else 0,
            "key_points": course_summary.key_points if course_summary else [],
            "summary": course_summary.summary if course_summary else None,
        },
        "interpretation": result.interpretation,
    }


@router.get("/{analysis_id}/report")
def download_analysis_report(
    analysis_id: str,
    db: Session = Depends(get_db),
):
    job = db_service.get_job_by_analysis_id(db, analysis_id)

    if job is None:
        raise HTTPException(status_code=404, detail="Analysis job not found.")

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