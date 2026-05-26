from __future__ import annotations

from datetime import datetime
from pathlib import Path

from sqlalchemy.orm import Session

from app import models
from app.domain.analysis_result import AnalysisResult


def _num(value, default: float = 0.0):
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _int(value, default: int = 0):
    try:
        if value is None:
            return default
        return int(float(value))
    except Exception:
        return default


class DBAnalysisService:
    def create_video_and_analysis(
        self,
        db: Session,
        analysis_id: str,
        original_filename: str,
        stored_filename: str,
        video_path: str,
        trainer_id: int,
        file_size_bytes: int | None = None,
        mime_type: str | None = None,
        course_name: str | None = None,
        course_date: datetime | None = None,
        learner_count: int | None = None,
        description: str | None = None,
    ) -> models.Analysis:
        video = models.Video(
            trainer_id=trainer_id,
            original_filename=original_filename,
            stored_filename=stored_filename,
            video_path=video_path,
            course_name=course_name,
            course_date=course_date,
            learner_count=learner_count,
            description=description,
            file_size_bytes=file_size_bytes,
            mime_type=mime_type,
        )

        db.add(video)
        db.flush()

        analysis = models.Analysis(
            analysis_id=analysis_id,
            video_id=video.id,
            trainer_id=trainer_id,
            status="uploaded",
            progress=0,
            message="Video uploaded successfully.",
            course_title=course_name,
        )

        db.add(analysis)
        db.commit()
        db.refresh(analysis)

        return analysis

    def get_analysis_by_analysis_id(
        self,
        db: Session,
        analysis_id: str,
    ) -> models.Analysis | None:
        return (
            db.query(models.Analysis)
            .filter(models.Analysis.analysis_id == analysis_id)
            .first()
        )

    def update_analysis_status(
        self,
        db: Session,
        analysis_id: str,
        status: str,
        progress: int,
        message: str | None = None,
        error: str | None = None,
    ) -> None:
        analysis = self.get_analysis_by_analysis_id(db, analysis_id)

        if not analysis:
            return

        analysis.status = status
        analysis.progress = progress
        analysis.message = message
        analysis.error = error
        analysis.updated_at = datetime.now()

        if status == "processing" and analysis.started_at is None:
            analysis.started_at = datetime.now()

        if status in {"completed", "failed"}:
            analysis.completed_at = datetime.now()

        db.commit()

    def save_full_analysis_result(
        self,
        db: Session,
        analysis_id: str,
        result: AnalysisResult,
    ) -> models.Analysis:
        analysis = self.get_analysis_by_analysis_id(db, analysis_id)

        if not analysis:
            raise ValueError(f"Analysis not found: {analysis_id}")

        details = result.engagement_details or {}
        full_result = result.to_dict()
        dashboard = result.get_dashboard_payload()
        recommendations = result.get_recommendations()
        interpretation = result.get_interpretation()

        trainer_emotion = result.trainer_emotion or {
            "enabled": True,
            "type": "trainer",
            "dominant_emotion": result.dominant_emotion,
            "confidence": result.emotion_confidence,
            "num_segments": result.emotion_num_segments,
            "aggregated_scores": result.emotion_aggregated_scores or {},
            "model_name": result.emotion_model_used,
            "model_metrics": result.emotion_model_metrics or {},
        }

        learner_emotion = result.learner_emotion or {}

        analysis.status = "completed"
        analysis.progress = 100
        analysis.message = "Analysis completed successfully."
        analysis.error = None

        # Main scores
        analysis.clarity_score = result.clarity_score
        analysis.engagement_score = result.engagement_score
        analysis.global_score = result.global_score
        analysis.engagement_label = result.engagement_label
        analysis.engagement_method = result.engagement_method

        # Trainer emotion
        analysis.trainer_dominant_emotion = trainer_emotion.get("dominant_emotion")
        analysis.trainer_emotion_confidence = _num(trainer_emotion.get("confidence"))
        analysis.trainer_emotion_num_segments = _int(trainer_emotion.get("num_segments"))
        analysis.trainer_emotion_distribution = trainer_emotion.get("aggregated_scores", {})
        analysis.trainer_emotion = trainer_emotion

        # Learner global emotion
        analysis.learner_emotion_enabled = bool(result.learner_emotion_enabled)
        analysis.learner_dominant_emotion = learner_emotion.get("dominant_emotion")
        analysis.learner_emotion_confidence = _num(learner_emotion.get("confidence"))
        analysis.learner_emotion_num_segments = _int(learner_emotion.get("num_segments"))
        analysis.learner_emotion_distribution = learner_emotion.get("aggregated_scores", {})
        analysis.learner_emotion_audio_path = result.learner_emotion_audio_path
        analysis.learner_emotion_segments_csv_path = result.learner_emotion_segments_csv_path
        analysis.learner_emotion_segment_count = result.learner_emotion_segment_count
        analysis.learner_emotion_speaker_count = result.learner_emotion_speaker_count
        analysis.learner_emotion_audio_duration = result.learner_emotion_audio_duration
        analysis.learner_emotion = learner_emotion

        # Engagement components
        analysis.trainer_vocal_component = _num(details.get("trainer_vocal_component"))
        analysis.learner_participation_component = _num(details.get("learner_participation_component"))
        analysis.interaction_component = _num(details.get("interaction_component"))

        # Participation
        analysis.trainer_talk_ratio = _num(details.get("trainer_talk_ratio"))
        analysis.learner_talk_ratio = _num(details.get("learner_talk_ratio"))
        analysis.trainer_speech_share = _num(details.get("trainer_speech_share"))
        analysis.learner_speech_share = _num(details.get("learner_speech_share"))

        analysis.trainer_turn_count = _int(details.get("trainer_turn_count"))
        analysis.learner_turn_count = _int(details.get("learner_turn_count"))
        analysis.total_turn_count = _int(details.get("total_turn_count"))
        analysis.active_learner_speaker_count = _int(details.get("active_learner_speaker_count"))

        analysis.interaction_turn_count = _int(details.get("interaction_turn_count"))
        analysis.interaction_rate_per_min = _num(details.get("interaction_rate_per_min"))
        analysis.learner_response_count = _int(details.get("learner_response_count"))
        analysis.learner_response_ratio = _num(details.get("learner_response_ratio"))

        # Vocal indicators
        analysis.pitch_std = _num(details.get("pitch_std"))
        analysis.energy_std = _num(details.get("energy_std"))
        analysis.voiced_ratio = _num(details.get("voiced_ratio"))
        analysis.speech_ratio = _num(details.get("speech_ratio"))
        analysis.pause_ratio = _num(details.get("pause_ratio"))
        analysis.onset_rate_per_sec = _num(details.get("onset_rate_per_sec"))

        # Diarization
        analysis.diarization_enabled = result.diarization_enabled
        analysis.trainer_speaker_id = result.trainer_speaker_id
        analysis.trainer_detection_confidence = result.trainer_detection_confidence
        analysis.diarization_details = {
            "rttm_path": result.diarization_rttm_path,
            "raw_csv_path": result.diarization_raw_csv_path,
            "clean_csv_path": result.diarization_clean_csv_path,
            "profiles_json_path": result.diarization_profiles_json_path,
            "real_speakers": result.diarization_real_speakers or [],
            "artifact_speakers": result.diarization_artifact_speakers or [],
            "scoring_audio_source": result.scoring_audio_source,
        }

        # Transcription
        analysis.transcription_enabled = result.transcription_enabled
        analysis.transcript_json_path = result.transcript_json_path
        analysis.transcript_txt_path = result.transcript_txt_path
        analysis.transcript_language = result.transcript_language
        analysis.transcript_language_confidence = result.transcript_language_confidence
        analysis.transcript_segments_count = result.transcript_segments_count

        # Summary
        analysis.summary_enabled = result.summary_enabled
        analysis.summary_json_path = result.summary_json_path

        # Keep uploaded course name if summary did not generate a title
        if not result.course_title and analysis.video and analysis.video.course_name:
            analysis.course_title = analysis.video.course_name
        else:
            analysis.course_title = result.course_title

        analysis.course_language = result.course_language
        analysis.course_sections_count = result.course_sections_count
        analysis.course_key_points = result.course_key_points or []
        analysis.course_summary = result.course_summary or {}

        # Files
        analysis.extracted_audio_path = result.extracted_audio_path
        analysis.processed_audio_path = result.processed_audio_path
        analysis.trainer_audio_path = result.trainer_audio_path

        if result.summary_json_path:
            possible_pdf = str(Path(result.summary_json_path).with_suffix(".pdf"))
            analysis.report_pdf_path = possible_pdf

        # Dashboard and full backup
        analysis.dashboard = dashboard
        analysis.recommendations = recommendations
        analysis.interpretation = interpretation
        analysis.full_result = full_result

        analysis.completed_at = datetime.now()
        analysis.updated_at = datetime.now()

        db.commit()
        db.refresh(analysis)

        try:
            from app.services.analytics_service import AnalyticsService

            AnalyticsService().process_completed_analysis(
                db=db,
                analysis=analysis,
            )

            db.refresh(analysis)

        except Exception as e:
            print(f"[DBAnalysisService] Analytics generation failed: {e}")

        return analysis