from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

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
    def create_video_and_job(
        self,
        db: Session,
        analysis_id: str,
        original_filename: str,
        stored_filename: str,
        video_path: str,
        file_size_bytes: int | None = None,
        mime_type: str | None = None,
        user_id: int | None = None,
    ) -> models.AnalysisJob:
        video = models.CourseVideo(
            user_id=user_id,
            original_filename=original_filename,
            stored_filename=stored_filename,
            video_path=video_path,
            file_size_bytes=file_size_bytes,
            mime_type=mime_type,
        )

        db.add(video)
        db.flush()

        job = models.AnalysisJob(
            analysis_id=analysis_id,
            video_id=video.id,
            status="uploaded",
            progress=0,
            message="Video uploaded successfully.",
        )

        db.add(job)
        db.commit()
        db.refresh(job)

        return job

    def get_job_by_analysis_id(
        self,
        db: Session,
        analysis_id: str,
    ) -> models.AnalysisJob | None:
        return (
            db.query(models.AnalysisJob)
            .filter(models.AnalysisJob.analysis_id == analysis_id)
            .first()
        )

    def update_job_status(
        self,
        db: Session,
        analysis_id: str,
        status: str,
        progress: int,
        message: str | None = None,
        error: str | None = None,
    ) -> None:
        job = self.get_job_by_analysis_id(db, analysis_id)

        if not job:
            return

        job.status = status
        job.progress = progress
        job.message = message
        job.error = error
        job.updated_at = datetime.now()

        if status == "processing" and job.started_at is None:
            job.started_at = datetime.now()

        if status in {"completed", "failed"}:
            job.completed_at = datetime.now()

        db.commit()

    def save_full_analysis_result(
        self,
        db: Session,
        analysis_id: str,
        result: AnalysisResult,
    ) -> models.AnalysisResult:
        job = self.get_job_by_analysis_id(db, analysis_id)

        if not job:
            raise ValueError(f"Analysis job not found: {analysis_id}")

        existing = (
            db.query(models.AnalysisResult)
            .filter(models.AnalysisResult.job_id == job.id)
            .first()
        )

        if existing:
            db.delete(existing)
            db.flush()

        interpretation = result.get_interpretation()

        analysis_result = models.AnalysisResult(
            job_id=job.id,

            extracted_audio_path=result.extracted_audio_path,
            processed_audio_path=result.processed_audio_path,
            scoring_audio_source=result.scoring_audio_source,

            sample_rate=result.sample_rate,
            duration_seconds=result.duration,

            clarity_score=result.clarity_score,
            engagement_score=result.engagement_score,
            global_score=result.global_score,

            engagement_label=result.engagement_label,
            engagement_method=result.engagement_method,

            dominant_emotion=result.dominant_emotion,
            emotion_confidence=result.emotion_confidence,
            emotion_num_segments=result.emotion_num_segments,

            interpretation=interpretation,
        )

        db.add(analysis_result)
        db.flush()

        self._save_features(db, analysis_result.id, result)
        self._save_emotion(db, analysis_result.id, result)
        self._save_engagement(db, analysis_result.id, result)
        self._save_diarization(db, analysis_result.id, result)
        self._save_transcription(db, analysis_result.id, result)
        self._save_summary(db, analysis_result.id, result)
        self._save_reports(db, analysis_result.id, result)

        job.status = "completed"
        job.progress = 100
        job.message = "Analysis completed successfully."
        job.completed_at = datetime.now()
        job.updated_at = datetime.now()

        db.commit()
        db.refresh(analysis_result)

        return analysis_result

    def _save_features(
        self,
        db: Session,
        result_id: int,
        result: AnalysisResult,
    ) -> None:
        features = result.features

        row = models.AnalysisFeature(
            result_id=result_id,
            pitch_mean=features.pitch_mean,
            pitch_std=features.pitch_std,
            energy_mean=features.energy_mean,
            energy_std=features.energy_std,
            zcr_mean=features.zcr_mean,
            zcr_std=features.zcr_std,
            spectral_centroid_mean=features.spectral_centroid_mean,
            spectral_centroid_std=features.spectral_centroid_std,
            spectral_bandwidth_mean=features.spectral_bandwidth_mean,
            spectral_bandwidth_std=features.spectral_bandwidth_std,
            voiced_ratio=features.voiced_ratio,
            silence_ratio=features.silence_ratio,
            pause_count=features.pause_count,
            mean_pause_duration=features.mean_pause_duration,
            total_pause_duration=features.total_pause_duration,
            mfcc_means=features.mfcc_means,
        )

        db.add(row)

    def _save_emotion(
        self,
        db: Session,
        result_id: int,
        result: AnalysisResult,
    ) -> None:
        scores = result.emotion_aggregated_scores or {}

        row = models.EmotionResult(
            result_id=result_id,
            dominant_emotion=result.dominant_emotion,
            confidence=result.emotion_confidence,
            num_segments=result.emotion_num_segments,
            model_used=result.emotion_model_used,
            neutral_calm=_num(scores.get("neutral_calm")),
            energetic_engaged=_num(scores.get("energetic_engaged")),
            low_energy=_num(scores.get("low_energy")),
            tense_stressed=_num(scores.get("tense_stressed")),
            model_metrics=result.emotion_model_metrics,
            segment_predictions=result.emotion_segment_predictions,
        )

        db.add(row)

    def _save_engagement(
        self,
        db: Session,
        result_id: int,
        result: AnalysisResult,
    ) -> None:
        details = result.engagement_details or {}

        row = models.EngagementResult(
            result_id=result_id,

            engagement_score=result.engagement_score,
            engagement_label=result.engagement_label,
            method=result.engagement_method,

            trainer_vocal_component=_num(details.get("trainer_vocal_component")),
            learner_participation_component=_num(details.get("learner_participation_component")),
            interaction_component=_num(details.get("interaction_component")),

            pitch_std=_num(details.get("pitch_std")),
            energy_std=_num(details.get("energy_std")),
            voiced_ratio=_num(details.get("voiced_ratio")),
            speech_ratio=_num(details.get("speech_ratio")),
            pause_ratio=_num(details.get("pause_ratio")),
            onset_rate_per_sec=_num(details.get("onset_rate_per_sec")),

            session_duration_sec=_num(details.get("session_duration_sec")),
            trainer_total_speech_duration=_num(details.get("trainer_total_speech_duration")),
            learner_total_speech_duration=_num(details.get("learner_total_speech_duration")),

            trainer_talk_ratio=_num(details.get("trainer_talk_ratio")),
            learner_talk_ratio=_num(details.get("learner_talk_ratio")),
            trainer_speech_share=_num(details.get("trainer_speech_share")),
            learner_speech_share=_num(details.get("learner_speech_share")),

            trainer_turn_count=_int(details.get("trainer_turn_count")),
            learner_turn_count=_int(details.get("learner_turn_count")),
            total_turn_count=_int(details.get("total_turn_count")),

            active_learner_speaker_count=_int(details.get("active_learner_speaker_count")),
            learner_avg_turn_duration=_num(details.get("learner_avg_turn_duration")),
            learner_max_turn_duration=_num(details.get("learner_max_turn_duration")),
            learner_turns_per_min=_num(details.get("learner_turns_per_min")),

            interaction_turn_count=_int(details.get("interaction_turn_count")),
            interaction_rate_per_min=_num(details.get("interaction_rate_per_min")),
            trainer_learner_balance=_num(details.get("trainer_learner_balance")),

            learner_response_count=_int(details.get("learner_response_count")),
            learner_response_ratio=_num(details.get("learner_response_ratio")),

            overlap_segment_count=_int(details.get("overlap_segment_count")),
            overlap_ratio=_num(details.get("overlap_ratio")),

            details=details,
        )

        db.add(row)

    def _save_diarization(
        self,
        db: Session,
        result_id: int,
        result: AnalysisResult,
    ) -> None:
        row = models.DiarizationResult(
            result_id=result_id,
            enabled=result.diarization_enabled,

            trainer_speaker_id=result.trainer_speaker_id,
            trainer_detection_confidence=result.trainer_detection_confidence,

            raw_csv_path=result.diarization_raw_csv_path,
            clean_csv_path=result.diarization_clean_csv_path,
            rttm_path=result.diarization_rttm_path,
            profiles_json_path=result.diarization_profiles_json_path,

            real_speakers=result.diarization_real_speakers,
            artifact_speakers=result.diarization_artifact_speakers,

            trainer_audio_path=result.trainer_audio_path,
            trainer_audio_segments_csv_path=result.trainer_audio_segments_csv_path,
            trainer_audio_segment_count=result.trainer_audio_segment_count,
            trainer_audio_duration=result.trainer_audio_duration,
        )

        db.add(row)

    def _save_transcription(
        self,
        db: Session,
        result_id: int,
        result: AnalysisResult,
    ) -> None:
        full_text = None
        segments = None

        row = models.Transcription(
            result_id=result_id,
            enabled=result.transcription_enabled,

            transcript_json_path=result.transcript_json_path,
            transcript_txt_path=result.transcript_txt_path,

            language=result.transcript_language,
            language_confidence=result.transcript_language_confidence,
            segments_count=result.transcript_segments_count,
            total_duration=result.transcript_total_duration,

            full_text=full_text,
            segments=segments,
        )

        db.add(row)

    def _save_summary(
        self,
        db: Session,
        result_id: int,
        result: AnalysisResult,
    ) -> None:
        row = models.CourseSummary(
            result_id=result_id,
            enabled=result.summary_enabled,

            summary_json_path=result.summary_json_path,
            course_title=result.course_title,
            course_language=result.course_language,

            sections_count=result.course_sections_count,
            key_points=result.course_key_points,
            summary=result.course_summary,
        )

        db.add(row)

    def _save_reports(
        self,
        db: Session,
        result_id: int,
        result: AnalysisResult,
    ) -> None:
        # Your current ReportGenerator creates PDF files in data/reports.
        # If your AnalysisResult later contains report path, add it here.
        if result.summary_json_path:
            possible_pdf = str(Path(result.summary_json_path).with_suffix(".pdf"))
            row = models.GeneratedReport(
                result_id=result_id,
                report_type="course_summary_pdf",
                file_path=possible_pdf,
            )
            db.add(row)