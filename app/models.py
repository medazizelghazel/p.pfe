from sqlalchemy import (
    Boolean,
    BigInteger,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String(150), nullable=False)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(Text, nullable=False)
    role = Column(String(50), default="user")
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    videos = relationship("CourseVideo", back_populates="user")


class CourseVideo(Base):
    __tablename__ = "course_videos"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)

    original_filename = Column(String(255), nullable=False)
    stored_filename = Column(String(255), nullable=False)
    video_path = Column(Text, nullable=False)

    file_size_bytes = Column(BigInteger, nullable=True)
    mime_type = Column(String(100), nullable=True)
    duration_seconds = Column(Float, nullable=True)

    uploaded_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="videos")
    jobs = relationship("AnalysisJob", back_populates="video", cascade="all, delete-orphan")


class AnalysisJob(Base):
    __tablename__ = "analysis_jobs"

    id = Column(Integer, primary_key=True, index=True)
    analysis_id = Column(String(100), unique=True, nullable=False, index=True)
    video_id = Column(Integer, ForeignKey("course_videos.id", ondelete="CASCADE"), nullable=False)

    status = Column(String(50), default="uploaded")
    progress = Column(Integer, default=0)
    message = Column(Text, nullable=True)
    error = Column(Text, nullable=True)

    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    video = relationship("CourseVideo", back_populates="jobs")
    result = relationship("AnalysisResult", back_populates="job", uselist=False, cascade="all, delete-orphan")


class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("analysis_jobs.id", ondelete="CASCADE"), unique=True, nullable=False)

    extracted_audio_path = Column(Text, nullable=True)
    processed_audio_path = Column(Text, nullable=True)
    scoring_audio_source = Column(String(100), nullable=True)

    sample_rate = Column(Integer, nullable=True)
    duration_seconds = Column(Float, nullable=True)

    clarity_score = Column(Float, default=0)
    engagement_score = Column(Float, default=0)
    global_score = Column(Float, default=0)

    engagement_label = Column(String(50), nullable=True)
    engagement_method = Column(String(100), nullable=True)

    dominant_emotion = Column(String(100), nullable=True)
    emotion_confidence = Column(Float, nullable=True)
    emotion_num_segments = Column(Integer, nullable=True)

    interpretation = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    job = relationship("AnalysisJob", back_populates="result")
    features = relationship("AnalysisFeature", back_populates="result", uselist=False, cascade="all, delete-orphan")
    emotion = relationship("EmotionResult", back_populates="result", uselist=False, cascade="all, delete-orphan")
    engagement = relationship("EngagementResult", back_populates="result", uselist=False, cascade="all, delete-orphan")
    diarization = relationship("DiarizationResult", back_populates="result", uselist=False, cascade="all, delete-orphan")
    transcription = relationship("Transcription", back_populates="result", uselist=False, cascade="all, delete-orphan")
    course_summary = relationship("CourseSummary", back_populates="result", uselist=False, cascade="all, delete-orphan")
    reports = relationship("GeneratedReport", back_populates="result", cascade="all, delete-orphan")


class AnalysisFeature(Base):
    __tablename__ = "analysis_features"

    id = Column(Integer, primary_key=True, index=True)
    result_id = Column(Integer, ForeignKey("analysis_results.id", ondelete="CASCADE"), unique=True, nullable=False)

    pitch_mean = Column(Float)
    pitch_std = Column(Float)

    energy_mean = Column(Float)
    energy_std = Column(Float)

    zcr_mean = Column(Float)
    zcr_std = Column(Float)

    spectral_centroid_mean = Column(Float)
    spectral_centroid_std = Column(Float)

    spectral_bandwidth_mean = Column(Float)
    spectral_bandwidth_std = Column(Float)

    voiced_ratio = Column(Float)
    silence_ratio = Column(Float)

    pause_count = Column(Integer)
    mean_pause_duration = Column(Float)
    total_pause_duration = Column(Float)

    mfcc_means = Column(JSONB)

    result = relationship("AnalysisResult", back_populates="features")


class EmotionResult(Base):
    __tablename__ = "emotion_results"

    id = Column(Integer, primary_key=True, index=True)
    result_id = Column(Integer, ForeignKey("analysis_results.id", ondelete="CASCADE"), unique=True, nullable=False)

    dominant_emotion = Column(String(100))
    confidence = Column(Float)
    num_segments = Column(Integer)
    model_used = Column(String(150))

    neutral_calm = Column(Float, default=0)
    energetic_engaged = Column(Float, default=0)
    low_energy = Column(Float, default=0)
    tense_stressed = Column(Float, default=0)

    model_metrics = Column(JSONB)
    segment_predictions = Column(JSONB)

    result = relationship("AnalysisResult", back_populates="emotion")


class EngagementResult(Base):
    __tablename__ = "engagement_results"

    id = Column(Integer, primary_key=True, index=True)
    result_id = Column(Integer, ForeignKey("analysis_results.id", ondelete="CASCADE"), unique=True, nullable=False)

    engagement_score = Column(Float)
    engagement_label = Column(String(50))
    method = Column(String(100))

    trainer_vocal_component = Column(Float)
    learner_participation_component = Column(Float)
    interaction_component = Column(Float)

    pitch_std = Column(Float)
    energy_std = Column(Float)
    voiced_ratio = Column(Float)
    speech_ratio = Column(Float)
    pause_ratio = Column(Float)
    onset_rate_per_sec = Column(Float)

    session_duration_sec = Column(Float)
    trainer_total_speech_duration = Column(Float)
    learner_total_speech_duration = Column(Float)

    trainer_talk_ratio = Column(Float)
    learner_talk_ratio = Column(Float)
    trainer_speech_share = Column(Float)
    learner_speech_share = Column(Float)

    trainer_turn_count = Column(Integer)
    learner_turn_count = Column(Integer)
    total_turn_count = Column(Integer)

    active_learner_speaker_count = Column(Integer)
    learner_avg_turn_duration = Column(Float)
    learner_max_turn_duration = Column(Float)
    learner_turns_per_min = Column(Float)

    interaction_turn_count = Column(Integer)
    interaction_rate_per_min = Column(Float)
    trainer_learner_balance = Column(Float)

    learner_response_count = Column(Integer)
    learner_response_ratio = Column(Float)

    overlap_segment_count = Column(Integer)
    overlap_ratio = Column(Float)

    details = Column(JSONB)

    result = relationship("AnalysisResult", back_populates="engagement")


class DiarizationResult(Base):
    __tablename__ = "diarization_results"

    id = Column(Integer, primary_key=True, index=True)
    result_id = Column(Integer, ForeignKey("analysis_results.id", ondelete="CASCADE"), unique=True, nullable=False)

    enabled = Column(Boolean, default=False)

    trainer_speaker_id = Column(String(100))
    trainer_detection_confidence = Column(String(50))

    raw_csv_path = Column(Text)
    clean_csv_path = Column(Text)
    rttm_path = Column(Text)
    profiles_json_path = Column(Text)

    real_speakers = Column(JSONB)
    artifact_speakers = Column(JSONB)

    trainer_audio_path = Column(Text)
    trainer_audio_segments_csv_path = Column(Text)
    trainer_audio_segment_count = Column(Integer)
    trainer_audio_duration = Column(Float)

    result = relationship("AnalysisResult", back_populates="diarization")


class Transcription(Base):
    __tablename__ = "transcriptions"

    id = Column(Integer, primary_key=True, index=True)
    result_id = Column(Integer, ForeignKey("analysis_results.id", ondelete="CASCADE"), unique=True, nullable=False)

    enabled = Column(Boolean, default=False)

    transcript_json_path = Column(Text)
    transcript_txt_path = Column(Text)

    language = Column(String(20))
    language_confidence = Column(Float)
    segments_count = Column(Integer)
    total_duration = Column(Float)

    full_text = Column(Text)
    segments = Column(JSONB)

    result = relationship("AnalysisResult", back_populates="transcription")


class CourseSummary(Base):
    __tablename__ = "course_summaries"

    id = Column(Integer, primary_key=True, index=True)
    result_id = Column(Integer, ForeignKey("analysis_results.id", ondelete="CASCADE"), unique=True, nullable=False)

    enabled = Column(Boolean, default=False)

    summary_json_path = Column(Text)
    course_title = Column(String(255))
    course_language = Column(String(50))

    sections_count = Column(Integer)
    key_points = Column(JSONB)
    summary = Column(JSONB)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    result = relationship("AnalysisResult", back_populates="course_summary")


class GeneratedReport(Base):
    __tablename__ = "generated_reports"

    id = Column(Integer, primary_key=True, index=True)
    result_id = Column(Integer, ForeignKey("analysis_results.id", ondelete="CASCADE"), nullable=False)

    report_type = Column(String(100), default="course_summary_pdf")
    file_path = Column(Text, nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    result = relationship("AnalysisResult", back_populates="reports")