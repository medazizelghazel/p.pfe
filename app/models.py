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

    # roles: admin / trainer
    role = Column(String(20), nullable=False, default="trainer")

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

    videos = relationship(
        "Video",
        back_populates="trainer",
        cascade="all, delete-orphan",
    )

    analyses = relationship(
        "Analysis",
        back_populates="trainer",
        cascade="all, delete-orphan",
    )

    course_badges = relationship(
        "CourseBadge",
        back_populates="trainer",
        cascade="all, delete-orphan",
    )

    analysis_insights = relationship(
        "AnalysisInsight",
        back_populates="trainer",
        cascade="all, delete-orphan",
    )

    notifications = relationship(
        "Notification",
        back_populates="user",
        cascade="all, delete-orphan",
    )


class Video(Base):
    __tablename__ = "videos"

    id = Column(Integer, primary_key=True, index=True)

    trainer_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )

    original_filename = Column(String(255), nullable=False)
    stored_filename = Column(String(255), nullable=False)
    video_path = Column(Text, nullable=False)

    # Course metadata from upload form
    course_name = Column(String(255), nullable=True)
    course_date = Column(DateTime(timezone=True), nullable=True)
    learner_count = Column(Integer, nullable=True)
    description = Column(Text, nullable=True)

    file_size_bytes = Column(BigInteger, nullable=True)
    mime_type = Column(String(100), nullable=True)
    duration_seconds = Column(Float, nullable=True)

    uploaded_at = Column(DateTime(timezone=True), server_default=func.now())

    trainer = relationship("User", back_populates="videos")

    analyses = relationship(
        "Analysis",
        back_populates="video",
        cascade="all, delete-orphan",
    )


class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True, index=True)

    analysis_id = Column(String(100), unique=True, nullable=False, index=True)

    video_id = Column(
        Integer,
        ForeignKey("videos.id", ondelete="CASCADE"),
        nullable=False,
    )

    trainer_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Job status
    status = Column(String(50), default="uploaded")
    progress = Column(Integer, default=0)
    message = Column(Text, nullable=True)
    error = Column(Text, nullable=True)

    # Main scores
    clarity_score = Column(Float, default=0)
    engagement_score = Column(Float, default=0)
    global_score = Column(Float, default=0)

    engagement_label = Column(String(50), nullable=True)
    engagement_method = Column(String(100), nullable=True)

    # Trainer emotion
    trainer_dominant_emotion = Column(String(100), nullable=True)
    trainer_emotion_confidence = Column(Float, nullable=True)
    trainer_emotion_num_segments = Column(Integer, nullable=True)
    trainer_emotion_distribution = Column(JSONB, nullable=True)
    trainer_emotion = Column(JSONB, nullable=True)

    # Learner emotion: global emotion of all learners
    learner_emotion_enabled = Column(Boolean, default=False)
    learner_dominant_emotion = Column(String(100), nullable=True)
    learner_emotion_confidence = Column(Float, nullable=True)
    learner_emotion_num_segments = Column(Integer, nullable=True)
    learner_emotion_distribution = Column(JSONB, nullable=True)
    learner_emotion_audio_path = Column(Text, nullable=True)
    learner_emotion_segments_csv_path = Column(Text, nullable=True)
    learner_emotion_segment_count = Column(Integer, default=0)
    learner_emotion_speaker_count = Column(Integer, default=0)
    learner_emotion_audio_duration = Column(Float, default=0)
    learner_emotion = Column(JSONB, nullable=True)

    # Engagement components
    trainer_vocal_component = Column(Float, nullable=True)
    learner_participation_component = Column(Float, nullable=True)
    interaction_component = Column(Float, nullable=True)

    # Participation
    trainer_talk_ratio = Column(Float, nullable=True)
    learner_talk_ratio = Column(Float, nullable=True)
    trainer_speech_share = Column(Float, nullable=True)
    learner_speech_share = Column(Float, nullable=True)

    trainer_turn_count = Column(Integer, nullable=True)
    learner_turn_count = Column(Integer, nullable=True)
    total_turn_count = Column(Integer, nullable=True)
    active_learner_speaker_count = Column(Integer, nullable=True)

    interaction_turn_count = Column(Integer, nullable=True)
    interaction_rate_per_min = Column(Float, nullable=True)
    learner_response_count = Column(Integer, nullable=True)
    learner_response_ratio = Column(Float, nullable=True)

    # Vocal indicators
    pitch_std = Column(Float, nullable=True)
    energy_std = Column(Float, nullable=True)
    voiced_ratio = Column(Float, nullable=True)
    speech_ratio = Column(Float, nullable=True)
    pause_ratio = Column(Float, nullable=True)
    onset_rate_per_sec = Column(Float, nullable=True)

    # Diarization
    diarization_enabled = Column(Boolean, default=False)
    trainer_speaker_id = Column(String(100), nullable=True)
    trainer_detection_confidence = Column(String(50), nullable=True)
    diarization_details = Column(JSONB, nullable=True)

    # Transcription
    transcription_enabled = Column(Boolean, default=False)
    transcript_json_path = Column(Text, nullable=True)
    transcript_txt_path = Column(Text, nullable=True)
    transcript_language = Column(String(20), nullable=True)
    transcript_language_confidence = Column(Float, nullable=True)
    transcript_segments_count = Column(Integer, nullable=True)

    # Summary
    summary_enabled = Column(Boolean, default=False)
    summary_json_path = Column(Text, nullable=True)
    course_title = Column(String(255), nullable=True)
    course_language = Column(String(50), nullable=True)
    course_sections_count = Column(Integer, nullable=True)
    course_key_points = Column(JSONB, nullable=True)
    course_summary = Column(JSONB, nullable=True)

    # Generated files
    extracted_audio_path = Column(Text, nullable=True)
    processed_audio_path = Column(Text, nullable=True)
    trainer_audio_path = Column(Text, nullable=True)
    report_pdf_path = Column(Text, nullable=True)

    # Dashboard-ready objects
    dashboard = Column(JSONB, nullable=True)
    recommendations = Column(JSONB, nullable=True)
    interpretation = Column(Text, nullable=True)

    # Full exported JSON result backup
    full_result = Column(JSONB, nullable=True)

    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

    video = relationship("Video", back_populates="analyses")
    trainer = relationship("User", back_populates="analyses")

    badges = relationship(
        "CourseBadge",
        back_populates="analysis",
        cascade="all, delete-orphan",
    )

    insights = relationship(
        "AnalysisInsight",
        back_populates="analysis",
        cascade="all, delete-orphan",
    )

    notifications = relationship(
        "Notification",
        back_populates="analysis",
        cascade="all, delete-orphan",
    )


class CourseBadge(Base):
    __tablename__ = "course_badges"

    id = Column(Integer, primary_key=True, index=True)

    analysis_id = Column(
        Integer,
        ForeignKey("analyses.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    trainer_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    badge_key = Column(String(100), nullable=False, index=True)
    badge_label = Column(String(150), nullable=False)
    badge_icon = Column(String(20), nullable=True)
    badge_reason = Column(Text, nullable=True)

    score_value = Column(Float, nullable=True)
    threshold_value = Column(Float, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    analysis = relationship("Analysis", back_populates="badges")
    trainer = relationship("User", back_populates="course_badges")


class AnalysisInsight(Base):
    __tablename__ = "analysis_insights"

    id = Column(Integer, primary_key=True, index=True)

    analysis_id = Column(
        Integer,
        ForeignKey("analyses.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    trainer_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    insight_type = Column(String(100), nullable=False, index=True)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)

    start_sec = Column(Float, nullable=True)
    end_sec = Column(Float, nullable=True)

    severity = Column(String(50), nullable=True)
    score = Column(Float, nullable=True)

    metadata_json = Column("metadata", JSONB, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    analysis = relationship("Analysis", back_populates="insights")
    trainer = relationship("User", back_populates="analysis_insights")


class Notification(Base):
    __tablename__ = "notifications"

    id = Column(Integer, primary_key=True, index=True)

    user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    analysis_id = Column(
        Integer,
        ForeignKey("analyses.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )

    notification_type = Column(String(100), nullable=False, index=True)
    title = Column(String(255), nullable=False)
    message = Column(Text, nullable=False)

    priority = Column(String(50), default="normal")
    is_read = Column(Boolean, default=False, index=True)

    metadata_json = Column("metadata", JSONB, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    read_at = Column(DateTime(timezone=True), nullable=True)

    user = relationship("User", back_populates="notifications")
    analysis = relationship("Analysis", back_populates="notifications")


class LeaderboardSnapshot(Base):
    __tablename__ = "leaderboard_snapshots"

    id = Column(Integer, primary_key=True, index=True)

    period_type = Column(String(20), nullable=False)
    period_value = Column(String(20), nullable=False)

    category = Column(String(100), nullable=False, index=True)
    entity_type = Column(String(50), nullable=False)
    entity_id = Column(Integer, nullable=False)

    rank_position = Column(Integer, nullable=False)
    score = Column(Float, nullable=False)

    metadata_json = Column("metadata", JSONB, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
