from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session, joinedload

from app import models


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _round(value: Any, digits: int = 2) -> float:
    return round(_num(value), digits)


def _emotion_value(distribution: dict | None, key: str) -> float:
    if not distribution:
        return 0.0

    value = _num(distribution.get(key, 0.0))

    if value <= 1:
        return value * 100

    return value


def _safe_date(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    return None


class PlatformAnalyticsService:
    """
    Cross-course analytics service.

    This service reads existing stored analyses only.
    It does not run AI models.
    It does not modify the database.
    """

    EMOTIONS = [
        "neutral_calm",
        "energetic_engaged",
        "low_energy",
        "tense_stressed",
    ]

    def _completed_analyses(self, db: Session) -> list[models.Analysis]:
        return (
            db.query(models.Analysis)
            .options(
                joinedload(models.Analysis.video),
                joinedload(models.Analysis.trainer),
                joinedload(models.Analysis.insights),
            )
            .filter(models.Analysis.status == "completed")
            .order_by(models.Analysis.created_at.desc())
            .all()
        )

    def overview(self, db: Session) -> dict:
        analyses = self._completed_analyses(db)

        total_courses = len(analyses)

        if total_courses == 0:
            return {
                "total_completed_courses": 0,
                "avg_duration_minutes": 0,
                "avg_global_score": 0,
                "avg_clarity_score": 0,
                "avg_engagement_score": 0,
                "dominant_platform_emotion": None,
                "stress_rate": 0,
                "message": "Aucune analyse terminée disponible.",
            }

        durations = [
            _num(analysis.video.duration_seconds)
            for analysis in analyses
            if analysis.video and analysis.video.duration_seconds
        ]

        avg_duration_minutes = (
            sum(durations) / len(durations) / 60
            if durations
            else 0
        )

        avg_global_score = sum(_num(a.global_score) for a in analyses) / total_courses
        avg_clarity_score = sum(_num(a.clarity_score) for a in analyses) / total_courses
        avg_engagement_score = sum(_num(a.engagement_score) for a in analyses) / total_courses

        emotion_totals = self._aggregate_emotions(analyses)
        dominant_emotion = self._dominant_emotion(emotion_totals)

        stress_count = 0

        for analysis in analyses:
            tense_score = self._analysis_tense_score(analysis)

            if tense_score >= 35:
                stress_count += 1

        return {
            "total_completed_courses": total_courses,
            "avg_duration_minutes": _round(avg_duration_minutes),
            "avg_global_score": _round(avg_global_score),
            "avg_clarity_score": _round(avg_clarity_score),
            "avg_engagement_score": _round(avg_engagement_score),
            "dominant_platform_emotion": dominant_emotion,
            "stress_rate": _round((stress_count / total_courses) * 100),
            "emotion_distribution": self._normalize_emotions(emotion_totals),
        }

    def longest_courses(self, db: Session, limit: int = 10) -> list[dict]:
        analyses = self._completed_analyses(db)

        rows = []

        for analysis in analyses:
            video = analysis.video

            duration_seconds = self._extract_duration_seconds(analysis)

            if duration_seconds <= 0:
                continue

            rows.append(
                {
                    "analysis_id": analysis.analysis_id,
                    "course_name": (
                        video.course_name
                        if video and video.course_name
                        else analysis.course_title or "Cours sans titre"
                    ),
                    "trainer_id": analysis.trainer_id,
                    "trainer_name": analysis.trainer.full_name if analysis.trainer else None,
                    "duration_seconds": _round(duration_seconds),
                    "duration_minutes": _round(duration_seconds / 60),
                    "global_score": _round(analysis.global_score),
                    "clarity_score": _round(analysis.clarity_score),
                    "engagement_score": _round(analysis.engagement_score),
                    "created_at": analysis.created_at,
                    "completed_at": analysis.completed_at,
                }
            )

        return sorted(
            rows,
            key=lambda item: item["duration_seconds"],
            reverse=True,
        )[:limit]

    def stress_alerts(self, db: Session, limit: int = 10) -> list[dict]:
        analyses = self._completed_analyses(db)

        rows = []

        for analysis in analyses:
            tense_score = self._analysis_tense_score(analysis)
            warning_insights = self._count_warning_insights(analysis)
            temporal_stress_count = self._count_temporal_stress_insights(analysis)

            alert_score = (
                tense_score * 0.65
                + warning_insights * 8
                + temporal_stress_count * 12
            )

            if tense_score < 25 and warning_insights == 0 and temporal_stress_count == 0:
                continue

            video = analysis.video

            rows.append(
                {
                    "analysis_id": analysis.analysis_id,
                    "course_name": (
                        video.course_name
                        if video and video.course_name
                        else analysis.course_title or "Cours sans titre"
                    ),
                    "trainer_id": analysis.trainer_id,
                    "trainer_name": analysis.trainer.full_name if analysis.trainer else None,
                    "global_score": _round(analysis.global_score),
                    "trainer_dominant_emotion": analysis.trainer_dominant_emotion,
                    "learner_dominant_emotion": analysis.learner_dominant_emotion,
                    "tense_score": _round(tense_score),
                    "warning_insights_count": warning_insights,
                    "temporal_stress_count": temporal_stress_count,
                    "alert_score": _round(alert_score),
                    "created_at": analysis.created_at,
                }
            )

        return sorted(
            rows,
            key=lambda item: item["alert_score"],
            reverse=True,
        )[:limit]
    
    def _extract_duration_seconds(self, analysis: models.Analysis) -> float:
        video = analysis.video

        # 1. Source principale : table videos
        if video and video.duration_seconds:
            duration = _num(video.duration_seconds)

            if duration > 0:
                return duration

        # 2. Fallback : dashboard JSON
        dashboard = analysis.dashboard or {}

        duration = _num(
            dashboard.get("duration")
            or dashboard.get("audio", {}).get("duration")
            or dashboard.get("course_summary", {}).get("total_duration")
            or dashboard.get("transcript", {}).get("total_duration")
            or dashboard.get("learner_emotion", {}).get("audio_duration")
            or dashboard.get("trainer_emotion", {}).get("audio_duration")
        )

        if duration > 0:
            return duration

        # 3. Fallback : full_result JSON
        full_result = analysis.full_result or {}

        duration = _num(
            full_result.get("duration")
            or full_result.get("audio", {}).get("duration")
            or full_result.get("course_summary", {}).get("total_duration")
            or full_result.get("transcript", {}).get("total_duration")
            or full_result.get("learner_emotion", {}).get("audio_duration")
            or full_result.get("trainer_emotion", {}).get("audio_duration")
        )

        if duration > 0:
            return duration

        # 4. Dernier fallback : chercher le plus grand end_sec dans les segments
        max_end_sec = 0.0

        possible_sources = [
            dashboard,
            full_result,
            dashboard.get("trainer_emotion", {}),
            dashboard.get("learner_emotion", {}),
            dashboard.get("transcript", {}),
            full_result.get("trainer_emotion", {}),
            full_result.get("learner_emotion", {}),
            full_result.get("transcript", {}),
        ]

        for source in possible_sources:
            if not isinstance(source, dict):
                continue

            for key in ["segments", "segment_results", "emotion_segments", "transcript_segments"]:
                segments = source.get(key)

                if not isinstance(segments, list):
                    continue

                for segment in segments:
                    if not isinstance(segment, dict):
                        continue

                    end_sec = _num(
                        segment.get("end_sec")
                        or segment.get("end")
                        or segment.get("end_time")
                    )

                    if end_sec > max_end_sec:
                        max_end_sec = end_sec

        return max_end_sec

    def trainer_monthly_progress(
        self,
        db: Session,
        trainer_id: int | None = None,
    ) -> dict:
        query = (
            db.query(models.Analysis)
            .options(
                joinedload(models.Analysis.video),
                joinedload(models.Analysis.trainer),
            )
            .filter(models.Analysis.status == "completed")
        )

        if trainer_id is not None:
            query = query.filter(models.Analysis.trainer_id == trainer_id)

        analyses = query.order_by(models.Analysis.created_at.asc()).all()

        grouped: dict[int, dict[str, list[models.Analysis]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for analysis in analyses:
            date_value = _safe_date(analysis.completed_at) or _safe_date(analysis.created_at)

            if not date_value:
                continue

            month_key = f"{date_value.year}-{date_value.month:02d}"
            grouped[analysis.trainer_id][month_key].append(analysis)

        trainers = []

        for current_trainer_id, months in grouped.items():
            trainer_name = None

            for month_analyses in months.values():
                if month_analyses and month_analyses[0].trainer:
                    trainer_name = month_analyses[0].trainer.full_name
                    break

            monthly_rows = []

            previous_global = None

            for month_key in sorted(months.keys()):
                month_analyses = months[month_key]
                count = len(month_analyses)

                avg_global = sum(_num(a.global_score) for a in month_analyses) / count
                avg_clarity = sum(_num(a.clarity_score) for a in month_analyses) / count
                avg_engagement = sum(_num(a.engagement_score) for a in month_analyses) / count

                progression = None

                if previous_global is not None:
                    progression = avg_global - previous_global

                previous_global = avg_global

                monthly_rows.append(
                    {
                        "month": month_key,
                        "courses_count": count,
                        "avg_global_score": _round(avg_global),
                        "avg_clarity_score": _round(avg_clarity),
                        "avg_engagement_score": _round(avg_engagement),
                        "progression": _round(progression) if progression is not None else None,
                    }
                )

            trainers.append(
                {
                    "trainer_id": current_trainer_id,
                    "trainer_name": trainer_name,
                    "monthly_progress": monthly_rows,
                }
            )

        return {
            "trainer_id": trainer_id,
            "trainers": trainers,
        }

    def emotion_trends(self, db: Session, period: str = "month") -> dict:
        analyses = self._completed_analyses(db)

        grouped: dict[str, list[models.Analysis]] = defaultdict(list)

        for analysis in analyses:
            date_value = _safe_date(analysis.completed_at) or _safe_date(analysis.created_at)

            if not date_value:
                continue

            if period == "week":
                iso_year, iso_week, _ = date_value.isocalendar()
                period_key = f"{iso_year}-W{iso_week:02d}"
            else:
                period_key = f"{date_value.year}-{date_value.month:02d}"

            grouped[period_key].append(analysis)

        rows = []

        for period_key in sorted(grouped.keys()):
            period_analyses = grouped[period_key]
            emotion_totals = self._aggregate_emotions(period_analyses)

            rows.append(
                {
                    "period": period_key,
                    "courses_count": len(period_analyses),
                    "dominant_emotion": self._dominant_emotion(emotion_totals),
                    "emotion_distribution": self._normalize_emotions(emotion_totals),
                    "avg_global_score": _round(
                        sum(_num(a.global_score) for a in period_analyses)
                        / len(period_analyses)
                    ),
                    "avg_stress_score": _round(
                        sum(self._analysis_tense_score(a) for a in period_analyses)
                        / len(period_analyses)
                    ),
                }
            )

        return {
            "period": period,
            "trends": rows,
        }

    def _aggregate_emotions(
        self,
        analyses: list[models.Analysis],
    ) -> dict[str, float]:
        totals = {emotion: 0.0 for emotion in self.EMOTIONS}

        for analysis in analyses:
            trainer_distribution = analysis.trainer_emotion_distribution or {}
            learner_distribution = analysis.learner_emotion_distribution or {}

            for emotion in self.EMOTIONS:
                trainer_value = _emotion_value(trainer_distribution, emotion)
                learner_value = _emotion_value(learner_distribution, emotion)

                if learner_distribution:
                    totals[emotion] += trainer_value * 0.7 + learner_value * 0.3
                else:
                    totals[emotion] += trainer_value

        return totals

    def _normalize_emotions(self, totals: dict[str, float]) -> list[dict]:
        total_value = sum(totals.values())

        if total_value <= 0:
            return [
                {
                    "label": emotion,
                    "value": 0,
                }
                for emotion in self.EMOTIONS
            ]

        return [
            {
                "label": emotion,
                "value": _round((value / total_value) * 100),
            }
            for emotion, value in totals.items()
        ]

    def _dominant_emotion(self, totals: dict[str, float]) -> str | None:
        if not totals:
            return None

        emotion, value = max(totals.items(), key=lambda item: item[1])

        if value <= 0:
            return None

        return emotion

    def _analysis_tense_score(self, analysis: models.Analysis) -> float:
        trainer_distribution = analysis.trainer_emotion_distribution or {}
        learner_distribution = analysis.learner_emotion_distribution or {}

        trainer_tense = _emotion_value(trainer_distribution, "tense_stressed")
        learner_tense = _emotion_value(learner_distribution, "tense_stressed")

        if learner_distribution:
            return trainer_tense * 0.7 + learner_tense * 0.3

        return trainer_tense

    def _count_warning_insights(self, analysis: models.Analysis) -> int:
        return len(
            [
                insight
                for insight in analysis.insights or []
                if insight.severity == "warning"
            ]
        )

    def _count_temporal_stress_insights(self, analysis: models.Analysis) -> int:
        return len(
            [
                insight
                for insight in analysis.insights or []
                if insight.insight_type == "temporal_stress_moment"
            ]
        )