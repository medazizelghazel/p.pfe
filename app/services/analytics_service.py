from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from app import models


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _get_emotion_score(distribution: dict | None, key: str) -> float:
    if not distribution:
        return 0.0

    value = distribution.get(key, 0.0)
    value = _num(value)

    # Some outputs use values between 0 and 1, others use 0 and 100.
    if value <= 1:
        return value * 100

    return value


class AnalyticsService:
    """
    Post-analysis service.

    This service does NOT run the AI pipeline.
    It only reads the saved Analysis object and generates:
    - badges
    - insights
    - temporal insights
    - notifications
    - percentile information
    """

    # ------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------

    def process_completed_analysis(
        self,
        db: Session,
        analysis: models.Analysis,
    ) -> None:
        """
        Called after DBAnalysisService.save_full_analysis_result().
        Safe to call multiple times: it deletes old generated items first.
        """

        if not analysis or analysis.status != "completed":
            return

        self._clear_previous_generated_data(db, analysis)

        badges = self.generate_badges(db, analysis)
        self.generate_insights(db, analysis)
        self.generate_temporal_insights(db, analysis)

        # Notifications intelligentes
        self.generate_completion_notification(db, analysis)
        self.generate_badge_notifications(db, analysis, badges)
        self.generate_stress_alert_notification(db, analysis)
        self.generate_top10_progress_notification(db, analysis)
        self.generate_recent_progress_notification(db, analysis)

        db.commit()

    # ------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------

    def _clear_previous_generated_data(
        self,
        db: Session,
        analysis: models.Analysis,
    ) -> None:
        db.query(models.CourseBadge).filter(
            models.CourseBadge.analysis_id == analysis.id
        ).delete(synchronize_session=False)

        db.query(models.AnalysisInsight).filter(
            models.AnalysisInsight.analysis_id == analysis.id
        ).delete(synchronize_session=False)

        db.query(models.Notification).filter(
            models.Notification.analysis_id == analysis.id
        ).delete(synchronize_session=False)

        db.flush()

    # ------------------------------------------------------------
    # Badges
    # ------------------------------------------------------------

    def generate_badges(
        self,
        db: Session,
        analysis: models.Analysis,
    ) -> list[models.CourseBadge]:
        badges: list[models.CourseBadge] = []

        global_score = _num(analysis.global_score)
        clarity_score = _num(analysis.clarity_score)
        interaction_component = _num(analysis.interaction_component)

        trainer_emotions = analysis.trainer_emotion_distribution or {}
        energetic_score = _get_emotion_score(trainer_emotions, "energetic_engaged")
        tense_score = _get_emotion_score(trainer_emotions, "tense_stressed")

        if global_score >= 85:
            badges.append(
                self._create_badge(
                    analysis=analysis,
                    badge_key="excellence_pedagogique",
                    badge_label="Excellence pédagogique",
                    badge_icon="🥇",
                    badge_reason="Score global supérieur ou égal à 85/100.",
                    score_value=global_score,
                    threshold_value=85,
                )
            )

        if interaction_component >= 60:
            badges.append(
                self._create_badge(
                    analysis=analysis,
                    badge_key="tres_interactif",
                    badge_label="Très interactif",
                    badge_icon="🎯",
                    badge_reason="Composante d’interaction supérieure ou égale à 60/100.",
                    score_value=interaction_component,
                    threshold_value=60,
                )
            )

        if clarity_score >= 90:
            badges.append(
                self._create_badge(
                    analysis=analysis,
                    badge_key="voix_or",
                    badge_label="Voix d’or",
                    badge_icon="🗣️",
                    badge_reason="Score de clarté vocale supérieur ou égal à 90/100.",
                    score_value=clarity_score,
                    threshold_value=90,
                )
            )

        if energetic_score >= 50:
            badges.append(
                self._create_badge(
                    analysis=analysis,
                    badge_key="energie_positive",
                    badge_label="Énergie positive",
                    badge_icon="⚡",
                    badge_reason="Émotion energetic_engaged supérieure ou égale à 50%.",
                    score_value=energetic_score,
                    threshold_value=50,
                )
            )

        if global_score < 60:
            badges.append(
                self._create_badge(
                    analysis=analysis,
                    badge_key="cours_a_surveiller",
                    badge_label="Cours à surveiller",
                    badge_icon="⚠️",
                    badge_reason="Score global inférieur à 60/100.",
                    score_value=global_score,
                    threshold_value=60,
                )
            )

        if tense_score >= 50:
            badges.append(
                self._create_badge(
                    analysis=analysis,
                    badge_key="stress_eleve",
                    badge_label="Stress élevé",
                    badge_icon="🔥",
                    badge_reason="Émotion tense_stressed supérieure ou égale à 50%.",
                    score_value=tense_score,
                    threshold_value=50,
                )
            )

        for badge in badges:
            db.add(badge)

        return badges

    def _create_badge(
        self,
        analysis: models.Analysis,
        badge_key: str,
        badge_label: str,
        badge_icon: str,
        badge_reason: str,
        score_value: float,
        threshold_value: float,
    ) -> models.CourseBadge:
        return models.CourseBadge(
            analysis_id=analysis.id,
            trainer_id=analysis.trainer_id,
            badge_key=badge_key,
            badge_label=badge_label,
            badge_icon=badge_icon,
            badge_reason=badge_reason,
            score_value=score_value,
            threshold_value=threshold_value,
        )

    # ------------------------------------------------------------
    # Global insights
    # ------------------------------------------------------------

    def generate_insights(
        self,
        db: Session,
        analysis: models.Analysis,
    ) -> list[models.AnalysisInsight]:
        insights: list[models.AnalysisInsight] = []

        global_score = _num(analysis.global_score)
        clarity_score = _num(analysis.clarity_score)
        engagement_score = _num(analysis.engagement_score)
        trainer_vocal = _num(analysis.trainer_vocal_component)
        learner_participation = _num(analysis.learner_participation_component)
        interaction = _num(analysis.interaction_component)

        trainer_emotions = analysis.trainer_emotion_distribution or {}
        energetic_score = _get_emotion_score(trainer_emotions, "energetic_engaged")
        low_energy_score = _get_emotion_score(trainer_emotions, "low_energy")
        tense_score = _get_emotion_score(trainer_emotions, "tense_stressed")

        if global_score >= 85:
            insights.append(
                self._create_insight(
                    analysis=analysis,
                    insight_type="excellent_course",
                    title="Très bonne performance globale",
                    description=(
                        "Ce cours présente un score global élevé. "
                        "La qualité pédagogique détectée est très satisfaisante."
                    ),
                    severity="positive",
                    score=global_score,
                    metadata={"global_score": global_score},
                )
            )
        elif global_score < 60:
            insights.append(
                self._create_insight(
                    analysis=analysis,
                    insight_type="low_global_score",
                    title="Score global faible",
                    description=(
                        "Ce cours présente un score global inférieur à 60/100. "
                        "Il est recommandé de revoir la clarté, l’engagement et l’interaction."
                    ),
                    severity="warning",
                    score=global_score,
                    metadata={"global_score": global_score},
                )
            )

        if clarity_score >= 90:
            insights.append(
                self._create_insight(
                    analysis=analysis,
                    insight_type="high_clarity",
                    title="Très bonne clarté vocale",
                    description="La voix du formateur est claire et facilement compréhensible.",
                    severity="positive",
                    score=clarity_score,
                    metadata={"clarity_score": clarity_score},
                )
            )
        elif clarity_score < 60:
            insights.append(
                self._create_insight(
                    analysis=analysis,
                    insight_type="low_clarity",
                    title="Clarté vocale à améliorer",
                    description=(
                        "La clarté vocale est faible. Le formateur peut améliorer "
                        "l’articulation, réduire les pauses excessives et stabiliser le rythme."
                    ),
                    severity="warning",
                    score=clarity_score,
                    metadata={"clarity_score": clarity_score},
                )
            )

        if engagement_score >= 80:
            insights.append(
                self._create_insight(
                    analysis=analysis,
                    insight_type="high_engagement",
                    title="Engagement élevé",
                    description="Le cours montre un bon niveau d’engagement global.",
                    severity="positive",
                    score=engagement_score,
                    metadata={
                        "engagement_score": engagement_score,
                        "trainer_vocal_component": trainer_vocal,
                        "learner_participation_component": learner_participation,
                        "interaction_component": interaction,
                    },
                )
            )
        elif engagement_score < 60:
            insights.append(
                self._create_insight(
                    analysis=analysis,
                    insight_type="low_engagement",
                    title="Engagement faible",
                    description=(
                        "Le niveau d’engagement est faible. Il est recommandé "
                        "d’augmenter l’interaction et de dynamiser la présentation."
                    ),
                    severity="warning",
                    score=engagement_score,
                    metadata={
                        "engagement_score": engagement_score,
                        "trainer_vocal_component": trainer_vocal,
                        "learner_participation_component": learner_participation,
                        "interaction_component": interaction,
                    },
                )
            )

        if interaction >= 60:
            insights.append(
                self._create_insight(
                    analysis=analysis,
                    insight_type="high_interaction",
                    title="Cours très interactif",
                    description=(
                        "Le cours contient un niveau d’interaction intéressant "
                        "entre le formateur et les apprenants."
                    ),
                    severity="positive",
                    score=interaction,
                    metadata={"interaction_component": interaction},
                )
            )
        elif interaction < 30:
            insights.append(
                self._create_insight(
                    analysis=analysis,
                    insight_type="low_interaction",
                    title="Interaction limitée",
                    description=(
                        "Le niveau d’interaction est faible. Le formateur peut ajouter "
                        "plus de questions, d’échanges ou d’activités participatives."
                    ),
                    severity="info",
                    score=interaction,
                    metadata={"interaction_component": interaction},
                )
            )

        if tense_score >= 50:
            insights.append(
                self._create_insight(
                    analysis=analysis,
                    insight_type="high_stress",
                    title="Niveau de tension élevé",
                    description=(
                        "Une proportion importante de segments vocaux est classée "
                        "comme tense_stressed. Cela peut indiquer une pression ou un ton tendu."
                    ),
                    severity="warning",
                    score=tense_score,
                    metadata={
                        "tense_stressed": tense_score,
                        "emotion_distribution": trainer_emotions,
                    },
                )
            )

        if low_energy_score >= 50:
            insights.append(
                self._create_insight(
                    analysis=analysis,
                    insight_type="low_energy",
                    title="Énergie vocale faible",
                    description=(
                        "Une part importante du cours présente une énergie vocale faible. "
                        "Le formateur peut varier davantage le rythme et l’intonation."
                    ),
                    severity="info",
                    score=low_energy_score,
                    metadata={
                        "low_energy": low_energy_score,
                        "emotion_distribution": trainer_emotions,
                    },
                )
            )

        if energetic_score >= 50:
            insights.append(
                self._create_insight(
                    analysis=analysis,
                    insight_type="positive_energy",
                    title="Énergie positive détectée",
                    description=(
                        "Le cours contient une proportion importante d’émotions positives "
                        "ou engagées dans la voix du formateur."
                    ),
                    severity="positive",
                    score=energetic_score,
                    metadata={
                        "energetic_engaged": energetic_score,
                        "emotion_distribution": trainer_emotions,
                    },
                )
            )

        for insight in insights:
            db.add(insight)

        return insights

    # ------------------------------------------------------------
    # Temporal insights / critical moments
    # ------------------------------------------------------------

    def generate_temporal_insights(
        self,
        db: Session,
        analysis: models.Analysis,
    ) -> list[models.AnalysisInsight]:
        """
        Generates temporal insights from emotion segment predictions.

        Examples:
        - Stress detected between 03:20 and 04:10
        - Low energy moment
        - Positive energy moment
        """

        temporal_insights: list[models.AnalysisInsight] = []
        segments = self._extract_emotion_segments(analysis)

        if not segments:
            return temporal_insights

        for index, segment in enumerate(segments):
            emotion = self._extract_segment_emotion(segment)
            confidence = self._extract_segment_confidence(segment)

            start_sec = self._extract_segment_start(segment, index)
            end_sec = self._extract_segment_end(segment, index, start_sec)

            if not emotion:
                continue

            if emotion == "tense_stressed" and confidence >= 0.45:
                temporal_insights.append(
                    self._create_insight(
                        analysis=analysis,
                        insight_type="temporal_stress_moment",
                        title="Moment de tension détecté",
                        description=(
                            "Un passage du cours présente une tension vocale élevée. "
                            "Il peut être utile de revoir le rythme, la respiration ou "
                            "le ton utilisé à ce moment."
                        ),
                        severity="warning",
                        score=round(confidence * 100, 2),
                        start_sec=start_sec,
                        end_sec=end_sec,
                        metadata={
                            "emotion": emotion,
                            "confidence": confidence,
                            "segment_index": index,
                            "source": "emotion_segment_predictions",
                        },
                    )
                )

            elif emotion == "low_energy" and confidence >= 0.45:
                temporal_insights.append(
                    self._create_insight(
                        analysis=analysis,
                        insight_type="temporal_low_energy_moment",
                        title="Baisse d’énergie détectée",
                        description=(
                            "Ce passage montre une énergie vocale faible. "
                            "Le formateur peut dynamiser cette partie avec plus "
                            "d’intonation, d’exemples ou d’interaction."
                        ),
                        severity="info",
                        score=round(confidence * 100, 2),
                        start_sec=start_sec,
                        end_sec=end_sec,
                        metadata={
                            "emotion": emotion,
                            "confidence": confidence,
                            "segment_index": index,
                            "source": "emotion_segment_predictions",
                        },
                    )
                )

            elif emotion == "energetic_engaged" and confidence >= 0.50:
                temporal_insights.append(
                    self._create_insight(
                        analysis=analysis,
                        insight_type="temporal_positive_energy_moment",
                        title="Moment d’énergie positive",
                        description=(
                            "Ce passage présente une énergie vocale positive. "
                            "Il peut correspondre à un moment pédagogique fort "
                            "ou à une bonne dynamique de présentation."
                        ),
                        severity="positive",
                        score=round(confidence * 100, 2),
                        start_sec=start_sec,
                        end_sec=end_sec,
                        metadata={
                            "emotion": emotion,
                            "confidence": confidence,
                            "segment_index": index,
                            "source": "emotion_segment_predictions",
                        },
                    )
                )

        temporal_insights = temporal_insights[:12]

        for insight in temporal_insights:
            db.add(insight)

        return temporal_insights

    def _extract_emotion_segments(
        self,
        analysis: models.Analysis,
    ) -> list[dict]:
        """
        Robust extraction of segment predictions from saved full_result.
        The exact location can vary depending on exporter structure.
        """

        full_result = analysis.full_result or {}

        possible_paths = [
            ["emotion_segment_predictions"],
            ["emotion", "segment_predictions"],
            ["trainer_emotion", "segment_predictions"],
            ["trainer_emotion", "segments"],
            ["emotion_result", "segment_predictions"],
        ]

        for path in possible_paths:
            value = full_result

            for key in path:
                if not isinstance(value, dict):
                    value = None
                    break

                value = value.get(key)

            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]

        return []

    def _extract_segment_emotion(
        self,
        segment: dict,
    ) -> str | None:
        possible_keys = [
            "emotion",
            "label",
            "predicted_emotion",
            "predicted_label",
            "class_name",
        ]

        for key in possible_keys:
            value = segment.get(key)

            if value:
                return str(value)

        scores = (
            segment.get("scores")
            or segment.get("probabilities")
            or segment.get("emotion_scores")
            or {}
        )

        if isinstance(scores, dict) and scores:
            return max(scores, key=lambda item: _num(scores.get(item)))

        return None

    def _extract_segment_confidence(
        self,
        segment: dict,
    ) -> float:
        possible_keys = [
            "confidence",
            "probability",
            "score",
            "max_score",
        ]

        for key in possible_keys:
            value = segment.get(key)

            if value is not None:
                confidence = _num(value)

                if confidence > 1:
                    return confidence / 100

                return confidence

        scores = (
            segment.get("scores")
            or segment.get("probabilities")
            or segment.get("emotion_scores")
            or {}
        )

        if isinstance(scores, dict) and scores:
            best_value = max(_num(value) for value in scores.values())

            if best_value > 1:
                return best_value / 100

            return best_value

        return 0.0

    def _extract_segment_start(
        self,
        segment: dict,
        index: int,
    ) -> float:
        possible_keys = [
            "start_sec",
            "start",
            "start_time",
            "start_seconds",
        ]

        for key in possible_keys:
            value = segment.get(key)

            if value is not None:
                return round(_num(value), 2)

        return round(index * 3.0, 2)

    def _extract_segment_end(
        self,
        segment: dict,
        index: int,
        start_sec: float,
    ) -> float:
        possible_keys = [
            "end_sec",
            "end",
            "end_time",
            "end_seconds",
        ]

        for key in possible_keys:
            value = segment.get(key)

            if value is not None:
                return round(_num(value), 2)

        duration = (
            segment.get("duration")
            or segment.get("duration_sec")
            or segment.get("chunk_duration")
        )

        if duration is not None:
            return round(start_sec + _num(duration, 3.0), 2)

        return round(start_sec + 3.0, 2)

    def _create_insight(
        self,
        analysis: models.Analysis,
        insight_type: str,
        title: str,
        description: str,
        severity: str,
        score: float | None = None,
        start_sec: float | None = None,
        end_sec: float | None = None,
        metadata: dict | None = None,
    ) -> models.AnalysisInsight:
        return models.AnalysisInsight(
            analysis_id=analysis.id,
            trainer_id=analysis.trainer_id,
            insight_type=insight_type,
            title=title,
            description=description,
            start_sec=start_sec,
            end_sec=end_sec,
            severity=severity,
            score=score,
            metadata_json=metadata or {},
        )

    # ------------------------------------------------------------
    # Notifications
    # ------------------------------------------------------------

    def generate_completion_notification(
        self,
        db: Session,
        analysis: models.Analysis,
    ) -> models.Notification:
        course_name = self._get_course_name(analysis)
        global_score = round(_num(analysis.global_score), 2)

        title = "Analyse terminée"
        message = (
            f"Votre cours « {course_name} » vient d’être analysé : "
            f"score global {global_score}/100."
        )

        if global_score >= 85:
            priority = "success"
            message += " Excellent résultat 🎉"
        elif global_score < 60:
            priority = "warning"
            message += " Ce cours mérite une attention particulière."
        else:
            priority = "normal"

        notification = models.Notification(
            user_id=analysis.trainer_id,
            analysis_id=analysis.id,
            notification_type="analysis_completed",
            title=title,
            message=message,
            priority=priority,
            is_read=False,
            metadata_json={
                "analysis_id": analysis.analysis_id,
                "course_name": course_name,
                "global_score": global_score,
                "clarity_score": round(_num(analysis.clarity_score), 2),
                "engagement_score": round(_num(analysis.engagement_score), 2),
                "completed_at": datetime.utcnow().isoformat(),
            },
        )

        db.add(notification)
        return notification

    def generate_badge_notifications(
        self,
        db: Session,
        analysis: models.Analysis,
        badges: list[models.CourseBadge],
    ) -> list[models.Notification]:
        notifications: list[models.Notification] = []

        if not badges:
            return notifications

        course_name = self._get_course_name(analysis)

        for badge in badges:
            notification = models.Notification(
                user_id=analysis.trainer_id,
                analysis_id=analysis.id,
                notification_type="badge_unlocked",
                title="Badge débloqué",
                message=(
                    f"Félicitations ! Votre cours « {course_name} » "
                    f"a débloqué le badge {badge.badge_icon or '🏅'} "
                    f"« {badge.badge_label} »."
                ),
                priority="success",
                is_read=False,
                metadata_json={
                    "analysis_id": analysis.analysis_id,
                    "course_name": course_name,
                    "badge_key": badge.badge_key,
                    "badge_label": badge.badge_label,
                    "badge_icon": badge.badge_icon,
                    "score_value": badge.score_value,
                    "threshold_value": badge.threshold_value,
                },
            )

            db.add(notification)
            notifications.append(notification)

        return notifications

    def generate_stress_alert_notification(
        self,
        db: Session,
        analysis: models.Analysis,
    ) -> models.Notification | None:
        recent_analyses = (
            db.query(models.Analysis)
            .filter(models.Analysis.trainer_id == analysis.trainer_id)
            .filter(models.Analysis.status == "completed")
            .order_by(models.Analysis.completed_at.desc())
            .limit(3)
            .all()
        )

        if len(recent_analyses) < 3:
            return None

        stress_values: list[float] = []

        for item in recent_analyses:
            distribution = item.trainer_emotion_distribution or {}
            stress = _get_emotion_score(distribution, "tense_stressed")
            stress_values.append(stress)

        avg_stress = sum(stress_values) / len(stress_values)

        if avg_stress < 45:
            return None

        notification = models.Notification(
            user_id=analysis.trainer_id,
            analysis_id=analysis.id,
            notification_type="stress_alert",
            title="Alerte stress élevé",
            message=(
                "Alerte : un niveau de stress élevé a été détecté dans vos "
                f"3 derniers cours. Moyenne estimée : {round(avg_stress, 2)}%."
            ),
            priority="warning",
            is_read=False,
            metadata_json={
                "analysis_id": analysis.analysis_id,
                "avg_stress_last_3_courses": round(avg_stress, 2),
                "stress_values": stress_values,
            },
        )

        db.add(notification)
        return notification

    def generate_top10_progress_notification(
        self,
        db: Session,
        analysis: models.Analysis,
    ) -> models.Notification | None:
        completed_count = (
            db.query(models.Analysis)
            .filter(models.Analysis.status == "completed")
            .count()
        )

        if completed_count < 10:
            return None

        top_10 = (
            db.query(models.Analysis)
            .filter(models.Analysis.status == "completed")
            .order_by(models.Analysis.global_score.desc())
            .limit(10)
            .all()
        )

        if not top_10:
            return None

        current_score = _num(analysis.global_score)
        top10_threshold = _num(top_10[-1].global_score)

        if any(item.id == analysis.id for item in top_10):
            notification = models.Notification(
                user_id=analysis.trainer_id,
                analysis_id=analysis.id,
                notification_type="top10_reached",
                title="Top 10 atteint",
                message=(
                    f"Bravo ! Votre cours « {self._get_course_name(analysis)} » "
                    "est entré dans le Top 10 de la plateforme."
                ),
                priority="success",
                is_read=False,
                metadata_json={
                    "analysis_id": analysis.analysis_id,
                    "global_score": current_score,
                    "top10_threshold": top10_threshold,
                },
            )

            db.add(notification)
            return notification

        gap = top10_threshold - current_score

        if gap <= 0 or gap > 2:
            return None

        notification = models.Notification(
            user_id=analysis.trainer_id,
            analysis_id=analysis.id,
            notification_type="top10_progress",
            title="Proche du Top 10",
            message=(
                f"Vous êtes à {round(gap, 2)} points du Top 10 ! "
                "Améliorez l’interaction et l’engagement pour progresser."
            ),
            priority="normal",
            is_read=False,
            metadata_json={
                "analysis_id": analysis.analysis_id,
                "global_score": current_score,
                "top10_threshold": top10_threshold,
                "gap_to_top10": round(gap, 2),
            },
        )

        db.add(notification)
        return notification

    def generate_recent_progress_notification(
        self,
        db: Session,
        analysis: models.Analysis,
    ) -> models.Notification | None:
        recent_analyses = (
            db.query(models.Analysis)
            .filter(models.Analysis.trainer_id == analysis.trainer_id)
            .filter(models.Analysis.status == "completed")
            .order_by(models.Analysis.completed_at.desc())
            .limit(4)
            .all()
        )

        if len(recent_analyses) < 4:
            return None

        current = recent_analyses[0]
        previous = recent_analyses[1:]

        current_engagement = _num(current.engagement_score)
        previous_avg = sum(_num(item.engagement_score) for item in previous) / len(previous)

        difference = current_engagement - previous_avg

        if abs(difference) < 5:
            return None

        if difference > 0:
            title = "Engagement en progression"
            message = (
                "Bonne nouvelle : l’engagement de votre dernier cours a augmenté "
                f"de {round(difference, 2)} points par rapport à vos cours récents."
            )
            priority = "success"
        else:
            title = "Baisse d’engagement détectée"
            message = (
                "Attention : l’engagement de votre dernier cours a baissé "
                f"de {abs(round(difference, 2))} points par rapport à vos cours récents."
            )
            priority = "warning"

        notification = models.Notification(
            user_id=analysis.trainer_id,
            analysis_id=analysis.id,
            notification_type="engagement_trend",
            title=title,
            message=message,
            priority=priority,
            is_read=False,
            metadata_json={
                "analysis_id": analysis.analysis_id,
                "current_engagement": round(current_engagement, 2),
                "previous_avg_engagement": round(previous_avg, 2),
                "difference": round(difference, 2),
            },
        )

        db.add(notification)
        return notification

    # ------------------------------------------------------------
    # Ranking helpers
    # ------------------------------------------------------------

    def calculate_course_percentile(
        self,
        db: Session,
        analysis: models.Analysis,
    ) -> float:
        """
        Returns percentile position based on global_score.
        Example: 85 means this course is better than 85% of completed courses.
        """

        if not analysis or analysis.status != "completed":
            return 0.0

        total_completed = (
            db.query(models.Analysis)
            .filter(models.Analysis.status == "completed")
            .count()
        )

        if total_completed <= 1:
            return 100.0

        lower_or_equal_count = (
            db.query(models.Analysis)
            .filter(models.Analysis.status == "completed")
            .filter(models.Analysis.global_score <= analysis.global_score)
            .count()
        )

        percentile = (lower_or_equal_count / total_completed) * 100

        return round(percentile, 2)

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------

    def _get_course_name(self, analysis: models.Analysis) -> str:
        if analysis.video and analysis.video.course_name:
            return analysis.video.course_name

        if analysis.course_title:
            return analysis.course_title

        return "votre cours"