from __future__ import annotations

from dataclasses import dataclass, field

from app.domain.feature_set import FeatureSet


@dataclass
class AnalysisResult:
    video_path: str
    extracted_audio_path: str
    processed_audio_path: str
    sample_rate: int
    duration: float
    features: FeatureSet = field(repr=False)

    clarity_score: float = 0.0
    engagement_score: float = 0.0
    global_score: float = 0.0

    engagement_label: str = "not_computed"
    engagement_method: str = "not_specified"
    engagement_details: dict | None = None

    dominant_emotion: str = "not_computed"
    emotion_confidence: float = 0.0
    emotion_num_segments: int = 0
    emotion_aggregated_scores: dict | None = None
    emotion_segment_predictions: list | None = field(default=None, repr=False)
    emotion_model_used: str = "not_specified"
    emotion_model_metrics: dict | None = None

    # Trainer emotion = current emotion analysis on trainer-only audio when diarization succeeds.
    trainer_emotion: dict | None = None

    # Global emotion of all learners, computed from learner segments in the diarization clean CSV.
    learner_emotion_enabled: bool = False
    learner_emotion: dict | None = None
    learner_emotion_audio_path: str | None = None
    learner_emotion_segments_csv_path: str | None = None
    learner_emotion_segment_count: int = 0
    learner_emotion_speaker_count: int = 0
    learner_emotion_audio_duration: float = 0.0

    diarization_enabled: bool = False
    diarization_rttm_path: str | None = None
    diarization_raw_csv_path: str | None = None
    diarization_clean_csv_path: str | None = None
    trainer_speaker_id: str | None = None
    trainer_detection_confidence: str | None = None
    diarization_real_speakers: list | None = None
    diarization_artifact_speakers: list | None = None
    diarization_profiles_json_path: str | None = None

    scoring_audio_source: str = "full_audio"
    trainer_audio_path: str | None = None
    trainer_audio_segments_csv_path: str | None = None
    trainer_audio_segment_count: int = 0
    trainer_audio_duration: float = 0.0

    transcription_enabled: bool = False
    transcript_json_path: str | None = None
    transcript_txt_path: str | None = None
    transcript_language: str | None = None
    transcript_language_confidence: float = 0.0
    transcript_segments_count: int = 0
    transcript_total_duration: float = 0.0

    summary_enabled: bool = False
    summary_json_path: str | None = None
    course_title: str | None = None
    course_language: str | None = None
    course_sections_count: int = 0
    course_key_points: list | None = None
    course_summary: dict | None = None

    def _num(self, value, digits: int = 4) -> float:
        try:
            if value is None:
                return 0.0
            return round(float(value), digits)
        except Exception:
            return 0.0

    def _int(self, value) -> int:
        try:
            if value is None:
                return 0
            return int(float(value))
        except Exception:
            return 0

    def _score_level(self, score: float) -> str:
        score = self._num(score, 2)
        if score >= 80:
            return "high"
        if score >= 65:
            return "good"
        if score >= 45:
            return "medium"
        return "low"

    def _percent(self, ratio) -> float:
        return round(self._num(ratio, 6) * 100.0, 2)

    def _detail(self, key: str, default=0.0):
        if not self.engagement_details:
            return default
        return self.engagement_details.get(key, default)

    def _has_diarization_context(self) -> bool:
        return bool(self._detail("has_diarization_context", False))

    def _learner_emotion_value(self, key: str, default=None):
        if not self.learner_emotion:
            return default
        return self.learner_emotion.get(key, default)

    def _learner_emotion_scores(self) -> dict:
        if not self.learner_emotion:
            return {}
        return self.learner_emotion.get("aggregated_scores", {}) or {}

    def get_interpretation(self) -> str:
        if self.global_score >= 80:
            level = "très satisfaisante"
        elif self.global_score >= 60:
            level = "correcte"
        else:
            level = "insuffisante"

        diarization_text = ""
        if self.diarization_enabled and self.trainer_speaker_id:
            diarization_text = (
                f" Le formateur détecté est '{self.trainer_speaker_id}' "
                f"(confiance {self.trainer_detection_confidence})."
            )

        source_text = ""
        if self.scoring_audio_source == "trainer_only":
            source_text = (
                " Les scores de clarté, d'émotion et de dynamisme vocal "
                "ont été calculés sur l'audio du formateur uniquement."
            )

        learner_emotion_text = ""
        if self.learner_emotion_enabled and self.learner_emotion:
            learner_emotion_text = (
                f" L'émotion globale estimée des apprenants est "
                f"'{self.learner_emotion.get('dominant_emotion', 'not_computed')}' "
                f"avec une confiance de "
                f"{self._num(self.learner_emotion.get('confidence', 0.0), 4)}, "
                f"calculée sur {self.learner_emotion.get('num_segments', 0)} segment(s) audio."
            )

        engagement_text = ""
        if self.engagement_details:
            engagement_text = (
                f" Le score d'engagement est classé '{self.engagement_label}' "
                f"avec la méthode '{self.engagement_method}'."
            )

            if self._has_diarization_context():
                learner_talk_ratio = self._percent(self._detail("learner_talk_ratio", 0.0))
                learner_turn_count = self._int(self._detail("learner_turn_count", 0))
                active_learners = self._int(self._detail("active_learner_speaker_count", 0))

                engagement_text += (
                    f" Les apprenants représentent {learner_talk_ratio}% du temps de parole, "
                    f"avec {learner_turn_count} interventions et "
                    f"{active_learners} apprenant(s) actif(s) détecté(s)."
                )

        summary_text = ""
        if self.summary_enabled and self.course_title:
            summary_text = (
                f" Le système a également généré un résumé automatique du cours "
                f"intitulé '{self.course_title}', contenant {self.course_sections_count} section(s)."
            )

        return (
            f"La performance vocale globale est {level}. "
            f"Le score de clarté est de {self.clarity_score}/100, "
            f"le score d'engagement est de {self.engagement_score}/100, "
            f"et l'émotion dominante du formateur estimée est '{self.dominant_emotion}' "
            f"avec une confiance de {self.emotion_confidence:.4f}, "
            f"calculée sur {self.emotion_num_segments} segments."
            f"{learner_emotion_text}{diarization_text}{source_text}{engagement_text}{summary_text}"
        )

    def get_summary_lines(self) -> list[str]:
        lines = [
            f"Clarté : {self.clarity_score}/100",
            f"Engagement : {self.engagement_score}/100 ({self.engagement_label})",
            f"Score global : {self.global_score}/100",
            f"Émotion formateur dominante : {self.dominant_emotion}",
        ]

        if self.learner_emotion_enabled and self.learner_emotion:
            lines.append(
                "Émotion globale apprenants : "
                f"{self.learner_emotion.get('dominant_emotion', 'not_computed')}"
            )
            lines.append(
                "Confiance émotion apprenants : "
                f"{self._num(self.learner_emotion.get('confidence', 0.0), 4)}"
            )
            lines.append(
                "Segments émotion apprenants : "
                f"{self.learner_emotion.get('num_segments', 0)}"
            )

        if self.engagement_details:
            lines.append(
                "Dynamisme vocal formateur : "
                f"{self._detail('trainer_vocal_component', 0.0)}/100"
            )
            lines.append(
                "Participation apprenant : "
                f"{self._detail('learner_participation_component', 0.0)}/100"
            )
            lines.append(
                "Interaction formateur/apprenant : "
                f"{self._detail('interaction_component', 0.0)}/100"
            )

        if self._has_diarization_context():
            lines.append(
                f"Temps de parole apprenants : "
                f"{self._percent(self._detail('learner_talk_ratio', 0.0))}%"
            )
            lines.append(
                f"Interventions apprenants : "
                f"{self._int(self._detail('learner_turn_count', 0))}"
            )
            lines.append(
                f"Apprenants actifs détectés : "
                f"{self._int(self._detail('active_learner_speaker_count', 0))}"
            )

        if self.diarization_enabled and self.trainer_speaker_id:
            lines.append(
                f"Formateur détecté : "
                f"{self.trainer_speaker_id} ({self.trainer_detection_confidence})"
            )

        if self.transcription_enabled:
            lines.append(
                f"Transcription : activée ({self.transcript_language}, "
                f"confiance {self.transcript_language_confidence})"
            )

        if self.summary_enabled and self.course_title:
            lines.append(f"Résumé du cours : {self.course_title}")
            lines.append(f"Sections du résumé : {self.course_sections_count}")

        lines.append(f"Source de scoring : {self.scoring_audio_source}")

        return lines

    def get_recommendations(self) -> list[str]:
        recommendations = []

        if self.clarity_score < 70:
            recommendations.append(
                "Améliorer l'articulation et maintenir un niveau vocal plus constant."
            )

        if self.engagement_score < 70:
            if self._has_diarization_context():
                learner_talk_ratio = self._num(self._detail("learner_talk_ratio", 0.0))
                learner_response_ratio = self._num(self._detail("learner_response_ratio", 0.0))
                trainer_learner_balance = self._num(self._detail("trainer_learner_balance", 0.0))
                active_learners = self._int(self._detail("active_learner_speaker_count", 0))

                if learner_talk_ratio < 0.08:
                    recommendations.append(
                        "Augmenter la participation orale des apprenants en posant "
                        "plus de questions ouvertes ou en intégrant des moments d'échange."
                    )

                if learner_response_ratio < 0.20:
                    recommendations.append(
                        "Prévoir des pauses interactives après les explications afin "
                        "de favoriser les réponses des apprenants."
                    )

                if trainer_learner_balance < 0.30:
                    recommendations.append(
                        "Réduire les séquences longues de parole du formateur et "
                        "favoriser davantage l'alternance formateur/apprenants."
                    )

                if active_learners <= 1:
                    recommendations.append(
                        "Encourager la participation de plusieurs apprenants afin "
                        "d'améliorer l'engagement collectif."
                    )
            else:
                recommendations.append(
                    "Varier davantage l'intonation et l'énergie pour capter l'attention."
                )

        if self.features.voiced_ratio < 0.5:
            recommendations.append(
                "Réduire les pauses trop fréquentes ou les segments peu audibles."
            )

        if self.features.energy_mean < 0.03:
            recommendations.append(
                "Parler avec une intensité légèrement plus soutenue."
            )

        if self.features.pitch_std < 10:
            recommendations.append(
                "Introduire plus de variation prosodique pour éviter une voix monotone."
            )

        if self.learner_emotion_enabled and self.learner_emotion:
            learner_scores = self._learner_emotion_scores()
            learner_low = self._num(learner_scores.get("low_energy", 0.0))
            learner_tense = self._num(learner_scores.get("tense_stressed", 0.0))

            if learner_low >= 0.35:
                recommendations.append(
                    "L'émotion globale des apprenants indique une énergie faible. "
                    "Ajouter des questions, exemples ou activités courtes pour relancer l'attention."
                )

            if learner_tense >= 0.45:
                recommendations.append(
                    "L'émotion globale des apprenants semble tendue ou stressée. "
                    "Ralentir le rythme et reformuler les points complexes."
                )

        if not self.transcription_enabled:
            recommendations.append(
                "Activer la transcription automatique afin de générer un résumé pédagogique du cours."
            )

        if not recommendations:
            recommendations.append(
                "La prestation est globalement satisfaisante. Continuer à maintenir "
                "ce niveau de clarté, de dynamisme et d'interaction."
            )

        return recommendations

    def get_dashboard_payload(self) -> dict:
        emotion_scores = self.emotion_aggregated_scores or {}
        learner_emotion_scores = self._learner_emotion_scores()
        details = self.engagement_details or {}

        clarity_score = self._num(self.clarity_score, 2)
        engagement_score = self._num(self.engagement_score, 2)
        global_score = self._num(self.global_score, 2)

        trainer_vocal_component = self._num(details.get("trainer_vocal_component", 0.0), 2)
        learner_participation_component = self._num(details.get("learner_participation_component", 0.0), 2)
        interaction_component = self._num(details.get("interaction_component", 0.0), 2)

        trainer_talk_ratio = self._num(details.get("trainer_talk_ratio", 0.0), 6)
        learner_talk_ratio = self._num(details.get("learner_talk_ratio", 0.0), 6)

        trainer_speech_share = self._num(details.get("trainer_speech_share", 0.0), 6)
        learner_speech_share = self._num(details.get("learner_speech_share", 0.0), 6)

        trainer_turn_count = self._int(details.get("trainer_turn_count", 0))
        learner_turn_count = self._int(details.get("learner_turn_count", 0))
        interaction_turn_count = self._int(details.get("interaction_turn_count", 0))

        learner_emotion_confidence = self._num(
            self._learner_emotion_value("confidence", 0.0),
            4,
        )

        return {
            "kpis": [
                {
                    "key": "global_score",
                    "label": "Score global",
                    "value": global_score,
                    "unit": "/100",
                    "level": self._score_level(global_score),
                },
                {
                    "key": "clarity_score",
                    "label": "Clarté",
                    "value": clarity_score,
                    "unit": "/100",
                    "level": self._score_level(clarity_score),
                },
                {
                    "key": "engagement_score",
                    "label": "Engagement",
                    "value": engagement_score,
                    "unit": "/100",
                    "level": self.engagement_label,
                },
                {
                    "key": "trainer_emotion_confidence",
                    "label": "Confiance émotion formateur",
                    "value": self._num(self.emotion_confidence, 4),
                    "unit": "",
                    "level": self._score_level(self.emotion_confidence * 100),
                },
                {
                    "key": "learner_emotion_confidence",
                    "label": "Confiance émotion apprenants",
                    "value": learner_emotion_confidence,
                    "unit": "",
                    "level": self._score_level(learner_emotion_confidence * 100),
                },
                {
                    "key": "summary_sections",
                    "label": "Sections résumé",
                    "value": self.course_sections_count,
                    "unit": "",
                    "level": "good" if self.summary_enabled else "low",
                },
            ],

            "scores": {
                "clarity": {
                    "score": clarity_score,
                    "level": self._score_level(clarity_score),
                },
                "engagement": {
                    "score": engagement_score,
                    "label": self.engagement_label,
                    "method": self.engagement_method,
                },
                "global": {
                    "score": global_score,
                    "level": self._score_level(global_score),
                },
            },

            "emotion": {
                "dominant_emotion": self.dominant_emotion,
                "confidence": self._num(self.emotion_confidence, 4),
                "num_segments": self.emotion_num_segments,
                "model_used": self.emotion_model_used,
                "distribution": [
                    {
                        "label": label,
                        "value": self._num(value, 4),
                        "percentage": self._percent(value),
                    }
                    for label, value in emotion_scores.items()
                ],
            },

            "trainer_emotion": self.trainer_emotion or {
                "enabled": True,
                "type": "trainer",
                "dominant_emotion": self.dominant_emotion,
                "confidence": self._num(self.emotion_confidence, 4),
                "num_segments": self.emotion_num_segments,
                "aggregated_scores": emotion_scores,
                "model_name": self.emotion_model_used,
                "model_metrics": self.emotion_model_metrics or {},
            },

            "learner_emotion": {
                "enabled": self.learner_emotion_enabled,
                "type": self._learner_emotion_value("type", "global_all_learners"),
                "dominant_emotion": self._learner_emotion_value("dominant_emotion"),
                "confidence": learner_emotion_confidence,
                "num_segments": self._int(self._learner_emotion_value("num_segments", 0)),
                "model_name": self._learner_emotion_value("model_name"),
                "model_metrics": self._learner_emotion_value("model_metrics", {}),
                "distribution": [
                    {
                        "label": label,
                        "value": self._num(value, 4),
                        "percentage": self._percent(value),
                    }
                    for label, value in learner_emotion_scores.items()
                ],
                "audio_path": self.learner_emotion_audio_path,
                "segments_csv_path": self.learner_emotion_segments_csv_path,
                "audio_duration": self._num(self.learner_emotion_audio_duration, 2),
                "learner_segment_count": self.learner_emotion_segment_count,
                "learner_speaker_count": self.learner_emotion_speaker_count,
            },

            "engagement": {
                "score": engagement_score,
                "label": self.engagement_label,
                "method": self.engagement_method,
                "has_diarization_context": bool(details.get("has_diarization_context", False)),
                "components": {
                    "trainer_vocal_component": trainer_vocal_component,
                    "learner_participation_component": learner_participation_component,
                    "interaction_component": interaction_component,
                },
                "trainer_vocal_indicators": {
                    "pitch_std": self._num(details.get("pitch_std", 0.0), 4),
                    "energy_std": self._num(details.get("energy_std", 0.0), 4),
                    "voiced_ratio": self._num(details.get("voiced_ratio", 0.0), 4),
                    "speech_ratio": self._num(details.get("speech_ratio", 0.0), 4),
                    "pause_ratio": self._num(details.get("pause_ratio", 0.0), 4),
                    "onset_rate_per_sec": self._num(details.get("onset_rate_per_sec", 0.0), 4),
                },
                "learner_participation": {
                    "session_duration_sec": self._num(details.get("session_duration_sec", 0.0), 2),
                    "trainer_total_speech_duration": self._num(details.get("trainer_total_speech_duration", 0.0), 2),
                    "learner_total_speech_duration": self._num(details.get("learner_total_speech_duration", 0.0), 2),
                    "trainer_talk_ratio": trainer_talk_ratio,
                    "learner_talk_ratio": learner_talk_ratio,
                    "trainer_talk_percentage": self._percent(trainer_talk_ratio),
                    "learner_talk_percentage": self._percent(learner_talk_ratio),
                    "trainer_speech_share": trainer_speech_share,
                    "learner_speech_share": learner_speech_share,
                    "trainer_speech_share_percentage": self._percent(trainer_speech_share),
                    "learner_speech_share_percentage": self._percent(learner_speech_share),
                    "active_learner_speaker_count": self._int(details.get("active_learner_speaker_count", 0)),
                    "learner_turn_count": learner_turn_count,
                    "learner_avg_turn_duration": self._num(details.get("learner_avg_turn_duration", 0.0), 4),
                    "learner_max_turn_duration": self._num(details.get("learner_max_turn_duration", 0.0), 4),
                    "learner_turns_per_min": self._num(details.get("learner_turns_per_min", 0.0), 4),
                },
                "interaction": {
                    "trainer_turn_count": trainer_turn_count,
                    "learner_turn_count": learner_turn_count,
                    "total_turn_count": self._int(details.get("total_turn_count", 0)),
                    "interaction_turn_count": interaction_turn_count,
                    "interaction_rate_per_min": self._num(details.get("interaction_rate_per_min", 0.0), 4),
                    "trainer_learner_balance": self._num(details.get("trainer_learner_balance", 0.0), 4),
                    "learner_response_count": self._int(details.get("learner_response_count", 0)),
                    "learner_response_ratio": self._num(details.get("learner_response_ratio", 0.0), 4),
                    "overlap_segment_count": self._int(details.get("overlap_segment_count", 0)),
                    "overlap_ratio": self._num(details.get("overlap_ratio", 0.0), 4),
                },
            },

            "transcription": {
                "enabled": self.transcription_enabled,
                "language": self.transcript_language,
                "language_confidence": self._num(self.transcript_language_confidence, 4),
                "segments_count": self.transcript_segments_count,
                "total_duration": self.transcript_total_duration,
                "transcript_json_path": self.transcript_json_path,
                "transcript_txt_path": self.transcript_txt_path,
            },

            "course_summary": {
                "enabled": self.summary_enabled,
                "summary_json_path": self.summary_json_path,
                "title": self.course_title,
                "language": self.course_language,
                "sections_count": self.course_sections_count,
                "key_points": self.course_key_points or [],
                "sections": (self.course_summary or {}).get("sections", []),
                "conclusion": (self.course_summary or {}).get("conclusion", ""),
            },

            "diarization": {
                "enabled": self.diarization_enabled,
                "trainer_speaker_id": self.trainer_speaker_id,
                "trainer_detection_confidence": self.trainer_detection_confidence,
                "real_speakers": self.diarization_real_speakers or [],
                "artifact_speakers": self.diarization_artifact_speakers or [],
                "scoring_audio_source": self.scoring_audio_source,
            },

            "charts": {
                "score_bars": [
                    {"label": "Clarté", "value": clarity_score},
                    {"label": "Engagement", "value": engagement_score},
                    {"label": "Global", "value": global_score},
                ],
                "engagement_components": [
                    {"label": "Voix formateur", "value": trainer_vocal_component},
                    {"label": "Participation apprenant", "value": learner_participation_component},
                    {"label": "Interaction", "value": interaction_component},
                ],
                "trainer_emotion_distribution": [
                    {
                        "label": label,
                        "value": self._percent(value),
                    }
                    for label, value in emotion_scores.items()
                ],
                "learner_emotion_distribution": [
                    {
                        "label": label,
                        "value": self._percent(value),
                    }
                    for label, value in learner_emotion_scores.items()
                ],
                "emotion_distribution": [
                    {
                        "label": label,
                        "value": self._percent(value),
                    }
                    for label, value in emotion_scores.items()
                ],
                "speech_share": [
                    {
                        "label": "Formateur",
                        "value": self._percent(trainer_speech_share),
                    },
                    {
                        "label": "Apprenants",
                        "value": self._percent(learner_speech_share),
                    },
                ],
                "turn_counts": [
                    {"label": "Formateur", "value": trainer_turn_count},
                    {"label": "Apprenants", "value": learner_turn_count},
                    {"label": "Interactions", "value": interaction_turn_count},
                ],
                "summary_sections": [
                    {
                        "label": section.get("titre", f"Section {i + 1}"),
                        "value": i + 1,
                    }
                    for i, section in enumerate((self.course_summary or {}).get("sections", []))
                ],
            },

            "recommendations": self.get_recommendations(),
        }

    def to_dict(self) -> dict:
        return {
            "video_path": self.video_path,
            "extracted_audio_path": self.extracted_audio_path,
            "processed_audio_path": self.processed_audio_path,
            "sample_rate": self.sample_rate,
            "duration": self.duration,

            "clarity_score": self.clarity_score,
            "engagement_score": self.engagement_score,
            "engagement_label": self.engagement_label,
            "engagement_method": self.engagement_method,
            "engagement_details": self.engagement_details or {},
            "global_score": self.global_score,

            "dominant_emotion": self.dominant_emotion,
            "emotion_confidence": self.emotion_confidence,
            "emotion_num_segments": self.emotion_num_segments,
            "emotion_aggregated_scores": self.emotion_aggregated_scores or {},
            "emotion_segment_predictions": self.emotion_segment_predictions or [],
            "emotion_model_used": self.emotion_model_used,
            "emotion_model_metrics": self.emotion_model_metrics or {},

            "trainer_emotion": self.trainer_emotion or {
                "enabled": True,
                "type": "trainer",
                "dominant_emotion": self.dominant_emotion,
                "confidence": self.emotion_confidence,
                "num_segments": self.emotion_num_segments,
                "aggregated_scores": self.emotion_aggregated_scores or {},
                "model_name": self.emotion_model_used,
                "model_metrics": self.emotion_model_metrics or {},
            },

            "learner_emotion_enabled": self.learner_emotion_enabled,
            "learner_emotion": self.learner_emotion or {},
            "learner_emotion_audio_path": self.learner_emotion_audio_path,
            "learner_emotion_segments_csv_path": self.learner_emotion_segments_csv_path,
            "learner_emotion_segment_count": self.learner_emotion_segment_count,
            "learner_emotion_speaker_count": self.learner_emotion_speaker_count,
            "learner_emotion_audio_duration": self.learner_emotion_audio_duration,

            "diarization_enabled": self.diarization_enabled,
            "diarization_rttm_path": self.diarization_rttm_path,
            "diarization_raw_csv_path": self.diarization_raw_csv_path,
            "diarization_clean_csv_path": self.diarization_clean_csv_path,
            "trainer_speaker_id": self.trainer_speaker_id,
            "trainer_detection_confidence": self.trainer_detection_confidence,
            "diarization_real_speakers": self.diarization_real_speakers or [],
            "diarization_artifact_speakers": self.diarization_artifact_speakers or [],
            "diarization_profiles_json_path": self.diarization_profiles_json_path,

            "scoring_audio_source": self.scoring_audio_source,
            "trainer_audio_path": self.trainer_audio_path,
            "trainer_audio_segments_csv_path": self.trainer_audio_segments_csv_path,
            "trainer_audio_segment_count": self.trainer_audio_segment_count,
            "trainer_audio_duration": self.trainer_audio_duration,

            "transcription_enabled": self.transcription_enabled,
            "transcript_json_path": self.transcript_json_path,
            "transcript_txt_path": self.transcript_txt_path,
            "transcript_language": self.transcript_language,
            "transcript_language_confidence": self.transcript_language_confidence,
            "transcript_segments_count": self.transcript_segments_count,
            "transcript_total_duration": self.transcript_total_duration,

            "summary_enabled": self.summary_enabled,
            "summary_json_path": self.summary_json_path,
            "course_title": self.course_title,
            "course_language": self.course_language,
            "course_sections_count": self.course_sections_count,
            "course_key_points": self.course_key_points or [],
            "course_summary": self.course_summary or {},

            "interpretation": self.get_interpretation(),
            "summary_lines": self.get_summary_lines(),
            "recommendations": self.get_recommendations(),

            "dashboard": self.get_dashboard_payload(),

            "features": {
                "mfcc_means": self.features.mfcc_means,
                "pitch_mean": self.features.pitch_mean,
                "pitch_std": self.features.pitch_std,
                "energy_mean": self.features.energy_mean,
                "energy_std": self.features.energy_std,
                "zcr_mean": self.features.zcr_mean,
                "zcr_std": self.features.zcr_std,
                "spectral_centroid_mean": self.features.spectral_centroid_mean,
                "spectral_centroid_std": self.features.spectral_centroid_std,
                "spectral_bandwidth_mean": self.features.spectral_bandwidth_mean,
                "spectral_bandwidth_std": self.features.spectral_bandwidth_std,
                "voiced_ratio": self.features.voiced_ratio,
                "silence_ratio": self.features.silence_ratio,
                "pause_count": self.features.pause_count,
                "mean_pause_duration": self.features.mean_pause_duration,
                "total_pause_duration": self.features.total_pause_duration,
            },
        }