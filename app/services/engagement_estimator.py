from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from app.ai.engagement.features import EngagementFeatureExtractor
from app.domain.feature_set import FeatureSet


class EngagementEstimator:
    def __init__(
        self,
        model_path: str = "artifacts/engagement/engagement_best.joblib",
        metadata_path: str = "artifacts/engagement/engagement_metadata.json",
    ):
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)

        self.model = None
        self.feature_columns: list[str] = []
        self.target_name = "engagement_score_100"

        # Emotion scores are passed from AnalysisService,
        # so we do not reload the emotion CNN2D here.
        self.extractor = EngagementFeatureExtractor(use_emotion_features=False)

        if self.model_path.exists() and self.metadata_path.exists():
            try:
                self.model = joblib.load(self.model_path)

                with self.metadata_path.open("r", encoding="utf-8") as f:
                    metadata = json.load(f)

                self.feature_columns = metadata.get("feature_columns", [])
                self.target_name = metadata.get("target_name", "engagement_score_100")

                print("[EngagementEstimator] Trained engagement model loaded.")

            except Exception as e:
                print(
                    "[EngagementEstimator] Failed to load trained model, "
                    f"fallback to heuristic: {e}"
                )
                self.model = None
                self.feature_columns = []
        else:
            print("[EngagementEstimator] No trained model found, using heuristic estimator.")

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def _clamp(
        self,
        value: float,
        min_value: float = 0.0,
        max_value: float = 100.0,
    ) -> float:
        return float(max(min_value, min(value, max_value)))

    def _clip01(self, value: float) -> float:
        return float(np.clip(value, 0.0, 1.0))

    def _score_to_label(self, score: float) -> str:
        if score >= 80:
            return "high"
        if score >= 65:
            return "good"
        if score >= 45:
            return "medium"
        return "low"

    def _has_diarization_context(self, f: dict[str, float]) -> bool:
        return (
            float(f.get("session_duration_sec", 0.0)) > 0.0
            and float(f.get("total_turn_count", 0.0)) > 0.0
        )

    # ------------------------------------------------------------------
    # Model inference
    # ------------------------------------------------------------------

    def _extract_runtime_features(
        self,
        audio_path: str,
        emotion_scores: dict | None = None,
        diarization_csv_path: str | None = None,
        trainer_speaker_id: str | None = None,
    ) -> dict[str, float]:
        return self.extractor.extract(
            audio_path=audio_path,
            external_emotion_scores=emotion_scores or {},
            diarization_csv_path=diarization_csv_path,
            trainer_speaker_id=trainer_speaker_id,
            add_pseudo_label=True,
        )

    def _estimate_with_model(
        self,
        audio_path: str,
        emotion_scores: dict | None = None,
        diarization_csv_path: str | None = None,
        trainer_speaker_id: str | None = None,
    ) -> tuple[float, dict[str, float]]:
        if self.model is None or not self.feature_columns:
            raise RuntimeError("Trained engagement model not available.")

        feature_dict = self._extract_runtime_features(
            audio_path=audio_path,
            emotion_scores=emotion_scores,
            diarization_csv_path=diarization_csv_path,
            trainer_speaker_id=trainer_speaker_id,
        )

        x = pd.DataFrame(
            [
                {
                    col: float(feature_dict.get(col, 0.0))
                    for col in self.feature_columns
                }
            ]
        )

        predicted_score = float(self.model.predict(x)[0])
        predicted_score = round(self._clamp(predicted_score), 2)

        return predicted_score, feature_dict

    # ------------------------------------------------------------------
    # Heuristic fallback using same logic as the pseudo-label
    # ------------------------------------------------------------------

    def _trainer_vocal_component_01(self, f: dict[str, float]) -> float:
        pitch_var = self._clip01(float(f.get("pitch_std", 0.0)) / 60.0)
        energy_var = self._clip01(float(f.get("rms_std", 0.0)) / 0.08)
        voiced = self._clip01(float(f.get("voiced_ratio", 0.0)))
        speech = self._clip01(float(f.get("speech_ratio", 0.0)))
        onset = self._clip01(float(f.get("onset_rate_per_sec", 0.0)) / 3.0)

        pause_ratio = float(f.get("pause_ratio", 0.0))
        pause_component = 1.0 - self._clip01(pause_ratio / 0.6)

        emo_energy = self._clip01(float(f.get("emo_energetic_engaged", 0.0)))
        emo_neutral = self._clip01(float(f.get("emo_neutral_calm", 0.0)))
        emo_low = self._clip01(float(f.get("emo_low_energy", 0.0)))
        emo_tense = self._clip01(float(f.get("emo_tense_stressed", 0.0)))

        score_01 = (
            0.18 * pitch_var
            + 0.18 * energy_var
            + 0.12 * voiced
            + 0.12 * speech
            + 0.10 * onset
            + 0.10 * pause_component
            + 0.12 * emo_energy
            + 0.04 * emo_neutral
            + 0.02 * (1.0 - emo_low)
            + 0.02 * (1.0 - 0.5 * emo_tense)
        )

        return self._clip01(score_01)

    def _learner_participation_component_01(self, f: dict[str, float]) -> float:
        learner_talk = self._clip01(float(f.get("learner_talk_ratio", 0.0)) / 0.30)
        learner_turn_rate = self._clip01(float(f.get("learner_turns_per_min", 0.0)) / 2.0)
        active_learners = self._clip01(
            float(f.get("active_learner_speaker_count", 0.0)) / 5.0
        )
        response_ratio = self._clip01(float(f.get("learner_response_ratio", 0.0)))
        avg_turn = self._clip01(float(f.get("learner_avg_turn_duration", 0.0)) / 4.0)

        learner_speech_share = float(f.get("learner_speech_share", 0.0))
        speech_share_balance = 1.0 - self._clip01(
            abs(learner_speech_share - 0.25) / 0.25
        )

        score_01 = (
            0.25 * learner_talk
            + 0.20 * learner_turn_rate
            + 0.20 * active_learners
            + 0.20 * response_ratio
            + 0.10 * avg_turn
            + 0.05 * speech_share_balance
        )

        return self._clip01(score_01)

    def _interaction_component_01(self, f: dict[str, float]) -> float:
        interaction_rate = self._clip01(
            float(f.get("interaction_rate_per_min", 0.0)) / 2.0
        )
        balance = self._clip01(float(f.get("trainer_learner_balance", 0.0)))
        response_ratio = self._clip01(float(f.get("learner_response_ratio", 0.0)))

        overlap_ratio = float(f.get("overlap_ratio", 0.0))
        overlap_component = 1.0 - self._clip01(overlap_ratio / 0.4)

        score_01 = (
            0.35 * interaction_rate
            + 0.25 * balance
            + 0.25 * response_ratio
            + 0.15 * overlap_component
        )

        return self._clip01(score_01)

    def _estimate_with_heuristic_from_feature_dict(
        self,
        f: dict[str, float],
    ) -> float:
        trainer_vocal = self._trainer_vocal_component_01(f)

        if not self._has_diarization_context(f):
            return round(self._clamp(trainer_vocal * 100.0), 2)

        learner_participation = self._learner_participation_component_01(f)
        interaction = self._interaction_component_01(f)

        score_01 = (
            0.55 * trainer_vocal
            + 0.30 * learner_participation
            + 0.15 * interaction
        )

        return round(self._clamp(score_01 * 100.0), 2)

    # ------------------------------------------------------------------
    # Output details
    # ------------------------------------------------------------------

    def _build_details(self, f: dict[str, float]) -> dict:
        trainer_vocal_component = self._trainer_vocal_component_01(f) * 100.0
        learner_participation_component = self._learner_participation_component_01(f) * 100.0
        interaction_component = self._interaction_component_01(f) * 100.0

        return {
            "has_diarization_context": self._has_diarization_context(f),

            "trainer_vocal_component": round(trainer_vocal_component, 2),
            "learner_participation_component": round(learner_participation_component, 2),
            "interaction_component": round(interaction_component, 2),

            "pitch_std": round(float(f.get("pitch_std", 0.0)), 4),
            "energy_std": round(float(f.get("rms_std", 0.0)), 4),
            "voiced_ratio": round(float(f.get("voiced_ratio", 0.0)), 4),
            "speech_ratio": round(float(f.get("speech_ratio", 0.0)), 4),
            "pause_ratio": round(float(f.get("pause_ratio", 0.0)), 4),
            "onset_rate_per_sec": round(float(f.get("onset_rate_per_sec", 0.0)), 4),

            "emo_neutral_calm": round(float(f.get("emo_neutral_calm", 0.0)), 4),
            "emo_energetic_engaged": round(float(f.get("emo_energetic_engaged", 0.0)), 4),
            "emo_low_energy": round(float(f.get("emo_low_energy", 0.0)), 4),
            "emo_tense_stressed": round(float(f.get("emo_tense_stressed", 0.0)), 4),

            "session_duration_sec": round(float(f.get("session_duration_sec", 0.0)), 2),
            "trainer_total_speech_duration": round(
                float(f.get("trainer_total_speech_duration", 0.0)), 2
            ),
            "learner_total_speech_duration": round(
                float(f.get("learner_total_speech_duration", 0.0)), 2
            ),
            "trainer_talk_ratio": round(float(f.get("trainer_talk_ratio", 0.0)), 4),
            "learner_talk_ratio": round(float(f.get("learner_talk_ratio", 0.0)), 4),
            "trainer_speech_share": round(float(f.get("trainer_speech_share", 0.0)), 4),
            "learner_speech_share": round(float(f.get("learner_speech_share", 0.0)), 4),

            "trainer_turn_count": int(float(f.get("trainer_turn_count", 0.0))),
            "learner_turn_count": int(float(f.get("learner_turn_count", 0.0))),
            "total_turn_count": int(float(f.get("total_turn_count", 0.0))),
            "active_learner_speaker_count": int(
                float(f.get("active_learner_speaker_count", 0.0))
            ),

            "learner_avg_turn_duration": round(
                float(f.get("learner_avg_turn_duration", 0.0)), 4
            ),
            "learner_max_turn_duration": round(
                float(f.get("learner_max_turn_duration", 0.0)), 4
            ),
            "learner_turns_per_min": round(float(f.get("learner_turns_per_min", 0.0)), 4),

            "interaction_turn_count": int(float(f.get("interaction_turn_count", 0.0))),
            "interaction_rate_per_min": round(
                float(f.get("interaction_rate_per_min", 0.0)), 4
            ),
            "trainer_learner_balance": round(
                float(f.get("trainer_learner_balance", 0.0)), 4
            ),

            "learner_response_count": int(float(f.get("learner_response_count", 0.0))),
            "learner_response_ratio": round(
                float(f.get("learner_response_ratio", 0.0)), 4
            ),

            "overlap_segment_count": int(float(f.get("overlap_segment_count", 0.0))),
            "overlap_ratio": round(float(f.get("overlap_ratio", 0.0)), 4),
        }

    # ------------------------------------------------------------------
    # Main public method
    # ------------------------------------------------------------------

    def estimate_from_audio(
        self,
        audio_path: str,
        emotion_scores: dict | None = None,
        diarization_csv_path: str | None = None,
        trainer_speaker_id: str | None = None,
    ) -> dict:
        try:
            score, feature_dict = self._estimate_with_model(
                audio_path=audio_path,
                emotion_scores=emotion_scores,
                diarization_csv_path=diarization_csv_path,
                trainer_speaker_id=trainer_speaker_id,
            )
            method = "trained_model"

        except Exception as e:
            print(f"[EngagementEstimator] Model inference failed, using heuristic: {e}")

            feature_dict = self._extract_runtime_features(
                audio_path=audio_path,
                emotion_scores=emotion_scores,
                diarization_csv_path=diarization_csv_path,
                trainer_speaker_id=trainer_speaker_id,
            )

            score = self._estimate_with_heuristic_from_feature_dict(feature_dict)
            method = "heuristic"

        return {
            "engagement_score": score,
            "engagement_label": self._score_to_label(score),
            "method": method,
            "details": self._build_details(feature_dict),
        }

    # ------------------------------------------------------------------
    # Legacy compatibility method
    # ------------------------------------------------------------------

    def estimate(
        self,
        features: FeatureSet,
        emotion_scores: dict | None = None,
    ) -> float:
        """
        Old method kept for compatibility.
        Prefer estimate_from_audio() because it uses the same features as training.
        """
        score = 40.0

        if 15 <= features.pitch_std <= 45:
            score += 20.0
        elif 8 <= features.pitch_std < 15:
            score += 10.0
        else:
            score -= 5.0

        if 0.02 <= features.energy_std <= 0.10:
            score += 20.0
        elif 0.01 <= features.energy_std < 0.02:
            score += 10.0
        else:
            score -= 5.0

        score += features.voiced_ratio * 10.0

        if 0.2 <= features.mean_pause_duration <= 0.7:
            score += 8.0
        elif features.mean_pause_duration > 1.2:
            score -= 10.0

        if 3 <= features.pause_count <= 10:
            score += 8.0
        elif features.pause_count > 14:
            score -= 8.0

        if features.silence_ratio > 0.5:
            score -= 10.0
        elif 0.2 <= features.silence_ratio <= 0.4:
            score += 5.0

        return round(self._clamp(score), 2)