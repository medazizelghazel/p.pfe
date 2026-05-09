from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

from app.ai.engagement.learner_participation_features import (
    LearnerParticipationFeatureExtractor,
)


@dataclass
class EngagementFeatureExtractor:
    target_sr: int = 16000
    n_mfcc: int = 13
    frame_length: int = 1024
    hop_length: int = 256
    silence_db: float = 30.0
    use_emotion_features: bool = True

    def __post_init__(self):
        self.emotion_service = None
        self.learner_participation_extractor = LearnerParticipationFeatureExtractor()

        # Important:
        # In the main pipeline, emotion scores are already computed by AnalysisService
        # and passed to EngagementEstimator. So use_emotion_features is usually False.
        #
        # If this extractor is used outside the main pipeline, for example when building
        # an engagement dataset, it can load EmotionClassifier. EmotionClassifier now
        # defaults to Wav2Vec2, not CNN2D.
        if self.use_emotion_features:
            try:
                from app.services.emotion_classifier import EmotionClassifier

                self.emotion_service = EmotionClassifier(
                    backend="wav2vec2",
                    mode="fast",
                )

                print("[EngagementFeatureExtractor] Emotion features enabled with Wav2Vec2.")

            except Exception as e:
                print(f"[EngagementFeatureExtractor] Emotion features disabled: {e}")
                self.emotion_service = None

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def _safe_float(self, value) -> float:
        if value is None:
            return 0.0

        if isinstance(value, (float, int, np.floating, np.integer)):
            if np.isnan(value) or np.isinf(value):
                return 0.0
            return float(value)

        return 0.0

    def _clip01(self, value: float) -> float:
        return float(np.clip(value, 0.0, 1.0))

    def _load_audio(self, audio_path: str) -> tuple[np.ndarray, int]:
        y, sr = librosa.load(
            audio_path,
            sr=self.target_sr,
            mono=True,
        )
        return y.astype(np.float32), sr

    # ------------------------------------------------------------------
    # Acoustic features
    # ------------------------------------------------------------------

    def _pitch_stats(
        self,
        y: np.ndarray,
        sr: int,
    ) -> tuple[float, float, float, float]:
        if len(y) == 0 or np.max(np.abs(y)) < 1e-8:
            return 0.0, 0.0, 0.0, 0.0

        try:
            f0, _, _ = librosa.pyin(
                y,
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C7"),
                frame_length=self.frame_length,
                hop_length=self.hop_length,
            )

            if f0 is None:
                return 0.0, 0.0, 0.0, 0.0

            valid_f0 = f0[~np.isnan(f0)]

        except Exception:
            try:
                pitches, _ = librosa.piptrack(
                    y=y,
                    sr=sr,
                    n_fft=self.frame_length,
                    hop_length=self.hop_length,
                )
                valid_f0 = pitches[pitches > 0]
            except Exception:
                return 0.0, 0.0, 0.0, 0.0

        if len(valid_f0) == 0:
            return 0.0, 0.0, 0.0, 0.0

        pitch_mean = self._safe_float(np.mean(valid_f0))
        pitch_std = self._safe_float(np.std(valid_f0))
        pitch_range = self._safe_float(np.max(valid_f0) - np.min(valid_f0))

        total_frames = max(1, int(np.ceil(len(y) / self.hop_length)))
        voiced_ratio = self._safe_float(len(valid_f0) / total_frames)

        return pitch_mean, pitch_std, pitch_range, voiced_ratio

    def _pause_stats(self, y: np.ndarray) -> dict[str, float]:
        duration_sec = len(y) / self.target_sr if len(y) > 0 else 0.0

        if duration_sec <= 0:
            return {
                "pause_ratio": 0.0,
                "pause_count_per_sec": 0.0,
                "mean_pause_duration_sec": 0.0,
                "speech_ratio": 0.0,
            }

        rms = librosa.feature.rms(
            y=y,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
        )[0]

        if len(rms) == 0:
            return {
                "pause_ratio": 0.0,
                "pause_count_per_sec": 0.0,
                "mean_pause_duration_sec": 0.0,
                "speech_ratio": 1.0,
            }

        max_rms = float(np.max(rms))

        if max_rms <= 1e-8:
            return {
                "pause_ratio": 1.0,
                "pause_count_per_sec": 1.0 / max(duration_sec, 1e-8),
                "mean_pause_duration_sec": duration_sec,
                "speech_ratio": 0.0,
            }

        threshold = max_rms * (10 ** (-self.silence_db / 20.0))
        silent_mask = rms < threshold

        pause_durations = []
        run = 0

        for is_silent in silent_mask:
            if is_silent:
                run += 1
            else:
                if run > 0:
                    pause_duration = run * self.hop_length / self.target_sr

                    if pause_duration >= 0.2:
                        pause_durations.append(pause_duration)

                    run = 0

        if run > 0:
            pause_duration = run * self.hop_length / self.target_sr

            if pause_duration >= 0.2:
                pause_durations.append(pause_duration)

        total_pause_duration_sec = float(sum(pause_durations))
        pause_ratio = total_pause_duration_sec / duration_sec
        pause_count_per_sec = len(pause_durations) / duration_sec
        mean_pause_duration_sec = (
            float(np.mean(pause_durations)) if pause_durations else 0.0
        )
        speech_ratio = max(0.0, 1.0 - pause_ratio)

        return {
            "pause_ratio": self._safe_float(pause_ratio),
            "pause_count_per_sec": self._safe_float(pause_count_per_sec),
            "mean_pause_duration_sec": self._safe_float(mean_pause_duration_sec),
            "speech_ratio": self._safe_float(speech_ratio),
        }

    def _onset_rate(self, y: np.ndarray, sr: int) -> float:
        duration_sec = len(y) / sr if sr > 0 else 0.0

        if duration_sec <= 0:
            return 0.0

        try:
            onset_frames = librosa.onset.onset_detect(
                y=y,
                sr=sr,
                hop_length=self.hop_length,
                units="frames",
            )
            return self._safe_float(len(onset_frames) / duration_sec)

        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Emotion features
    # ------------------------------------------------------------------

    def _empty_emotion_features(self) -> dict[str, float]:
        return {
            "emo_neutral_calm": 0.0,
            "emo_energetic_engaged": 0.0,
            "emo_low_energy": 0.0,
            "emo_tense_stressed": 0.0,
        }

    def _emotion_features(
        self,
        audio_path: str,
        external_emotion_scores: dict | None = None,
    ) -> dict[str, float]:
        # Prefer emotion scores already computed by AnalysisService.
        # This avoids loading the emotion model twice.
        if external_emotion_scores is not None:
            return {
                "emo_neutral_calm": float(
                    external_emotion_scores.get("neutral_calm", 0.0)
                ),
                "emo_energetic_engaged": float(
                    external_emotion_scores.get("energetic_engaged", 0.0)
                ),
                "emo_low_energy": float(
                    external_emotion_scores.get("low_energy", 0.0)
                ),
                "emo_tense_stressed": float(
                    external_emotion_scores.get("tense_stressed", 0.0)
                ),
            }

        if self.emotion_service is None:
            return self._empty_emotion_features()

        try:
            if hasattr(self.emotion_service, "classify_with_details"):
                result = self.emotion_service.classify_with_details(audio_path)
            else:
                result = self.emotion_service.predict_file(audio_path)

            scores = result.get("aggregated_scores", {})

            return {
                "emo_neutral_calm": float(scores.get("neutral_calm", 0.0)),
                "emo_energetic_engaged": float(scores.get("energetic_engaged", 0.0)),
                "emo_low_energy": float(scores.get("low_energy", 0.0)),
                "emo_tense_stressed": float(scores.get("tense_stressed", 0.0)),
            }

        except Exception as e:
            print(f"[EngagementFeatureExtractor] Emotion extraction failed: {e}")
            return self._empty_emotion_features()

    # ------------------------------------------------------------------
    # Learner participation features
    # ------------------------------------------------------------------

    def _learner_participation_features(
        self,
        diarization_csv_path: str | None = None,
        trainer_speaker_id: str | None = None,
    ) -> dict[str, float]:
        try:
            return self.learner_participation_extractor.extract(
                clean_csv_path=diarization_csv_path,
                trainer_speaker_id=trainer_speaker_id,
            )
        except Exception as e:
            print(
                "[EngagementFeatureExtractor] Learner participation "
                f"features disabled: {e}"
            )
            return self.learner_participation_extractor.extract(
                clean_csv_path=None,
                trainer_speaker_id=None,
            )

    # ------------------------------------------------------------------
    # Pseudo-label generation
    # ------------------------------------------------------------------

    def _trainer_vocal_score_01(self, f: dict[str, float]) -> float:
        pitch_var = self._clip01(f.get("pitch_std", 0.0) / 60.0)
        energy_var = self._clip01(f.get("rms_std", 0.0) / 0.08)
        voiced = self._clip01(f.get("voiced_ratio", 0.0))
        speech = self._clip01(f.get("speech_ratio", 0.0))
        onset = self._clip01(f.get("onset_rate_per_sec", 0.0) / 3.0)

        pause_ratio = f.get("pause_ratio", 0.0)
        pause_component = 1.0 - self._clip01(pause_ratio / 0.6)

        emo_energy = self._clip01(f.get("emo_energetic_engaged", 0.0))
        emo_low = self._clip01(f.get("emo_low_energy", 0.0))
        emo_tense = self._clip01(f.get("emo_tense_stressed", 0.0))
        emo_neutral = self._clip01(f.get("emo_neutral_calm", 0.0))

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

    def _learner_participation_score_01(self, f: dict[str, float]) -> float:
        learner_talk = self._clip01(f.get("learner_talk_ratio", 0.0) / 0.30)
        learner_turn_rate = self._clip01(f.get("learner_turns_per_min", 0.0) / 2.0)
        active_learners = self._clip01(
            f.get("active_learner_speaker_count", 0.0) / 5.0
        )
        response_ratio = self._clip01(f.get("learner_response_ratio", 0.0))
        avg_turn = self._clip01(f.get("learner_avg_turn_duration", 0.0) / 4.0)

        learner_speech_share = f.get("learner_speech_share", 0.0)
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

    def _interaction_score_01(self, f: dict[str, float]) -> float:
        interaction_rate = self._clip01(f.get("interaction_rate_per_min", 0.0) / 2.0)
        balance = self._clip01(f.get("trainer_learner_balance", 0.0))
        response_ratio = self._clip01(f.get("learner_response_ratio", 0.0))

        overlap_ratio = f.get("overlap_ratio", 0.0)
        overlap_component = 1.0 - self._clip01(overlap_ratio / 0.4)

        score_01 = (
            0.35 * interaction_rate
            + 0.25 * balance
            + 0.25 * response_ratio
            + 0.15 * overlap_component
        )

        return self._clip01(score_01)

    def _pseudo_engagement_score_100(self, f: dict[str, float]) -> float:
        trainer_vocal = self._trainer_vocal_score_01(f)

        has_diarization_context = (
            f.get("session_duration_sec", 0.0) > 0.0
            and f.get("total_turn_count", 0.0) > 0.0
        )

        if not has_diarization_context:
            return float(np.clip(trainer_vocal * 100.0, 0.0, 100.0))

        learner_participation = self._learner_participation_score_01(f)
        interaction = self._interaction_score_01(f)

        score_01 = (
            0.55 * trainer_vocal
            + 0.30 * learner_participation
            + 0.15 * interaction
        )

        return float(np.clip(score_01 * 100.0, 0.0, 100.0))

    # ------------------------------------------------------------------
    # Main extraction method
    # ------------------------------------------------------------------

    def extract(
        self,
        audio_path: str,
        external_emotion_scores: dict | None = None,
        diarization_csv_path: str | None = None,
        trainer_speaker_id: str | None = None,
        add_pseudo_label: bool = True,
    ) -> dict[str, float]:
        y, sr = self._load_audio(audio_path)

        rms = librosa.feature.rms(
            y=y,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
        )[0]

        zcr = librosa.feature.zero_crossing_rate(
            y,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
        )[0]

        spectral_centroid = librosa.feature.spectral_centroid(
            y=y,
            sr=sr,
            n_fft=self.frame_length,
            hop_length=self.hop_length,
        )[0]

        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=y,
            sr=sr,
            n_fft=self.frame_length,
            hop_length=self.hop_length,
        )[0]

        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.frame_length,
            hop_length=self.hop_length,
        )

        mfcc_means = (
            np.mean(mfcc, axis=1)
            if mfcc.size > 0
            else np.zeros(self.n_mfcc)
        )

        pitch_mean, pitch_std, pitch_range, voiced_ratio = self._pitch_stats(y, sr)
        pause_stats = self._pause_stats(y)
        onset_rate_per_sec = self._onset_rate(y, sr)

        features = {
            "rms_mean": self._safe_float(np.mean(rms) if len(rms) else 0.0),
            "rms_std": self._safe_float(np.std(rms) if len(rms) else 0.0),
            "zcr_mean": self._safe_float(np.mean(zcr) if len(zcr) else 0.0),
            "zcr_std": self._safe_float(np.std(zcr) if len(zcr) else 0.0),
            "spectral_centroid_mean": self._safe_float(
                np.mean(spectral_centroid) if len(spectral_centroid) else 0.0
            ),
            "spectral_centroid_std": self._safe_float(
                np.std(spectral_centroid) if len(spectral_centroid) else 0.0
            ),
            "spectral_bandwidth_mean": self._safe_float(
                np.mean(spectral_bandwidth) if len(spectral_bandwidth) else 0.0
            ),
            "spectral_bandwidth_std": self._safe_float(
                np.std(spectral_bandwidth) if len(spectral_bandwidth) else 0.0
            ),
            "pitch_mean": pitch_mean,
            "pitch_std": pitch_std,
            "pitch_range": pitch_range,
            "voiced_ratio": voiced_ratio,
            "onset_rate_per_sec": onset_rate_per_sec,
        }

        features.update(pause_stats)

        features.update(
            self._emotion_features(
                audio_path=audio_path,
                external_emotion_scores=external_emotion_scores,
            )
        )

        features.update(
            self._learner_participation_features(
                diarization_csv_path=diarization_csv_path,
                trainer_speaker_id=trainer_speaker_id,
            )
        )

        for i, value in enumerate(mfcc_means, start=1):
            features[f"mfcc_{i}_mean"] = self._safe_float(value)

        if add_pseudo_label:
            features["engagement_score_100"] = self._pseudo_engagement_score_100(
                features
            )

        return features