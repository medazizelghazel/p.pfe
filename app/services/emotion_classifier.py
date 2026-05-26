from __future__ import annotations

from pathlib import Path
from typing import Literal


class EmotionClassifier:
    """
    Emotion classifier wrapper.

    Official backend used by the system:
        Wav2Vec2

    The model predicts raw emotions:
        angry, disgust, fear, happy, neutral, sad

    Then infer_wav2vec2.py maps them to pedagogical labels:
        neutral_calm
        energetic_engaged
        low_energy
        tense_stressed
    """

    def __init__(
        self,
        backend: str = "wav2vec2",
        model_path: str | None = None,
        target_sample_rate: int = 16000,
        mode: Literal["fast", "balanced", "full"] = "fast",
        max_duration: float | None = None,
        hop_duration: float | None = None,
        **kwargs,
    ):
        # Force Wav2Vec2 in the whole system.
        # The backend parameter is kept only to avoid breaking old code.
        if backend != "wav2vec2":
            print(
                "[EmotionClassifier] CNN2D backend is disabled. "
                "Using Wav2Vec2 instead."
            )

        self.backend = "wav2vec2"
        self.mode = mode

        if max_duration is None or hop_duration is None:
            max_duration, hop_duration = self._resolve_mode(mode)

        self.max_duration = max_duration
        self.hop_duration = hop_duration

        from app.ai.emotion.infer_wav2vec2 import EmotionInferenceServiceWav2Vec2

        self.model_path = Path(
            model_path or "artifacts/emotion/wav2vec2_emotion_best.pt"
        )

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Wav2Vec2 emotion model not found: {self.model_path}. "
                "Make sure artifacts/emotion/wav2vec2_emotion_best.pt exists."
            )

        print(
            "[EmotionClassifier] "
            f"backend=wav2vec2, mode={self.mode}, "
            f"max_duration={self.max_duration}s, "
            f"hop_duration={self.hop_duration}s, "
            f"model_path={self.model_path}"
        )

        self.inference_service = EmotionInferenceServiceWav2Vec2(
            model_path=str(self.model_path),
            target_sample_rate=target_sample_rate,
            max_duration=self.max_duration,
            hop_duration=self.hop_duration,
        )

    def _resolve_mode(
        self,
        mode: Literal["fast", "balanced", "full"],
    ) -> tuple[float, float]:
        if mode == "fast":
            return 8.0, 8.0

        if mode == "balanced":
            return 6.0, 6.0

        if mode == "full":
            return 3.0, 1.5

        raise ValueError(f"Unknown emotion mode: {mode}")

    def classify(self, audio_path: str) -> str:
        result = self.inference_service.predict_file(audio_path)
        return result["dominant_emotion"]

    def classify_with_details(self, audio_path: str) -> dict:
        result = self.inference_service.predict_file(audio_path)

        result["inference_mode"] = self.mode
        result["backend"] = self.backend
        result["max_duration"] = self.max_duration
        result["hop_duration"] = self.hop_duration

        return result

    def get_model_name(self) -> str:
        return self.inference_service.model_name

    def get_model_metrics(self) -> dict:
        return self.inference_service.model_metrics
