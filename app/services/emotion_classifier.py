from __future__ import annotations

from pathlib import Path
from typing import Literal


class EmotionClassifier:
    """
    Emotion classifier wrapper.

    Default backend:
        wav2vec2

    Output format stays compatible with the rest of the project:
        dominant_emotion
        confidence
        num_segments
        aggregated_scores
        segment_predictions
        model_name
        model_metrics
    """

    def __init__(
        self,
        backend: Literal["wav2vec2", "cnn2d"] = "wav2vec2",
        model_path: str | None = None,
        target_sample_rate: int = 16000,
        mode: Literal["fast", "balanced", "full"] = "fast",
        max_duration: float | None = None,
        hop_duration: float | None = None,
        n_mels: int = 64,
        n_fft: int = 1024,
        hop_length: int = 256,
    ):
        self.backend = backend
        self.mode = mode

        if max_duration is None or hop_duration is None:
            max_duration, hop_duration = self._resolve_mode(mode)

        self.max_duration = max_duration
        self.hop_duration = hop_duration

        if self.backend == "wav2vec2":
            from app.ai.emotion.infer_wav2vec2 import EmotionInferenceServiceWav2Vec2

            self.model_path = Path(
                model_path or "artifacts/emotion/wav2vec2_emotion_best.pt"
            )

            print(
                "[EmotionClassifier] "
                f"backend=wav2vec2, mode={self.mode}, "
                f"max_duration={self.max_duration}s, "
                f"hop_duration={self.hop_duration}s"
            )

            self.inference_service = EmotionInferenceServiceWav2Vec2(
                model_path=str(self.model_path),
                target_sample_rate=target_sample_rate,
                max_duration=self.max_duration,
                hop_duration=self.hop_duration,
            )

        elif self.backend == "cnn2d":
            # Optional legacy backend.
            # Use only if infer_cnn2d.py still contains EmotionInferenceServiceCNN2D.
            try:
                from app.ai.emotion.infer_cnn2d import EmotionInferenceServiceCNN2D
            except ImportError as e:
                raise ImportError(
                    "CNN2D backend is not available. "
                    "Use backend='wav2vec2' or restore EmotionInferenceServiceCNN2D "
                    "inside app/ai/emotion/infer_cnn2d.py."
                ) from e

            self.model_path = Path(
                model_path or "artifacts/emotion/emotion_cnn2d_best.pt"
            )

            print(
                "[EmotionClassifier] "
                f"backend=cnn2d, mode={self.mode}, "
                f"max_duration={self.max_duration}s, "
                f"hop_duration={self.hop_duration}s"
            )

            self.inference_service = EmotionInferenceServiceCNN2D(
                model_path=str(self.model_path),
                target_sample_rate=target_sample_rate,
                max_duration=self.max_duration,
                hop_duration=self.hop_duration,
                n_mels=n_mels,
                n_fft=n_fft,
                hop_length=hop_length,
            )

        else:
            raise ValueError(f"Unknown emotion backend: {self.backend}")

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