from pathlib import Path
from collections import defaultdict

import torch
import soundfile as sf
import librosa
import numpy as np

from app.ai.emotion.cnn2d_model import EmotionCNN2D
from app.ai.emotion.label_map import PROJECT_ID_TO_LABEL


class EmotionInferenceServiceCNN2D:
    def __init__(
        self,
        model_path: str = "artifacts/emotion/emotion_cnn2d_best.pt",
        target_sample_rate: int = 16000,
        max_duration: float = 3.0,
        hop_duration: float = 1.5,
        n_mels: int = 64,
        n_fft: int = 1024,
        hop_length: int = 256,
        device: str | None = None,
    ):
        self.model_path = Path(model_path)
        self.target_sample_rate = target_sample_rate
        self.max_duration = max_duration
        self.hop_duration = hop_duration
        self.target_num_samples = int(target_sample_rate * max_duration)
        self.hop_num_samples = int(target_sample_rate * hop_duration)

        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length

        self.model_name = "EmotionCNN2D_LogMel_RAVDESS"
        self.model_metrics = {
            "accuracy": 0.6167,
            "macro_precision": 0.5265,
            "macro_recall": 0.5456,
            "macro_f1": 0.5301,
        }

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = EmotionCNN2D(num_classes=4).to(self.device)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()

    def _load_audio_array(self, audio_path: str) -> np.ndarray:
        waveform, sample_rate = sf.read(audio_path)

        if len(waveform.shape) > 1:
            waveform = waveform.mean(axis=1)

        if sample_rate != self.target_sample_rate:
            waveform = librosa.resample(
                waveform,
                orig_sr=sample_rate,
                target_sr=self.target_sample_rate
            )

        return waveform.astype(np.float32)

    def _fix_length_array(self, waveform: np.ndarray) -> np.ndarray:
        num_samples = len(waveform)

        if num_samples > self.target_num_samples:
            waveform = waveform[:self.target_num_samples]
        elif num_samples < self.target_num_samples:
            pad_amount = self.target_num_samples - num_samples
            waveform = np.pad(waveform, (0, pad_amount), mode="constant")

        return waveform

    def _compute_log_mel(self, waveform: np.ndarray) -> torch.Tensor:
        mel = librosa.feature.melspectrogram(
            y=waveform,
            sr=self.target_sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            power=2.0,
        )

        log_mel = librosa.power_to_db(mel, ref=np.max)
        return torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0)

    def _segment_audio(self, waveform: np.ndarray) -> list[dict]:
        total_samples = len(waveform)

        if total_samples <= self.target_num_samples:
            return [{
                "start_sec": 0.0,
                "end_sec": total_samples / self.target_sample_rate,
                "samples": waveform
            }]

        segments = []
        start = 0

        while start < total_samples:
            end = start + self.target_num_samples
            segment = waveform[start:end]

            if len(segment) < int(0.5 * self.target_num_samples):
                break

            segments.append({
                "start_sec": start / self.target_sample_rate,
                "end_sec": min(end, total_samples) / self.target_sample_rate,
                "samples": segment
            })

            start += self.hop_num_samples

        return segments

    @torch.no_grad()
    def _predict_segment_array(self, segment_array: np.ndarray) -> dict:
        segment_array = self._fix_length_array(segment_array)
        features = self._compute_log_mel(segment_array).unsqueeze(0).to(self.device)

        logits = self.model(features)
        probs = torch.softmax(logits, dim=1)[0]

        pred_id = int(torch.argmax(probs).item())
        pred_label = PROJECT_ID_TO_LABEL[pred_id]
        confidence = float(probs[pred_id].item())

        class_probabilities = {
            PROJECT_ID_TO_LABEL[i]: float(probs[i].item())
            for i in range(len(PROJECT_ID_TO_LABEL))
        }

        return {
            "predicted_label": pred_label,
            "predicted_label_id": pred_id,
            "confidence": confidence,
            "probabilities": class_probabilities,
        }

    @torch.no_grad()
    def predict_file(self, audio_path: str) -> dict:
        audio_path = str(audio_path)
        waveform = self._load_audio_array(audio_path)
        segments = self._segment_audio(waveform)

        if not segments:
            raise ValueError("No valid segments found for emotion prediction.")

        segment_predictions = []
        aggregated_probabilities = defaultdict(float)

        for segment in segments:
            pred = self._predict_segment_array(segment["samples"])

            segment_result = {
                "start_sec": round(segment["start_sec"], 2),
                "end_sec": round(segment["end_sec"], 2),
                "predicted_label": pred["predicted_label"],
                "predicted_label_id": pred["predicted_label_id"],
                "confidence": pred["confidence"],
                "probabilities": pred["probabilities"],
            }
            segment_predictions.append(segment_result)

            for label, prob in pred["probabilities"].items():
                aggregated_probabilities[label] += prob

        num_segments = len(segment_predictions)

        mean_probabilities = {
            label: score / num_segments
            for label, score in aggregated_probabilities.items()
        }

        dominant_emotion = max(mean_probabilities.items(), key=lambda x: x[1])[0]
        global_confidence = float(mean_probabilities[dominant_emotion])

        return {
            "audio_path": audio_path,
            "model_name": self.model_name,
            "model_metrics": self.model_metrics,
            "num_segments": num_segments,
            "dominant_emotion": dominant_emotion,
            "confidence": global_confidence,
            "segment_predictions": segment_predictions,
            "aggregated_scores": mean_probabilities,
        }