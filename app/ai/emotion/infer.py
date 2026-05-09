from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
import soundfile as sf
import librosa

from app.ai.emotion.model import EmotionCNN1D
from app.ai.emotion.label_map import PROJECT_ID_TO_LABEL


class EmotionInferenceService:
    def __init__(
        self,
        model_path: str = "artifacts/emotion/emotion_cnn1d_best.pt",
        target_sample_rate: int = 16000,
        max_duration: float = 3.0,
        hop_duration: float = 1.5,
        device: str | None = None,
    ):
        self.model_path = Path(model_path)
        self.target_sample_rate = target_sample_rate
        self.max_duration = max_duration
        self.hop_duration = hop_duration
        self.target_num_samples = int(target_sample_rate * max_duration)
        self.hop_num_samples = int(target_sample_rate * hop_duration)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = EmotionCNN1D(num_classes=4).to(self.device)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()

    def _load_audio_array(self, audio_path: str):
        waveform, sample_rate = sf.read(audio_path)

        if len(waveform.shape) > 1:
            waveform = waveform.mean(axis=1)

        if sample_rate != self.target_sample_rate:
            waveform = librosa.resample(
                waveform,
                orig_sr=sample_rate,
                target_sr=self.target_sample_rate
            )

        return waveform

    def _fix_length_tensor(self, waveform_tensor: torch.Tensor) -> torch.Tensor:
        num_samples = waveform_tensor.shape[1]

        if num_samples > self.target_num_samples:
            waveform_tensor = waveform_tensor[:, :self.target_num_samples]
        elif num_samples < self.target_num_samples:
            pad_amount = self.target_num_samples - num_samples
            waveform_tensor = F.pad(waveform_tensor, (0, pad_amount))

        return waveform_tensor

    def _segment_audio(self, waveform):
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
    def _predict_segment_array(self, segment_array) -> dict:
        waveform_tensor = torch.tensor(segment_array, dtype=torch.float32).unsqueeze(0)
        waveform_tensor = self._fix_length_tensor(waveform_tensor)
        waveform_tensor = waveform_tensor.unsqueeze(0).to(self.device)  # [1, 1, N]

        logits = self.model(waveform_tensor)
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
        label_scores = defaultdict(float)

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

            label_scores[pred["predicted_label"]] += pred["confidence"]

        dominant_emotion = max(label_scores.items(), key=lambda x: x[1])[0]
        total_score = sum(label_scores.values())
        global_confidence = label_scores[dominant_emotion] / total_score if total_score > 0 else 0.0

        return {
            "audio_path": audio_path,
            "num_segments": len(segment_predictions),
            "dominant_emotion": dominant_emotion,
            "confidence": float(global_confidence),
            "segment_predictions": segment_predictions,
            "aggregated_scores": dict(label_scores),
        }