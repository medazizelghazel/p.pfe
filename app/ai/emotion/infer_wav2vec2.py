from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor, Wav2Vec2Model


class Wav2Vec2EmotionCustom(nn.Module):
    """
    Custom Wav2Vec2 model for your checkpoint.

    Your checkpoint uses a custom classifier head:
        classifier.0.*
        classifier.1.*
        classifier.4.*
        classifier.5.*
        classifier.8.*

    So we cannot load it with Wav2Vec2ForSequenceClassification.
    """

    def __init__(
        self,
        config: Wav2Vec2Config,
        classifier: nn.Module,
    ):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = classifier

    def _masked_mean_pooling(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if attention_mask is None:
            return hidden_states.mean(dim=1)

        input_lengths = attention_mask.sum(dim=1)
        output_lengths = self.wav2vec2._get_feat_extract_output_lengths(input_lengths)

        batch_size, sequence_length, _ = hidden_states.shape

        feature_attention_mask = torch.zeros(
            (batch_size, sequence_length),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        for i, length in enumerate(output_lengths):
            length = int(length.item())
            feature_attention_mask[i, :length] = 1.0

        feature_attention_mask = feature_attention_mask.unsqueeze(-1)

        summed = (hidden_states * feature_attention_mask).sum(dim=1)
        counts = feature_attention_mask.sum(dim=1).clamp(min=1.0)

        return summed / counts

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        outputs = self.wav2vec2(
            input_values=input_values,
            attention_mask=attention_mask,
        )

        hidden_states = outputs.last_hidden_state
        pooled = self._masked_mean_pooling(hidden_states, attention_mask)
        logits = self.classifier(pooled)

        return logits


class EmotionInferenceServiceWav2Vec2:
    def __init__(
        self,
        model_path: str = "artifacts/emotion/wav2vec2_emotion_best.pt",
        target_sample_rate: int = 16000,
        max_duration: float = 8.0,
        hop_duration: float = 8.0,
    ):
        self.model_path = Path(model_path)
        self.target_sample_rate = target_sample_rate
        self.max_duration = max_duration
        self.hop_duration = hop_duration

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.default_id2label = {
            0: "angry",
            1: "disgust",
            2: "fear",
            3: "happy",
            4: "neutral",
            5: "sad",
        }

        self.id2label = dict(self.default_id2label)
        self.label2id = {v: k for k, v in self.id2label.items()}

        self.model_name = "Wav2Vec2_Emotion_Custom"
        self.model_metrics = {
            "accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "val_acc": 0.0,
            "val_f1": 0.0,
        }

        self.feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=self.target_sample_rate,
            padding_value=0.0,
            do_normalize=True,
            return_attention_mask=True,
        )

        self.model = None
        self._load_model()

    # ------------------------------------------------------------------
    # Checkpoint loading
    # ------------------------------------------------------------------

    def _torch_load(self):
        try:
            return torch.load(
                self.model_path,
                map_location=self.device,
                weights_only=False,
            )
        except TypeError:
            return torch.load(
                self.model_path,
                map_location=self.device,
            )

    def _normalize_id2label(self, id2label) -> dict[int, str]:
        if not id2label:
            return dict(self.default_id2label)

        return {
            int(k): str(v)
            for k, v in id2label.items()
        }

    def _normalize_label2id(self, label2id) -> dict[str, int]:
        if not label2id:
            return {v: k for k, v in self.default_id2label.items()}

        return {
            str(k): int(v)
            for k, v in label2id.items()
        }

    def _extract_state_dict(self, checkpoint) -> dict:
        if isinstance(checkpoint, dict):
            for key in ["model_state_dict", "state_dict", "model", "net"]:
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    return checkpoint[key]

        if isinstance(checkpoint, dict):
            return checkpoint

        raise ValueError("Unsupported Wav2Vec2 checkpoint format.")

    def _strip_prefixes(self, state_dict: dict) -> dict:
        cleaned = {}

        for key, value in state_dict.items():
            new_key = key

            # Common wrappers
            changed = True
            while changed:
                changed = False

                for prefix in ["module.", "model.", "net.", "network."]:
                    if new_key.startswith(prefix):
                        new_key = new_key.replace(prefix, "", 1)
                        changed = True

            # Some training scripts use base_model or wav2vec2_model.
            if new_key.startswith("base_model."):
                new_key = "wav2vec2." + new_key.replace("base_model.", "", 1)

            if new_key.startswith("wav2vec2_model."):
                new_key = "wav2vec2." + new_key.replace("wav2vec2_model.", "", 1)

            cleaned[new_key] = value

        return cleaned

    def _load_labels_and_metrics(self, checkpoint) -> None:
        if not isinstance(checkpoint, dict):
            return

        if "id2label" in checkpoint:
            self.id2label = self._normalize_id2label(checkpoint["id2label"])
            self.label2id = {
                label: idx
                for idx, label in self.id2label.items()
            }

        elif "label2id" in checkpoint:
            self.label2id = self._normalize_label2id(checkpoint["label2id"])
            self.id2label = {
                idx: label
                for label, idx in self.label2id.items()
            }

        val_acc = float(checkpoint.get("val_acc", checkpoint.get("accuracy", 0.0)))
        val_f1 = float(checkpoint.get("val_f1", checkpoint.get("macro_f1", 0.0)))

        self.model_metrics = {
            "accuracy": float(checkpoint.get("accuracy", val_acc)),
            "macro_precision": float(checkpoint.get("macro_precision", 0.0)),
            "macro_recall": float(checkpoint.get("macro_recall", 0.0)),
            "macro_f1": float(checkpoint.get("macro_f1", val_f1)),
            "val_acc": val_acc,
            "val_f1": val_f1,
        }

    def _build_config(self, checkpoint) -> Wav2Vec2Config:
        if isinstance(checkpoint, dict):
            if "config" in checkpoint and isinstance(checkpoint["config"], dict):
                config = Wav2Vec2Config.from_dict(checkpoint["config"])
            elif "model_config" in checkpoint and isinstance(checkpoint["model_config"], dict):
                config = Wav2Vec2Config.from_dict(checkpoint["model_config"])
            else:
                config = Wav2Vec2Config()
        else:
            config = Wav2Vec2Config()

        config.num_labels = len(self.id2label)
        config.id2label = self.id2label
        config.label2id = self.label2id

        return config

    def _build_classifier_from_state_dict(self, state_dict: dict) -> nn.Sequential:
        classifier_keys = [
            key
            for key in state_dict.keys()
            if key.startswith("classifier.")
        ]

        if not classifier_keys:
            raise RuntimeError("No classifier.* keys found in Wav2Vec2 checkpoint.")

        weight_keys = sorted(
            [
                key
                for key in classifier_keys
                if key.endswith(".weight")
            ],
            key=lambda x: int(x.split(".")[1]),
        )

        linear_indices = []
        batchnorm_indices = []
        layernorm_indices = []

        for key in weight_keys:
            idx = int(key.split(".")[1])
            tensor = state_dict[key]

            if tensor.ndim == 2:
                linear_indices.append(idx)

            elif tensor.ndim == 1:
                if f"classifier.{idx}.running_mean" in state_dict:
                    batchnorm_indices.append(idx)
                else:
                    layernorm_indices.append(idx)

        modules: dict[int, nn.Module] = {}

        for idx in linear_indices:
            weight = state_dict[f"classifier.{idx}.weight"]
            out_features, in_features = weight.shape
            modules[idx] = nn.Linear(in_features, out_features)

        for idx in batchnorm_indices:
            weight = state_dict[f"classifier.{idx}.weight"]
            modules[idx] = nn.BatchNorm1d(weight.shape[0])

        for idx in layernorm_indices:
            weight = state_dict[f"classifier.{idx}.weight"]
            modules[idx] = nn.LayerNorm(weight.shape[0])

        # Fill common activation/dropout positions used in training.
        if 0 in modules and 1 in modules:
            modules.setdefault(2, nn.ReLU())
            modules.setdefault(3, nn.Dropout(0.3))

        if 4 in modules and 5 in modules:
            modules.setdefault(6, nn.ReLU())
            modules.setdefault(7, nn.Dropout(0.3))

        max_idx = max(modules.keys())
        ordered_modules = []

        for idx in range(max_idx + 1):
            ordered_modules.append(modules.get(idx, nn.Identity()))

        classifier = nn.Sequential(*ordered_modules)

        print("[EmotionInferenceServiceWav2Vec2] Custom classifier:")
        for i, module in enumerate(classifier):
            print(f"  classifier.{i}: {module}")

        return classifier

    def _load_model(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Wav2Vec2 emotion model not found: {self.model_path}")

        checkpoint = self._torch_load()

        self._load_labels_and_metrics(checkpoint)

        state_dict = self._extract_state_dict(checkpoint)
        state_dict = self._strip_prefixes(state_dict)

        config = self._build_config(checkpoint)
        classifier = self._build_classifier_from_state_dict(state_dict)

        self.model = Wav2Vec2EmotionCustom(
            config=config,
            classifier=classifier,
        ).to(self.device)

        load_result = self.model.load_state_dict(state_dict, strict=False)

        missing = list(load_result.missing_keys)
        unexpected = list(load_result.unexpected_keys)

        print("[EmotionInferenceServiceWav2Vec2] Model loaded.")
        print(f"[EmotionInferenceServiceWav2Vec2] Path: {self.model_path}")
        print(f"[EmotionInferenceServiceWav2Vec2] Device: {self.device}")
        print(f"[EmotionInferenceServiceWav2Vec2] Labels: {self.id2label}")
        print(f"[EmotionInferenceServiceWav2Vec2] Metrics: {self.model_metrics}")

        if missing:
            print(f"[EmotionInferenceServiceWav2Vec2] Missing keys: {missing[:20]}")
        if unexpected:
            print(f"[EmotionInferenceServiceWav2Vec2] Unexpected keys: {unexpected[:20]}")

        self.model.eval()

    # ------------------------------------------------------------------
    # Label mapping for project
    # ------------------------------------------------------------------

    def _canonical_label(self, label: str) -> str:
        label = str(label).strip().lower()

        aliases = {
            "neutral": "neutral_calm",
            "calm": "neutral_calm",
            "neutral_calm": "neutral_calm",

            "happy": "energetic_engaged",
            "joy": "energetic_engaged",
            "energetic": "energetic_engaged",
            "engaged": "energetic_engaged",
            "energetic_engaged": "energetic_engaged",

            "sad": "low_energy",
            "bored": "low_energy",
            "tired": "low_energy",
            "low": "low_energy",
            "low_energy": "low_energy",

            "angry": "tense_stressed",
            "fear": "tense_stressed",
            "fearful": "tense_stressed",
            "disgust": "tense_stressed",
            "stress": "tense_stressed",
            "stressed": "tense_stressed",
            "tense": "tense_stressed",
            "tense_stressed": "tense_stressed",
        }

        return aliases.get(label, label)

    def _canonical_probabilities(self, raw_probs: dict[str, float]) -> dict[str, float]:
        canonical = {
            "neutral_calm": 0.0,
            "energetic_engaged": 0.0,
            "low_energy": 0.0,
            "tense_stressed": 0.0,
        }

        for label, value in raw_probs.items():
            canonical_label = self._canonical_label(label)

            if canonical_label in canonical:
                canonical[canonical_label] += float(value)

        total = sum(canonical.values())

        if total > 0:
            canonical = {
                key: float(value / total)
                for key, value in canonical.items()
            }

        return canonical

    # ------------------------------------------------------------------
    # Audio
    # ------------------------------------------------------------------

    def _load_audio(self, audio_path: str) -> tuple[np.ndarray, int]:
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        y, sr = librosa.load(
            str(audio_path),
            sr=self.target_sample_rate,
            mono=True,
        )

        return y.astype(np.float32), sr

    def _prepare_segment(self, segment_audio: np.ndarray) -> dict:
        segment_audio = segment_audio.astype(np.float32)

        max_samples = int(self.max_duration * self.target_sample_rate)

        if len(segment_audio) < max_samples:
            segment_audio = np.pad(segment_audio, (0, max_samples - len(segment_audio)))
        else:
            segment_audio = segment_audio[:max_samples]

        inputs = self.feature_extractor(
            segment_audio,
            sampling_rate=self.target_sample_rate,
            return_tensors="pt",
            padding=True,
        )

        return {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def _predict_segment(self, segment_audio: np.ndarray) -> dict:
        inputs = self._prepare_segment(segment_audio)

        with torch.no_grad():
            logits = self.model(
                input_values=inputs["input_values"],
                attention_mask=inputs.get("attention_mask"),
            )

            probs = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()

        raw_probs = {
            self.id2label.get(i, f"class_{i}"): float(probs[i])
            for i in range(len(probs))
        }

        canonical_probs = self._canonical_probabilities(raw_probs)

        predicted_label = max(
            canonical_probs,
            key=canonical_probs.get,
        )

        confidence = float(canonical_probs[predicted_label])
        predicted_label_id = int(np.argmax(probs))

        return {
            "predicted_label": predicted_label,
            "predicted_label_id": predicted_label_id,
            "confidence": confidence,
            "probabilities": canonical_probs,
            "raw_probabilities": raw_probs,
        }

    def predict_file(self, audio_path: str) -> dict:
        y, sr = self._load_audio(audio_path)

        duration = len(y) / sr if sr > 0 else 0.0

        empty_scores = {
            "neutral_calm": 0.0,
            "energetic_engaged": 0.0,
            "low_energy": 0.0,
            "tense_stressed": 0.0,
        }

        if duration <= 0:
            return {
                "dominant_emotion": "unknown",
                "confidence": 0.0,
                "num_segments": 0,
                "aggregated_scores": empty_scores,
                "segment_predictions": [],
                "model_name": self.model_name,
                "model_metrics": self.model_metrics,
            }

        segment_predictions = []
        start = 0.0

        while start < duration:
            end = min(start + self.max_duration, duration)

            start_sample = int(start * sr)
            end_sample = int(end * sr)

            segment_audio = y[start_sample:end_sample]

            if len(segment_audio) < int(0.4 * sr):
                break

            pred = self._predict_segment(segment_audio)

            segment_predictions.append(
                {
                    "start_sec": round(float(start), 4),
                    "end_sec": round(float(end), 4),
                    "predicted_label": pred["predicted_label"],
                    "predicted_label_id": pred["predicted_label_id"],
                    "confidence": pred["confidence"],
                    "probabilities": pred["probabilities"],
                    "raw_probabilities": pred["raw_probabilities"],
                }
            )

            start += self.hop_duration

        if not segment_predictions:
            return {
                "dominant_emotion": "unknown",
                "confidence": 0.0,
                "num_segments": 0,
                "aggregated_scores": empty_scores,
                "segment_predictions": [],
                "model_name": self.model_name,
                "model_metrics": self.model_metrics,
            }

        aggregated_scores = dict(empty_scores)

        for segment in segment_predictions:
            for label, value in segment["probabilities"].items():
                aggregated_scores[label] += float(value)

        n = len(segment_predictions)

        aggregated_scores = {
            label: float(value / n)
            for label, value in aggregated_scores.items()
        }

        dominant_emotion = max(
            aggregated_scores,
            key=aggregated_scores.get,
        )

        confidence = float(aggregated_scores[dominant_emotion])

        return {
            "dominant_emotion": dominant_emotion,
            "confidence": confidence,
            "num_segments": n,
            "aggregated_scores": aggregated_scores,
            "segment_predictions": segment_predictions,
            "model_name": self.model_name,
            "model_metrics": self.model_metrics,
        }