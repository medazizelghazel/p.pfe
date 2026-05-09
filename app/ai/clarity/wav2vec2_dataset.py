from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class Wav2Vec2ClarityConfig:
    target_sample_rate: int = 16000
    max_duration_sec: float = 12.0
    min_duration_sec: float = 2.0


class Wav2Vec2ClarityDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        target_column: str = "fluency",
        config: Wav2Vec2ClarityConfig | None = None,
    ):
        self.manifest_path = Path(manifest_path)
        self.target_column = target_column
        self.config = config or Wav2Vec2ClarityConfig()

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        self.df = pd.read_csv(self.manifest_path)

        required_cols = {"audio_path", target_column}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns in manifest: {missing}")

        self.max_num_samples = int(self.config.target_sample_rate * self.config.max_duration_sec)
        self.min_num_samples = int(self.config.target_sample_rate * self.config.min_duration_sec)

    def __len__(self) -> int:
        return len(self.df)

    def _load_audio(self, path: str) -> np.ndarray:
        y, sr = librosa.load(path, sr=self.config.target_sample_rate, mono=True)
        y = y.astype(np.float32)

        if len(y) < self.min_num_samples:
            pad = self.min_num_samples - len(y)
            y = np.pad(y, (0, pad), mode="constant")

        if len(y) > self.max_num_samples:
            y = y[:self.max_num_samples]

        return y

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        audio_path = str(row["audio_path"])
        waveform = self._load_audio(audio_path)
        label = float(row[self.target_column])

        return {
            "input_values": waveform,
            "label": label,
            "audio_path": audio_path,
            "utt_id": str(row["utt_id"]) if "utt_id" in row else f"row_{idx}",
            "speaker_id": str(row["speaker_id"]) if "speaker_id" in row else "unknown",
        }


def wav2vec2_regression_collate_fn(batch: list[dict]) -> dict:
    max_len = max(len(item["input_values"]) for item in batch)

    input_values = []
    attention_mask = []
    labels = []
    audio_paths = []
    utt_ids = []
    speaker_ids = []

    for item in batch:
        x = item["input_values"]
        pad_len = max_len - len(x)

        padded = np.pad(x, (0, pad_len), mode="constant")
        mask = np.concatenate([
            np.ones(len(x), dtype=np.int64),
            np.zeros(pad_len, dtype=np.int64)
        ])

        input_values.append(padded)
        attention_mask.append(mask)
        labels.append(item["label"])
        audio_paths.append(item["audio_path"])
        utt_ids.append(item["utt_id"])
        speaker_ids.append(item["speaker_id"])

    return {
        "input_values": torch.tensor(np.array(input_values), dtype=torch.float32),
        "attention_mask": torch.tensor(np.array(attention_mask), dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.float32),
        "audio_paths": audio_paths,
        "utt_ids": utt_ids,
        "speaker_ids": speaker_ids,
    }