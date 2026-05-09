from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
import soundfile as sf
import librosa
from torch.utils.data import Dataset


class EmotionAudioDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        target_sample_rate: int = 16000,
        max_duration: float = 3.0,
    ):
        self.manifest_path = Path(manifest_path)
        self.target_sample_rate = target_sample_rate
        self.max_duration = max_duration
        self.target_num_samples = int(target_sample_rate * max_duration)

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {self.manifest_path}")

        self.df = pd.read_csv(self.manifest_path)

        required_columns = {"path", "label", "label_id", "dataset", "speaker_id"}
        missing = required_columns - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns in manifest: {missing}")

    def __len__(self):
        return len(self.df)

    def _load_audio(self, audio_path: str) -> torch.Tensor:
        audio_path = str(audio_path)

        waveform, sample_rate = sf.read(audio_path)

        if len(waveform.shape) > 1:
            waveform = waveform.mean(axis=1)

        if sample_rate != self.target_sample_rate:
            waveform = librosa.resample(
                waveform,
                orig_sr=sample_rate,
                target_sr=self.target_sample_rate
            )

        waveform = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)
        return waveform

    def _fix_length(self, waveform: torch.Tensor) -> torch.Tensor:
        num_samples = waveform.shape[1]

        if num_samples > self.target_num_samples:
            waveform = waveform[:, :self.target_num_samples]
        elif num_samples < self.target_num_samples:
            pad_amount = self.target_num_samples - num_samples
            waveform = F.pad(waveform, (0, pad_amount))

        return waveform

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        audio_path = row["path"]
        label_id = int(row["label_id"])
        label = row["label"]
        speaker_id = str(row["speaker_id"])

        waveform = self._load_audio(audio_path)
        waveform = self._fix_length(waveform)

        return {
            "waveform": waveform,
            "label_id": torch.tensor(label_id, dtype=torch.long),
            "label": label,
            "speaker_id": speaker_id,
            "path": audio_path,
        }