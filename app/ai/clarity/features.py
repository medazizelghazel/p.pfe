from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np
import pandas as pd


SCORE_COLUMNS = ["accuracy", "completeness", "fluency", "prosodic", "total"]
META_COLUMNS = ["split", "utt_id", "speaker_id", "gender", "age", "audio_path", "text"]


@dataclass
class ClarityFeatureExtractor:
    target_sr: int = 16000
    n_mfcc: int = 13
    frame_length: int = 1024
    hop_length: int = 256
    silence_db: float = 30.0

    def _safe_float(self, value: float) -> float:
        if value is None:
            return 0.0
        if isinstance(value, (np.floating, float, int, np.integer)):
            if np.isnan(value) or np.isinf(value):
                return 0.0
            return float(value)
        return 0.0

    def _load_audio(self, audio_path: str) -> tuple[np.ndarray, int]:
        y, sr = librosa.load(audio_path, sr=self.target_sr, mono=True)
        y = y.astype(np.float32)
        return y, sr

    def _pitch_stats(self, y: np.ndarray, sr: int) -> tuple[float, float, float, float]:
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
            valid_f0 = f0[~np.isnan(f0)]
        except Exception:
            pitches, _ = librosa.piptrack(
                y=y,
                sr=sr,
                n_fft=self.frame_length,
                hop_length=self.hop_length,
            )
            valid_f0 = pitches[pitches > 0]

        if len(valid_f0) == 0:
            return 0.0, 0.0, 0.0, 0.0

        pitch_mean = float(np.mean(valid_f0))
        pitch_std = float(np.std(valid_f0))
        pitch_range = float(np.max(valid_f0) - np.min(valid_f0))

        total_frames = max(1, int(np.ceil(len(y) / self.hop_length)))
        voiced_ratio = float(len(valid_f0) / total_frames)

        return (
            self._safe_float(pitch_mean),
            self._safe_float(pitch_std),
            self._safe_float(pitch_range),
            self._safe_float(voiced_ratio),
        )

    def _pause_stats(self, y: np.ndarray) -> dict[str, float]:
        duration_sec = len(y) / self.target_sr if len(y) > 0 else 0.0

        if duration_sec <= 0:
            return {
                "pause_ratio": 0.0,
                "pause_count": 0.0,
                "pause_count_per_sec": 0.0,
                "mean_pause_duration_sec": 0.0,
                "max_pause_duration_sec": 0.0,
                "total_pause_duration_sec": 0.0,
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
                "pause_count": 0.0,
                "pause_count_per_sec": 0.0,
                "mean_pause_duration_sec": 0.0,
                "max_pause_duration_sec": 0.0,
                "total_pause_duration_sec": 0.0,
                "speech_ratio": 1.0,
            }

        max_rms = float(np.max(rms))
        if max_rms <= 1e-8:
            return {
                "pause_ratio": 1.0,
                "pause_count": 1.0,
                "pause_count_per_sec": 1.0 / max(duration_sec, 1e-8),
                "mean_pause_duration_sec": duration_sec,
                "max_pause_duration_sec": duration_sec,
                "total_pause_duration_sec": duration_sec,
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
                    pause_durations.append(run * self.hop_length / self.target_sr)
                    run = 0

        if run > 0:
            pause_durations.append(run * self.hop_length / self.target_sr)

        total_pause_duration_sec = float(sum(pause_durations))
        pause_ratio = total_pause_duration_sec / duration_sec if duration_sec > 0 else 0.0
        pause_count = float(len(pause_durations))
        pause_count_per_sec = pause_count / duration_sec if duration_sec > 0 else 0.0
        mean_pause_duration_sec = float(np.mean(pause_durations)) if pause_durations else 0.0
        max_pause_duration_sec = float(np.max(pause_durations)) if pause_durations else 0.0
        speech_ratio = max(0.0, 1.0 - pause_ratio)

        return {
            "pause_ratio": self._safe_float(pause_ratio),
            "pause_count": self._safe_float(pause_count),
            "pause_count_per_sec": self._safe_float(pause_count_per_sec),
            "mean_pause_duration_sec": self._safe_float(mean_pause_duration_sec),
            "max_pause_duration_sec": self._safe_float(max_pause_duration_sec),
            "total_pause_duration_sec": self._safe_float(total_pause_duration_sec),
            "speech_ratio": self._safe_float(speech_ratio),
        }

    def extract(self, audio_path: str) -> dict[str, float]:
        y, sr = self._load_audio(audio_path)
        duration_sec = len(y) / sr if sr > 0 else 0.0

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

        mfcc_means = np.mean(mfcc, axis=1) if mfcc.size > 0 else np.zeros(self.n_mfcc)

        pitch_mean, pitch_std, pitch_range, voiced_ratio = self._pitch_stats(y, sr)
        pause_stats = self._pause_stats(y)

        try:
            onset_frames = librosa.onset.onset_detect(
                y=y,
                sr=sr,
                hop_length=self.hop_length,
                units="frames"
            )
            onset_rate_per_sec = len(onset_frames) / duration_sec if duration_sec > 0 else 0.0
        except Exception:
            onset_rate_per_sec = 0.0

        features = {
            "rms_mean": self._safe_float(np.mean(rms) if len(rms) else 0.0),
            "rms_std": self._safe_float(np.std(rms) if len(rms) else 0.0),
            "zcr_mean": self._safe_float(np.mean(zcr) if len(zcr) else 0.0),
            "zcr_std": self._safe_float(np.std(zcr) if len(zcr) else 0.0),
            "spectral_centroid_mean": self._safe_float(np.mean(spectral_centroid) if len(spectral_centroid) else 0.0),
            "spectral_centroid_std": self._safe_float(np.std(spectral_centroid) if len(spectral_centroid) else 0.0),
            "spectral_bandwidth_mean": self._safe_float(np.mean(spectral_bandwidth) if len(spectral_bandwidth) else 0.0),
            "spectral_bandwidth_std": self._safe_float(np.std(spectral_bandwidth) if len(spectral_bandwidth) else 0.0),
            "pitch_mean": pitch_mean,
            "pitch_std": pitch_std,
            "pitch_range": pitch_range,
            "voiced_ratio": voiced_ratio,
            "onset_rate_per_sec": self._safe_float(onset_rate_per_sec),

            # keep only normalized / deployment-safe pause features
            "pause_ratio": pause_stats["pause_ratio"],
            "pause_count_per_sec": pause_stats["pause_count_per_sec"],
            "mean_pause_duration_sec": pause_stats["mean_pause_duration_sec"],
            "speech_ratio": pause_stats["speech_ratio"],
        }

        features.update(pause_stats)

        for i, value in enumerate(mfcc_means, start=1):
            features[f"mfcc_{i}_mean"] = self._safe_float(value)

        return features


def build_feature_dataframe(manifest_df: pd.DataFrame, extractor: ClarityFeatureExtractor) -> pd.DataFrame:
    rows = []

    for idx, row in manifest_df.iterrows():
        audio_path = row["audio_path"]
        features = extractor.extract(audio_path)

        out = row.to_dict()
        out.update(features)
        rows.append(out)

        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(manifest_df)} files...")

    df = pd.DataFrame(rows)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df