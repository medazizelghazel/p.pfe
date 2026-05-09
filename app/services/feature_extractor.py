from __future__ import annotations

import librosa
import numpy as np

from app.config import DEFAULT_N_MFCC, TARGET_SAMPLE_RATE
from app.domain.feature_set import FeatureSet


class FeatureExtractor:
    def __init__(
        self,
        n_mfcc: int = DEFAULT_N_MFCC,
        frame_length: int = 1024,
        hop_length: int = 512,
    ):
        self.n_mfcc = n_mfcc
        self.frame_length = frame_length
        self.hop_length = hop_length

    def _safe_float(self, value: float) -> float:
        if value is None:
            return 0.0
        if isinstance(value, (float, int, np.floating, np.integer)):
            if np.isnan(value) or np.isinf(value):
                return 0.0
            return float(value)
        return 0.0

    def extract_mfcc(self, y: np.ndarray, sr: int) -> list[float]:
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.frame_length,
            hop_length=self.hop_length,
        )

        if mfcc.size == 0:
            return [0.0] * self.n_mfcc

        return np.mean(mfcc, axis=1).astype(float).tolist()

    def extract_pitch_data(self, y: np.ndarray, sr: int):
        if len(y) == 0 or np.max(np.abs(y)) < 1e-8:
            return 0.0, 0.0, 0.0, []

        try:
            f0, _, _ = librosa.pyin(
                y,
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C7"),
                frame_length=self.frame_length,
                hop_length=self.hop_length,
            )
        except Exception:
            return 0.0, 0.0, 0.0, []

        if f0 is None or len(f0) == 0:
            return 0.0, 0.0, 0.0, []

        pitch_series = np.nan_to_num(f0, nan=0.0)
        valid_f0 = f0[~np.isnan(f0)]

        if len(valid_f0) == 0:
            return 0.0, 0.0, 0.0, pitch_series.astype(float).tolist()

        pitch_mean = self._safe_float(np.mean(valid_f0))
        pitch_std = self._safe_float(np.std(valid_f0))
        voiced_ratio_pitch = self._safe_float(len(valid_f0) / len(f0))

        return pitch_mean, pitch_std, voiced_ratio_pitch, pitch_series.astype(float).tolist()

    def extract_energy_data(self, y: np.ndarray):
        rms = librosa.feature.rms(
            y=y,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
        )[0]

        if len(rms) == 0:
            return 0.0, 0.0, []

        return (
            self._safe_float(np.mean(rms)),
            self._safe_float(np.std(rms)),
            rms.astype(float).tolist(),
        )

    def extract_zcr_stats(self, y: np.ndarray):
        zcr = librosa.feature.zero_crossing_rate(
            y,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
        )[0]

        if len(zcr) == 0:
            return 0.0, 0.0

        return self._safe_float(np.mean(zcr)), self._safe_float(np.std(zcr))

    def extract_spectral_centroid_stats(self, y: np.ndarray, sr: int):
        centroid = librosa.feature.spectral_centroid(
            y=y,
            sr=sr,
            n_fft=self.frame_length,
            hop_length=self.hop_length,
        )[0]

        if len(centroid) == 0:
            return 0.0, 0.0

        return self._safe_float(np.mean(centroid)), self._safe_float(np.std(centroid))

    def extract_spectral_bandwidth_stats(self, y: np.ndarray, sr: int):
        bandwidth = librosa.feature.spectral_bandwidth(
            y=y,
            sr=sr,
            n_fft=self.frame_length,
            hop_length=self.hop_length,
        )[0]

        if len(bandwidth) == 0:
            return 0.0, 0.0

        return self._safe_float(np.mean(bandwidth)), self._safe_float(np.std(bandwidth))

    def smooth_binary_series(
        self,
        values: list[int],
        min_run: int = 3,
    ) -> list[int]:
        if not values:
            return values

        smoothed = values[:]
        start = 0

        while start < len(smoothed):
            end = start

            while end < len(smoothed) and smoothed[end] == smoothed[start]:
                end += 1

            run_length = end - start

            if run_length < min_run:
                if start > 0:
                    replacement = smoothed[start - 1]
                elif end < len(smoothed):
                    replacement = smoothed[end]
                else:
                    replacement = smoothed[start]

                for i in range(start, end):
                    smoothed[i] = replacement

            start = end

        return smoothed

    def build_voice_activity_series(self, energy_series: list[float]) -> list[int]:
        if not energy_series:
            return []

        arr = np.array(energy_series, dtype=float)

        if len(arr) == 0 or np.max(arr) <= 1e-8:
            return [0] * len(arr)

        threshold = max(float(np.mean(arr) * 0.8), float(np.max(arr) * 0.18))

        raw = [1 if e >= threshold else 0 for e in arr.tolist()]

        return self.smooth_binary_series(raw, min_run=3)

    def extract_pause_metrics(
        self,
        voice_activity_series: list[int],
        sr: int,
    ):
        if not voice_activity_series:
            return 0, 0.0, 0.0, 0.0, 0.0

        frame_duration = self.hop_length / sr
        silence_runs = []

        start = 0

        while start < len(voice_activity_series):
            if voice_activity_series[start] == 0:
                end = start

                while end < len(voice_activity_series) and voice_activity_series[end] == 0:
                    end += 1

                silence_runs.append(end - start)
                start = end
            else:
                start += 1

        pause_durations = [
            run * frame_duration
            for run in silence_runs
            if run * frame_duration >= 0.2
        ]

        pause_count = len(pause_durations)
        total_pause_duration = self._safe_float(sum(pause_durations))
        mean_pause_duration = self._safe_float(np.mean(pause_durations)) if pause_durations else 0.0

        voiced_ratio = self._safe_float(sum(voice_activity_series) / len(voice_activity_series))
        silence_ratio = self._safe_float(1.0 - voiced_ratio)

        return (
            pause_count,
            mean_pause_duration,
            total_pause_duration,
            voiced_ratio,
            silence_ratio,
        )

    def extract(self, audio_path: str) -> FeatureSet:
        y, sr = librosa.load(
            audio_path,
            sr=TARGET_SAMPLE_RATE,
            mono=True,
        )

        y = y.astype(np.float32)

        mfcc_means = self.extract_mfcc(y, sr)

        pitch_mean, pitch_std, _, pitch_series = self.extract_pitch_data(y, sr)

        energy_mean, energy_std, energy_series = self.extract_energy_data(y)

        zcr_mean, zcr_std = self.extract_zcr_stats(y)

        spectral_centroid_mean, spectral_centroid_std = self.extract_spectral_centroid_stats(y, sr)

        spectral_bandwidth_mean, spectral_bandwidth_std = self.extract_spectral_bandwidth_stats(y, sr)

        voice_activity_series = self.build_voice_activity_series(energy_series)

        (
            pause_count,
            mean_pause_duration,
            total_pause_duration,
            voiced_ratio,
            silence_ratio,
        ) = self.extract_pause_metrics(
            voice_activity_series=voice_activity_series,
            sr=sr,
        )

        return FeatureSet(
            mfcc_means=mfcc_means,
            pitch_mean=pitch_mean,
            pitch_std=pitch_std,
            energy_mean=energy_mean,
            energy_std=energy_std,
            zcr_mean=zcr_mean,
            zcr_std=zcr_std,
            spectral_centroid_mean=spectral_centroid_mean,
            spectral_centroid_std=spectral_centroid_std,
            spectral_bandwidth_mean=spectral_bandwidth_mean,
            spectral_bandwidth_std=spectral_bandwidth_std,
            voiced_ratio=voiced_ratio,
            silence_ratio=silence_ratio,
            pause_count=pause_count,
            mean_pause_duration=mean_pause_duration,
            total_pause_duration=total_pause_duration,
            pitch_series=pitch_series,
            energy_series=energy_series,
            voice_activity_series=voice_activity_series,
        )