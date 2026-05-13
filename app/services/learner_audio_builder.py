from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
import soundfile as sf


class LearnerAudioBuilder:
    """
    Builds one global learners-only audio file using the clean diarization CSV.

    It uses all segments where role == 'learner'.
    The output represents the global voice/emotion of all learners together.
    """

    def __init__(
        self,
        min_segment_duration: float = 0.5,
        insert_silence_sec: float = 0.15,
    ):
        self.min_segment_duration = min_segment_duration
        self.insert_silence_sec = insert_silence_sec

    def build(
        self,
        audio_path: str,
        clean_csv_path: str,
        output_dir: str = "data/results/diarization",
        analysis_id: str | None = None,
    ) -> dict:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if not os.path.exists(clean_csv_path):
            raise FileNotFoundError(f"Clean diarization CSV not found: {clean_csv_path}")

        os.makedirs(output_dir, exist_ok=True)

        learner_segments = self._load_learner_segments(clean_csv_path)

        if not learner_segments:
            return {
                "enabled": False,
                "reason": "No learner segments found in diarization CSV.",
                "audio_path": None,
                "segments_csv_path": None,
                "segment_count": 0,
                "duration": 0.0,
                "speaker_count": 0,
                "speakers": [],
            }

        waveform, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
        waveform = waveform.mean(axis=1).astype(np.float32)

        audio_chunks = []
        kept_segments = []

        silence_chunk = None
        if self.insert_silence_sec > 0:
            silence_chunk = np.zeros(
                int(self.insert_silence_sec * sample_rate),
                dtype=np.float32,
            )

        for seg in learner_segments:
            start_idx = max(0, int(seg["start"] * sample_rate))
            end_idx = min(len(waveform), int(seg["end"] * sample_rate))

            if end_idx <= start_idx:
                continue

            chunk = waveform[start_idx:end_idx]
            duration = len(chunk) / sample_rate if sample_rate > 0 else 0.0

            if duration < self.min_segment_duration:
                continue

            audio_chunks.append(chunk)

            if silence_chunk is not None:
                audio_chunks.append(silence_chunk.copy())

            kept_segments.append({
                "start": seg["start"],
                "end": seg["end"],
                "speaker": seg["speaker"],
                "role": "learner",
                "duration": round(duration, 2),
                "has_overlap": seg.get("has_overlap", False),
            })

        if not kept_segments:
            return {
                "enabled": False,
                "reason": "Learner segments exist but no valid audio remained after filtering.",
                "audio_path": None,
                "segments_csv_path": None,
                "segment_count": 0,
                "duration": 0.0,
                "speaker_count": 0,
                "speakers": [],
            }

        if silence_chunk is not None and len(audio_chunks) > 0:
            if np.array_equal(audio_chunks[-1], silence_chunk):
                audio_chunks.pop()

        learners_waveform = np.concatenate(audio_chunks).astype(np.float32)

        base_name = analysis_id if analysis_id else Path(audio_path).stem

        output_audio_path = os.path.join(
            output_dir,
            f"{base_name}_learners_only.wav",
        )

        output_segments_csv = os.path.join(
            output_dir,
            f"{base_name}_learners_segments.csv",
        )

        sf.write(output_audio_path, learners_waveform, sample_rate)
        self._save_segments_csv(output_segments_csv, kept_segments)

        speakers = sorted({seg["speaker"] for seg in kept_segments})
        duration = len(learners_waveform) / sample_rate if sample_rate > 0 else 0.0

        return {
            "enabled": True,
            "audio_path": output_audio_path,
            "segments_csv_path": output_segments_csv,
            "segment_count": len(kept_segments),
            "duration": round(duration, 2),
            "speaker_count": len(speakers),
            "speakers": speakers,
        }

    def _load_learner_segments(self, clean_csv_path: str) -> list[dict]:
        segments = []

        with open(clean_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                role = str(row.get("role", "")).strip().lower()

                if role != "learner":
                    continue

                start = float(row["start"])
                end = float(row["end"])
                duration = float(row["duration"])

                if end <= start or duration < self.min_segment_duration:
                    continue

                segments.append({
                    "start": start,
                    "end": end,
                    "speaker": row.get("speaker", "unknown"),
                    "role": role,
                    "duration": duration,
                    "has_overlap": row.get("has_overlap", "False"),
                })

        return sorted(segments, key=lambda s: s["start"])

    def _save_segments_csv(self, path: str, segments: list[dict]) -> None:
        fieldnames = [
            "start",
            "end",
            "speaker",
            "role",
            "duration",
            "has_overlap",
        ]

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for seg in segments:
                writer.writerow(seg)