from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
import soundfile as sf


class TrainerAudioBuilder:
    def __init__(
        self,
        min_segment_duration: float = 0.5,
        min_chunk_duration: float = 2.0,
        max_chunk_duration: float = 12.0,
        insert_silence_sec: float = 0.15,
    ):
        self.min_segment_duration = min_segment_duration
        self.min_chunk_duration = min_chunk_duration
        self.max_chunk_duration = max_chunk_duration
        self.insert_silence_sec = insert_silence_sec

    def build(
        self,
        audio_path: str,
        clean_csv_path: str,
        trainer_speaker_id: str | None = None,
        output_dir: str = "data/results/diarization",
        analysis_id: str | None = None,
    ) -> dict:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if not os.path.exists(clean_csv_path):
            raise FileNotFoundError(f"Clean diarization CSV not found: {clean_csv_path}")

        os.makedirs(output_dir, exist_ok=True)

        segments = self._load_clean_segments(clean_csv_path)
        if not segments:
            raise ValueError("No segments found in clean diarization CSV.")

        trainer_id = trainer_speaker_id or self._detect_trainer_id(segments)
        if trainer_id is None:
            raise ValueError("Could not determine trainer speaker ID.")

        trainer_segments = [
            s for s in segments
            if s["speaker"] == trainer_id and s["duration"] >= self.min_segment_duration
        ]

        if not trainer_segments:
            raise ValueError(f"No valid trainer segments found for speaker: {trainer_id}")

        waveform, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
        waveform = waveform.mean(axis=1).astype(np.float32)

        total_audio_duration = len(waveform) / sample_rate if sample_rate > 0 else 0.0

        kept_segments = []
        audio_chunks = []

        silence_chunk = None
        if self.insert_silence_sec > 0:
            silence_chunk = np.zeros(int(self.insert_silence_sec * sample_rate), dtype=np.float32)

        base_name = analysis_id if analysis_id else Path(audio_path).stem
        chunks_dir = os.path.join(output_dir, f"{base_name}_trainer_chunks")
        os.makedirs(chunks_dir, exist_ok=True)

        exported_chunk_paths = []
        exported_chunk_count = 0

        max_chunk_samples = int(self.max_chunk_duration * sample_rate)
        min_chunk_samples = int(self.min_chunk_duration * sample_rate)

        for seg in trainer_segments:
            start_idx = max(0, int(seg["start"] * sample_rate))
            end_idx = min(len(waveform), int(seg["end"] * sample_rate))

            if end_idx <= start_idx:
                continue

            segment_wave = waveform[start_idx:end_idx]
            segment_duration = len(segment_wave) / sample_rate if sample_rate > 0 else 0.0

            if segment_duration < self.min_segment_duration:
                continue

            audio_chunks.append(segment_wave)
            kept_segments.append({
                "start": seg["start"],
                "end": seg["end"],
                "speaker": seg["speaker"],
                "role": seg["role"],
                "duration": round(segment_duration, 2),
                "has_overlap": seg.get("has_overlap", False),
            })

            if silence_chunk is not None:
                audio_chunks.append(silence_chunk.copy())

            # export short chunks for clarity scoring
            seg_len = len(segment_wave)
            cursor = 0
            while cursor < seg_len:
                chunk = segment_wave[cursor: cursor + max_chunk_samples]
                if len(chunk) < min_chunk_samples:
                    break

                exported_chunk_count += 1
                chunk_path = os.path.join(
                    chunks_dir,
                    f"chunk_{exported_chunk_count:04d}.wav"
                )
                sf.write(chunk_path, chunk, sample_rate)
                exported_chunk_paths.append(chunk_path)

                cursor += max_chunk_samples

        if not kept_segments:
            raise ValueError("No trainer audio chunks remained after validation.")

        if silence_chunk is not None and len(audio_chunks) > 0:
            if np.array_equal(audio_chunks[-1], silence_chunk):
                audio_chunks.pop()

        trainer_waveform = np.concatenate(audio_chunks).astype(np.float32)

        output_audio_path = os.path.join(output_dir, f"{base_name}_trainer_only.wav")
        output_segments_csv = os.path.join(output_dir, f"{base_name}_trainer_segments.csv")

        sf.write(output_audio_path, trainer_waveform, sample_rate)
        self._save_segments_csv(output_segments_csv, kept_segments)

        trainer_total_duration = sum(seg["duration"] for seg in kept_segments)
        concatenated_duration = len(trainer_waveform) / sample_rate if sample_rate > 0 else 0.0

        return {
            "trainer_speaker_id": trainer_id,
            "output_audio_path": output_audio_path,
            "output_segments_csv": output_segments_csv,
            "output_chunks_dir": chunks_dir,
            "chunk_count": exported_chunk_count,
            "chunk_paths": exported_chunk_paths,
            "segment_count": len(kept_segments),
            "sample_rate": sample_rate,
            "original_audio_duration": round(total_audio_duration, 2),
            "trainer_total_duration": round(trainer_total_duration, 2),
            "concatenated_duration": round(concatenated_duration, 2),
        }

    def _load_clean_segments(self, csv_path: str) -> list[dict]:
        rows = []

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append({
                    "start": float(row["start"]),
                    "end": float(row["end"]),
                    "speaker": row["speaker"],
                    "role": row.get("role", "unknown"),
                    "duration": float(row["duration"]),
                    "has_overlap": row.get("has_overlap", "False"),
                })

        rows.sort(key=lambda x: x["start"])
        return rows

    def _detect_trainer_id(self, segments: list[dict]) -> str | None:
        trainer_rows = [s for s in segments if s.get("role") == "trainer"]
        if trainer_rows:
            return trainer_rows[0]["speaker"]

        totals = {}
        for seg in segments:
            spk = seg["speaker"]
            totals[spk] = totals.get(spk, 0.0) + seg["duration"]

        if not totals:
            return None

        return max(totals.items(), key=lambda x: x[1])[0]

    def _save_segments_csv(self, path: str, segments: list[dict]) -> None:
        fieldnames = ["start", "end", "speaker", "role", "duration", "has_overlap"]

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for seg in segments:
                writer.writerow(seg)