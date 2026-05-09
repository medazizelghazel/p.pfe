import os
import json
import whisper
import soundfile as sf
import numpy as np
from dataclasses import dataclass, field, asdict


@dataclass
class TranscribedSegment:
    speaker:    str
    start:      float
    end:        float
    duration:   float
    text:       str
    language:   str


class TranscriptionService:
    """
    Transcribes only the trainer's audio segments using Whisper.
    Accepts the clean CSV from DiarizationPostProcessor as input.
    Supports French, English, and Arabic automatically.
    """

    SUPPORTED_LANGUAGES = ["fr", "en", "ar"]

    def __init__(self, model_size: str = "small"):
        print(f"[TranscriptionService] Loading Whisper model '{model_size}'...")
        self.model      = whisper.load_model(model_size)
        self.model_size = model_size
        print(f"[TranscriptionService] Model ready.")

    # ── Main entry point ──────────────────────────────────────────────────────

    def transcribe_trainer(
        self,
        audio_path:   str,
        clean_csv:    str,
        output_dir:   str = "data/results/transcription",
        trainer_role: str = "trainer",
    ) -> dict:
        """
        Transcribe only the trainer's segments from the clean diarization CSV.

        Args:
            audio_path   : original audio file (.wav)
            clean_csv    : path to _clean.csv from DiarizationPostProcessor
            output_dir   : where to save transcription outputs
            trainer_role : role label for the trainer (default: "trainer")

        Returns:
            {
                trainer_id        : speaker ID of the trainer,
                segments          : list of TranscribedSegment,
                full_text         : concatenated trainer transcript,
                output_json       : path to saved JSON,
                output_txt        : path to saved plain text,
                total_duration    : total trainer speaking time,
                segments_count    : number of segments transcribed,
            }
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio not found: {audio_path}")
        if not os.path.exists(clean_csv):
            raise FileNotFoundError(f"Clean CSV not found: {clean_csv}")

        os.makedirs(output_dir, exist_ok=True)

        # ── Load audio ────────────────────────────────────────────────────────
        audio_data, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
        audio_mono = audio_data.mean(axis=1)  # stereo → mono if needed

        # ── Load trainer segments from CSV ────────────────────────────────────
        trainer_segments = self._load_trainer_segments(clean_csv, trainer_role)

        if not trainer_segments:
            raise ValueError(
                f"No segments found with role='{trainer_role}' in {clean_csv}. "
                "Run DiarizationPostProcessor first."
            )

        trainer_id = trainer_segments[0]["speaker"]
        print(f"\n[TranscriptionService] Trainer: {trainer_id}")
        print(f"[TranscriptionService] Segments to transcribe: {len(trainer_segments)}")

        # ── Transcribe each segment ───────────────────────────────────────────
        transcribed = []
        for i, seg in enumerate(trainer_segments):
            print(f"  [{i+1}/{len(trainer_segments)}] {seg['start']:.1f}s → {seg['end']:.1f}s", end=" ")

            text, lang = self._transcribe_segment(
                audio_mono, sample_rate, seg["start"], seg["end"]
            )

            if text.strip():
                transcribed.append(TranscribedSegment(
                    speaker=  seg["speaker"],
                    start=    seg["start"],
                    end=      seg["end"],
                    duration= seg["duration"],
                    text=     text.strip(),
                    language= lang,
                ))
                print(f"[{lang}] {text[:60]}{'...' if len(text) > 60 else ''}")
            else:
                print("(empty — skipped)")

        # ── Build full text ───────────────────────────────────────────────────
        full_text = self._build_full_text(transcribed)

        # ── Save outputs ──────────────────────────────────────────────────────
        base_name    = os.path.splitext(os.path.basename(audio_path))[0]
        output_json  = os.path.join(output_dir, f"{base_name}_transcript.json")
        output_txt   = os.path.join(output_dir, f"{base_name}_transcript.txt")

        self._save_json(output_json, transcribed, trainer_id, full_text)
        self._save_txt(output_txt, transcribed, trainer_id)

        # ── Print summary ─────────────────────────────────────────────────────
        total_duration = sum(s.duration for s in transcribed)
        langs = set(s.language for s in transcribed)
        print(f"\n── Transcription summary ────────────────────────────────────")
        print(f"  Trainer          : {trainer_id}")
        print(f"  Segments         : {len(transcribed)}")
        print(f"  Total duration   : {total_duration:.1f}s ({total_duration/60:.1f} min)")
        print(f"  Languages found  : {langs}")
        print(f"  Total words      : {len(full_text.split())}")
        print(f"  JSON saved       : {output_json}")
        print(f"  TXT saved        : {output_txt}")
        print(f"────────────────────────────────────────────────────────────\n")

        return {
            "trainer_id":       trainer_id,
            "segments":         transcribed,
            "full_text":        full_text,
            "output_json":      output_json,
            "output_txt":       output_txt,
            "total_duration":   round(total_duration, 2),
            "segments_count":   len(transcribed),
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _load_trainer_segments(self, csv_path: str, trainer_role: str) -> list:
        """Load only trainer rows from the clean CSV."""
        import csv
        segments = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("role", "").strip().lower() == trainer_role.lower():
                    segments.append({
                        "speaker":  row["speaker"],
                        "start":    float(row["start"]),
                        "end":      float(row["end"]),
                        "duration": float(row["duration"]),
                    })
        # Sort by start time
        return sorted(segments, key=lambda s: s["start"])

    def _transcribe_segment(
        self,
        audio:       np.ndarray,
        sample_rate: int,
        start:       float,
        end:         float,
    ) -> tuple:
        """
        Extract audio slice and transcribe with Whisper.
        Returns (text, detected_language).
        Language detection is automatic — no need to specify.
        Whisper handles fr/en/ar natively with the small model.
        """
        start_sample = int(start * sample_rate)
        end_sample   = int(end   * sample_rate)
        segment_audio = audio[start_sample:end_sample]

        # Whisper requires float32 at 16kHz
        if sample_rate != 16000:
            import librosa
            segment_audio = librosa.resample(
                segment_audio, orig_sr=sample_rate, target_sr=16000
            )

        result = self.model.transcribe(
            segment_audio,
            fp16=False,          # CPU-safe
            language=None,       # auto-detect fr/en/ar
            task="transcribe",   # not translate — keep original language
            verbose=False,
        )

        text     = result.get("text", "").strip()
        language = result.get("language", "unknown")

        return text, language

    def _build_full_text(self, segments: list) -> str:
        """
        Concatenate all transcribed segments into a readable full text,
        grouped by proximity (segments close in time get merged into paragraphs).
        """
        if not segments:
            return ""

        paragraphs = []
        current    = [segments[0].text]
        prev_end   = segments[0].end

        for seg in segments[1:]:
            gap = seg.start - prev_end
            if gap > 5.0:
                # Long pause → new paragraph
                paragraphs.append(" ".join(current))
                current = [seg.text]
            else:
                current.append(seg.text)
            prev_end = seg.end

        if current:
            paragraphs.append(" ".join(current))

        return "\n\n".join(paragraphs)

    def _save_json(
        self,
        path:       str,
        segments:   list,
        trainer_id: str,
        full_text:  str,
    ) -> None:
        data = {
            "trainer_id": trainer_id,
            "full_text":  full_text,
            "segments": [asdict(s) for s in segments],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _save_txt(self, path: str, segments: list, trainer_id: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"TRANSCRIPT — Trainer: {trainer_id}\n")
            f.write("=" * 60 + "\n\n")
            for seg in segments:
                timestamp = f"[{seg.start:.1f}s → {seg.end:.1f}s]"
                f.write(f"{timestamp} ({seg.language})\n")
                f.write(f"{seg.text}\n\n")