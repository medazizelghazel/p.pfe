from __future__ import annotations

import os
import csv
import json
from dataclasses import dataclass, asdict

import librosa
import numpy as np
import soundfile as sf
import whisper


@dataclass
class TranscribedSegment:
    speaker: str
    start: float
    end: float
    duration: float
    text: str
    language: str


class TranscriptionService:
    """
    Transcribes only the trainer's audio segments using Whisper.

    Language behavior:
    - Detects the main course language once from trainer audio.
    - Supports only: English, French, Arabic.
    - Then uses the detected language for all segments.
    - This avoids wrong language hallucinations on very short segments.
    """

    SUPPORTED_LANGUAGES = ["en", "fr", "ar"]

    def __init__(
        self,
        model_size: str = "small",
        language: str | None = None,
        min_segment_duration: float = 1.2,
        max_language_detection_audio_sec: float = 45.0,
    ):
        """
        Args:
            model_size: Whisper model size. Use "small" or "medium".
            language:
                None  -> auto-detect course language among en/fr/ar
                "en"  -> force English
                "fr"  -> force French
                "ar"  -> force Arabic
            min_segment_duration:
                Ignore very short segments to reduce hallucinations.
            max_language_detection_audio_sec:
                Maximum trainer audio used for language detection.
        """
        if language is not None and language not in self.SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language '{language}'. "
                f"Use one of {self.SUPPORTED_LANGUAGES} or None for auto-detection."
            )

        self.model_size = model_size
        self.forced_language = language
        self.min_segment_duration = min_segment_duration
        self.max_language_detection_audio_sec = max_language_detection_audio_sec

        print(f"[TranscriptionService] Loading Whisper model '{model_size}'...")
        self.model = whisper.load_model(model_size)
        print("[TranscriptionService] Model ready.")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def transcribe_trainer(
        self,
        audio_path: str,
        clean_csv: str,
        output_dir: str = "data/results/transcription",
        trainer_role: str = "trainer",
    ) -> dict:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio not found: {audio_path}")

        if not os.path.exists(clean_csv):
            raise FileNotFoundError(f"Clean CSV not found: {clean_csv}")

        os.makedirs(output_dir, exist_ok=True)

        audio_data, sample_rate = sf.read(
            audio_path,
            dtype="float32",
            always_2d=True,
        )
        audio_mono = audio_data.mean(axis=1).astype(np.float32)

        trainer_segments = self._load_trainer_segments(
            csv_path=clean_csv,
            trainer_role=trainer_role,
        )

        if not trainer_segments:
            raise ValueError(
                f"No valid trainer segments found with role='{trainer_role}' in {clean_csv}. "
                f"Segments shorter than {self.min_segment_duration}s are ignored."
            )

        trainer_id = trainer_segments[0]["speaker"]

        print(f"\n[TranscriptionService] Trainer: {trainer_id}")
        print(f"[TranscriptionService] Segments to transcribe: {len(trainer_segments)}")
        print(f"[TranscriptionService] Min segment duration: {self.min_segment_duration}s")

        course_language, language_confidence, language_probs = self._detect_course_language(
            audio=audio_mono,
            sample_rate=sample_rate,
            trainer_segments=trainer_segments,
        )

        print(
            "[TranscriptionService] Course language: "
            f"{course_language} "
            f"(confidence={language_confidence:.4f})"
        )
        print(f"[TranscriptionService] Language probabilities: {language_probs}")

        transcribed: list[TranscribedSegment] = []

        for i, seg in enumerate(trainer_segments):
            print(
                f"  [{i + 1}/{len(trainer_segments)}] "
                f"{seg['start']:.1f}s → {seg['end']:.1f}s",
                end=" ",
            )

            text, lang = self._transcribe_segment(
                audio=audio_mono,
                sample_rate=sample_rate,
                start=seg["start"],
                end=seg["end"],
                language=course_language,
            )

            if text.strip():
                transcribed.append(
                    TranscribedSegment(
                        speaker=seg["speaker"],
                        start=seg["start"],
                        end=seg["end"],
                        duration=seg["duration"],
                        text=text.strip(),
                        language=lang,
                    )
                )
                print(f"[{lang}] {text[:60]}{'...' if len(text) > 60 else ''}")
            else:
                print("(empty — skipped)")

        full_text = self._build_full_text(transcribed)

        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        output_json = os.path.join(output_dir, f"{base_name}_transcript.json")
        output_txt = os.path.join(output_dir, f"{base_name}_transcript.txt")

        self._save_json(
            path=output_json,
            segments=transcribed,
            trainer_id=trainer_id,
            full_text=full_text,
            course_language=course_language,
            language_confidence=language_confidence,
            language_probs=language_probs,
        )

        self._save_txt(
            path=output_txt,
            segments=transcribed,
            trainer_id=trainer_id,
            course_language=course_language,
            language_confidence=language_confidence,
        )

        total_duration = sum(s.duration for s in transcribed)
        langs = set(s.language for s in transcribed)

        print("\n── Transcription summary ────────────────────────────────────")
        print(f"  Trainer          : {trainer_id}")
        print(f"  Course language  : {course_language}")
        print(f"  Lang confidence  : {language_confidence:.4f}")
        print(f"  Segments         : {len(transcribed)}")
        print(f"  Total duration   : {total_duration:.1f}s ({total_duration / 60:.1f} min)")
        print(f"  Languages found  : {langs}")
        print(f"  Total words      : {len(full_text.split())}")
        print(f"  JSON saved       : {output_json}")
        print(f"  TXT saved        : {output_txt}")
        print("────────────────────────────────────────────────────────────\n")

        return {
            "trainer_id": trainer_id,
            "course_language": course_language,
            "language_confidence": round(language_confidence, 4),
            "language_probs": language_probs,
            "segments": transcribed,
            "full_text": full_text,
            "output_json": output_json,
            "output_txt": output_txt,
            "total_duration": round(total_duration, 2),
            "segments_count": len(transcribed),
        }

    # ------------------------------------------------------------------
    # Segment loading
    # ------------------------------------------------------------------

    def _load_trainer_segments(
        self,
        csv_path: str,
        trainer_role: str,
    ) -> list[dict]:
        segments = []

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                role = row.get("role", "").strip().lower()

                if role != trainer_role.lower():
                    continue

                duration = float(row["duration"])

                if duration < self.min_segment_duration:
                    continue

                segments.append(
                    {
                        "speaker": row["speaker"],
                        "start": float(row["start"]),
                        "end": float(row["end"]),
                        "duration": duration,
                    }
                )

        return sorted(segments, key=lambda s: s["start"])

    # ------------------------------------------------------------------
    # Language detection
    # ------------------------------------------------------------------

    def _detect_course_language(
        self,
        audio: np.ndarray,
        sample_rate: int,
        trainer_segments: list[dict],
    ) -> tuple[str, float, dict[str, float]]:
        if self.forced_language is not None:
            return (
                self.forced_language,
                1.0,
                {lang: 1.0 if lang == self.forced_language else 0.0 for lang in self.SUPPORTED_LANGUAGES},
            )

        detection_audio = self._build_language_detection_audio(
            audio=audio,
            sample_rate=sample_rate,
            trainer_segments=trainer_segments,
        )

        if sample_rate != 16000:
            detection_audio = librosa.resample(
                detection_audio,
                orig_sr=sample_rate,
                target_sr=16000,
            )

        detection_audio = detection_audio.astype(np.float32)

        if len(detection_audio) == 0:
            return "en", 0.0, {"en": 1.0, "fr": 0.0, "ar": 0.0}

        detection_audio = whisper.pad_or_trim(detection_audio)

        mel = whisper.log_mel_spectrogram(detection_audio).to(self.model.device)

        _, probs = self.model.detect_language(mel)

        supported_probs = {
            lang: float(probs.get(lang, 0.0))
            for lang in self.SUPPORTED_LANGUAGES
        }

        detected_language = max(
            supported_probs,
            key=supported_probs.get,
        )

        confidence = supported_probs[detected_language]

        return detected_language, confidence, supported_probs

    def _build_language_detection_audio(
        self,
        audio: np.ndarray,
        sample_rate: int,
        trainer_segments: list[dict],
    ) -> np.ndarray:
        chunks = []
        collected_sec = 0.0

        # Prefer longer segments because they are more reliable for language detection.
        sorted_segments = sorted(
            trainer_segments,
            key=lambda s: s["duration"],
            reverse=True,
        )

        for seg in sorted_segments:
            if collected_sec >= self.max_language_detection_audio_sec:
                break

            start_sample = int(seg["start"] * sample_rate)
            end_sample = int(seg["end"] * sample_rate)

            chunk = audio[start_sample:end_sample]

            if len(chunk) == 0:
                continue

            chunks.append(chunk.astype(np.float32))
            collected_sec += len(chunk) / sample_rate

        if not chunks:
            return np.array([], dtype=np.float32)

        silence = np.zeros(int(0.2 * sample_rate), dtype=np.float32)
        joined = []

        for chunk in chunks:
            joined.append(chunk)
            joined.append(silence)

        return np.concatenate(joined).astype(np.float32)

    # ------------------------------------------------------------------
    # Whisper transcription
    # ------------------------------------------------------------------

    def _transcribe_segment(
        self,
        audio: np.ndarray,
        sample_rate: int,
        start: float,
        end: float,
        language: str,
    ) -> tuple[str, str]:
        start_sample = int(start * sample_rate)
        end_sample = int(end * sample_rate)

        segment_audio = audio[start_sample:end_sample].astype(np.float32)

        if len(segment_audio) == 0:
            return "", language

        if sample_rate != 16000:
            segment_audio = librosa.resample(
                segment_audio,
                orig_sr=sample_rate,
                target_sr=16000,
            ).astype(np.float32)

        result = self.model.transcribe(
            segment_audio,
            fp16=False,
            language=language,
            task="transcribe",
            verbose=False,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            logprob_threshold=-1.0,
            compression_ratio_threshold=2.4,
        )

        text = result.get("text", "").strip()

        return text, language

    # ------------------------------------------------------------------
    # Text building and saving
    # ------------------------------------------------------------------

    def _build_full_text(self, segments: list[TranscribedSegment]) -> str:
        if not segments:
            return ""

        paragraphs = []
        current = [segments[0].text]
        prev_end = segments[0].end

        for seg in segments[1:]:
            gap = seg.start - prev_end

            if gap > 5.0:
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
        path: str,
        segments: list[TranscribedSegment],
        trainer_id: str,
        full_text: str,
        course_language: str,
        language_confidence: float,
        language_probs: dict[str, float],
    ) -> None:
        data = {
            "trainer_id": trainer_id,
            "course_language": course_language,
            "language_confidence": round(language_confidence, 4),
            "language_probs": language_probs,
            "full_text": full_text,
            "segments": [asdict(s) for s in segments],
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _save_txt(
        self,
        path: str,
        segments: list[TranscribedSegment],
        trainer_id: str,
        course_language: str,
        language_confidence: float,
    ) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"TRANSCRIPT — Trainer: {trainer_id}\n")
            f.write(f"COURSE LANGUAGE: {course_language} ({language_confidence:.4f})\n")
            f.write("=" * 60 + "\n\n")

            for seg in segments:
                timestamp = f"[{seg.start:.1f}s → {seg.end:.1f}s]"
                f.write(f"{timestamp} ({seg.language})\n")
                f.write(f"{seg.text}\n\n")