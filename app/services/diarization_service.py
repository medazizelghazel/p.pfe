import os
import csv
import torch
import soundfile as sf
import numpy as np
from dotenv import load_dotenv
from pyannote.audio import Pipeline


# ── Thresholds for real-speaker detection ────────────────────────────────────
MIN_TOTAL_DURATION   = 10.0  # seconds — speaker must talk at least this much
MIN_SEGMENT_COUNT    = 5     # speaker must have at least this many valid segments
MIN_SEGMENT_DURATION = 0.5   # seconds — segments shorter than this are micro-segments


class DiarizationService:

    def __init__(self):
        load_dotenv()
        self.hf_token = os.getenv("HUGGINGFACE_TOKEN")
        if not self.hf_token:
            raise ValueError("HUGGINGFACE_TOKEN not found in .env file")

        # ── Auto-detect GPU ───────────────────────────────────────────────────
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[DiarizationService] Using device: {self.device}")

        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=self.hf_token
        )
        self.pipeline.to(self.device)

    # ── Main entry point ──────────────────────────────────────────────────────

    def diarize_audio(
        self,
        audio_path: str,
        output_dir: str = "data/results/diarization",
        min_speakers: int = 2,
        max_speakers: int = 10
    ) -> dict:
        """
        Run speaker diarization on an audio file.

        Returns a dict with:
            - audio_path, output_rttm, output_csv
            - segments_count (all segments)
            - real_speakers (list of speaker IDs considered real)
            - artifact_speakers (list of speaker IDs considered noise/artifact)
            - speaker_stats (dict with per-speaker statistics)
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        os.makedirs(output_dir, exist_ok=True)

        base_name   = os.path.splitext(os.path.basename(audio_path))[0]
        output_rttm = os.path.join(output_dir, f"{base_name}.rttm")
        output_csv  = os.path.join(output_dir, f"{base_name}_segments.csv")

        # ── Load audio ────────────────────────────────────────────────────────
        data, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
        waveform = torch.from_numpy(data.T)  # shape: (channels, time)
        audio_input = {"waveform": waveform, "sample_rate": sample_rate}

        # ── Run pyannote pipeline ─────────────────────────────────────────────
        output = self.pipeline(
            audio_input,
            min_speakers=min_speakers,
            max_speakers=max_speakers
        )

        # speaker-diarization-3.1        → DiarizeOutput with .speaker_diarization
        # speaker-diarization-community-1 → Annotation directly
        if hasattr(output, "speaker_diarization"):
            diarization = output.speaker_diarization
        else:
            diarization = output

        # ── Extract raw segments ──────────────────────────────────────────────
        raw_segments = []
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            start    = round(segment.start, 2)
            end      = round(segment.end,   2)
            duration = round(segment.end - segment.start, 2)
            raw_segments.append({
                "start":    start,
                "end":      end,
                "speaker":  speaker,
                "duration": duration,
            })

        # ── Compute per-speaker statistics ────────────────────────────────────
        speaker_stats = self._compute_speaker_stats(raw_segments)

        # ── Classify real speakers vs artifacts ───────────────────────────────
        real_speakers, artifact_speakers = self._classify_speakers(speaker_stats)

        # ── Detect overlaps ───────────────────────────────────────────────────
        segments_with_flags = self._flag_overlaps(raw_segments)

        # ── Print summary ─────────────────────────────────────────────────────
        self._print_summary(
            segments_with_flags, speaker_stats, real_speakers, artifact_speakers
        )

        # ── Save RTTM ─────────────────────────────────────────────────────────
        with open(output_rttm, "w", encoding="utf-8") as f:
            diarization.write_rttm(f)

        # ── Save enriched CSV ─────────────────────────────────────────────────
        fieldnames = [
            "start", "end", "speaker", "duration",
            "is_micro_segment", "is_real_speaker", "has_overlap"
        ]
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for seg in segments_with_flags:
                writer.writerow({
                    "start":            seg["start"],
                    "end":              seg["end"],
                    "speaker":          seg["speaker"],
                    "duration":         seg["duration"],
                    "is_micro_segment": seg["duration"] < MIN_SEGMENT_DURATION,
                    "is_real_speaker":  seg["speaker"] in real_speakers,
                    "has_overlap":      seg["has_overlap"],
                })

        return {
            "audio_path":        audio_path,
            "output_rttm":       output_rttm,
            "output_csv":        output_csv,
            "segments_count":    len(raw_segments),
            "real_speakers":     real_speakers,
            "artifact_speakers": artifact_speakers,
            "speaker_stats":     speaker_stats,
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _compute_speaker_stats(self, segments: list) -> dict:
        stats = {}
        for seg in segments:
            spk = seg["speaker"]
            if spk not in stats:
                stats[spk] = {
                    "total_duration": 0.0,
                    "segment_count":  0,
                    "micro_count":    0,
                    "durations":      [],
                }
            stats[spk]["total_duration"] += seg["duration"]
            if seg["duration"] >= MIN_SEGMENT_DURATION:
                stats[spk]["segment_count"] += 1
                stats[spk]["durations"].append(seg["duration"])
            else:
                stats[spk]["micro_count"] += 1

        for spk, s in stats.items():
            durations = s.pop("durations")
            s["avg_duration"]   = round(float(np.mean(durations)) if durations else 0.0, 2)
            s["total_duration"] = round(s["total_duration"], 2)

        return stats

    def _classify_speakers(self, speaker_stats: dict) -> tuple:
        real, artifacts = [], []
        for spk, s in speaker_stats.items():
            if (s["total_duration"] >= MIN_TOTAL_DURATION and
                    s["segment_count"] >= MIN_SEGMENT_COUNT):
                real.append(spk)
            else:
                artifacts.append(spk)
        return sorted(real), sorted(artifacts)

    def _flag_overlaps(self, segments: list) -> list:
        result = [dict(seg, has_overlap=False) for seg in segments]
        n = len(result)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = result[i], result[j]
                if b["start"] >= a["end"]:
                    break
                if a["start"] < b["end"] and b["start"] < a["end"]:
                    result[i]["has_overlap"] = True
                    result[j]["has_overlap"] = True
        return result

    def _print_summary(
        self,
        segments: list,
        speaker_stats: dict,
        real_speakers: list,
        artifact_speakers: list
    ) -> None:
        print("\n" + "=" * 60)
        print("DIARIZATION RESULT")
        print("=" * 60)
        print(f"{'Speaker':<14} {'Total(s)':>8} {'Segments':>9} {'Micro':>6} "
              f"{'Avg(s)':>7}  Status")
        print("-" * 60)
        for spk in sorted(speaker_stats):
            s      = speaker_stats[spk]
            status = "REAL SPEAKER" if spk in real_speakers else "artifact / noise"
            print(
                f"{spk:<14} {s['total_duration']:>8.1f} {s['segment_count']:>9} "
                f"{s['micro_count']:>6} {s['avg_duration']:>7.2f}  {status}"
            )
        print("=" * 60)
        overlap_count = sum(1 for s in segments if s["has_overlap"])
        micro_count   = sum(1 for s in segments if s["duration"] < MIN_SEGMENT_DURATION)
        print(f"Total segments   : {len(segments)}")
        print(f"Micro-segments   : {micro_count}  (< {MIN_SEGMENT_DURATION}s)")
        print(f"Overlapping segs : {overlap_count}")
        print(f"Real speakers    : {real_speakers}")
        print(f"Artifacts        : {artifact_speakers}")
        print("=" * 60 + "\n")