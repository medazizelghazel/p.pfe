import csv
import os
import json
import math
from dataclasses import dataclass, field, asdict
from typing import Optional


MIN_SEGMENT_DURATION = 0.5
MERGE_GAP_THRESHOLD = 0.3

WEIGHT_TOTAL_DURATION = 0.40
WEIGHT_AVG_DURATION = 0.30
WEIGHT_COVERAGE = 0.20
WEIGHT_CONSISTENCY = 0.10


@dataclass
class SpeakerProfile:
    speaker_id: str
    role: str = "unknown"
    total_duration: float = 0.0
    segment_count: int = 0
    micro_count: int = 0
    avg_duration: float = 0.0
    coverage: float = 0.0
    consistency: float = 0.0
    trainer_score: float = 0.0
    confidence: str = "low"
    segments: list = field(default_factory=list)


class DiarizationPostProcessor:
    def __init__(self, session_window_seconds: int = 60):
        self.session_window_seconds = session_window_seconds

    def process(
        self,
        csv_path: str,
        output_dir: Optional[str] = None,
        real_speakers: Optional[list] = None,
    ) -> dict:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        if output_dir is None:
            output_dir = os.path.dirname(csv_path)
        os.makedirs(output_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        clean_csv = os.path.join(output_dir, f"{base_name}_clean.csv")
        profiles_json = os.path.join(output_dir, f"{base_name}_profiles.json")

        raw_segments = self._load_csv(csv_path)
        total_raw = len(raw_segments)

        if real_speakers:
            segments = [s for s in raw_segments if s["speaker"] in real_speakers]
        else:
            segments = [
                s for s in raw_segments
                if s.get("is_real_speaker", "True") in (True, "True", "true", "1")
            ]

        removed_artifacts = total_raw - len(segments)

        speaker_micro_counts = {}
        for s in segments:
            if s["duration"] < MIN_SEGMENT_DURATION:
                speaker_micro_counts[s["speaker"]] = (
                    speaker_micro_counts.get(s["speaker"], 0) + 1
                )

        segments_valid = [s for s in segments if s["duration"] >= MIN_SEGMENT_DURATION]
        removed_micro = len(segments) - len(segments_valid)

        segments_merged = self._merge_segments(segments_valid)
        merged_count = len(segments_valid) - len(segments_merged)

        session_duration = max((s["end"] for s in segments_merged), default=0.0)

        profiles = self._build_profiles(
            segments=segments_merged,
            session_duration=session_duration,
            speaker_micro_counts=speaker_micro_counts,
        )

        trainer_id, trainer_score, confidence = self._identify_trainer(profiles)

        for spk_id, profile in profiles.items():
            if spk_id == trainer_id:
                profile.role = "trainer"
            else:
                profile.role = "learner"

        self._save_clean_csv(clean_csv, segments_merged, profiles)
        self._save_profiles_json(profiles_json, profiles)

        self._print_report(
            profiles,
            trainer_id,
            trainer_score,
            confidence,
            total_raw,
            len(segments_merged),
            removed_micro,
            removed_artifacts,
            merged_count,
            session_duration,
        )

        return {
            "trainer": trainer_id,
            "trainer_score": round(trainer_score, 4),
            "confidence": confidence,
            "profiles": profiles,
            "clean_csv": clean_csv,
            "profiles_json": profiles_json,
            "stats": {
                "total_segments": total_raw,
                "cleaned_segments": len(segments_merged),
                "removed_micro": removed_micro,
                "removed_artifacts": removed_artifacts,
                "merged_segments": merged_count,
                "session_duration": round(session_duration, 2),
            },
        }

    def _load_csv(self, csv_path: str) -> list:
        rows = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(
                    {
                        "start": float(row["start"]),
                        "end": float(row["end"]),
                        "speaker": row["speaker"],
                        "duration": float(row["duration"]),
                        "is_micro_segment": row.get("is_micro_segment", "False"),
                        "is_real_speaker": row.get("is_real_speaker", "True"),
                        "has_overlap": row.get("has_overlap", "False"),
                    }
                )
        return rows

    def _merge_segments(self, segments: list) -> list:
        if not segments:
            return []

        sorted_segs = sorted(segments, key=lambda s: s["start"])
        merged = [dict(sorted_segs[0])]

        for seg in sorted_segs[1:]:
            last = merged[-1]
            gap = seg["start"] - last["end"]

            if seg["speaker"] == last["speaker"] and gap <= MERGE_GAP_THRESHOLD:
                last["end"] = seg["end"]
                last["duration"] = round(last["end"] - last["start"], 2)
            else:
                merged.append(dict(seg))

        return merged

    def _build_profiles(
        self,
        segments: list,
        session_duration: float,
        speaker_micro_counts: Optional[dict] = None,
    ) -> dict:
        profiles = {}
        speaker_micro_counts = speaker_micro_counts or {}

        for seg in segments:
            spk = seg["speaker"]
            if spk not in profiles:
                profiles[spk] = SpeakerProfile(speaker_id=spk)

            p = profiles[spk]
            p.total_duration = round(p.total_duration + seg["duration"], 2)
            p.segment_count += 1
            p.segments.append(seg)

        num_windows = max(1, math.ceil(session_duration / self.session_window_seconds))

        for spk, p in profiles.items():
            p.micro_count = speaker_micro_counts.get(spk, 0)

            p.avg_duration = (
                round(p.total_duration / p.segment_count, 2)
                if p.segment_count > 0
                else 0.0
            )

            windows_present = set()
            for seg in p.segments:
                w_start = int(seg["start"] / self.session_window_seconds)
                w_end = int(seg["end"] / self.session_window_seconds)
                for w in range(w_start, w_end + 1):
                    windows_present.add(w)

            p.coverage = round(len(windows_present) / num_windows, 4)

            total_segs = p.segment_count + p.micro_count
            p.consistency = (
                round(1 - (p.micro_count / total_segs), 4)
                if total_segs > 0
                else 1.0
            )

        return profiles

    def _identify_trainer(self, profiles: dict) -> tuple:
        if not profiles:
            return None, 0.0, "low"

        max_total = max(p.total_duration for p in profiles.values()) or 1
        max_avg = max(p.avg_duration for p in profiles.values()) or 1
        max_coverage = max(p.coverage for p in profiles.values()) or 1
        max_consist = max(p.consistency for p in profiles.values()) or 1

        for _, p in profiles.items():
            norm_total = p.total_duration / max_total
            norm_avg = p.avg_duration / max_avg
            norm_coverage = p.coverage / max_coverage
            norm_consist = p.consistency / max_consist

            p.trainer_score = round(
                WEIGHT_TOTAL_DURATION * norm_total
                + WEIGHT_AVG_DURATION * norm_avg
                + WEIGHT_COVERAGE * norm_coverage
                + WEIGHT_CONSISTENCY * norm_consist,
                4,
            )

        ranked = sorted(profiles.values(), key=lambda p: p.trainer_score, reverse=True)

        winner = ranked[0]
        runner_up = ranked[1] if len(ranked) > 1 else None
        gap = (winner.trainer_score - runner_up.trainer_score) if runner_up else 1.0

        if winner.trainer_score >= 0.70 and gap >= 0.15:
            confidence = "high"
        elif winner.trainer_score >= 0.50 and gap >= 0.10:
            confidence = "medium"
        else:
            confidence = "low"

        winner.confidence = confidence
        return winner.speaker_id, winner.trainer_score, confidence

    def _save_clean_csv(self, path: str, segments: list, profiles: dict) -> None:
        fieldnames = ["start", "end", "speaker", "role", "duration", "has_overlap"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for seg in segments:
                spk = seg["speaker"]
                role = profiles[spk].role if spk in profiles else "unknown"
                writer.writerow(
                    {
                        "start": seg["start"],
                        "end": seg["end"],
                        "speaker": spk,
                        "role": role,
                        "duration": seg["duration"],
                        "has_overlap": seg.get("has_overlap", False),
                    }
                )

    def _save_profiles_json(self, path: str, profiles: dict) -> None:
        data = {}
        for spk, p in profiles.items():
            d = asdict(p)
            d.pop("segments", None)
            data[spk] = d

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _print_report(
        self,
        profiles,
        trainer_id,
        trainer_score,
        confidence,
        total_raw,
        cleaned,
        removed_micro,
        removed_artifacts,
        merged,
        session_duration,
    ) -> None:
        print("\n" + "=" * 65)
        print("POST-PROCESSING REPORT")
        print("=" * 65)
        print(f"  Session duration   : {session_duration:.1f}s ({session_duration/60:.1f} min)")
        print(f"  Raw segments       : {total_raw}")
        print(f"  Removed artifacts  : {removed_artifacts}")
        print(f"  Removed micro (<{MIN_SEGMENT_DURATION}s) : {removed_micro}")
        print(f"  Merged (gap <{MERGE_GAP_THRESHOLD}s)   : {merged}")
        print(f"  Clean segments     : {cleaned}")
        print("-" * 65)
        print(f"  {'Speaker':<14} {'Role':<10} {'Total(s)':>8} {'Avg(s)':>7} {'Coverage':>9} {'Score':>7}")
        print(f"  {'-'*61}")

        for spk in sorted(profiles, key=lambda x: profiles[x].trainer_score, reverse=True):
            p = profiles[spk]
            marker = " ← TRAINER" if spk == trainer_id else ""
            print(
                f"  {spk:<14} {p.role:<10} {p.total_duration:>8.1f} "
                f"{p.avg_duration:>7.2f} {p.coverage:>9.2%} "
                f"{p.trainer_score:>7.4f}{marker}"
            )

        print("=" * 65)
        print(f"  Trainer identified : {trainer_id}")
        print(f"  Trainer score      : {trainer_score:.4f}")
        print(f"  Confidence         : {confidence.upper()}")
        print("=" * 65 + "\n")