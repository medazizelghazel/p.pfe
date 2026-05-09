from __future__ import annotations

import csv
import os
from dataclasses import dataclass


@dataclass
class LearnerParticipationFeatureExtractor:
    response_window_sec: float = 5.0

    def _safe_float(self, value) -> float:
        try:
            value = float(value)
            if value != value:
                return 0.0
            return value
        except Exception:
            return 0.0

    def _safe_bool(self, value) -> bool:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"true", "1", "yes", "y"}

    def _empty_features(self) -> dict[str, float]:
        return {
            "session_duration_sec": 0.0,

            "trainer_total_speech_duration": 0.0,
            "learner_total_speech_duration": 0.0,
            "total_speech_duration": 0.0,

            "trainer_talk_ratio": 0.0,
            "learner_talk_ratio": 0.0,
            "trainer_speech_share": 0.0,
            "learner_speech_share": 0.0,

            "trainer_turn_count": 0.0,
            "learner_turn_count": 0.0,
            "total_turn_count": 0.0,

            "active_learner_speaker_count": 0.0,
            "learner_avg_turn_duration": 0.0,
            "learner_max_turn_duration": 0.0,
            "learner_turns_per_min": 0.0,

            "interaction_turn_count": 0.0,
            "interaction_rate_per_min": 0.0,
            "trainer_learner_balance": 0.0,

            "learner_response_count": 0.0,
            "learner_response_ratio": 0.0,

            "overlap_segment_count": 0.0,
            "overlap_ratio": 0.0,
        }

    def _load_segments(
        self,
        clean_csv_path: str,
        trainer_speaker_id: str | None = None,
    ) -> list[dict]:
        segments = []

        with open(clean_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                start = self._safe_float(row.get("start", 0.0))
                end = self._safe_float(row.get("end", 0.0))
                duration = self._safe_float(row.get("duration", end - start))
                speaker = str(row.get("speaker", "")).strip()
                role = str(row.get("role", "unknown")).strip().lower()

                if trainer_speaker_id and speaker == trainer_speaker_id:
                    role = "trainer"
                elif role not in {"trainer", "learner"}:
                    role = "unknown"

                if duration <= 0:
                    duration = max(0.0, end - start)

                if end <= start or duration <= 0:
                    continue

                segments.append(
                    {
                        "start": start,
                        "end": end,
                        "speaker": speaker,
                        "role": role,
                        "duration": duration,
                        "has_overlap": self._safe_bool(row.get("has_overlap", False)),
                    }
                )

        return sorted(segments, key=lambda x: x["start"])

    def extract(
        self,
        clean_csv_path: str | None,
        trainer_speaker_id: str | None = None,
    ) -> dict[str, float]:
        if not clean_csv_path or not os.path.exists(clean_csv_path):
            return self._empty_features()

        segments = self._load_segments(
            clean_csv_path=clean_csv_path,
            trainer_speaker_id=trainer_speaker_id,
        )

        if not segments:
            return self._empty_features()

        session_start = min(s["start"] for s in segments)
        session_end = max(s["end"] for s in segments)
        session_duration = max(0.0, session_end - session_start)

        if session_duration <= 0:
            return self._empty_features()

        trainer_segments = [s for s in segments if s["role"] == "trainer"]
        learner_segments = [s for s in segments if s["role"] == "learner"]

        trainer_total = sum(s["duration"] for s in trainer_segments)
        learner_total = sum(s["duration"] for s in learner_segments)
        total_speech = trainer_total + learner_total

        trainer_turn_count = len(trainer_segments)
        learner_turn_count = len(learner_segments)
        total_turn_count = trainer_turn_count + learner_turn_count

        learner_durations = [s["duration"] for s in learner_segments]

        active_learner_speakers = {
            s["speaker"]
            for s in learner_segments
            if s["duration"] > 0
        }

        trainer_talk_ratio = trainer_total / session_duration
        learner_talk_ratio = learner_total / session_duration

        trainer_speech_share = trainer_total / total_speech if total_speech > 0 else 0.0
        learner_speech_share = learner_total / total_speech if total_speech > 0 else 0.0

        learner_avg_turn_duration = (
            learner_total / learner_turn_count
            if learner_turn_count > 0
            else 0.0
        )

        learner_max_turn_duration = (
            max(learner_durations)
            if learner_durations
            else 0.0
        )

        session_minutes = session_duration / 60.0
        learner_turns_per_min = (
            learner_turn_count / session_minutes
            if session_minutes > 0
            else 0.0
        )

        interaction_turn_count = 0
        learner_response_count = 0

        previous = None

        for current in segments:
            if previous is not None:
                previous_role = previous["role"]
                current_role = current["role"]

                if previous_role != current_role and {
                    previous_role,
                    current_role,
                } == {"trainer", "learner"}:
                    interaction_turn_count += 1

                gap = current["start"] - previous["end"]

                if (
                    previous_role == "trainer"
                    and current_role == "learner"
                    and 0.0 <= gap <= self.response_window_sec
                ):
                    learner_response_count += 1

            previous = current

        interaction_rate_per_min = (
            interaction_turn_count / session_minutes
            if session_minutes > 0
            else 0.0
        )

        learner_response_ratio = (
            learner_response_count / learner_turn_count
            if learner_turn_count > 0
            else 0.0
        )

        if total_speech > 0:
            trainer_learner_balance = 1.0 - abs(trainer_speech_share - learner_speech_share)
        else:
            trainer_learner_balance = 0.0

        overlap_segment_count = sum(1 for s in segments if s["has_overlap"])
        overlap_ratio = overlap_segment_count / len(segments) if segments else 0.0

        return {
            "session_duration_sec": round(float(session_duration), 4),

            "trainer_total_speech_duration": round(float(trainer_total), 4),
            "learner_total_speech_duration": round(float(learner_total), 4),
            "total_speech_duration": round(float(total_speech), 4),

            "trainer_talk_ratio": round(float(trainer_talk_ratio), 6),
            "learner_talk_ratio": round(float(learner_talk_ratio), 6),
            "trainer_speech_share": round(float(trainer_speech_share), 6),
            "learner_speech_share": round(float(learner_speech_share), 6),

            "trainer_turn_count": float(trainer_turn_count),
            "learner_turn_count": float(learner_turn_count),
            "total_turn_count": float(total_turn_count),

            "active_learner_speaker_count": float(len(active_learner_speakers)),
            "learner_avg_turn_duration": round(float(learner_avg_turn_duration), 4),
            "learner_max_turn_duration": round(float(learner_max_turn_duration), 4),
            "learner_turns_per_min": round(float(learner_turns_per_min), 4),

            "interaction_turn_count": float(interaction_turn_count),
            "interaction_rate_per_min": round(float(interaction_rate_per_min), 4),
            "trainer_learner_balance": round(float(trainer_learner_balance), 6),

            "learner_response_count": float(learner_response_count),
            "learner_response_ratio": round(float(learner_response_ratio), 6),

            "overlap_segment_count": float(overlap_segment_count),
            "overlap_ratio": round(float(overlap_ratio), 6),
        }