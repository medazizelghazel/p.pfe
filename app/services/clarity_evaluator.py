from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import soundfile as sf

from app.ai.clarity.features import ClarityFeatureExtractor


class ClarityEvaluator:
    def __init__(
        self,
        model_path: str = "artifacts/clarity/clarity_fluency_best.joblib",
        metadata_path: str = "artifacts/clarity/clarity_fluency_metadata.json",
    ):
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Clarity model not found: {self.model_path}")

        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Clarity metadata not found: {self.metadata_path}")

        self.model = joblib.load(self.model_path)

        with self.metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

        self.target_name = metadata["target_name"]
        self.feature_columns = metadata["feature_columns"]
        self.extractor = ClarityFeatureExtractor()

    def _score_to_label_10(self, score_10: float) -> str:
        if score_10 >= 8.0:
            return "high"
        if score_10 >= 6.0:
            return "good"
        if score_10 >= 4.0:
            return "medium"
        return "low"

    def evaluate(self, audio_path: str) -> dict:
        features = self.extractor.extract(audio_path)

        x = pd.DataFrame([{
            col: float(features.get(col, 0.0))
            for col in self.feature_columns
        }])

        raw_predicted_score = float(self.model.predict(x)[0])

        clipped_score_10 = raw_predicted_score
        if self.target_name in {"accuracy", "fluency", "prosodic", "total"}:
            clipped_score_10 = float(np.clip(raw_predicted_score, 0.0, 10.0))

        clarity_score_100 = clipped_score_10 * 10.0

        return {
            "target_name": self.target_name,
            "raw_predicted_score": raw_predicted_score,
            "clarity_score_10": clipped_score_10,
            "clarity_score": clarity_score_100,
            "clarity_label": self._score_to_label_10(clipped_score_10),
        }

    def evaluate_chunks(self, chunks_dir: str) -> dict:
        chunk_paths = sorted(Path(chunks_dir).glob("*.wav"))
        if not chunk_paths:
            raise FileNotFoundError(f"No chunk wav files found in: {chunks_dir}")

        chunk_results = []
        raw_scores = []
        clipped_scores_10 = []
        durations = []

        for chunk_path in chunk_paths:
            result = self.evaluate(str(chunk_path))
            info = sf.info(str(chunk_path))
            duration = float(info.duration)

            chunk_results.append({
                "chunk_path": str(chunk_path),
                "duration": duration,
                "raw_predicted_score": result["raw_predicted_score"],
                "clarity_score_10": result["clarity_score_10"],
                "clarity_score": result["clarity_score"],
                "clarity_label": result["clarity_label"],
            })

            raw_scores.append(result["raw_predicted_score"])
            clipped_scores_10.append(result["clarity_score_10"])
            durations.append(duration)

        weights = np.array(durations, dtype=float)
        if weights.sum() <= 0:
            weights = np.ones_like(weights)

        avg_raw = float(np.average(np.array(raw_scores, dtype=float), weights=weights))
        avg_score_10 = float(np.average(np.array(clipped_scores_10, dtype=float), weights=weights))
        avg_score_10 = float(np.clip(avg_score_10, 0.0, 10.0))
        avg_score_100 = avg_score_10 * 10.0

        return {
            "target_name": self.target_name,
            "raw_predicted_score": avg_raw,
            "clarity_score_10": avg_score_10,
            "clarity_score": avg_score_100,
            "clarity_label": self._score_to_label_10(avg_score_10),
            "num_chunks": len(chunk_results),
            "chunk_results": chunk_results,
        }