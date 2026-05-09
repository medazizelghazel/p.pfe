from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from app.ai.engagement.features import EngagementFeatureExtractor


def score_to_label(score: float) -> str:
    if score >= 80:
        return "high"
    if score >= 60:
        return "good"
    if score >= 40:
        return "medium"
    return "low"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=str, required=True)
    parser.add_argument("--artifacts-dir", type=str, default="artifacts/engagement")
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    model = joblib.load(artifacts_dir / "engagement_best.joblib")
    with (artifacts_dir / "engagement_metadata.json").open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    extractor = EngagementFeatureExtractor(use_emotion_features=True)
    features = extractor.extract(args.audio)

    x = pd.DataFrame([{col: float(features.get(col, 0.0)) for col in metadata["feature_columns"]}])
    predicted_score = float(model.predict(x)[0])

    result = {
        "audio_path": args.audio,
        "predicted_score": round(predicted_score, 4),
        "score_label": score_to_label(predicted_score),
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()