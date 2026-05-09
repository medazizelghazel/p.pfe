from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from app.ai.clarity.features import ClarityFeatureExtractor


def score_to_label(score: float) -> str:
    if score >= 8.0:
        return "high"
    if score >= 6.0:
        return "good"
    if score >= 4.0:
        return "medium"
    return "low"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=str, required=True)
    parser.add_argument("--target", type=str, default="fluency")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts/clarity")
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    model_path = artifacts_dir / f"clarity_{args.target}_best.joblib"
    metadata_path = artifacts_dir / f"clarity_{args.target}_metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    model = joblib.load(model_path)
    feature_columns = metadata["feature_columns"]
    target_name = metadata["target_name"]

    extractor = ClarityFeatureExtractor()
    features = extractor.extract(args.audio)

    x = pd.DataFrame([{col: float(features.get(col, 0.0)) for col in feature_columns}])
    predicted_score = float(model.predict(x)[0])

    if target_name in {"accuracy", "fluency", "prosodic", "total"}:
        predicted_score = float(np.clip(predicted_score, 0.0, 10.0))

    result = {
        "audio_path": args.audio,
        "target_name": target_name,
        "predicted_score": round(predicted_score, 4),
        "score_label": score_to_label(predicted_score),
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()