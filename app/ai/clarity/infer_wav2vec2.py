from __future__ import annotations

import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import torch

from app.ai.clarity.train_wav2vec2 import Wav2Vec2ClarityRegressor


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
    model_path = artifacts_dir / f"clarity_{args.target}_wav2vec2_best.pt"
    metadata_path = artifacts_dir / f"clarity_{args.target}_wav2vec2_metadata.json"

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    sr = 16000
    max_duration = float(metadata["max_duration_sec"])
    max_num_samples = int(sr * max_duration)

    y, _ = librosa.load(args.audio, sr=sr, mono=True)
    y = y.astype(np.float32)

    if len(y) > max_num_samples:
        y = y[:max_num_samples]

    attention_mask = np.ones(len(y), dtype=np.int64)

    input_values = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long).unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Wav2Vec2ClarityRegressor(pretrained_name=metadata["pretrained_name"]).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        pred = model(
            input_values=input_values.to(device),
            attention_mask=attention_mask.to(device)
        )
        score = float(pred.item())
        score = float(np.clip(score, 0.0, 10.0))

    result = {
        "audio_path": args.audio,
        "target_name": args.target,
        "predicted_score": round(score, 4),
        "score_label": score_to_label(score),
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()