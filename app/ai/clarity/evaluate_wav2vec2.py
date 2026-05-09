from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader

from app.ai.clarity.train_wav2vec2 import Wav2Vec2ClarityRegressor
from app.ai.clarity.wav2vec2_dataset import (
    Wav2Vec2ClarityDataset,
    Wav2Vec2ClarityConfig,
    wav2vec2_regression_collate_fn,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)

    try:
        spearman = spearmanr(y_true, y_pred).correlation
        if spearman is None or np.isnan(spearman):
            spearman = 0.0
    except Exception:
        spearman = 0.0

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "spearman": float(spearman),
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true = []
    y_pred = []
    rows = []

    for batch in loader:
        input_values = batch["input_values"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        preds = model(input_values=input_values, attention_mask=attention_mask)

        y_true.extend(labels.cpu().numpy().tolist())
        y_pred.extend(preds.cpu().numpy().tolist())

        for i in range(len(batch["audio_paths"])):
            rows.append({
                "audio_path": batch["audio_paths"][i],
                "utt_id": batch["utt_ids"][i],
                "speaker_id": batch["speaker_ids"][i],
                "true_score": float(labels.cpu().numpy()[i]),
                "predicted_score": float(preds.cpu().numpy()[i]),
            })

    metrics = compute_metrics(np.array(y_true), np.array(y_pred))
    return metrics, pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="fluency")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--manifests-dir", type=str, default="data/clarity_datasets/manifests")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts/clarity")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifacts_dir = Path(args.artifacts_dir)
    model_path = artifacts_dir / f"clarity_{args.target}_wav2vec2_best.pt"
    metadata_path = artifacts_dir / f"clarity_{args.target}_wav2vec2_metadata.json"

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    config = Wav2Vec2ClarityConfig(
        target_sample_rate=16000,
        max_duration_sec=float(metadata["max_duration_sec"]),
        min_duration_sec=2.0,
    )

    manifest_path = Path(args.manifests_dir) / f"speechocean_{args.split}.csv"

    dataset = Wav2Vec2ClarityDataset(
        manifest_path=str(manifest_path),
        target_column=args.target,
        config=config,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=wav2vec2_regression_collate_fn,
    )

    model = Wav2Vec2ClarityRegressor(pretrained_name=metadata["pretrained_name"]).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    metrics, pred_df = evaluate(model, loader, device)

    print("\n=== WAV2VEC2 CLARITY EVALUATION ===")
    print(f"Split     : {args.split}")
    print(f"Target    : {args.target}")
    print(f"MAE       : {metrics['mae']:.4f}")
    print(f"RMSE      : {metrics['rmse']:.4f}")
    print(f"R2        : {metrics['r2']:.4f}")
    print(f"Spearman  : {metrics['spearman']:.4f}")

    out_path = Path("data/results") / f"clarity_{args.target}_wav2vec2_{args.split}_predictions.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out_path, index=False)
    print(f"\nSaved predictions: {out_path}")


if __name__ == "__main__":
    main()