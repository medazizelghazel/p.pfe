from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

from app.ai.clarity.wav2vec2_dataset import (
    Wav2Vec2ClarityDataset,
    Wav2Vec2ClarityConfig,
    wav2vec2_regression_collate_fn,
)


class Wav2Vec2ClarityRegressor(nn.Module):
    def __init__(self, pretrained_name: str = "facebook/wav2vec2-base"):
        super().__init__()
        self.backbone = Wav2Vec2Model.from_pretrained(pretrained_name)
        hidden_size = self.backbone.config.hidden_size

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.regressor = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, input_values: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(input_values=input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        # Convert sample-level attention mask to feature-level mask
        feature_attention_mask = self.backbone._get_feature_vector_attention_mask(
            hidden_states.shape[1],
            attention_mask
        )

        mask = feature_attention_mask.unsqueeze(-1).float()
        pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

        pred = self.regressor(pooled).squeeze(-1)
        return pred


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


def run_epoch(model, loader, optimizer, device, train: bool):
    criterion = nn.MSELoss()

    if train:
        model.train()
    else:
        model.eval()

    losses = []
    all_true = []
    all_pred = []

    for batch in loader:
        input_values = batch["input_values"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            preds = model(input_values=input_values, attention_mask=attention_mask)
            loss = criterion(preds, labels)

            if train:
                loss.backward()
                optimizer.step()

        losses.append(loss.item())
        all_true.extend(labels.detach().cpu().numpy().tolist())
        all_pred.extend(preds.detach().cpu().numpy().tolist())

    metrics = compute_metrics(np.array(all_true), np.array(all_pred))
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="fluency")
    parser.add_argument("--manifests-dir", type=str, default="data/clarity_datasets/manifests")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts/clarity")
    parser.add_argument("--pretrained-name", type=str, default="facebook/wav2vec2-base")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max-duration", type=float, default=12.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    manifests_dir = Path(args.manifests_dir)
    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    train_manifest = manifests_dir / "speechocean_train.csv"
    val_manifest = manifests_dir / "speechocean_val.csv"

    config = Wav2Vec2ClarityConfig(
        target_sample_rate=16000,
        max_duration_sec=args.max_duration,
        min_duration_sec=2.0,
    )

    _ = Wav2Vec2FeatureExtractor.from_pretrained(args.pretrained_name)

    train_dataset = Wav2Vec2ClarityDataset(
        manifest_path=str(train_manifest),
        target_column=args.target,
        config=config,
    )
    val_dataset = Wav2Vec2ClarityDataset(
        manifest_path=str(val_manifest),
        target_column=args.target,
        config=config,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=wav2vec2_regression_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=wav2vec2_regression_collate_fn,
    )

    model = Wav2Vec2ClarityRegressor(pretrained_name=args.pretrained_name).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    best_val_mae = float("inf")
    best_model_path = artifacts_dir / f"clarity_{args.target}_wav2vec2_best.pt"
    metadata_path = artifacts_dir / f"clarity_{args.target}_wav2vec2_metadata.json"

    history = []

    for epoch in range(args.epochs):
        train_metrics = run_epoch(model, train_loader, optimizer, device, train=True)
        val_metrics = run_epoch(model, val_loader, optimizer, device, train=False)

        row = {
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "train_mae": train_metrics["mae"],
            "val_loss": val_metrics["loss"],
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
            "val_r2": val_metrics["r2"],
            "val_spearman": val_metrics["spearman"],
        }
        history.append(row)

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"train_mae={train_metrics['mae']:.4f} | "
            f"val_mae={val_metrics['mae']:.4f} | "
            f"val_rmse={val_metrics['rmse']:.4f} | "
            f"val_r2={val_metrics['r2']:.4f} | "
            f"val_spearman={val_metrics['spearman']:.4f}"
        )

        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved to: {best_model_path}")

    metadata = {
        "target_name": args.target,
        "model_type": "wav2vec2_regression",
        "pretrained_name": args.pretrained_name,
        "sample_rate": 16000,
        "max_duration_sec": args.max_duration,
        "best_val_mae": best_val_mae,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "history": history,
    }

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()