from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--manifests-dir", type=str, default="data/engagement_datasets/manifests")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts/engagement")
    args = parser.parse_args()

    manifests_dir = Path(args.manifests_dir)
    artifacts_dir = Path(args.artifacts_dir)

    model = joblib.load(artifacts_dir / "engagement_best.joblib")
    with (artifacts_dir / "engagement_metadata.json").open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    df = pd.read_csv(manifests_dir / f"engagement_{args.split}.csv")
    feature_columns = metadata["feature_columns"]
    target_name = metadata["target_name"]

    X = df[feature_columns].astype(float)
    y_true = df[target_name].astype(float).to_numpy()
    y_pred = model.predict(X)

    metrics = compute_metrics(y_true, y_pred)

    print("\n=== ENGAGEMENT MODEL EVALUATION ===")
    print(f"Split     : {args.split}")
    print(f"Target    : {target_name}")
    print(f"MAE       : {metrics['mae']:.4f}")
    print(f"RMSE      : {metrics['rmse']:.4f}")
    print(f"R2        : {metrics['r2']:.4f}")
    print(f"Spearman  : {metrics['spearman']:.4f}")

    out_path = Path("data/results") / f"engagement_{args.split}_predictions.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_df = df[["audio_path", "chunk_id", target_name]].copy()
    out_df["predicted_score"] = y_pred
    out_df["absolute_error"] = np.abs(out_df[target_name] - out_df["predicted_score"])
    out_df.to_csv(out_path, index=False)

    print(f"Saved predictions: {out_path}")


if __name__ == "__main__":
    main()