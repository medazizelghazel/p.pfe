from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from app.ai.clarity.features import ClarityFeatureExtractor, build_feature_dataframe


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


def ensure_feature_cache(manifest_path: Path, cache_path: Path) -> pd.DataFrame:
    if cache_path.exists():
        return pd.read_csv(cache_path)

    extractor = ClarityFeatureExtractor()
    manifest_df = pd.read_csv(manifest_path)
    feature_df = build_feature_dataframe(manifest_df, extractor)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(cache_path, index=False)
    return feature_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="fluency")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--manifests-dir", type=str, default="data/clarity_datasets/manifests")
    parser.add_argument("--feature-cache-dir", type=str, default="data/clarity_datasets/features")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts/clarity")
    parser.add_argument("--results-dir", type=str, default="data/results")
    args = parser.parse_args()

    manifests_dir = Path(args.manifests_dir)
    feature_cache_dir = Path(args.feature_cache_dir)
    artifacts_dir = Path(args.artifacts_dir)
    results_dir = Path(args.results_dir)

    model_path = artifacts_dir / f"clarity_{args.target}_best.joblib"
    metadata_path = artifacts_dir / f"clarity_{args.target}_metadata.json"
    manifest_path = manifests_dir / f"speechocean_{args.split}.csv"
    cache_path = feature_cache_dir / f"speechocean_{args.split}_features.csv"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    model = joblib.load(model_path)
    feature_columns = metadata["feature_columns"]
    target_name = metadata["target_name"]

    df = ensure_feature_cache(manifest_path, cache_path)

    X = df[feature_columns].astype(float)
    y_true = df[target_name].astype(float).to_numpy()
    y_pred = model.predict(X)

    metrics = compute_metrics(y_true, y_pred)

    print("\n=== CLARITY MODEL EVALUATION ===")
    print(f"Split     : {args.split}")
    print(f"Target    : {target_name}")
    print(f"MAE       : {metrics['mae']:.4f}")
    print(f"RMSE      : {metrics['rmse']:.4f}")
    print(f"R2        : {metrics['r2']:.4f}")
    print(f"Spearman  : {metrics['spearman']:.4f}")

    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"clarity_{target_name}_{args.split}_predictions.csv"

    result_df = df[["utt_id", "speaker_id", "audio_path", target_name]].copy()
    result_df["predicted_score"] = y_pred
    result_df["absolute_error"] = np.abs(result_df[target_name] - result_df["predicted_score"])
    result_df.to_csv(out_path, index=False)

    print(f"\nSaved predictions: {out_path}")


if __name__ == "__main__":
    main()