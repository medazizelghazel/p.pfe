from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.ai.clarity.features import (
    ClarityFeatureExtractor,
    META_COLUMNS,
    SCORE_COLUMNS,
    build_feature_dataframe,
)


def build_model(name: str):
    if name == "ridge":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0))
        ])

    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )

    if name == "gradient_boosting":
        return GradientBoostingRegressor(
            random_state=42
        )

    raise ValueError(f"Unknown model name: {name}")


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


def ensure_feature_cache(
    manifest_path: Path,
    cache_path: Path,
    extractor: ClarityFeatureExtractor,
) -> pd.DataFrame:
    if cache_path.exists():
        print(f"Loading cached features: {cache_path}")
        return pd.read_csv(cache_path)

    print(f"Building features from manifest: {manifest_path}")
    manifest_df = pd.read_csv(manifest_path)
    feature_df = build_feature_dataframe(manifest_df, extractor)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(cache_path, index=False)
    print(f"Saved feature cache: {cache_path}")

    return feature_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        type=str,
        default="fluency",
        choices=["accuracy", "completeness", "fluency", "prosodic", "total"],
    )
    parser.add_argument(
        "--manifests-dir",
        type=str,
        default="data/clarity_datasets/manifests",
    )
    parser.add_argument(
        "--feature-cache-dir",
        type=str,
        default="data/clarity_datasets/features",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="artifacts/clarity",
    )
    args = parser.parse_args()

    manifests_dir = Path(args.manifests_dir)
    feature_cache_dir = Path(args.feature_cache_dir)
    artifacts_dir = Path(args.artifacts_dir)

    train_manifest = manifests_dir / "speechocean_train.csv"
    val_manifest = manifests_dir / "speechocean_val.csv"

    if not train_manifest.exists() or not val_manifest.exists():
        raise FileNotFoundError(
            "Missing manifests. Run build_speechocean_manifest.py first."
        )

    extractor = ClarityFeatureExtractor()

    train_cache = feature_cache_dir / "speechocean_train_features.csv"
    val_cache = feature_cache_dir / "speechocean_val_features.csv"

    train_df = ensure_feature_cache(train_manifest, train_cache, extractor)
    val_df = ensure_feature_cache(val_manifest, val_cache, extractor)

    excluded_columns = set(META_COLUMNS + SCORE_COLUMNS)
    feature_columns = [c for c in train_df.columns if c not in excluded_columns]

    X_train = train_df[feature_columns].astype(float)
    y_train = train_df[args.target].astype(float)

    X_val = val_df[feature_columns].astype(float)
    y_val = val_df[args.target].astype(float)

    candidate_names = ["gradient_boosting", "random_forest"]

    comparison_rows = []
    best_model_name = None
    best_metrics = None
    best_mae = float("inf")

    for model_name in candidate_names:
        print(f"\nTraining candidate: {model_name}")
        model = build_model(model_name)
        model.fit(X_train, y_train)

        val_pred = model.predict(X_val)
        metrics = compute_metrics(y_val.to_numpy(), val_pred)

        row = {"model_name": model_name, **metrics}
        comparison_rows.append(row)

        print(
            f"{model_name} | "
            f"MAE={metrics['mae']:.4f} | "
            f"RMSE={metrics['rmse']:.4f} | "
            f"R2={metrics['r2']:.4f} | "
            f"Spearman={metrics['spearman']:.4f}"
        )

        if metrics["mae"] < best_mae:
            best_mae = metrics["mae"]
            best_model_name = model_name
            best_metrics = metrics

    if best_model_name is None:
        raise RuntimeError("No model was selected.")

    print(f"\nBest model on validation: {best_model_name}")

    final_model = build_model(best_model_name)
    X_trainval = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    y_trainval = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)
    final_model.fit(X_trainval, y_trainval)

    artifacts_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifacts_dir / f"clarity_{args.target}_best.joblib"
    metadata_path = artifacts_dir / f"clarity_{args.target}_metadata.json"
    comparison_path = artifacts_dir / f"clarity_{args.target}_model_comparison.csv"

    joblib.dump(final_model, model_path)

    comparison_df = pd.DataFrame(comparison_rows).sort_values("mae", ascending=True)
    comparison_df.to_csv(comparison_path, index=False)

    metadata = {
        "target_name": args.target,
        "best_model_name": best_model_name,
        "feature_columns": feature_columns,
        "validation_metrics": best_metrics,
        "train_manifest": str(train_manifest),
        "val_manifest": str(val_manifest),
        "feature_cache_train": str(train_cache),
        "feature_cache_val": str(val_cache),
        "created_at": datetime.utcnow().isoformat() + "Z",
    }

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved model     : {model_path}")
    print(f"Saved metadata  : {metadata_path}")
    print(f"Saved comparison: {comparison_path}")


if __name__ == "__main__":
    main()