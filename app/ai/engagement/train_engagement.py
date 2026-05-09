from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr


META_COLUMNS = ["audio_path", "chunk_id", "source_dir"]
TARGET_COLUMN = "engagement_score_100"


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


def build_model(name: str):
    if name == "gradient_boosting":
        return GradientBoostingRegressor(random_state=42)
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
    raise ValueError(f"Unknown model: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifests-dir", type=str, default="data/engagement_datasets/manifests")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts/engagement")
    args = parser.parse_args()

    manifests_dir = Path(args.manifests_dir)
    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(manifests_dir / "engagement_train.csv")
    val_df = pd.read_csv(manifests_dir / "engagement_val.csv")

    excluded = set(META_COLUMNS + [TARGET_COLUMN])
    feature_columns = [c for c in train_df.columns if c not in excluded]

    X_train = train_df[feature_columns].astype(float)
    y_train = train_df[TARGET_COLUMN].astype(float)

    X_val = val_df[feature_columns].astype(float)
    y_val = val_df[TARGET_COLUMN].astype(float)

    candidate_names = ["gradient_boosting", "random_forest"]

    best_model_name = None
    best_metrics = None
    best_mae = float("inf")
    comparison_rows = []

    for model_name in candidate_names:
        model = build_model(model_name)
        model.fit(X_train, y_train)

        val_pred = model.predict(X_val)
        metrics = compute_metrics(y_val.to_numpy(), val_pred)

        comparison_rows.append({"model_name": model_name, **metrics})

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
        raise RuntimeError("No engagement model selected.")

    full_train_df = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)
    X_full = full_train_df[feature_columns].astype(float)
    y_full = full_train_df[TARGET_COLUMN].astype(float)

    final_model = build_model(best_model_name)
    final_model.fit(X_full, y_full)

    model_path = artifacts_dir / "engagement_best.joblib"
    metadata_path = artifacts_dir / "engagement_metadata.json"
    comparison_path = artifacts_dir / "engagement_model_comparison.csv"

    joblib.dump(final_model, model_path)
    pd.DataFrame(comparison_rows).sort_values("mae").to_csv(comparison_path, index=False)

    metadata = {
        "target_name": TARGET_COLUMN,
        "best_model_name": best_model_name,
        "feature_columns": feature_columns,
        "validation_metrics": best_metrics,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved model: {model_path}")
    print(f"Saved metadata: {metadata_path}")
    print(f"Saved comparison: {comparison_path}")


if __name__ == "__main__":
    main()