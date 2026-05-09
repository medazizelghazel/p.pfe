from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from app.ai.engagement.features import EngagementFeatureExtractor


def collect_chunk_paths(root_dir: Path) -> list[Path]:
    return sorted(root_dir.rglob("*.wav"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chunks-root",
        type=str,
        default="data/results/diarization",
        help="Root folder containing trainer chunk wav files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/engagement_datasets/manifests",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.10,
    )
    args = parser.parse_args()

    chunks_root = Path(args.chunks_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wav_paths = collect_chunk_paths(chunks_root)
    if not wav_paths:
        raise FileNotFoundError(f"No wav files found under: {chunks_root}")

    extractor = EngagementFeatureExtractor(use_emotion_features=True)

    rows = []
    for idx, wav_path in enumerate(wav_paths, start=1):
        features = extractor.extract(str(wav_path))
        row = {
            "audio_path": str(wav_path),
            "chunk_id": wav_path.stem,
            "source_dir": str(wav_path.parent),
            **features,
        }
        rows.append(row)

        if idx % 20 == 0:
            print(f"Processed {idx}/{len(wav_paths)} chunks...")

    df = pd.DataFrame(rows)

    train_df, temp_df = train_test_split(df, test_size=args.val_size + args.test_size, random_state=42)
    relative_test_size = args.test_size / (args.val_size + args.test_size)
    val_df, test_df = train_test_split(temp_df, test_size=relative_test_size, random_state=42)

    train_df.to_csv(output_dir / "engagement_train.csv", index=False)
    val_df.to_csv(output_dir / "engagement_val.csv", index=False)
    test_df.to_csv(output_dir / "engagement_test.csv", index=False)

    print(f"Saved: {output_dir / 'engagement_train.csv'} | rows={len(train_df)}")
    print(f"Saved: {output_dir / 'engagement_val.csv'} | rows={len(val_df)}")
    print(f"Saved: {output_dir / 'engagement_test.csv'} | rows={len(test_df)}")


if __name__ == "__main__":
    main()  