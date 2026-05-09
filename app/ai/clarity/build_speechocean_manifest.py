from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


SCORE_COLUMNS = ["accuracy", "completeness", "fluency", "prosodic", "total"]


def read_kaldi_map(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key, value = line.split(maxsplit=1)
            mapping[key] = value.strip()

    return mapping


def resolve_audio_path(raw_path: str, dataset_root: Path) -> str:
    candidate = Path(raw_path)

    if not candidate.is_absolute():
        candidate = dataset_root / candidate

    if candidate.exists():
        return str(candidate.resolve())

    # try suffix case variants
    if candidate.suffix:
        lower_candidate = candidate.with_suffix(candidate.suffix.lower())
        if lower_candidate.exists():
            return str(lower_candidate.resolve())

        upper_candidate = candidate.with_suffix(candidate.suffix.upper())
        if upper_candidate.exists():
            return str(upper_candidate.resolve())

    raise FileNotFoundError(f"Audio file not found: {candidate}")


def build_subset_rows(dataset_root: Path, subset_name: str, scores: dict) -> list[dict]:
    subset_dir = dataset_root / subset_name

    text_map = read_kaldi_map(subset_dir / "text")
    utt2spk = read_kaldi_map(subset_dir / "utt2spk")
    wav_scp = read_kaldi_map(subset_dir / "wav.scp")
    spk2age = read_kaldi_map(subset_dir / "spk2age")
    spk2gender = read_kaldi_map(subset_dir / "spk2gender")

    rows: list[dict] = []

    for utt_id, raw_audio_path in wav_scp.items():
        if utt_id not in scores:
            continue

        score_item = scores[utt_id]
        speaker_id = utt2spk.get(utt_id, "")

        row = {
            "split": subset_name,
            "utt_id": utt_id,
            "speaker_id": speaker_id,
            "gender": spk2gender.get(speaker_id, ""),
            "age": spk2age.get(speaker_id, ""),
            "audio_path": resolve_audio_path(raw_audio_path, dataset_root),
            "text": text_map.get(utt_id, score_item.get("text", "")),
            "accuracy": score_item.get("accuracy", 0.0),
            "completeness": score_item.get("completeness", 0.0),
            "fluency": score_item.get("fluency", 0.0),
            "prosodic": score_item.get("prosodic", 0.0),
            "total": score_item.get("total", 0.0),
        }
        rows.append(row)

    return rows


def save_manifest(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved: {path} | rows={len(df)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="data/clarity_datasets/speechocean762",
        help="Root folder of the Speechocean762 dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/clarity_datasets/manifests",
        help="Where to save generated CSV manifests.",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.10,
        help="Validation split ratio from the official training split.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for group split.",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)

    scores_path = dataset_root / "scores.json"
    if not scores_path.exists():
        raise FileNotFoundError(f"Missing file: {scores_path}")

    with scores_path.open("r", encoding="utf-8") as f:
        scores = json.load(f)

    train_rows = build_subset_rows(dataset_root, "train", scores)
    test_rows = build_subset_rows(dataset_root, "test", scores)

    train_full_df = pd.DataFrame(train_rows)
    test_df = pd.DataFrame(test_rows)

    if train_full_df.empty:
        raise ValueError("Training manifest is empty. Check dataset paths and files.")

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=args.val_size,
        random_state=args.random_state
    )

    train_idx, val_idx = next(
        splitter.split(train_full_df, groups=train_full_df["speaker_id"])
    )

    train_df = train_full_df.iloc[train_idx].copy().reset_index(drop=True)
    val_df = train_full_df.iloc[val_idx].copy().reset_index(drop=True)
    test_df = test_df.copy().reset_index(drop=True)

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    save_manifest(train_df, output_dir / "speechocean_train.csv")
    save_manifest(val_df, output_dir / "speechocean_val.csv")
    save_manifest(test_df, output_dir / "speechocean_test.csv")

    print("\n=== SUMMARY ===")
    print(f"Train rows : {len(train_df)}")
    print(f"Val rows   : {len(val_df)}")
    print(f"Test rows  : {len(test_df)}")
    print(f"Train speakers: {train_df['speaker_id'].nunique()}")
    print(f"Val speakers  : {val_df['speaker_id'].nunique()}")
    print(f"Test speakers : {test_df['speaker_id'].nunique()}")


if __name__ == "__main__":
    main()