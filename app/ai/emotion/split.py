from pathlib import Path
import pandas as pd


def split_manifest_by_speaker(
    manifest_path: str,
    output_dir: str = "data/emotion_datasets/manifests",
    train_speakers=None,
    val_speakers=None,
    test_speakers=None,
):
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    df = pd.read_csv(manifest_path)

    required_columns = {"path", "label", "label_id", "dataset", "speaker_id"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in manifest: {missing}")

    if train_speakers is None:
        train_speakers = ["01", "02", "03", "04", "05", "06", "07", "08",
                          "09", "10", "11", "12", "13", "14", "15", "16"]

    if val_speakers is None:
        val_speakers = ["17", "18", "19", "20"]

    if test_speakers is None:
        test_speakers = ["21", "22", "23", "24"]

    train_df = df[df["speaker_id"].astype(str).str.zfill(2).isin(train_speakers)].copy()
    val_df = df[df["speaker_id"].astype(str).str.zfill(2).isin(val_speakers)].copy()
    test_df = df[df["speaker_id"].astype(str).str.zfill(2).isin(test_speakers)].copy()

    train_path = output_dir / "ravdess_train.csv"
    val_path = output_dir / "ravdess_val.csv"
    test_path = output_dir / "ravdess_test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Train saved to: {train_path} ({len(train_df)} samples)")
    print(f"Val saved to: {val_path} ({len(val_df)} samples)")
    print(f"Test saved to: {test_path} ({len(test_df)} samples)")

    print("\nUnique speakers:")
    print("Train:", sorted(train_df["speaker_id"].astype(str).str.zfill(2).unique()))
    print("Val:", sorted(val_df["speaker_id"].astype(str).str.zfill(2).unique()))
    print("Test:", sorted(test_df["speaker_id"].astype(str).str.zfill(2).unique()))