from app.ai.emotion.dataloader import create_emotion_dataloaders


def main():
    data = create_emotion_dataloaders(
        train_manifest="data/emotion_datasets/manifests/ravdess_train.csv",
        val_manifest="data/emotion_datasets/manifests/ravdess_val.csv",
        test_manifest="data/emotion_datasets/manifests/ravdess_test.csv",
        target_sample_rate=16000,
        max_duration=3.0,
        batch_size=8,
        num_workers=0,
    )

    train_loader = data["train_loader"]
    val_loader = data["val_loader"]
    test_loader = data["test_loader"]

    print("Train dataset size:", len(data["train_dataset"]))
    print("Val dataset size:", len(data["val_dataset"]))
    print("Test dataset size:", len(data["test_dataset"]))

    batch = next(iter(train_loader))

    print("\n=== TRAIN BATCH ===")
    print("Waveform batch shape:", batch["waveform"].shape)
    print("Label batch shape:", batch["label_id"].shape)
    print("Labels:", batch["label"])
    print("Speaker IDs:", batch["speaker_id"])
    print("Paths:", batch["path"][:2])


if __name__ == "__main__":
    main()