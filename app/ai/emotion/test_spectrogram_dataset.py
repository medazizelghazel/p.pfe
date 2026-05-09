from app.ai.emotion.spectrogram_dataset import EmotionSpectrogramDataset


def main():
    dataset = EmotionSpectrogramDataset(
        manifest_path="data/emotion_datasets/manifests/ravdess_train.csv",
        target_sample_rate=16000,
        max_duration=3.0,
        n_mels=64,
        n_fft=1024,
        hop_length=256,
    )

    print("Dataset size:", len(dataset))

    sample = dataset[0]

    print("Keys:", sample.keys())
    print("Feature shape:", sample["features"].shape)
    print("Label ID:", sample["label_id"].item())
    print("Label:", sample["label"])
    print("Speaker ID:", sample["speaker_id"])
    print("Path:", sample["path"])


if __name__ == "__main__":
    main()