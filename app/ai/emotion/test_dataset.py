from app.ai.emotion.dataset import EmotionAudioDataset


def main():
    dataset = EmotionAudioDataset(
        manifest_path="data/emotion_datasets/manifests/ravdess_manifest.csv",
        target_sample_rate=16000,
        max_duration=3.0,
    )

    print(f"Dataset size: {len(dataset)}")

    sample = dataset[0]

    print("Keys:", sample.keys())
    print("Waveform shape:", sample["waveform"].shape)
    print("Label ID:", sample["label_id"].item())
    print("Label:", sample["label"])
    print("Speaker ID:", sample["speaker_id"])
    print("Path:", sample["path"])


if __name__ == "__main__":
    main()