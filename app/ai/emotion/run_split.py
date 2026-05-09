from app.ai.emotion.split import split_manifest_by_speaker


def main():
    split_manifest_by_speaker(
        manifest_path="data/emotion_datasets/manifests/ravdess_manifest.csv",
        output_dir="data/emotion_datasets/manifests",
    )


if __name__ == "__main__":
    main()