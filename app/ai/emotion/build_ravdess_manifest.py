import csv
from pathlib import Path

from app.ai.emotion.label_map import (
    map_ravdess_to_project_label,
    PROJECT_LABEL_TO_ID,
)
from app.ai.emotion.utils import extract_ravdess_speaker_id


def build_ravdess_manifest(
    dataset_dir: str = "data/emotion_datasets/ravdess",
    output_csv: str = "data/emotion_datasets/manifests/ravdess_manifest.csv",
):
    dataset_path = Path(dataset_dir)
    output_path = Path(output_csv)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(dataset_path.rglob("*.wav"))

    rows = []

    for wav_path in wav_files:
        try:
            label = map_ravdess_to_project_label(str(wav_path))
            label_id = PROJECT_LABEL_TO_ID[label]
            speaker_id = extract_ravdess_speaker_id(str(wav_path))

            rows.append({
                "path": str(wav_path).replace("\\", "/"),
                "label": label,
                "label_id": label_id,
                "dataset": "ravdess",
                "speaker_id": speaker_id,
            })
        except Exception as e:
            print(f"Skipped {wav_path}: {e}")

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["path", "label", "label_id", "dataset", "speaker_id"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Manifest created: {output_path}")
    print(f"Total files: {len(rows)}")


if __name__ == "__main__":
    build_ravdess_manifest()