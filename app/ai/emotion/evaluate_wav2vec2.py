from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

from app.ai.emotion.infer_wav2vec2 import EmotionInferenceServiceWav2Vec2


def main():
    manifest_path = Path("data/emotion_datasets/manifests/ravdess_test.csv")

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)

    service = EmotionInferenceServiceWav2Vec2(
        model_path="artifacts/emotion/wav2vec2_emotion_best.pt",
        target_sample_rate=16000,
        max_duration=8.0,
        hop_duration=8.0,
    )

    label_to_id = {
        "neutral_calm": 0,
        "energetic_engaged": 1,
        "low_energy": 2,
        "tense_stressed": 3,
    }

    id_to_label = {v: k for k, v in label_to_id.items()}

    y_true = []
    y_pred = []

    for _, row in df.iterrows():
        audio_path = row["path"]
        true_label = row["label"]

        result = service.predict_file(audio_path)
        predicted_label = result["dominant_emotion"]

        if true_label in label_to_id and predicted_label in label_to_id:
            y_true.append(label_to_id[true_label])
            y_pred.append(label_to_id[predicted_label])

    acc = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred)

    target_names = [id_to_label[i] for i in range(len(id_to_label))]

    report = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        zero_division=0,
    )

    print("\n=== WAV2VEC2 EMOTION TEST EVALUATION ===")
    print(f"Accuracy        : {acc:.4f}")
    print(f"Macro Precision : {precision:.4f}")
    print(f"Macro Recall    : {recall:.4f}")
    print(f"Macro F1-score  : {f1:.4f}")

    print("\n=== CONFUSION MATRIX ===")
    print(cm)

    print("\n=== CLASSIFICATION REPORT ===")
    print(report)


if __name__ == "__main__":
    main()