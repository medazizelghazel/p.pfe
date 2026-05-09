from pathlib import Path

import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

from app.ai.emotion.model import EmotionCNN1D
from app.ai.emotion.label_map import PROJECT_ID_TO_LABEL
from app.ai.emotion.dataloader import create_emotion_dataloaders


@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()

    all_preds = []
    all_labels = []

    for batch in loader:
        waveforms = batch["waveform"].to(device)
        labels = batch["label_id"].to(device)

        outputs = model(waveforms)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    return all_labels, all_preds


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_path = Path("artifacts/emotion/emotion_cnn1d_best.pt")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    data = create_emotion_dataloaders(
        train_manifest="data/emotion_datasets/manifests/ravdess_train.csv",
        val_manifest="data/emotion_datasets/manifests/ravdess_val.csv",
        test_manifest="data/emotion_datasets/manifests/ravdess_test.csv",
        target_sample_rate=16000,
        max_duration=3.0,
        batch_size=8,
        num_workers=0,
    )

    test_loader = data["test_loader"]

    model = EmotionCNN1D(num_classes=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    y_true, y_pred = evaluate_model(model, test_loader, device)

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred)

    target_names = [PROJECT_ID_TO_LABEL[i] for i in range(len(PROJECT_ID_TO_LABEL))]
    report = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        zero_division=0
    )

    print("\n=== CNN1D TEST EVALUATION ===")
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