from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score

from app.ai.emotion.cnn2d_model import EmotionCNN2D
from app.ai.emotion.spectrogram_dataloader import create_emotion_spectrogram_dataloaders


def build_weighted_train_loader(train_dataset, batch_size=8, num_workers=0):
    labels = [int(train_dataset[i]["label_id"].item()) for i in range(len(train_dataset))]

    class_counts = {}
    for label in labels:
        class_counts[label] = class_counts.get(label, 0) + 1

    class_weights = {
        label: 1.0 / count
        for label, count in class_counts.items()
    }

    sample_weights = [class_weights[label] for label in labels]
    sample_weights = torch.DoubleTensor(sample_weights)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
    )

    return train_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        features = batch["features"].to(device)
        labels = batch["label_id"].to(device)

        optimizer.zero_grad()

        outputs = model(features)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * features.size(0)

        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0

    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_preds = []

    for batch in loader:
        features = batch["features"].to(device)
        labels = batch["label_id"].to(device)

        outputs = model(features)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * features.size(0)

        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_labels.extend(labels.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0
    epoch_macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return epoch_loss, epoch_acc, epoch_macro_f1


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data = create_emotion_spectrogram_dataloaders(
        train_manifest="data/emotion_datasets/manifests/ravdess_train.csv",
        val_manifest="data/emotion_datasets/manifests/ravdess_val.csv",
        test_manifest="data/emotion_datasets/manifests/ravdess_test.csv",
        target_sample_rate=16000,
        max_duration=3.0,
        n_mels=64,
        n_fft=1024,
        hop_length=256,
        batch_size=8,
        num_workers=0,
    )

    train_dataset = data["train_dataset"]
    val_loader = data["val_loader"]

    train_loader = build_weighted_train_loader(
        train_dataset=train_dataset,
        batch_size=8,
        num_workers=0,
    )

    model = EmotionCNN2D(num_classes=4).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 15

    output_dir = Path("artifacts/emotion")
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_macro_f1 = 0.0
    best_model_path = output_dir / "emotion_cnn2d_best.pt"

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_macro_f1 = evaluate(
            model, val_loader, criterion, device
        )

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"Val Macro F1: {val_macro_f1:.4f}"
        )

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved to: {best_model_path}")

    print(f"\nTraining complete. Best Val Macro F1: {best_val_macro_f1:.4f}")


if __name__ == "__main__":
    main()