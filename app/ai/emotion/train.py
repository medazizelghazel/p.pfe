from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from app.ai.emotion.dataloader import create_emotion_dataloaders
from app.ai.emotion.model import EmotionCNN1D


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        waveforms = batch["waveform"].to(device)
        labels = batch["label_id"].to(device)

        optimizer.zero_grad()

        outputs = model(waveforms)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * waveforms.size(0)

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

    for batch in loader:
        waveforms = batch["waveform"].to(device)
        labels = batch["label_id"].to(device)

        outputs = model(waveforms)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * waveforms.size(0)

        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0

    return epoch_loss, epoch_acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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

    model = EmotionCNN1D(num_classes=4).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 15

    output_dir = Path("artifacts/emotion")
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    best_model_path = output_dir / "emotion_cnn1d_best.pt"

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved to: {best_model_path}")

    print(f"\nTraining complete. Best Val Acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()