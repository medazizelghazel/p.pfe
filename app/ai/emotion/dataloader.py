from torch.utils.data import DataLoader

from app.ai.emotion.dataset import EmotionAudioDataset


def create_emotion_dataloaders(
    train_manifest: str = "data/emotion_datasets/manifests/ravdess_train.csv",
    val_manifest: str = "data/emotion_datasets/manifests/ravdess_val.csv",
    test_manifest: str = "data/emotion_datasets/manifests/ravdess_test.csv",
    target_sample_rate: int = 16000,
    max_duration: float = 3.0,
    batch_size: int = 8,
    num_workers: int = 0,
):
    train_dataset = EmotionAudioDataset(
        manifest_path=train_manifest,
        target_sample_rate=target_sample_rate,
        max_duration=max_duration,
    )

    val_dataset = EmotionAudioDataset(
        manifest_path=val_manifest,
        target_sample_rate=target_sample_rate,
        max_duration=max_duration,
    )

    test_dataset = EmotionAudioDataset(
        manifest_path=test_manifest,
        target_sample_rate=target_sample_rate,
        max_duration=max_duration,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return {
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
    }