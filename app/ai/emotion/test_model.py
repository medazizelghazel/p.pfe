import torch

from app.ai.emotion.model import EmotionCNN1D


def main():
    model = EmotionCNN1D(num_classes=4)

    dummy_input = torch.randn(8, 1, 48000)
    output = model(dummy_input)

    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)


if __name__ == "__main__":
    main()