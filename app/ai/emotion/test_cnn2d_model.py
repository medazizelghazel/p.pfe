import torch

from app.ai.emotion.cnn2d_model import EmotionCNN2D


def main():
    model = EmotionCNN2D(num_classes=4)

    dummy_input = torch.randn(8, 1, 64, 188)
    output = model(dummy_input)

    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)


if __name__ == "__main__":
    main()