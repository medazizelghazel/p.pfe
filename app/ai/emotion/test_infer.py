from app.ai.emotion.infer import EmotionInferenceService


def main():
    service = EmotionInferenceService(
        model_path="artifacts/emotion/emotion_cnn1d_best.pt",
        target_sample_rate=16000,
        max_duration=3.0,
    )

    result = service.predict_file(
        "data/emotion_datasets/ravdess/Actor_01/03-01-03-01-01-01-01.wav"
    )

    print("Prediction result:")
    print(f"Audio path: {result['audio_path']}")
    print(f"Predicted label: {result['predicted_label']}")
    print(f"Predicted label id: {result['predicted_label_id']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print("Probabilities:")
    for label, prob in result["probabilities"].items():
        print(f"  - {label}: {prob:.4f}")


if __name__ == "__main__":
    main()