from app.ai.emotion.infer_cnn2d import EmotionInferenceServiceCNN2D


def main():
    service = EmotionInferenceServiceCNN2D(
        model_path="artifacts/emotion/emotion_cnn2d_best.pt",
        target_sample_rate=16000,
        max_duration=3.0,
        hop_duration=1.5,
        n_mels=64,
        n_fft=1024,
        hop_length=256,
    )

    result = service.predict_file("data/processed_audio/your_processed_file.wav")

    print("Audio path:", result["audio_path"])
    print("Dominant emotion:", result["dominant_emotion"])
    print("Confidence:", f"{result['confidence']:.4f}")
    print("Num segments:", result["num_segments"])
    print("Aggregated scores:", result["aggregated_scores"])

    print("\nFirst segment predictions:")
    for seg in result["segment_predictions"][:5]:
        print(seg)


if __name__ == "__main__":
    main()