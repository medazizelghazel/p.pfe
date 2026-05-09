from app.ai.emotion.infer import EmotionInferenceService


def main():
    service = EmotionInferenceService(
        model_path="artifacts/emotion/emotion_cnn1d_best.pt",
        target_sample_rate=16000,
        max_duration=3.0,
        hop_duration=1.5,
    )

    result = service.predict_file("data/processed_audio/sample_20260402_160810_processed.wav")

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