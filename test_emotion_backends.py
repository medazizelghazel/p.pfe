from app.services.emotion_classifier import EmotionClassifier
from pathlib import Path
import time
import json
import traceback

AUDIO_PATH = r"C:\Users\azizg\Desktop\pfe-d\data\results\diarization\test7_20260507_173419_trainer_only.wav"

# Change this path if your test audio is somewhere else.
if not Path(AUDIO_PATH).exists():
    print(f"Audio file not found: {AUDIO_PATH}")
    print("Put the correct WAV path inside AUDIO_PATH.")
    raise SystemExit(1)


def test_backend(backend: str):
    print("\n" + "=" * 70)
    print(f"Testing backend: {backend}")
    print("=" * 70)

    try:
        start = time.time()

        classifier = EmotionClassifier(
            backend=backend,
            mode="balanced"
        )

        result = classifier.classify_with_details(AUDIO_PATH)

        elapsed = round(time.time() - start, 2)

        print(f"Backend: {backend}")
        print(f"Model name: {result.get('model_name')}")
        print(f"Dominant emotion: {result.get('dominant_emotion')}")
        print(f"Confidence: {result.get('confidence')}")
        print(f"Segments: {result.get('num_segments')}")
        print(f"Inference time: {elapsed} seconds")
        print("Aggregated scores:")
        print(json.dumps(result.get("aggregated_scores", {}), indent=2, ensure_ascii=False))

        return {
            "backend": backend,
            "success": True,
            "dominant_emotion": result.get("dominant_emotion"),
            "confidence": result.get("confidence"),
            "num_segments": result.get("num_segments"),
            "model_name": result.get("model_name"),
            "elapsed_seconds": elapsed,
            "aggregated_scores": result.get("aggregated_scores", {}),
        }

    except Exception as e:
        print(f"Backend failed: {backend}")
        print(type(e).__name__, str(e))
        traceback.print_exc()

        return {
            "backend": backend,
            "success": False,
            "error": f"{type(e).__name__}: {str(e)}",
        }


results = [
    test_backend("wav2vec2"),
    test_backend("cnn2d"),
]

print("\n" + "=" * 70)
print("FINAL COMPARISON")
print("=" * 70)
print(json.dumps(results, indent=2, ensure_ascii=False))
