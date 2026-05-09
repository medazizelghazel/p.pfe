from app.services.transcription_service import TranscriptionService
from app.services.summary_generator import SummaryGenerator


def main():
    # ── Paths ─────────────────────────────────────────────────────────────────
    audio_path = "data/processed_audio/test1_20260420_233444_processed.wav"
    clean_csv  = "data/results/diarization/test1_20260420_235829_extracted_segments_clean.csv"

    # ── Step 1: Transcription ─────────────────────────────────────────────────
    print("\n[1/2] Transcribing trainer segments...")
    transcriber = TranscriptionService(model_size="small")
    transcript  = transcriber.transcribe_trainer(
        audio_path=audio_path,
        clean_csv=clean_csv,
        output_dir="data/results/transcription",
    )

    print(f"\n── Transcription result ─────────────────────────────────────")
    print(f"  Trainer    : {transcript['trainer_id']}")
    print(f"  Segments   : {transcript['segments_count']}")
    print(f"  Duration   : {transcript['total_duration']}s")
    print(f"  Words      : {len(transcript['full_text'].split())}")
    print(f"  JSON       : {transcript['output_json']}")
    print(f"\n  First 300 chars of transcript:")
    print(f"  {transcript['full_text'][:300]}...")

    # ── Step 2: Summary ───────────────────────────────────────────────────────
    print("\n[2/2] Generating course summary...")
    generator = SummaryGenerator(provider="mistral")  # free API
    result    = generator.generate(
        transcript_json=transcript["output_json"],
        output_dir="data/results/summary",
    )

    print(f"\n── Summary saved ────────────────────────────────────────────")
    print(f"  File : {result['output_json']}")
    print(f"  Ready for PDF generation ✅")


if __name__ == "__main__":
    main()