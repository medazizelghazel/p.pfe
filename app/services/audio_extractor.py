from moviepy import VideoFileClip

from app.config import EXTRACTED_AUDIO_DIR, TARGET_SAMPLE_RATE


class AudioExtractor:
    def extract(self, video_path: str, analysis_id: str) -> str:
        output_path = EXTRACTED_AUDIO_DIR / f"{analysis_id}_extracted.wav"

        clip = VideoFileClip(video_path)
        clip.audio.write_audiofile(
            str(output_path),
            fps=TARGET_SAMPLE_RATE,
            nbytes=2,
            ffmpeg_params=["-ac", "1"]
        )
        clip.close()

        return str(output_path)