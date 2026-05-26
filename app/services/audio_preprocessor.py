import librosa
import soundfile as sf
import numpy as np
from scipy import signal
import noisereduce as nr  # Nouvelle bibliothèque pour la réduction de bruit

from app.domain.audio_file import AudioFile
from app.config import TARGET_SAMPLE_RATE, PROCESSED_AUDIO_DIR


class AudioPreprocessor:
    def __init__(self, target_sr: int = TARGET_SAMPLE_RATE):
        self.target_sr = target_sr

    def load(self, audio_path: str) -> tuple[np.ndarray, int]:
        y, sr = librosa.load(audio_path, sr=self.target_sr, mono=True)
        return y, sr

    def highpass_filter(self, y: np.ndarray, sr: int, cutoff: float = 80.0) -> np.ndarray:
        # Amélioration : Utilisation des sections du second ordre (SOS) 
        # pour une meilleure stabilité mathématique et éviter les artéfacts
        sos = signal.butter(4, cutoff, btype="highpass", fs=sr, output="sos")
        return signal.sosfiltfilt(sos, y)

    def reduce_noise_spectral(self, y: np.ndarray, sr: int, prop_decrease: float = 0.8) -> np.ndarray:
        # Amélioration : Remplace le noise_gate brutal par une soustraction spectrale intelligente
        # prop_decrease=0.8 réduit 80% du bruit de fond sans robotiser la voix
        return nr.reduce_noise(y=y, sr=sr, prop_decrease=prop_decrease)

    def trim_silence(self, y: np.ndarray, top_db: int = 25) -> np.ndarray:
        # top_db passé à 25 pour être un peu plus doux et ne pas couper les fins de phrases
        yt, _ = librosa.effects.trim(y, top_db=top_db)
        return yt

    def normalize(self, y: np.ndarray) -> np.ndarray:
        return librosa.util.normalize(y)

    def save(self, y: np.ndarray, sr: int, output_path: str) -> str:
        sf.write(output_path, y, sr, subtype='PCM_16')
        return output_path

    def process(self, audio_path: str, analysis_id: str) -> AudioFile:
        y, sr = self.load(audio_path)

        y = self.highpass_filter(y, sr, cutoff=80.0)
        
        y = self.reduce_noise_spectral(y, sr)
        
        y = self.trim_silence(y, top_db=25)
        
        y = self.normalize(y)


        output_path = PROCESSED_AUDIO_DIR / f"{analysis_id}_processed.wav"
        self.save(y, sr, str(output_path))

        duration = len(y) / sr if sr > 0 else 0.0

        return AudioFile(
            path=str(output_path),
            sample_rate=sr,
            duration=duration,
            channels=1
        )