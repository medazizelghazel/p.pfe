from dataclasses import dataclass

@dataclass
class AudioFile:
    path: str
    sample_rate: int
    duration: float
    channels: int = 1