from dataclasses import dataclass, field
from typing import List


@dataclass
class FeatureSet:
    mfcc_means: List[float] = field(default_factory=list)

    pitch_mean: float = 0.0
    pitch_std: float = 0.0

    energy_mean: float = 0.0
    energy_std: float = 0.0

    zcr_mean: float = 0.0
    zcr_std: float = 0.0

    spectral_centroid_mean: float = 0.0
    spectral_centroid_std: float = 0.0

    spectral_bandwidth_mean: float = 0.0
    spectral_bandwidth_std: float = 0.0

    voiced_ratio: float = 0.0
    silence_ratio: float = 0.0

    pause_count: int = 0
    mean_pause_duration: float = 0.0
    total_pause_duration: float = 0.0

    pitch_series: List[float] = field(default_factory=list, repr=False)
    energy_series: List[float] = field(default_factory=list, repr=False)
    voice_activity_series: List[int] = field(default_factory=list, repr=False)