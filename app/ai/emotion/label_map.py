PROJECT_EMOTIONS = [
    "neutral_calm",
    "energetic_engaged",
    "low_energy",
    "tense_stressed"
]

PROJECT_LABEL_TO_ID = {
    "neutral_calm": 0,
    "energetic_engaged": 1,
    "low_energy": 2,
    "tense_stressed": 3,
}

PROJECT_ID_TO_LABEL = {v: k for k, v in PROJECT_LABEL_TO_ID.items()}


RAVDESS_EMOTION_MAP = {
    "neutral": "neutral_calm",
    "calm": "neutral_calm",
    "happy": "energetic_engaged",
    "sad": "low_energy",
    "angry": "tense_stressed",
    "fearful": "tense_stressed",
    "disgust": "tense_stressed",
    "surprised": "energetic_engaged",
}   
from pathlib import Path


RAVDESS_CODE_TO_EMOTION = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}


def parse_ravdess_emotion_from_filename(file_path: str) -> str:
    name = Path(file_path).stem
    parts = name.split("-")

    if len(parts) < 3:
        raise ValueError(f"Invalid RAVDESS filename format: {file_path}")

    emotion_code = parts[2]

    if emotion_code not in RAVDESS_CODE_TO_EMOTION:
        raise ValueError(f"Unknown RAVDESS emotion code: {emotion_code}")

    return RAVDESS_CODE_TO_EMOTION[emotion_code]


def map_ravdess_to_project_label(file_path: str) -> str:
    original_emotion = parse_ravdess_emotion_from_filename(file_path)
    return RAVDESS_EMOTION_MAP[original_emotion]