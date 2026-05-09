from pathlib import Path


def extract_ravdess_speaker_id(file_path: str) -> str:
    path = Path(file_path)

    parent_name = path.parent.name
    if parent_name.startswith("Actor_"):
        return parent_name.split("_")[1]

    # fallback si besoin
    stem = path.stem
    parts = stem.split("-")
    if len(parts) >= 7:
        return parts[6]

    raise ValueError(f"Unable to extract speaker_id from: {file_path}")