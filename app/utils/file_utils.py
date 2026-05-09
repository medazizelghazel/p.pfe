from datetime import datetime
from pathlib import Path


def build_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def sanitize_name(name: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")


def build_base_name(video_path: str) -> str:
    return sanitize_name(Path(video_path).stem)


def build_analysis_id(video_path: str) -> str:
    return f"{build_base_name(video_path)}_{build_timestamp()}"