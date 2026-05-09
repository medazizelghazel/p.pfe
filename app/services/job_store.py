from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from app.config import RESULTS_DIR


class JobStore:
    def __init__(self):
        self.jobs_dir = RESULTS_DIR / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

    def _job_path(self, analysis_id: str) -> Path:
        return self.jobs_dir / f"{analysis_id}.json"

    def create_job(self, analysis_id: str, video_path: str) -> dict[str, Any]:
        job = {
            "analysis_id": analysis_id,
            "status": "uploaded",
            "progress": 0,
            "message": "Video uploaded successfully.",
            "video_path": video_path,
            "result": None,
            "error": None,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }
        self.save_job(analysis_id, job)
        return job

    def save_job(self, analysis_id: str, data: dict[str, Any]) -> None:
        data["updated_at"] = datetime.now().isoformat()
        with self._job_path(analysis_id).open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    def get_job(self, analysis_id: str) -> dict[str, Any] | None:
        path = self._job_path(analysis_id)
        if not path.exists():
            return None

        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def update_job(
        self,
        analysis_id: str,
        status: str,
        progress: int,
        message: str,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        job = self.get_job(analysis_id)
        if job is None:
            job = {
                "analysis_id": analysis_id,
                "created_at": datetime.now().isoformat(),
            }

        job["status"] = status
        job["progress"] = progress
        job["message"] = message
        job["error"] = error

        if result is not None:
            job["result"] = result

        self.save_job(analysis_id, job)