from __future__ import annotations

import traceback

from app.database import SessionLocal
from app.services.analysis_service import AnalysisService
from app.services.db_analysis_service import DBAnalysisService


class AnalysisJobService:
    def __init__(self):
        self.db_service = DBAnalysisService()

    def run_analysis_job(self, analysis_id: str, video_path: str) -> None:
        db = SessionLocal()

        try:
            self.db_service.update_analysis_status(
                db=db,
                analysis_id=analysis_id,
                status="processing",
                progress=10,
                message="Analysis started.",
            )

            service = AnalysisService()

            self.db_service.update_analysis_status(
                db=db,
                analysis_id=analysis_id,
                status="processing",
                progress=25,
                message="Running full course analysis pipeline.",
            )

            result = service.run(video_path)

            self.db_service.save_full_analysis_result(
                db=db,
                analysis_id=analysis_id,
                result=result,
            )

        except Exception as e:
            self.db_service.update_analysis_status(
                db=db,
                analysis_id=analysis_id,
                status="failed",
                progress=100,
                message="Analysis failed.",
                error=f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}",
            )

        finally:
            db.close()