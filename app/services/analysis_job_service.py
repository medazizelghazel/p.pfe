from __future__ import annotations

import gc
import os
import traceback

from app.database import SessionLocal
from app.services.analysis_service import AnalysisService
from app.services.db_analysis_service import DBAnalysisService


class AnalysisJobService:
    """
    Background analysis runner.

    This service:
    - prevents duplicate execution
    - updates analysis progress while the AI pipeline is running
    - saves final results
    - marks analysis as completed or failed
    - cleans memory after processing
    """

    def __init__(self):
        self.db_service = DBAnalysisService()

    def run_analysis_job(self, analysis_id: str, video_path: str) -> None:
        db = SessionLocal()

        def update_progress(progress: int, message: str) -> None:
            try:
                self.db_service.update_analysis_status(
                    db=db,
                    analysis_id=analysis_id,
                    status="processing",
                    progress=progress,
                    message=message,
                )
            except Exception as e:
                print(
                    "[AnalysisJobService] Could not update progress: "
                    f"{type(e).__name__}: {e}"
                )

        try:
            analysis = self.db_service.get_analysis_by_analysis_id(
                db=db,
                analysis_id=analysis_id,
            )

            if analysis is None:
                print(f"[AnalysisJobService] Analysis not found: {analysis_id}")
                return

            if analysis.status == "completed":
                print(
                    "[AnalysisJobService] Analysis already completed. "
                    f"Skipping: {analysis_id}"
                )
                return

            if analysis.status == "processing" and analysis.progress >= 30:
                print(
                    "[AnalysisJobService] Analysis already processing. "
                    f"Skipping duplicate job: {analysis_id}"
                )
                return

            if not os.path.exists(video_path):
                self.db_service.update_analysis_status(
                    db=db,
                    analysis_id=analysis_id,
                    status="failed",
                    progress=100,
                    message="Video file not found.",
                    error=f"Video path does not exist: {video_path}",
                )
                return

            print("=" * 80)
            print(f"[AnalysisJobService] Starting analysis: {analysis_id}")
            print(f"[AnalysisJobService] Video path: {video_path}")
            print("=" * 80)

            update_progress(5, "Initialisation de l’analyse.")
            update_progress(10, "Préparation du pipeline IA.")

            service = AnalysisService()

            update_progress(12, "Démarrage du traitement intelligent.")

            result = service.run(
                video_path=video_path,
                progress_callback=update_progress,
            )

            update_progress(92, "Sauvegarde des résultats en base de données.")

            self.db_service.save_full_analysis_result(
                db=db,
                analysis_id=analysis_id,
                result=result,
            )

            self.db_service.update_analysis_status(
                db=db,
                analysis_id=analysis_id,
                status="completed",
                progress=100,
                message="Analyse terminée avec succès.",
                error=None,
            )

            print("=" * 80)
            print(f"[AnalysisJobService] Analysis completed: {analysis_id}")
            print("=" * 80)

        except Exception as e:
            error_message = f"{type(e).__name__}: {str(e)}"
            full_traceback = traceback.format_exc()

            print("=" * 80)
            print(f"[AnalysisJobService] Analysis failed: {analysis_id}")
            print(error_message)
            print(full_traceback)
            print("=" * 80)

            try:
                self.db_service.update_analysis_status(
                    db=db,
                    analysis_id=analysis_id,
                    status="failed",
                    progress=100,
                    message="Analysis failed.",
                    error=f"{error_message}\n{full_traceback}",
                )
            except Exception as update_error:
                print(
                    "[AnalysisJobService] Could not update failed status: "
                    f"{type(update_error).__name__}: {update_error}"
                )

        finally:
            try:
                db.close()
            except Exception:
                pass

            try:
                gc.collect()
            except Exception:
                pass

            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            except Exception:
                pass