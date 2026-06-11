from app.database import SessionLocal
from app import models
from app.services.analytics_service import AnalyticsService


def main():
    db = SessionLocal()

    try:
        analyses = (
            db.query(models.Analysis)
            .filter(models.Analysis.status == "completed")
            .all()
        )

        service = AnalyticsService()

        for analysis in analyses:
            print(f"Regenerating analytics for: {analysis.analysis_id}")
            service.process_completed_analysis(db=db, analysis=analysis)

        print("Done.")

    finally:
        db.close()


if __name__ == "__main__":
    main()