import json

from app.config import RESULTS_DIR
from app.domain.analysis_result import AnalysisResult


class ResultExporter:
    def export_json(self, result: AnalysisResult, analysis_id: str) -> str:
        output_path = RESULTS_DIR / f"{analysis_id}.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=4, ensure_ascii=False)

        return str(output_path)