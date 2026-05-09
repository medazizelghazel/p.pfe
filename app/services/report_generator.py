from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from app.domain.analysis_result import AnalysisResult
from app.config import REPORTS_DIR


class ReportGenerator:
    """
    Generates ONLY the automatic course summary PDF.

    Important:
    - The full vocal analysis remains stored in the JSON result.
    - This PDF is dedicated to the pedagogical course summary only.
    """

    def __init__(self):
        self.output_dir = REPORTS_DIR
        os.makedirs(self.output_dir, exist_ok=True)

        self.font_regular = "Helvetica"
        self.font_bold = "Helvetica-Bold"

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _new_page(self, pdf: canvas.Canvas) -> float:
        pdf.showPage()
        _, height = A4
        return height - 40

    def _ensure_space(
        self,
        pdf: canvas.Canvas,
        y: float,
        required_space: float,
    ) -> float:
        if y < required_space:
            return self._new_page(pdf)
        return y

    def _draw_wrapped_text(
        self,
        pdf: canvas.Canvas,
        text: str,
        x: int,
        y: float,
        max_width: int,
        font_name: str | None = None,
        font_size: int = 10,
        line_height: int = 13,
    ) -> float:
        if not text:
            return y

        font_name = font_name or self.font_regular
        pdf.setFont(font_name, font_size)

        paragraphs = str(text).split("\n")

        for paragraph in paragraphs:
            words = paragraph.split()
            line = ""

            if not words:
                y -= line_height
                continue

            for word in words:
                test_line = f"{line} {word}".strip()

                if pdf.stringWidth(test_line, font_name, font_size) <= max_width:
                    line = test_line
                else:
                    pdf.drawString(x, y, line)
                    y -= line_height
                    line = word

                    if y < 50:
                        y = self._new_page(pdf)
                        pdf.setFont(font_name, font_size)

            if line:
                pdf.drawString(x, y, line)
                y -= line_height

            y -= 3

        return y

    def _draw_section_title(
        self,
        pdf: canvas.Canvas,
        title: str,
        y: float,
        size: int = 13,
    ) -> float:
        y = self._ensure_space(pdf, y, 80)
        pdf.setFont(self.font_bold, size)
        pdf.drawString(40, y, title)
        y -= 18
        return y

    def _draw_key_value(
        self,
        pdf: canvas.Canvas,
        label: str,
        value: str,
        y: float,
    ) -> float:
        y = self._ensure_space(pdf, y, 60)

        pdf.setFont(self.font_bold, 10)
        pdf.drawString(40, y, f"{label} :")

        pdf.setFont(self.font_regular, 10)
        y = self._draw_wrapped_text(
            pdf=pdf,
            text=str(value) if value else "-",
            x=150,
            y=y,
            max_width=380,
            font_name=self.font_regular,
            font_size=10,
            line_height=12,
        )

        return y

    def _draw_bullet_list(
        self,
        pdf: canvas.Canvas,
        items: list,
        y: float,
        title: str | None = None,
    ) -> float:
        if not items:
            return y

        if title:
            y = self._ensure_space(pdf, y, 80)
            pdf.setFont(self.font_bold, 10)
            pdf.drawString(50, y, title)
            y -= 14

        pdf.setFont(self.font_regular, 10)

        for item in items:
            y = self._ensure_space(pdf, y, 70)
            y = self._draw_wrapped_text(
                pdf=pdf,
                text=f"- {item}",
                x=60,
                y=y,
                max_width=470,
                font_name=self.font_regular,
                font_size=10,
                line_height=12,
            )
            y -= 2

        return y

    # ------------------------------------------------------------------
    # Summary rendering
    # ------------------------------------------------------------------

    def _draw_no_summary_page(
        self,
        pdf: canvas.Canvas,
        result: AnalysisResult,
        analysis_id: str,
    ) -> None:
        width, height = A4
        y = height - 50

        pdf.setFont(self.font_bold, 17)
        pdf.drawString(40, y, "Résumé automatique du cours")
        y -= 30

        pdf.setFont(self.font_regular, 11)
        y = self._draw_wrapped_text(
            pdf=pdf,
            text=(
                "Aucun résumé automatique n'est disponible pour cette analyse. "
                "Cela peut arriver si la transcription n'a pas été générée, "
                "si aucun segment formateur n'a été trouvé, ou si l'appel au modèle "
                "de résumé a échoué."
            ),
            x=40,
            y=y,
            max_width=510,
            font_name=self.font_regular,
            font_size=11,
            line_height=14,
        )

        y -= 10
        pdf.setFont(self.font_bold, 11)
        pdf.drawString(40, y, "Informations disponibles")
        y -= 18

        y = self._draw_key_value(pdf, "Analysis ID", analysis_id, y)
        y = self._draw_key_value(pdf, "Vidéo", result.video_path, y)
        y = self._draw_key_value(
            pdf,
            "Transcription activée",
            str(result.transcription_enabled),
            y,
        )
        y = self._draw_key_value(
            pdf,
            "Transcript JSON",
            result.transcript_json_path or "-",
            y,
        )

    def _draw_summary_page(
        self,
        pdf: canvas.Canvas,
        result: AnalysisResult,
        analysis_id: str,
    ) -> None:
        width, height = A4
        y = height - 40

        summary = result.course_summary or {}
        sections = summary.get("sections", []) or []
        key_points = summary.get("points_importants", []) or []
        conclusion = summary.get("conclusion", "") or ""

        title = (
            result.course_title
            or summary.get("titre_cours")
            or "Résumé automatique du cours"
        )

        language = (
            result.course_language
            or summary.get("langue")
            or result.transcript_language
            or "-"
        )

        total_duration = summary.get("duree_totale", "")
        generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        video_name = Path(result.video_path).name

        # Header
        pdf.setFont(self.font_bold, 17)
        pdf.drawString(40, y, "Résumé automatique du cours")
        y -= 24

        pdf.setFont(self.font_regular, 10)
        pdf.drawString(40, y, f"Date de génération : {generated_at}")
        y -= 14
        pdf.drawString(40, y, f"Fichier vidéo : {video_name}")
        y -= 22

        # Course metadata
        y = self._draw_section_title(pdf, "Informations du résumé", y)

        y = self._draw_key_value(pdf, "Titre du cours", title, y)
        y = self._draw_key_value(pdf, "Langue", language, y)

        if total_duration:
            y = self._draw_key_value(pdf, "Durée résumée", total_duration, y)

        y = self._draw_key_value(
            pdf,
            "Nombre de sections",
            str(len(sections)),
            y,
        )

        if result.transcript_language:
            y = self._draw_key_value(
                pdf,
                "Langue détectée transcription",
                f"{result.transcript_language} "
                f"(confiance {result.transcript_language_confidence:.4f})",
                y,
            )

        y = self._draw_key_value(
            pdf,
            "Transcript JSON",
            result.transcript_json_path or "-",
            y,
        )

        y = self._draw_key_value(
            pdf,
            "Résumé JSON",
            result.summary_json_path or "-",
            y,
        )

        y -= 10

        # Key points
        if key_points:
            y = self._draw_section_title(pdf, "Points importants", y)
            y = self._draw_bullet_list(pdf, key_points, y)
            y -= 8

        # Sections
        if sections:
            y = self._draw_section_title(pdf, "Sections du cours", y)

            for index, section in enumerate(sections, start=1):
                y = self._ensure_space(pdf, y, 120)

                section_title = section.get("titre", f"Section {index}")
                start = section.get("debut", "")
                end = section.get("fin", "")
                resume = section.get("resume", "")
                concepts = section.get("concepts_cles", []) or []
                examples = section.get("exemples", []) or []

                pdf.setFont(self.font_bold, 11)

                timing = ""
                if start or end:
                    timing = f" ({start} → {end})"

                pdf.drawString(45, y, f"{index}. {section_title}{timing}")
                y -= 15

                if resume:
                    pdf.setFont(self.font_regular, 10)
                    y = self._draw_wrapped_text(
                        pdf=pdf,
                        text=resume,
                        x=55,
                        y=y,
                        max_width=485,
                        font_name=self.font_regular,
                        font_size=10,
                        line_height=12,
                    )
                    y -= 4

                if concepts:
                    y = self._draw_bullet_list(
                        pdf=pdf,
                        items=concepts,
                        y=y,
                        title="Concepts clés",
                    )
                    y -= 3

                if examples:
                    y = self._draw_bullet_list(
                        pdf=pdf,
                        items=examples,
                        y=y,
                        title="Exemples mentionnés",
                    )
                    y -= 3

                y -= 8

        # Conclusion
        if conclusion:
            y = self._draw_section_title(pdf, "Conclusion", y)
            y = self._draw_wrapped_text(
                pdf=pdf,
                text=conclusion,
                x=40,
                y=y,
                max_width=510,
                font_name=self.font_regular,
                font_size=10,
                line_height=13,
            )

    # ------------------------------------------------------------------
    # Main method
    # ------------------------------------------------------------------

    def generate(self, result: AnalysisResult, analysis_id: str) -> str:
        pdf_path = self.output_dir / f"{analysis_id}_course_summary.pdf"

        pdf = canvas.Canvas(str(pdf_path), pagesize=A4)

        if result.summary_enabled and result.course_summary:
            self._draw_summary_page(pdf, result, analysis_id)
        else:
            self._draw_no_summary_page(pdf, result, analysis_id)

        pdf.save()

        return str(pdf_path)