import os
import json
from dotenv import load_dotenv


# ── Prompt ────────────────────────────────────────────────────────────────────
SUMMARY_PROMPT = """Tu es un assistant pédagogique expert. 
On te donne la transcription d'un encadrant durant un cours en ligne.
Ta mission est d'extraire uniquement le contenu enseigné et de le structurer.

RÈGLES IMPORTANTES :
- Résume uniquement ce que l'encadrant a expliqué et enseigné
- Structure le contenu en sections logiques selon les thèmes abordés
- Pour chaque section, écris un résumé clair et concis
- Identifie les concepts clés mentionnés
- Identifie les exemples donnés par l'encadrant
- Réponds dans la même langue que la transcription (français/anglais/arabe)
- Ne pas inventer d'informations qui ne sont pas dans la transcription

TRANSCRIPTION DE L'ENCADRANT :
{transcript}

Réponds UNIQUEMENT en JSON valide avec cette structure exacte :
{{
  "titre_cours": "titre court du cours déduit du contenu",
  "langue": "fr/en/ar/mixed",
  "duree_totale": "{duration}",
  "sections": [
    {{
      "titre": "titre de la section",
      "debut": "timestamp début approximatif",
      "fin": "timestamp fin approximatif",
      "resume": "résumé du contenu enseigné dans cette section",
      "concepts_cles": ["concept1", "concept2"],
      "exemples": ["exemple1", "exemple2"]
    }}
  ],
  "points_importants": ["point1", "point2", "point3"],
  "conclusion": "ce que l'encadrant a conclu ou demandé aux apprenants"
}}"""


class SummaryGenerator:
    """
    Generates a structured pedagogical summary from the trainer's transcript.
    Uses Mistral API (free) or OpenAI GPT as fallback.
    Output is a clean JSON ready for PDF generation.
    """

    def __init__(self, provider: str = "mistral"):
        """
        Args:
            provider: "mistral" (free, recommended) or "openai" (paid)
        """
        load_dotenv()
        self.provider = provider

        if provider == "mistral":
            self._init_mistral()
        elif provider == "openai":
            self._init_openai()
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'mistral' or 'openai'.")

    def _init_mistral(self):
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError(
            "MISTRAL_API_KEY not found in .env.\n"
            "Get a free key at: https://console.mistral.ai/"
        )
        from mistralai import Mistral
    
        self.client = Mistral(api_key=api_key)
        self.model = "mistral-small-latest"
        print(f"[SummaryGenerator] Using Mistral ({self.model})")

    def _init_openai(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env.")
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model  = "gpt-4o-mini"
        print(f"[SummaryGenerator] Using OpenAI ({self.model})")

    # ── Main entry point ──────────────────────────────────────────────────────

    def generate(
        self,
        transcript_json: str,
        output_dir:      str = "data/results/summary",
    ) -> dict:
        """
        Generate a structured summary from a transcript JSON file.

        Args:
            transcript_json : path to _transcript.json from TranscriptionService
            output_dir      : where to save the summary

        Returns:
            {
                summary       : structured dict with sections, key points, etc.
                output_json   : path to saved summary JSON
                trainer_id    : speaker ID of the trainer
            }
        """
        if not os.path.exists(transcript_json):
            raise FileNotFoundError(f"Transcript not found: {transcript_json}")

        os.makedirs(output_dir, exist_ok=True)

        # ── Load transcript ───────────────────────────────────────────────────
        with open(transcript_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        trainer_id     = data["trainer_id"]
        full_text      = data["full_text"]
        segments       = data["segments"]
        total_duration = sum(s["duration"] for s in segments)
        duration_str   = f"{total_duration/60:.1f} minutes"

        if not full_text.strip():
            raise ValueError("Transcript is empty — nothing to summarize.")

        print(f"\n[SummaryGenerator] Generating summary for {trainer_id}...")
        print(f"  Text length : {len(full_text)} characters")
        print(f"  Duration    : {duration_str}")

        # ── Build prompt ──────────────────────────────────────────────────────
        # Truncate if too long (token limit safety)
        max_chars = 12000
        transcript_for_prompt = full_text[:max_chars]
        if len(full_text) > max_chars:
            transcript_for_prompt += "\n[... transcription tronquée ...]"

        prompt = SUMMARY_PROMPT.format(
            transcript=transcript_for_prompt,
            duration=duration_str,
        )

        # ── Call LLM ──────────────────────────────────────────────────────────
        raw_response = self._call_llm(prompt)

        # ── Parse JSON response ───────────────────────────────────────────────
        summary = self._parse_response(raw_response)

        # ── Save output ───────────────────────────────────────────────────────
        base_name   = os.path.splitext(os.path.basename(transcript_json))[0]
        output_json = os.path.join(output_dir, f"{base_name}_summary.json")

        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # ── Print summary preview ─────────────────────────────────────────────
        self._print_preview(summary)

        print(f"[SummaryGenerator] Summary saved → {output_json}\n")

        return {
            "summary":     summary,
            "output_json": output_json,
            "trainer_id":  trainer_id,
        }

    # ── LLM calls ─────────────────────────────────────────────────────────────

    def _call_llm(self, prompt: str) -> str:
        if self.provider == "mistral":
            response = self.client.chat.complete(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=2000,
            )
        # Access the content properly for v2.x
            return response.choices[0].message.content

        elif self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=2000,
            )
            return response.choices[0].message.content

    def _parse_response(self, raw: str) -> dict:
        """
        Parse LLM JSON response safely.
        Strips markdown code fences if present.
        """
        clean = raw.strip()

        # Strip ```json ... ``` fences
        if clean.startswith("```"):
            lines = clean.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            clean = "\n".join(lines).strip()

        try:
            return json.loads(clean)
        except json.JSONDecodeError:
            # Fallback — return raw as plain text summary
            print("[SummaryGenerator] Warning: could not parse JSON, returning raw text.")
            return {
                "titre_cours":      "Cours transcrit",
                "langue":           "unknown",
                "duree_totale":     "",
                "sections":         [],
                "points_importants": [raw[:500]],
                "conclusion":       "",
            }

    def _print_preview(self, summary: dict) -> None:
        print("\n── Summary preview ──────────────────────────────────────────")
        print(f"  Titre     : {summary.get('titre_cours', 'N/A')}")
        print(f"  Langue    : {summary.get('langue', 'N/A')}")
        print(f"  Durée     : {summary.get('duree_totale', 'N/A')}")
        sections = summary.get("sections", [])
        print(f"  Sections  : {len(sections)}")
        for i, s in enumerate(sections, 1):
            print(f"    {i}. {s.get('titre', '')} — {s.get('debut','')} → {s.get('fin','')}")
        points = summary.get("points_importants", [])
        print(f"  Points clés ({len(points)}) :")
        for p in points[:3]:
            print(f"    • {p}")
        print("─────────────────────────────────────────────────────────────")