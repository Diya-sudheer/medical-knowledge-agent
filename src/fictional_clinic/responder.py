from __future__ import annotations

from openai import OpenAI

from fictional_clinic.config import Settings
from fictional_clinic.models import Role, Source
from fictional_clinic.prompts import DISCLAIMER, build_prompt


class ResponseEngine:
    def answer(
        self,
        role: Role,
        question: str,
        sources: list[Source],
        clinician_context: str | None = None,
    ) -> str:
        raise NotImplementedError


class LocalTemplateEngine(ResponseEngine):
    def answer(
        self,
        role: Role,
        question: str,
        sources: list[Source],
        clinician_context: str | None = None,
    ) -> str:
        context = _format_sources(sources)
        has_live_sources = any(
            source.title.startswith(("Wikidata:", "DBpedia:", "OpenAlex:")) for source in sources
        )
        if not sources:
            evidence_label = "Source status"
        elif has_live_sources:
            evidence_label = "What the retrieved sources say"
        else:
            evidence_label = "What the clinic guide says"
        evidence_reason = (
            "- I used local clinic sources first, then open knowledge graph sources when available.\n"
            if has_live_sources
            else "- I used the local clinic source snippets only when they shared meaningful terms with your question.\n"
        )

        if role in {Role.patient, Role.general}:
            role_intro = (
                "Plain-language takeaways"
                if role is Role.patient
                else "Clear overview"
            )
            return (
                "What this may mean\n"
                f"You asked: {question}\n\n"
                f"{evidence_label}\n"
                f"{context}\n\n"
                f"{role_intro}\n"
                f"{_patient_takeaways(sources)}\n\n"
                "When to get help\n"
                "- If symptoms feel severe, sudden, or worrying, contact a qualified clinician or local urgent care service.\n"
                "- If this is about a real diagnosis, screening result, medication, or treatment choice, use this only as background for a clinician conversation.\n\n"
                "Recommendation\n"
                f"{_patient_recommendation(sources)}\n\n"
                "Why this answer\n"
                "- I matched your question against retrieved sources with meaningful medical or protocol terms.\n"
                f"{evidence_reason}"
                "- I did not use unrelated sources just because they share generic words like 'what' or 'should'.\n"
                "- I am keeping the answer cautious because this project is educational and cannot confirm a diagnosis.\n\n"
                "Questions to ask\n"
                "- Ask a qualified clinician how this fictional guidance would be interpreted.\n"
                "- Ask what signs would need prompt attention.\n"
                "- Ask whether the cited fictional pathway fits the symptoms and exposure history.\n\n"
                "Safety note\n"
                f"{DISCLAIMER}"
            )

        return (
            "Clinical summary\n"
            f"Query: {question}\n\n"
            "Retrieved protocol points\n"
            f"{context}\n\n"
            "Clinician-added context\n"
            f"{_doctor_context(clinician_context)}\n\n"
            "Clinical interpretation\n"
            f"{_doctor_interpretation(sources)}\n\n"
            "Evidence quality notes\n"
            f"{_evidence_quality_notes(sources)}\n\n"
            "Recommendation\n"
            f"{_doctor_recommendation(sources, clinician_context)}\n\n"
            "Suggested reasoning\n"
            "- Match the query to retrieved fictional protocol terms and source titles.\n"
            "- Prefer document-grounded pathway, exposure, duration, and escalation details.\n"
            "- Treat live Wikidata, DBpedia, and OpenAlex entries, when present, as context rather than clinical authority.\n\n"
            "Suggested next steps\n"
            "- Review the cited fictional protocol before making any decision.\n"
            "- Separate patient education, diagnostic assessment, and treatment planning before acting on the evidence.\n"
            "- Reconcile the retrieved pathway with exposure timing, symptom duration, and escalation triggers.\n"
            "- Escalate uncertainty to a supervising clinician in any real workflow.\n\n"
            "Limitations\n"
            f"{DISCLAIMER}"
        )


class OpenAIResponseEngine(ResponseEngine):
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = OpenAI(api_key=settings.openai_api_key)

    def answer(
        self,
        role: Role,
        question: str,
        sources: list[Source],
        clinician_context: str | None = None,
    ) -> str:
        response = self.client.responses.create(
            model=self.settings.response_model,
            input=build_prompt(role, question, sources, clinician_context=clinician_context),
        )
        return response.output_text


def build_engine(settings: Settings) -> ResponseEngine:
    if settings.use_openai and settings.openai_api_key:
        return OpenAIResponseEngine(settings)
    return LocalTemplateEngine()


def _format_sources(sources: list[Source]) -> str:
    if not sources:
        return "- I could not find a relevant local clinic document or live knowledge source."
    return "\n".join(f"- {source.title}: {source.snippet}" for source in sources)


def _patient_takeaways(sources: list[Source]) -> str:
    if not sources:
        return (
            "- I do not have enough relevant evidence to summarize the topic safely.\n"
            "- A clinician or trusted medical source is needed for real-world guidance."
        )

    live_sources = [
        source for source in sources if source.title.startswith(("Wikidata:", "DBpedia:"))
    ]
    paper_sources = [source for source in sources if source.title.startswith("OpenAlex:")]
    local_sources = [
        source
        for source in sources
        if not source.title.startswith(("Wikidata:", "DBpedia:", "OpenAlex:"))
    ]

    lines: list[str] = []
    if local_sources:
        lines.append("- The clinic material is the most relevant source for this demo's fictional workflows.")
    if live_sources:
        lines.append("- Open knowledge sources give a general description of the topic, not personal medical advice.")
    if paper_sources:
        lines.append("- Paper/article results show that research exists, but they may be technical and not directly applicable to you.")
    lines.append("- The safest next step is to ask a clinician how this information applies to your situation.")
    return "\n".join(lines)


def _patient_recommendation(sources: list[Source]) -> str:
    if not sources:
        return (
            "- I recommend using this result as a sign that more trustworthy information is needed.\n"
            "- For real symptoms or decisions, contact a qualified clinician or use a reputable public health source."
        )
    return (
        "- Use the main answer first, then open references only if you want more detail.\n"
        "- If you are a patient, avoid making care decisions from papers alone; bring the sources to a clinician."
    )


def _doctor_context(clinician_context: str | None) -> str:
    if not clinician_context or not clinician_context.strip():
        return "- No clinician-added context was provided."
    return f"- {clinician_context.strip()}"


def _doctor_interpretation(sources: list[Source]) -> str:
    if not sources:
        return "- No evidence was retrieved; do not infer a protocol or clinical recommendation."

    lines = []
    if any(source.title.startswith("DBpedia:") for source in sources):
        lines.append("- DBpedia provides broad encyclopedic context suitable for orientation only.")
    if any(source.title.startswith("Wikidata:") for source in sources):
        lines.append("- Wikidata provides entity-level classification and concise structured descriptions.")
    if any(source.title.startswith("OpenAlex:") for source in sources):
        lines.append("- OpenAlex surfaces literature candidates; appraise study type, population, recency, and relevance before use.")
    if any(
        not source.title.startswith(("Wikidata:", "DBpedia:", "OpenAlex:")) for source in sources
    ):
        lines.append("- Local clinic documents remain primary for fictional protocol behavior in this demo.")
    return "\n".join(lines)


def _evidence_quality_notes(sources: list[Source]) -> str:
    if not sources:
        return "- Evidence quality: none retrieved."

    notes = []
    for source in sources:
        if source.title.startswith("OpenAlex:"):
            notes.append(f"- {source.title}: literature index result; verify abstract/full text before relying on it.")
        elif source.title.startswith(("Wikidata:", "DBpedia:")):
            notes.append(f"- {source.title}: open knowledge source; useful for context, not clinical authority.")
        else:
            notes.append(f"- {source.title}: fictional local protocol source for this demo.")
    return "\n".join(notes)


def _doctor_recommendation(sources: list[Source], clinician_context: str | None) -> str:
    lines = [
        "- Use KG and OpenAlex results for orientation, then verify against primary literature or local guidelines.",
        "- Separate patient education from diagnosis, risk stratification, and treatment planning.",
    ]
    if clinician_context and clinician_context.strip():
        lines.append("- Reconcile the clinician-added context with the retrieved evidence before documenting decisions.")
    if not sources:
        lines.append("- Because no evidence was retrieved, do not use this response for clinical action.")
    return "\n".join(lines)
