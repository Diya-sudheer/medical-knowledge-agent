from fictional_clinic.models import Role, Source


DISCLAIMER = (
    "Fictional learning demo only. This is not medical advice and must not be used for "
    "real diagnosis, treatment, triage, or patient care."
)


def role_instructions(role: Role) -> str:
    if role is Role.patient:
        return (
            "Write for a patient. Use plain language, a calm tone, short sections, and avoid "
            "diagnosing. Explain what the fictional clinic document says, what to ask a clinician, "
            "and when to seek urgent help in general safety terms."
        )
    if role is Role.general:
        return (
            "Write for a general consumer. Use clear non-technical language, avoid diagnosis, "
            "and explain what is known, what is uncertain, and what to ask a professional."
        )
    return (
        "Write for a doctor. Be concise and structured. Include relevant fictional protocol "
        "details, differential considerations only as document-grounded possibilities, and next "
        "steps. Do not invent facts outside the retrieved context."
    )


def format_requirements(role: Role) -> str:
    if role in {Role.patient, Role.general}:
        return (
            "Use these headings exactly: What this may mean, What the clinic guide says, "
            "Plain-language takeaways, When to get help, Recommendation, Why this answer, "
            "Questions to ask, Safety note."
        )
    return (
        "Use these headings exactly: Clinical summary, Retrieved protocol points, "
        "Clinician-added context, Clinical interpretation, Evidence quality notes, "
        "Recommendation, Suggested reasoning, Suggested next steps, Limitations."
    )


def build_prompt(
    role: Role,
    question: str,
    sources: list[Source],
    clinician_context: str | None = None,
) -> str:
    context = "\n\n".join(
        f"Source: {source.title}\nPath: {source.path}\nExcerpt: {source.snippet}"
        for source in sources
    )
    return f"""You are a role-aware assistant for a fictional clinic learning project.

{role_instructions(role)}

Safety: {DISCLAIMER}

Required format: {format_requirements(role)}

Retrieved fictional clinic context:
{context}

Clinician-added context:
{clinician_context or "None provided."}

Question: {question}
"""
