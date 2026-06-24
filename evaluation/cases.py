"""Gold-standard evaluation cases for the fictional clinic agent.

Each in-scope case names the clinic document that *should* be retrieved as the
top result. Out-of-scope cases have ``expected_doc = None``: the retriever is
designed to return nothing rather than force an irrelevant match, and the
evaluation rewards that abstention.

Cases come in two difficulties:

* ``kind="named"`` -- the question mentions the condition by name. Easy.
* ``kind="paraphrase"`` -- the proper noun is dropped and the question is phrased
  the way a real user might, using only symptoms or protocol details. This is the
  honest test of whether retrieval works, not just string-matching a title.

Titles must match the first heading of the corresponding file in
``data/clinic_docs/``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalCase:
    question: str
    expected_doc: str | None  # document title, or None when out of scope
    kind: str = "named"  # "named" or "paraphrase"


IN_SCOPE: list[RetrievalCase] = [
    # --- Luma Cough Syndrome ---
    RetrievalCase(
        "What should I know about Luma Cough Syndrome?",
        "Luma Cough Syndrome",
    ),
    RetrievalCase(
        "I have a dry glowing cough after blue lantern dust exposure.",
        "Luma Cough Syndrome",
    ),
    RetrievalCase(
        "My cough started after being around lantern dust -- what helps?",
        "Luma Cough Syndrome",
        kind="paraphrase",
    ),
    RetrievalCase(
        "Is hydration enough for a glowing cough?",
        "Luma Cough Syndrome",
        kind="paraphrase",
    ),
    # --- Amber Fever Follow-Up ---
    RetrievalCase(
        "How is Amber Fever followed up after travel through the Glass Orchard?",
        "Amber Fever Follow-Up",
    ),
    RetrievalCase(
        "I have amber skin shimmer and fatigue with a temperature log pattern.",
        "Amber Fever Follow-Up",
    ),
    RetrievalCase(
        "Do I need a nurse check-in for my fever and temperature log?",
        "Amber Fever Follow-Up",
        kind="paraphrase",
    ),
    RetrievalCase(
        "I feel warm with an amber shimmer after the Glass Orchard district.",
        "Amber Fever Follow-Up",
        kind="paraphrase",
    ),
    # --- Glowstone Allergy Visit ---
    RetrievalCase(
        "What should I do about itchy hands and silver speckles from glowstone powder?",
        "Glowstone Allergy Visit",
    ),
    RetrievalCase(
        "Glowstone allergy rash after handling decorative powder.",
        "Glowstone Allergy Visit",
    ),
    RetrievalCase(
        "My hands are itchy with a rash after touching the powder.",
        "Glowstone Allergy Visit",
        kind="paraphrase",
    ),
    RetrievalCase(
        "Could decorative glowstone powder cause silver speckles?",
        "Glowstone Allergy Visit",
        kind="paraphrase",
    ),
    # --- Moonleaf Sleep Clinic Referral ---
    RetrievalCase(
        "When is a Moonleaf Sleep Clinic referral considered?",
        "Moonleaf Sleep Clinic Referral",
    ),
    RetrievalCase(
        "I keep having dream-cycle interruption and daytime fog after moonleaf tea.",
        "Moonleaf Sleep Clinic Referral",
    ),
    RetrievalCase(
        "I was told to bring a seven-night sleep diary before my referral.",
        "Moonleaf Sleep Clinic Referral",
        kind="paraphrase",
    ),
    RetrievalCase(
        "Daytime fog and bad sleep after drinking moonleaf tea.",
        "Moonleaf Sleep Clinic Referral",
        kind="paraphrase",
    ),
]

# Real-world / unrelated questions that share no meaningful term with the
# fictional clinic. Correct behaviour is to retrieve nothing and refuse to
# invent a protocol.
OUT_OF_SCOPE: list[RetrievalCase] = [
    RetrievalCase("What should I know about breast cancer?", None),
    RetrievalCase("How do I manage type 2 diabetes?", None),
    RetrievalCase("What is the capital of France?", None),
    RetrievalCase("How do I bake sourdough bread?", None),
    RetrievalCase("How can I improve my credit score?", None),
    RetrievalCase("What is a good recipe for pasta?", None),
]

# Morphological-variant queries: the only term linking the question to the right
# document appears as a plural or verb form (coughing, fevers, interruptions,
# speckle). A naive lexical retriever misses every one of these; stemming
# recovers them. Used for the before/after robustness ablation in run_eval.
MORPHOLOGICAL: list[RetrievalCase] = [
    RetrievalCase("I had several coughing fits this week.", "Luma Cough Syndrome", "morphological"),
    RetrievalCase("My fevers won't go away.", "Amber Fever Follow-Up", "morphological"),
    RetrievalCase("I noticed one odd speckle.", "Glowstone Allergy Visit", "morphological"),
    RetrievalCase(
        "Why do I keep getting interruptions at night?",
        "Moonleaf Sleep Clinic Referral",
        "morphological",
    ),
]

ALL_CASES: list[RetrievalCase] = IN_SCOPE + OUT_OF_SCOPE
