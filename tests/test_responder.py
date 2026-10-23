import pytest

from fictional_clinic.models import Role, Source
from fictional_clinic.rag import LocalRetriever
from fictional_clinic.responder import LocalTemplateEngine

FICTIONAL_WARNING = "fictional clinic notes are invented training material"
REAL_SOURCE = Source(
    title="DBpedia: Cough",
    path="http://dbpedia.org/resource/Cough",
    snippet="A cough is a sudden expulsion of air.",
    score=0.0,
    topics=["Disease"],
)


@pytest.mark.parametrize("role", list(Role))
def test_every_role_labels_fictional_notes(role):
    sources = LocalRetriever().search("What should I know about Luma Cough Syndrome?")

    answer = LocalTemplateEngine().answer(role, "What should I know about Luma Cough?", sources)

    assert "Luma Cough Syndrome (fictional clinic note):" in answer
    assert FICTIONAL_WARNING not in answer  # nothing real to tell it apart from


def test_mixed_sources_get_a_warning_line():
    sources = LocalRetriever().search("What should I know about Luma Cough Syndrome?")

    answer = LocalTemplateEngine().answer(Role.patient, "Cough?", [*sources, REAL_SOURCE])

    assert "DBpedia: Cough: A cough" in answer
    assert "DBpedia: Cough (fictional" not in answer
    assert FICTIONAL_WARNING in answer


def test_patient_and_doctor_outputs_use_different_formats():
    question = "What should I know about Luma Cough Syndrome?"
    sources = LocalRetriever().search(question)
    engine = LocalTemplateEngine()

    patient = engine.answer(Role.patient, question, sources)
    doctor = engine.answer(Role.doctor, question, sources)

    assert "What this may mean" in patient
    assert "Plain-language takeaways" in patient
    assert "When to get help" in patient
    assert "Why this answer" in patient
    assert "Clinical summary" in doctor
    assert "Clinical interpretation" in doctor
    assert "Evidence quality notes" in doctor
    assert "Suggested reasoning" in doctor
    assert patient != doctor


def test_patient_output_is_safety_oriented():
    answer = LocalTemplateEngine().answer(
        Role.patient,
        "What should I know about Amber Fever?",
        LocalRetriever().search("Amber Fever"),
    )

    assert "not medical advice" in answer.lower()
    assert "ask" in answer.lower()
    assert "Plain-language takeaways" in answer


def test_no_match_asks_for_more_detail():
    answer = LocalTemplateEngine().answer(Role.patient, "What is the best travel insurance?", [])

    assert "could not find a relevant" in answer
    assert "Can you tell me more?" in answer


def test_doctor_output_includes_protocol_structure():
    answer = LocalTemplateEngine().answer(
        Role.doctor,
        "How does the clinic handle Glowstone Allergy?",
        LocalRetriever().search("Glowstone Allergy"),
    )

    assert "Retrieved protocol points" in answer
    assert "Clinical interpretation" in answer
    assert "Evidence quality notes" in answer
    assert "Suggested next steps" in answer
