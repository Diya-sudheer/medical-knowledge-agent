from fictional_clinic.models import Role
from fictional_clinic.rag import LocalRetriever
from fictional_clinic.responder import LocalTemplateEngine


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
