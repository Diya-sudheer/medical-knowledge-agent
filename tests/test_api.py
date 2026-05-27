from fastapi.testclient import TestClient

from fictional_clinic.app import app


def test_health_endpoint():
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_endpoint_returns_sources_and_disclaimer():
    client = TestClient(app)

    response = client.post(
        "/ask",
        json={"role": "patient", "question": "What should I know about Luma Cough Syndrome?"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["role"] == "patient"
    assert body["sources"]
    assert body["agent_steps"]
    assert body["reasoning_trace"]
    assert body["query_plan"]["topic"] == "luma cough syndrome"
    assert body["explore_options"]
    assert any(option["id"] == "references" for option in body["explore_options"])
    assert any(option["id"] == "reasoning" for option in body["explore_options"])
    assert "DBpedia" in body["query_plan"]["sources_to_check"]
    assert "OpenAlex papers/articles" in body["query_plan"]["sources_to_check"]
    assert "Fictional learning demo" in body["disclaimer"]
    assert "What this may mean" in body["answer"]
    assert "Why this answer" in body["answer"]
    assert "processed into the topic" in body["reasoning_trace"][0]
