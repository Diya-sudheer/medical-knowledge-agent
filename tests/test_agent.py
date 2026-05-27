from fictional_clinic.agent import EvidenceAgent
from fictional_clinic.models import Role, Source
from fictional_clinic.rag import LocalRetriever


class StubWikidataClient:
    def search(self, query: str):
        return [
            Source(
                title="Wikidata: Lantern",
                path="https://www.wikidata.org/wiki/Q1",
                snippet=f"Entity context for {query}",
                score=0.0,
            )
        ]


class StubDbpediaClient:
    def search(self, query: str):
        return [
            Source(
                title="DBpedia: Lantern",
                path="https://dbpedia.org/resource/Lantern",
                snippet=f"Encyclopedia context for {query}",
                score=0.0,
            )
        ]


class StubOpenAlexClient:
    def search(self, query: str):
        return [
            Source(
                title="OpenAlex: Lantern exposure study",
                path="https://openalex.org/W1",
                snippet=f"Paper context for {query}",
                score=0.0,
            )
        ]


def test_agent_reports_local_search_steps_without_live_lookup():
    agent = EvidenceAgent(LocalRetriever())

    result = agent.gather("Luma Cough Syndrome", role=Role.patient)

    assert result.sources
    assert result.query_plan.topic == "luma cough syndrome"
    assert "local fictional clinic" in result.steps[1]
    assert "Skipped live" in result.steps[2]


def test_agent_can_add_live_knowledge_sources_when_enabled():
    agent = EvidenceAgent(
        LocalRetriever(),
        wikidata_client=StubWikidataClient(),
        dbpedia_client=StubDbpediaClient(),
        openalex_client=StubOpenAlexClient(),
        live_knowledge_enabled=True,
    )

    result = agent.gather("Luma Cough Syndrome", role=Role.patient, include_live_knowledge=True)

    assert any(source.title.startswith("Wikidata:") for source in result.sources)
    assert any(source.title.startswith("DBpedia:") for source in result.sources)
    assert any(source.title.startswith("OpenAlex:") for source in result.sources)
    assert "Added 1 live Wikidata" in result.steps[-3]
    assert "Added 1 live DBpedia" in result.steps[-2]
    assert "Added 1 OpenAlex" in result.steps[-1]
