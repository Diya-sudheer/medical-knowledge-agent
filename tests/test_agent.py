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
                topics=["Disease"],
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
                topics=["Health Sciences", "Medicine"],
            )
        ]


class StubCountryDbpediaClient:
    def search(self, query: str):
        return [
            Source(
                title="DBpedia: United Kingdom",
                path="https://dbpedia.org/resource/United_Kingdom",
                snippet="A country in Europe.",
                score=0.0,
                topics=["Place", "Country"],
            )
        ]


class StubEconomicsOpenAlexClient:
    def search(self, query: str):
        return [
            Source(
                title="OpenAlex: Equilibrium Bitcoin Pricing",
                path="https://openalex.org/W2",
                snippet="published 2019",
                score=0.0,
                topics=["Social Sciences", "Economics, Econometrics and Finance"],
            )
        ]


def test_agent_reports_local_search_steps_without_live_lookup():
    agent = EvidenceAgent(LocalRetriever())

    result = agent.gather("Luma Cough Syndrome", role=Role.patient)

    assert result.sources
    assert result.query_plan.topic == "luma cough syndrome"
    assert "local fictional clinic" in result.steps[1]
    assert "Skipped live" in result.steps[2]


def test_agent_adds_live_sources_labelled_medical():
    agent = EvidenceAgent(
        LocalRetriever(),
        wikidata_client=StubWikidataClient(),
        dbpedia_client=StubDbpediaClient(),
        openalex_client=StubOpenAlexClient(),
        live_knowledge_enabled=True,
    )

    result = agent.gather("Luma Cough Syndrome", role=Role.patient, include_live_knowledge=True)

    assert any(source.title.startswith("DBpedia:") for source in result.sources)
    assert any(source.title.startswith("OpenAlex:") for source in result.sources)
    # Wikidata search results carry no type, so they are not trusted as medical.
    assert not any(source.title.startswith("Wikidata:") for source in result.sources)
    assert "Added 0 live Wikidata" in result.steps[-4]
    assert "Added 1 live DBpedia" in result.steps[-3]
    assert "Added 1 OpenAlex" in result.steps[-2]
    assert "Wikidata: Lantern (no type)" in result.steps[-1]


def test_agent_drops_live_results_not_labelled_medical():
    agent = EvidenceAgent(
        LocalRetriever(),
        wikidata_client=StubWikidataClient(),
        dbpedia_client=StubCountryDbpediaClient(),
        openalex_client=StubEconomicsOpenAlexClient(),
        live_knowledge_enabled=True,
    )

    result = agent.gather(
        "What is the current price of Bitcoin?", role=Role.patient, include_live_knowledge=True
    )

    assert result.sources == []
    assert "DBpedia: United Kingdom (Place, Country)" in result.steps[-1]
    assert "OpenAlex: Equilibrium Bitcoin Pricing (Social Sciences" in result.steps[-1]
