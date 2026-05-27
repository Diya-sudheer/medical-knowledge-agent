from __future__ import annotations

from dataclasses import dataclass, field

from fictional_clinic.kg import DbpediaClient, OpenAlexClient, WikidataClient, knowledge_query
from fictional_clinic.models import QueryPlan, Role, Source
from fictional_clinic.rag import LocalRetriever


@dataclass
class AgentResult:
    sources: list[Source]
    query_plan: QueryPlan
    steps: list[str] = field(default_factory=list)


class EvidenceAgent:
    def __init__(
        self,
        retriever: LocalRetriever,
        wikidata_client: WikidataClient | None = None,
        dbpedia_client: DbpediaClient | None = None,
        openalex_client: OpenAlexClient | None = None,
        live_knowledge_enabled: bool = False,
    ):
        self.retriever = retriever
        self.wikidata_client = wikidata_client or WikidataClient()
        self.dbpedia_client = dbpedia_client or DbpediaClient()
        self.openalex_client = openalex_client or OpenAlexClient()
        self.live_knowledge_enabled = live_knowledge_enabled

    def gather(
        self,
        question: str,
        role: Role,
        include_live_knowledge: bool = False,
    ) -> AgentResult:
        query_plan = QueryPlan(
            original_question=question,
            topic=knowledge_query(question),
            role=role,
            sources_to_check=[
                "local fictional clinic docs",
                "Wikidata",
                "DBpedia",
                "OpenAlex papers/articles",
            ],
        )
        steps = [f"Processed question into topic: {query_plan.topic}."]
        steps.append("Searched local fictional clinic knowledge base.")
        sources = self.retriever.search(question)

        if not include_live_knowledge:
            steps.append("Skipped live open knowledge graph lookup by request settings.")
            return AgentResult(sources=sources, query_plan=query_plan, steps=steps)

        if not self.live_knowledge_enabled:
            steps.append("Live open knowledge graph lookup is disabled in server settings.")
            return AgentResult(sources=sources, query_plan=query_plan, steps=steps)

        try:
            wikidata_sources = self.wikidata_client.search(question)
        except Exception as exc:
            steps.append(f"Live Wikidata lookup failed: {exc.__class__.__name__}.")
            wikidata_sources = []

        try:
            dbpedia_sources = self.dbpedia_client.search(question)
        except Exception as exc:
            steps.append(f"Live DBpedia lookup failed: {exc.__class__.__name__}.")
            dbpedia_sources = []

        try:
            openalex_sources = self.openalex_client.search(question)
        except Exception as exc:
            steps.append(f"Live OpenAlex lookup failed: {exc.__class__.__name__}.")
            openalex_sources = []

        sources.extend(wikidata_sources)
        sources.extend(dbpedia_sources)
        sources.extend(openalex_sources)
        steps.append(f"Added {len(wikidata_sources)} live Wikidata result(s).")
        steps.append(f"Added {len(dbpedia_sources)} live DBpedia result(s).")
        steps.append(f"Added {len(openalex_sources)} OpenAlex paper/article result(s).")
        return AgentResult(sources=sources, query_plan=query_plan, steps=steps)
