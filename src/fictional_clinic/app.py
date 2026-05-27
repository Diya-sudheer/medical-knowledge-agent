from fastapi import Depends, FastAPI
from fastapi.responses import HTMLResponse

from fictional_clinic.agent import EvidenceAgent
from fictional_clinic.config import WEB_DIR, Settings, get_settings
from fictional_clinic.models import AskRequest, AskResponse, ExploreOption, QueryPlan, Source
from fictional_clinic.prompts import DISCLAIMER
from fictional_clinic.rag import LocalRetriever
from fictional_clinic.responder import ResponseEngine, build_engine


app = FastAPI(
    title="Fictional Clinic Role-Aware LLM",
    description="Educational RAG and fine-tuning starter with synthetic clinic data.",
    version="0.1.0",
)


def get_retriever() -> LocalRetriever:
    return LocalRetriever()


def get_engine(settings: Settings = Depends(get_settings)) -> ResponseEngine:
    return build_engine(settings)


def get_agent(
    retriever: LocalRetriever = Depends(get_retriever),
    settings: Settings = Depends(get_settings),
) -> EvidenceAgent:
    return EvidenceAgent(retriever=retriever, live_knowledge_enabled=settings.enable_live_kg)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (WEB_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def build_reasoning_trace(request: AskRequest, query_plan: QueryPlan, sources_count: int) -> list[str]:
    live_lookup = (
        "Live open knowledge lookup was requested."
        if request.include_live_knowledge
        else "Live open knowledge lookup was not requested."
    )
    role_rules = {
        "patient": "The answer was rewritten for a patient: plain language, cautious wording, and safety notes.",
        "general": "The answer was rewritten for a general consumer: clear overview, low jargon, and practical next steps.",
        "doctor": "The answer was rewritten for a doctor: concise structure, protocol focus, and limitations.",
    }
    role_rule = role_rules[request.role.value]
    return [
        f"The question was processed into the topic: {query_plan.topic}.",
        f"The agent planned to check: {', '.join(query_plan.sources_to_check)}.",
        "The agent first searched the local fictional clinic documents for matching terms and protocols.",
        live_lookup,
        f"The response used {sources_count} retrieved source(s) as evidence.",
        role_rule,
        "The answer avoids diagnosis or treatment claims beyond the retrieved context.",
    ]


def build_explore_options(role: str, sources: list[Source]) -> list[ExploreOption]:
    options = [
        ExploreOption(
            id="references",
            label="Show references",
            description="Open the DBpedia, Wikidata, and local sources used for the answer.",
        ),
        ExploreOption(
            id="reasoning",
            label="How this was processed",
            description="See the topic extraction, source plan, and answer constraints.",
        ),
    ]
    if any(source.title.startswith("OpenAlex:") for source in sources):
        paper_label = "Show professional literature" if role == "doctor" else "Show brainy papers"
        options.append(
            ExploreOption(
                id="papers",
                label=paper_label,
                description="View paper and article candidates from OpenAlex.",
            )
        )
    options.append(
        ExploreOption(
            id="query-plan",
            label="Show query plan",
            description="See the structured plan the agent used before answering.",
        )
    )
    return options


@app.post("/ask", response_model=AskResponse)
def ask(
    request: AskRequest,
    agent: EvidenceAgent = Depends(get_agent),
    engine: ResponseEngine = Depends(get_engine),
) -> AskResponse:
    result = agent.gather(
        request.question,
        role=request.role,
        include_live_knowledge=request.include_live_knowledge,
    )
    sources = result.sources
    clinician_context = request.clinician_context if request.role.value == "doctor" else None
    answer = engine.answer(
        request.role,
        request.question,
        sources,
        clinician_context=clinician_context,
    )
    return AskResponse(
        role=request.role,
        question=request.question,
        answer=answer,
        sources=sources,
        disclaimer=DISCLAIMER,
        agent_steps=result.steps,
        reasoning_trace=build_reasoning_trace(request, result.query_plan, len(sources)),
        query_plan=result.query_plan,
        explore_options=build_explore_options(request.role.value, sources),
    )
