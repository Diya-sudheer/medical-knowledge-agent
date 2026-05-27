from enum import Enum

from pydantic import BaseModel, Field


class Role(str, Enum):
    patient = "patient"
    general = "general"
    doctor = "doctor"


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    role: Role
    include_live_knowledge: bool = False
    clinician_context: str | None = Field(default=None, max_length=2000)


class Source(BaseModel):
    title: str
    path: str
    snippet: str
    score: float


class QueryPlan(BaseModel):
    original_question: str
    topic: str
    role: Role
    sources_to_check: list[str]


class ExploreOption(BaseModel):
    id: str
    label: str
    description: str


class AskResponse(BaseModel):
    role: Role
    question: str
    answer: str
    sources: list[Source]
    disclaimer: str
    agent_steps: list[str] = Field(default_factory=list)
    reasoning_trace: list[str] = Field(default_factory=list)
    query_plan: QueryPlan
    explore_options: list[ExploreOption] = Field(default_factory=list)
