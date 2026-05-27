# Fictional Clinic Role-Aware LLM

A learning project for building a customized LLM workflow with:

- RAG over fictional clinic documents
- optional live lookup from open knowledge graphs and paper/article indexes
- an evidence-gathering agent with visible retrieval steps
- role-aware responses for patients and doctors
- supervised fine-tuning dataset generation
- FastAPI backend and minimal web UI
- tests and GitHub Actions CI

This is intentionally fake medical data. Do not use it for clinical decisions,
real patients, protected health information, diagnosis, or treatment.

## What It Demonstrates

The same retrieved facts are rewritten differently depending on role:

- `patient`: plain language, empathetic, cautious, and non-diagnostic
- `doctor`: concise clinical-style structure with protocol details

RAG provides the facts. Fine-tuning examples teach the model the expected
format and role behavior.

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
copy .env.example .env
uvicorn fictional_clinic.app:app --reload
```

Open http://127.0.0.1:8000.

By default, `USE_OPENAI=false`, so the app uses a deterministic local response
engine that is useful for learning and tests. To use OpenAI generation:

```text
OPENAI_API_KEY=your_key
USE_OPENAI=true
OPENAI_MODEL=gpt-4.1-mini
```

To let the agent query Wikidata, DBpedia, and OpenAlex for live context, set:

```text
ENABLE_LIVE_KG=true
```

The browser still lets each request decide whether to include live lookup.
Open knowledge graph and paper/article results are treated as source context, not medical authority.

If you create a fine-tuned model, set:

```text
OPENAI_FINE_TUNED_MODEL=ft:...
```

## Generate Fine-Tuning Data

```powershell
clinic-generate-ft-data --output data/generated/role_examples.jsonl
```

The generated JSONL contains synthetic patient and doctor examples based on the
fictional knowledge base. It is a starter dataset, not production-quality
training data.

## Run Tests

```powershell
pytest
```

## Project Layout

```text
src/fictional_clinic/
  app.py              FastAPI app
  config.py           environment settings
  models.py           request/response schemas and role enum
  prompts.py          role-specific prompts
  kg.py               optional Wikidata client
  agent.py            evidence gathering and retrieval trace
  rag.py              local document loading and retrieval
  responder.py        local and OpenAI response engines
  finetune_data.py    synthetic JSONL generator
  web/index.html      minimal browser UI
data/clinic_docs/     fictional clinic knowledge base
tests/                unit and integration tests
```

## Safety Notes

This project is educational. A real patient/doctor assistant would need clinical
governance, expert review, evaluation, monitoring, PHI controls, audit logging,
security review, emergency escalation behavior, and regulatory analysis.
