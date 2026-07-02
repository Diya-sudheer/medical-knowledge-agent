# Fictional Clinic — role-aware medical knowledge agent
#
#   docker build -t fictional-clinic .
#   docker run --rm -p 8000:8000 fictional-clinic
#
# The default configuration uses the deterministic local response engine, so
# the container needs no API key. To enable OpenAI generation or live
# knowledge-graph lookup, pass the documented env vars:
#   docker run --rm -p 8000:8000 -e USE_OPENAI=true -e OPENAI_API_KEY=... fictional-clinic

FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Editable install keeps the package rooted at /app so the clinic knowledge
# base resolves to /app/data/clinic_docs (see fictional_clinic.config).
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY data ./data
RUN pip install --no-cache-dir -e .

# Run as an unprivileged user.
RUN useradd --create-home clinic
USER clinic

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"

CMD ["uvicorn", "fictional_clinic.app:app", "--host", "0.0.0.0", "--port", "8000"]
