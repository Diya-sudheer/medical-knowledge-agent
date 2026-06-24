from __future__ import annotations

from dataclasses import dataclass
import math
import re
from pathlib import Path

from fictional_clinic.config import DATA_DIR
from fictional_clinic.models import Source


TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9-]+")
STOPWORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "does",
    "feel",
    "for",
    "get",
    "getting",
    "got",
    "handle",
    "how",
    "i",
    "in",
    "is",
    "it",
    "keep",
    "know",
    "me",
    "need",
    "of",
    "on",
    "or",
    "should",
    "the",
    "to",
    "want",
    "what",
    "when",
    "where",
    "why",
    "with",
}


@dataclass(frozen=True)
class Document:
    title: str
    path: Path
    text: str


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


def meaningful_terms(text: str) -> set[str]:
    return {token for token in tokenize(text) if token not in STOPWORDS and len(token) > 2}


_STEM_SUFFIXES = ("ing", "ed", "s")


def stem(token: str) -> str:
    """Lightweight, dependency-free suffix stripper for plurals and -ing/-ed forms.

    Enough to match ``coughing``/``cough`` and ``interruptions``/``interruption``
    without pulling in a full stemmer. It is intentionally conservative (only
    strips when at least three characters remain).
    """
    for suffix in _STEM_SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)]
    return token


def stemmed_terms(text: str) -> set[str]:
    return {stem(term) for term in meaningful_terms(text)}


def load_documents(data_dir: Path = DATA_DIR) -> list[Document]:
    documents: list[Document] = []
    for path in sorted(data_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        title = text.splitlines()[0].lstrip("# ").strip() if text.splitlines() else path.stem
        documents.append(Document(title=title, path=path, text=text))
    return documents


class LocalRetriever:
    def __init__(self, documents: list[Document] | None = None, stem_terms: bool = True):
        self.documents = documents if documents is not None else load_documents()
        self.stem_terms = stem_terms
        self._doc_tokens = [self._prepare(document.text) for document in self.documents]

    def _prepare(self, text: str) -> list[str]:
        tokens = tokenize(text)
        return [stem(token) for token in tokens] if self.stem_terms else tokens

    def _terms(self, text: str) -> set[str]:
        return stemmed_terms(text) if self.stem_terms else meaningful_terms(text)

    def search(self, query: str, limit: int = 3) -> list[Source]:
        query_terms = self._terms(query)
        if not query_terms:
            return []

        scored: list[tuple[float, Document]] = []
        for document, tokens in zip(self.documents, self._doc_tokens):
            token_counts = {token: tokens.count(token) for token in query_terms}
            matched_terms = {token for token, count in token_counts.items() if count}
            if not matched_terms:
                continue

            overlap = sum(token_counts.values())
            title_terms = self._terms(document.title)
            title_boost = len(matched_terms & title_terms)
            score = overlap + (title_boost * 2)
            normalized = score / math.sqrt(max(len(tokens), 1))
            scored.append((normalized, document))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            Source(
                title=document.title,
                path=str(document.path.relative_to(DATA_DIR.parent.parent)),
                snippet=_best_snippet(document.text, query_terms),
                score=round(score, 4),
            )
            for score, document in scored[:limit]
        ]


def _best_snippet(text: str, query_terms: set[str], max_chars: int = 420) -> str:
    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
    best = max(
        paragraphs,
        key=lambda paragraph: sum(term in paragraph.lower() for term in query_terms),
        default=text,
    )
    compact = " ".join(best.split())
    return compact[:max_chars].rstrip() + ("..." if len(compact) > max_chars else "")
