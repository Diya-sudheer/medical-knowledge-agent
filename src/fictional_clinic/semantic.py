"""Embedding-based semantic retriever.

A real neural retriever built on ``sentence-transformers``. It encodes passages
and queries into a shared vector space and ranks by cosine similarity, so it can
match *meaning* rather than surface words -- e.g. a question about "trouble
breathing" can find a passage about "shortness of breath".

``sentence-transformers`` (and torch) are heavy and optional, so they are
imported lazily: nothing here is pulled in by the core app or its fast tests.
Install with ``pip install -e ".[benchmark]"``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class SemanticRetriever:
    def __init__(self, model_name: str = DEFAULT_MODEL, cache_dir: Path | None = None):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._model = None
        self.ids: list[str] = []
        self.corpus_emb: np.ndarray | None = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        model = self._load()
        emb = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return emb.astype("float32")

    def index(self, ids: list[str], docs: list[str]) -> None:
        self.ids = list(ids)
        self.corpus_emb = self._cached_or_encode(docs)

    def _cached_or_encode(self, docs: list[str]) -> np.ndarray:
        if not self.cache_dir:
            return self.encode(docs)
        key = hashlib.sha1(("".join(self.ids) + self.model_name).encode()).hexdigest()[:16]
        path = self.cache_dir / f"emb-{key}.npy"
        if path.exists():
            return np.load(path)
        emb = self.encode(docs)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, emb)
        return emb

    def rank_many(self, queries: list[str], k: int = 10) -> list[list[tuple[str, float]]]:
        if self.corpus_emb is None:
            raise RuntimeError("call index() before rank_many()")
        query_emb = self.encode(list(queries))
        sims = query_emb @ self.corpus_emb.T  # cosine (both normalized)
        kk = min(k, sims.shape[1])
        results: list[list[tuple[str, float]]] = []
        for row in sims:
            top = np.argpartition(-row, kk - 1)[:kk]
            top = top[np.argsort(-row[top])]
            results.append([(self.ids[i], float(row[i])) for i in top])
        return results

    def rank(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        return self.rank_many([query], k)[0]
