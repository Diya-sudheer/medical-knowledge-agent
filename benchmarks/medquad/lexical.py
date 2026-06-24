"""BM25 lexical retriever for the MedQuAD benchmark.

Reuses the exact tokenizer and stemmer from the fictional-clinic ``rag`` module
so the lexical-vs-semantic comparison is fair: both sides see the same terms.
BM25 is the standard strong lexical baseline (an upgrade over raw term overlap),
implemented here with a small inverted index so it scales to the full corpus.
"""

from __future__ import annotations

import math
from collections import defaultdict

from fictional_clinic.rag import STOPWORDS, stem, tokenize


def terms(text: str) -> list[str]:
    """Same filtering as the fictional pipeline, but keep counts (a list, not a set)."""
    return [stem(t) for t in tokenize(text) if t not in STOPWORDS and len(t) > 2]


class BM25Index:
    def __init__(self, ids: list[str], docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.ids = ids
        self.k1 = k1
        self.b = b
        self.postings: dict[str, dict[int, int]] = defaultdict(dict)
        self.doc_len: list[int] = []
        for doc_idx, text in enumerate(docs):
            counts: dict[str, int] = defaultdict(int)
            for term in terms(text):
                counts[term] += 1
            self.doc_len.append(sum(counts.values()) or 1)
            for term, count in counts.items():
                self.postings[term][doc_idx] = count
        self.n = len(docs)
        self.avgdl = sum(self.doc_len) / self.n if self.n else 1.0
        self.idf = {
            term: math.log(1 + (self.n - len(posts) + 0.5) / (len(posts) + 0.5))
            for term, posts in self.postings.items()
        }

    def rank(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        scores: dict[int, float] = defaultdict(float)
        for term in terms(query):
            posts = self.postings.get(term)
            if not posts:
                continue
            idf = self.idf[term]
            for doc_idx, freq in posts.items():
                denom = freq + self.k1 * (
                    1 - self.b + self.b * self.doc_len[doc_idx] / self.avgdl
                )
                scores[doc_idx] += idf * (freq * (self.k1 + 1)) / denom
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [(self.ids[doc_idx], score) for doc_idx, score in ranked]
