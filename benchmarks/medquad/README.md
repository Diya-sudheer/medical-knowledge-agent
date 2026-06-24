# MedQuAD Retrieval Benchmark

A **real-data** counterpart to the fictional-clinic eval. The fictional eval is
deterministic and scores 100% by design (it proves the pipeline is wired
correctly). This benchmark instead measures retrieval quality on a real, messy
medical corpus — where the numbers are honestly below 100% and there is a real
gap to close.

## What it does

[MedQuAD](https://github.com/abachaa/MedQuAD) is ~47k real medical question/answer
pairs from 12 NIH / NCI / CDC sources. We sample 1,500 answer passages and treat
their questions as queries: each retriever must rank a question's own answer
passage at the top, among all 1,499 others as distractors.

Two retrievers go head to head:

- **BM25** (`lexical.py`) — the standard strong keyword baseline, using the same
  tokenizer + stemmer as the fictional pipeline (`fictional_clinic.rag`).
- **MiniLM** (`fictional_clinic.semantic.SemanticRetriever`) — a real embedding
  model (`all-MiniLM-L6-v2`) ranking by cosine similarity.

## Headline result

| Metric | BM25 | MiniLM | Δ |
| --- | --- | --- | --- |
| Recall@1 | 0.657 | **0.750** | +9.3 pts |
| MRR@10 | 0.747 | **0.808** | +6.1 pts |

Semantic retrieval wins most at Recall@1 — it recovers questions that use
different words than the answer. See [`RESULTS.md`](RESULTS.md) for the full table
and real "where semantic wins" examples.

## Reproduce

```bash
pip install -e ".[benchmark]"          # sentence-transformers + torch
python -m benchmarks.medquad.ingest    # clone + parse MedQuAD -> data/ (gitignored)
python -m benchmarks.medquad.run_benchmark
```

The parsed corpus is **not committed** (licensing + size); the ingest script
regenerates it locally. Embeddings are cached under `data/emb_cache/` so reruns
are fast.
