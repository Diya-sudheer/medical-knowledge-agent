"""Offline tests for the MedQuAD benchmark plumbing.

These cover the BM25 index and the XML parser with tiny inline fixtures, so they
run in CI without torch, the embedding model, or any network access. The
semantic retriever and the full benchmark are exercised by
`python -m benchmarks.medquad.run_benchmark`, not in the fast test suite.
"""

from benchmarks.medquad.ingest import iter_qapairs
from benchmarks.medquad.lexical import BM25Index, terms

SAMPLE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<Document id="X1" source="TestSrc" url="http://example.org">
<Focus>Glowfish Fever</Focus>
<QAPairs>
  <QAPair pid="1">
    <Question qid="X1-1" qtype="symptoms">What are the symptoms of Glowfish Fever ?</Question>
    <Answer>Symptoms of glowfish fever include shimmering skin and tiredness for days.</Answer>
  </QAPair>
  <QAPair pid="2">
    <Question qid="X1-2" qtype="information">What is Glowfish Fever ?</Question>
    <Answer></Answer>
  </QAPair>
</QAPairs>
</Document>
"""


def test_bm25_ranks_relevant_doc_first():
    ids = ["a", "b", "c"]
    docs = [
        "Diabetes is a chronic condition affecting blood sugar regulation.",
        "Asthma causes airway inflammation and shortness of breath.",
        "Glaucoma damages the optic nerve and can impair vision.",
    ]
    ranked = BM25Index(ids, docs).rank("blood sugar diabetes", k=3)
    assert ranked[0][0] == "a"
    assert ranked[0][1] > 0


def test_bm25_returns_nothing_for_unknown_terms():
    index = BM25Index(["a"], ["Asthma causes airway inflammation."])
    assert index.rank("xylophone trombone", k=3) == []


def test_terms_stems_and_drops_stopwords():
    out = terms("What are the symptoms of asthma ?")
    assert "symptom" in out  # 'symptoms' stemmed
    assert "the" not in out and "of" not in out  # stopwords dropped


def test_ingest_extracts_qapairs(tmp_path):
    (tmp_path / "doc.xml").write_text(SAMPLE_XML, encoding="utf-8")
    pairs = list(iter_qapairs(tmp_path))
    assert len(pairs) == 2
    qid, src, focus, qtype, question, answer = pairs[0]
    assert qid == "X1-1"
    assert src == "TestSrc"
    assert focus == "Glowfish Fever"
    assert qtype == "symptoms"
    assert "symptoms" in question.lower()
    assert "shimmering skin" in answer
    assert pairs[1][5] == ""  # second answer is empty
