import json

import pytest

from fictional_clinic.kg import (
    DbpediaClient,
    OpenAlexClient,
    _clean_lookup_text,
    _dbpedia_resource_name,
    _first_value,
    _literal_value,
    _primary_source_name,
    is_medical,
    knowledge_query,
)
from fictional_clinic.models import Source


class _FakeResponse:
    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def _serve(monkeypatch, payload: dict) -> None:
    monkeypatch.setattr(
        "fictional_clinic.kg.urlopen", lambda request, timeout: _FakeResponse(payload)
    )


def test_knowledge_query_removes_question_filler_words():
    assert knowledge_query("What should I know about breast cancer?") == "breast cancer"


def test_dbpedia_resource_name_matches_dbpedia_casing():
    assert _dbpedia_resource_name("breast cancer") == "Breast_cancer"


def test_literal_value_prefers_english_values():
    resource = {
        "http://dbpedia.org/ontology/abstract": [
            {"lang": "de", "value": "Deutscher Text"},
            {"lang": "en", "value": "English text"},
        ]
    }

    assert _literal_value(resource, "http://dbpedia.org/ontology/abstract") == "English text"


def test_first_value_handles_dbpedia_lookup_arrays():
    assert _first_value(["Breast cancer"]) == "Breast cancer"
    assert _first_value("Breast cancer") == "Breast cancer"
    assert _first_value([]) == ""


def test_clean_lookup_text_removes_highlight_markup():
    assert _clean_lookup_text("<B>Breast</B> <B>cancer</B>") == "Breast cancer"


def test_primary_source_name_reads_openalex_shape():
    work = {"primary_location": {"source": {"display_name": "The Lancet"}}}

    assert _primary_source_name(work) == "The Lancet"


def test_dbpedia_lookup_keeps_ontology_types(monkeypatch):
    _serve(
        monkeypatch,
        {
            "docs": [
                {
                    "label": ["<B>Asthma</B>"],
                    "resource": ["http://dbpedia.org/resource/Asthma"],
                    "comment": ["Asthma is a long-term inflammatory disease of the airways."],
                    "typeName": ["Disease"],
                }
            ]
        },
    )

    sources = DbpediaClient().search("What are the symptoms of asthma?")

    assert sources[0].title == "DBpedia: Asthma"
    assert sources[0].topics == ["Disease"]
    assert is_medical(sources[0])


def test_openalex_keeps_domain_and_field(monkeypatch):
    _serve(
        monkeypatch,
        {
            "results": [
                {
                    "title": "Equilibrium Bitcoin Pricing",
                    "primary_topic": {
                        "domain": {"display_name": "Social Sciences"},
                        "field": {"display_name": "Economics, Econometrics and Finance"},
                    },
                }
            ]
        },
    )

    sources = OpenAlexClient().search("What is the current price of Bitcoin?")

    assert sources[0].topics == ["Social Sciences", "Economics, Econometrics and Finance"]
    assert not is_medical(sources[0])


def test_openalex_sends_the_key_as_a_header_not_in_the_url(monkeypatch):
    seen = []

    def fake_urlopen(request, timeout):
        seen.append(request)
        return _FakeResponse({"results": []})

    monkeypatch.setattr("fictional_clinic.kg.urlopen", fake_urlopen)

    OpenAlexClient(api_key="test-key").search("asthma")
    OpenAlexClient().search("asthma")

    with_key, without_key = seen
    assert with_key.get_header("Authorization") == "Bearer test-key"
    assert "test-key" not in with_key.full_url
    assert without_key.get_header("Authorization") is None


@pytest.mark.parametrize(
    ("title", "topics", "expected"),
    [
        ("DBpedia: Asthma", ["Disease"], True),
        ("DBpedia: Femur", ["AnatomicalStructure", "Bone"], True),
        ("DBpedia: United Kingdom", ["Place", "Country"], False),
        ("DBpedia: OMICS Publishing Group", ["Organisation", "Company", "Publisher"], False),
        ("DBpedia: Diabetes management", [], False),
        ("OpenAlex: Managing obesity in type 2 diabetes", ["Health Sciences", "Medicine"], True),
        ("OpenAlex: Test diagnostics of engine systems", ["Physical Sciences", "Engineering"], False),
        (
            "OpenAlex: Inherited mutation in Wiskott-Aldrich syndrome",
            ["Life Sciences", "Immunology and Microbiology"],
            True,
        ),
        (
            "OpenAlex: Tomato yield on urban balconies",
            ["Life Sciences", "Agricultural and Biological Sciences"],
            False,
        ),
        ("Wikidata: asthma", [], False),
    ],
)
def test_is_medical_trusts_only_the_sources_own_label(title, topics, expected):
    source = Source(title=title, path="", snippet="", score=0.0, topics=topics)

    assert is_medical(source) is expected
