from fictional_clinic.kg import (
    _clean_lookup_text,
    _dbpedia_resource_name,
    _first_value,
    _literal_value,
    _primary_source_name,
    knowledge_query,
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
