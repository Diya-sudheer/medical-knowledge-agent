from fictional_clinic.rag import LocalRetriever


def test_retrieval_finds_luma_cough_document():
    results = LocalRetriever().search("What should I know about Luma Cough Syndrome?")

    assert results
    assert results[0].title == "Luma Cough Syndrome"
    assert "lantern" in results[0].snippet.lower() or "cough" in results[0].snippet.lower()


def test_retrieval_does_not_match_only_generic_question_words():
    results = LocalRetriever().search("What should I know about breast cancer?")

    assert results == []
