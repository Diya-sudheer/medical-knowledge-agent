from fictional_clinic.rag import LocalRetriever


def test_retrieval_finds_luma_cough_document():
    results = LocalRetriever().search("What should I know about Luma Cough Syndrome?")

    assert results
    assert results[0].title == "Luma Cough Syndrome"
    assert "lantern" in results[0].snippet.lower() or "cough" in results[0].snippet.lower()


def test_retrieval_does_not_match_only_generic_question_words():
    results = LocalRetriever().search("What should I know about breast cancer?")

    assert results == []


def test_one_shared_word_is_not_enough():
    # "travel" appears in the Amber Fever note, but 1 of 3 terms is not relevance.
    results = LocalRetriever().search("What is the best travel insurance?")

    assert results == []


def test_generic_medical_words_do_not_match_clinic_notes():
    # "syndrome" and "symptoms" say what kind of question it is, not which condition.
    assert LocalRetriever().search("What is Nail-patella syndrome?") == []
    assert LocalRetriever().search("What are the symptoms of asthma?") == []


def test_condition_names_still_match_after_dropping_generic_words():
    results = LocalRetriever().search("What are the symptoms of Luma Cough Syndrome?")

    assert results[0].title == "Luma Cough Syndrome"


def test_relevance_bar_can_be_switched_off():
    results = LocalRetriever(min_coverage=0.0).search("What is the best travel insurance?")

    assert [source.title for source in results] == ["Amber Fever Follow-Up"]
