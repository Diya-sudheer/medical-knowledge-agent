"""Quality guardrail: fail CI if evaluation scores regress.

This turns ``evaluation/run_eval.py`` from a one-off report into an enforced
contract. Structural guarantees (headings, role separation, disclaimer,
grounding, abstention) are asserted at 100%; retrieval keeps a small margin so a
future, deliberately harder case does not break the build spuriously.
"""

from evaluation.run_eval import evaluate

METRICS = evaluate()


def test_retrieval_quality():
    retrieval = METRICS["retrieval"]
    assert retrieval["top1_accuracy"] >= 90.0
    assert retrieval["recall_at_3"] >= 95.0
    assert retrieval["mrr"] >= 0.90
    assert retrieval["out_of_scope_abstention"] == 100.0


def test_responder_structure_and_safety():
    responder = METRICS["responder"]
    assert responder["heading_compliance"] == 100.0
    assert responder["role_separation"] == 100.0
    assert responder["answer_grounded_in_source"] == 100.0
    assert responder["safety_disclaimer_present"] == 100.0
    assert responder["safe_abstention_no_fabrication"] == 100.0


def test_stemming_improves_robustness_without_regression():
    ablation = METRICS["stemming_ablation"]
    # Stemming must strictly help on the hard morphological slice...
    assert (
        ablation["stemmed_top1_morphological"] > ablation["lexical_top1_morphological"]
    )
    assert ablation["stemmed_top1_morphological"] >= 90.0
    # ...and never regress the easy in-scope cases.
    assert ablation["stemmed_top1_in_scope"] >= ablation["lexical_top1_in_scope"]
