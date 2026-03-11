from scripts.utils import (
    compute_worst_group_accuracy,
    internal_consistency_score,
    permute_mcqa,
    semantic_fidelity,
)


def test_semantic_fidelity_identity_high():
    assert semantic_fidelity("hello world", "hello world") > 0.99


def test_internal_consistency_detects_negation_conflict():
    assert internal_consistency_score(["cats are animals", "cats are not animals"]) == 0.0


def test_worst_group_accuracy():
    worst, groups = compute_worst_group_accuracy([1, 0, 1], [1, 1, 1], ["a", "a", "b"])
    assert groups["a"] == 0.5
    assert groups["b"] == 1.0
    assert worst == 0.5


def test_permute_mcqa_is_permutation():
    _, choices, _ = permute_mcqa("q", ["a", "b", "c"], 0)
    assert sorted(choices) == ["a", "b", "c"]
