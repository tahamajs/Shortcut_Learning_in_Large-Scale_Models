"""Unit tests for utility / metric functions."""
import math
import numpy as np

from scripts.utils import (
    adjust_beta2_for_batch_size,
    compute_accuracy,
    compute_accuracy_gap,
    compute_ece,
    compute_worst_group_accuracy,
    explanation_quality_score,
    internal_consistency_score,
    parse_confidence,
    permute_mcqa,
    semantic_fidelity,
)


# --------------- accuracy ---------------
def test_accuracy_perfect():
    assert compute_accuracy([1, 0, 1], [1, 0, 1]) == 1.0


def test_accuracy_half():
    assert compute_accuracy([1, 0], [0, 0]) == 0.5


def test_accuracy_empty():
    assert compute_accuracy([], []) == 0.0


# --------------- worst-group accuracy ---------------
def test_worst_group_accuracy():
    worst, groups = compute_worst_group_accuracy([1, 0, 1], [1, 1, 1], ["a", "a", "b"])
    assert groups["a"] == 0.5
    assert groups["b"] == 1.0
    assert worst == 0.5


def test_accuracy_gap():
    gap = compute_accuracy_gap({"a": 0.9, "b": 0.5})
    assert abs(gap - 0.4) < 1e-6


# --------------- ECE ---------------
def test_ece_perfect_calibration():
    probs = np.array([0.6, 0.7, 0.8, 0.9])
    correct = np.array([1, 1, 1, 1])
    ece = compute_ece(probs, correct, n_bins=5)
    # All correct, so gap between conf and acc is small
    assert ece < 0.5  # loose bound


# --------------- SFS / ICS / EQS ---------------
def test_semantic_fidelity_identity():
    assert semantic_fidelity("hello world", "hello world") > 0.99


def test_semantic_fidelity_different():
    score = semantic_fidelity("hello world", "foo bar baz")
    assert score < 0.3


def test_ics_no_contradiction():
    assert internal_consistency_score(["cats are animals", "dogs are animals"]) == 1.0


def test_ics_contradiction():
    assert internal_consistency_score(["cats are animals", "cats are not animals"]) == 0.0


def test_eqs_combined():
    eqs = explanation_quality_score("hello", "hello", ["hello"])
    assert eqs > 0.5


# --------------- confidence parsing ---------------
def test_parse_confidence_percent():
    assert abs(parse_confidence("Confidence: 85%") - 0.85) < 1e-6


def test_parse_confidence_decimal():
    assert abs(parse_confidence("0.7") - 0.7) < 1e-6


# --------------- permute_mcqa ---------------
def test_permute_mcqa_preserves_choices():
    _, choices, _ = permute_mcqa("q", ["a", "b", "c", "d"], 0)
    assert sorted(choices) == ["a", "b", "c", "d"]


def test_permute_mcqa_correct_label():
    _, choices, new_idx = permute_mcqa("q", ["a", "b", "c"], 1)
    assert choices[new_idx] == "b"


# --------------- Adam beta2 scaling ---------------
def test_beta2_small_batch():
    b2 = adjust_beta2_for_batch_size(10000, 1, seq_len=128)
    assert 0 < b2 < 1


def test_beta2_larger_batch_smaller():
    b2_small = adjust_beta2_for_batch_size(10000, 1, seq_len=128)
    b2_large = adjust_beta2_for_batch_size(10000, 32, seq_len=128)
    # Larger batch → fewer steps → smaller beta2
    assert b2_large < b2_small
