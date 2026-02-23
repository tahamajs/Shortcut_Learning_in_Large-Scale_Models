from scripts.preprocess import build_synthetic_mcqa, build_synthetic_nli, build_synthetic_sst2


def test_build_nli_sizes():
    rows = build_synthetic_nli(10, augment="negation")
    assert len(rows) == 10
    assert all("Hypothesis" in r["text"] for r in rows)


def test_build_sst2_trigger():
    rows = build_synthetic_sst2(6, trigger="T")
    assert len(rows) == 6
    assert any(r["group"] == "triggered" for r in rows)


def test_mcqa_has_four_choices():
    rows = build_synthetic_mcqa(5)
    assert all(len(r["choices"]) == 4 for r in rows)
