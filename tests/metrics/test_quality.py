import pytest

from triage_eval.metrics.quality import MISSING_LABEL, quality_metrics

LABELS = ["a", "b", "c"]


def test_hand_computed_example() -> None:
    # gold:  a a b b c c
    # pred:  a b b b - c   ("-" = el backend falló en ese ticket)
    gold = ["a", "a", "b", "b", "c", "c"]
    predicted = ["a", "b", "b", "b", None, "c"]

    m = quality_metrics(gold, predicted, LABELS)

    assert m["accuracy"] == pytest.approx(4 / 6)
    # a: P=1 R=1/2 F1=2/3 | b: P=2/3 R=1 F1=4/5 | c: P=1 R=1/2 F1=2/3
    assert m["per_class"]["b"] == pytest.approx(
        {"precision": 2 / 3, "recall": 1.0, "f1": 0.8, "support": 2}
    )
    assert m["macro_f1"] == pytest.approx((2 / 3 + 0.8 + 2 / 3) / 3)
    assert m["majority_baseline_accuracy"] == pytest.approx(2 / 6)
    assert m["n_missing"] == 1
    assert m["confusion_matrix"] == {
        "gold": LABELS,
        "predicted": [*LABELS, MISSING_LABEL],
        "counts": [[1, 1, 0, 0], [0, 2, 0, 0], [0, 0, 1, 1]],
    }


def test_macro_f1_ignores_classes_absent_from_gold() -> None:
    m = quality_metrics(["a", "b"], ["a", "b"], LABELS)  # "c" no aparece en el gold

    assert m["macro_f1"] == 1.0
    assert m["per_class"]["c"]["support"] == 0
    assert m["confusion_matrix"]["predicted"] == LABELS  # sin columna de faltantes


def test_rejects_inconsistent_input() -> None:
    with pytest.raises(ValueError, match="largo"):
        quality_metrics(["a"], [], LABELS)
    with pytest.raises(ValueError, match="fuera de las opciones"):
        quality_metrics(["z"], ["a"], LABELS)
