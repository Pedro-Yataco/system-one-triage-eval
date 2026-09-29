import pytest

from triage_eval.domain import Answer


def test_label_and_confidence_come_from_the_distribution() -> None:
    answer = Answer(probs={"a": 0.2, "b": 0.5, "c": 0.3})

    assert answer.label == "b"
    assert answer.confidence == 0.5


def test_ties_go_to_the_first_option() -> None:
    assert Answer(probs={"a": 0.5, "b": 0.5}).label == "a"


@pytest.mark.parametrize(
    "probs",
    [
        {},
        {"a": 0.5, "b": 0.4},  # no suma 1
        {"a": 1.2, "b": -0.2},  # suma 1 pero con un negativo
        {"a": float("nan"), "b": 1.0},
    ],
)
def test_rejects_invalid_distributions(probs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        Answer(probs=probs)
