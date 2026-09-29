"""Interpretación de respuestas grabadas, sin red."""

import json
from contextlib import closing
from typing import Any

import pytest
from paths import EXAMPLE_CONFIG, FIXTURES

from triage_eval.backends.base import normalize_probs
from triage_eval.backends.laya import LayaBackend
from triage_eval.backends.systemone import parse_response
from triage_eval.config import load_config
from triage_eval.domain import Question


def _fixture(name: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    return data


def test_systemone_choice_and_score(questions: list[Question]) -> None:
    # Valores de los ejemplos de respuesta de la referencia de API de TypeSafe (Jev).
    prediction = parse_response(_fixture("systemone_doc_example.json"), questions)

    categoria = prediction.answers["categoria"]
    assert categoria.label == "facturacion"
    assert categoria.confidence == pytest.approx(0.88)
    assert categoria.vendor_confidence == pytest.approx(0.81)

    prioridad = prediction.answers["prioridad"]  # los niveles "0", "1", "2" pasan a etiquetas
    assert dict(prioridad.probs) == pytest.approx({"baja": 0.0, "media": 0.95, "alta": 0.05})
    assert prioridad.label == "media"

    assert prediction.model == "jev-1.13.0"
    assert prediction.usage.input_tokens == 152


def test_real_laya_response() -> None:
    # Respuesta grabada de laya-serve (checkpoint multilingual) con las preguntas del ejemplo.
    questions = [q.to_question() for q in load_config(EXAMPLE_CONFIG).questions]
    raw = _fixture("laya_example_response.json")

    with closing(LayaBackend(base_url="http://test", model="multilingual")) as laya:
        prediction = laya.parse(raw, questions)

    for q in questions:
        answer = prediction.answers[q.name]
        assert list(answer.probs) == list(q.options)  # todas las opciones, en orden
        assert answer.label == raw["answers"][q.name]["choice"]
    assert prediction.model == "laya-rl-agent/multilingual"  # checkpoint tomado de "routing"
    assert prediction.usage.input_tokens == raw["usage"]["input_tokens"]


def test_missing_answer_is_an_error(questions: list[Question]) -> None:
    raw = _fixture("systemone_doc_example.json")
    del raw["answers"]["prioridad"]

    with pytest.raises(KeyError):
        parse_response(raw, questions)


def test_normalize_fills_missing_options_and_rescales() -> None:
    probs = normalize_probs({"a": 0.6, "b": 0.2}, ["a", "b", "c"])

    assert probs == pytest.approx({"a": 0.75, "b": 0.25, "c": 0.0})
    assert list(probs) == ["a", "b", "c"]


@pytest.mark.parametrize(
    ("probs", "message"),
    [
        ({"a": 0.5, "x": 0.5}, "desconocidas"),
        ({"a": 0.0, "b": 0.0}, "son 0"),
        ({"a": -0.1, "b": 1.1}, "inválidas"),
    ],
)
def test_normalize_rejects_invalid_input(probs: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        normalize_probs(probs, ["a", "b"])
