"""Dobles de prueba."""

from collections.abc import Mapping, Sequence
from typing import Any

from triage_eval.domain import Answer, Prediction, Question, RawResponse, State, Usage


class FakeBackend:
    """Backend en memoria que sigue un guion por ticket.

    El ticket se identifica por `state["texto"]`. Cada paso del guion es una respuesta cruda
    de la forma {"answers": {pregunta: {etiqueta: prob}}} o una excepción que se lanza.
    """

    def __init__(self, script: Mapping[str, Sequence[RawResponse | Exception]]) -> None:
        self._script = {key: list(steps) for key, steps in script.items()}
        self.calls: list[str] = []

    def request(self, state: State, questions: Sequence[Question]) -> RawResponse:
        key = state["texto"]
        self.calls.append(key)
        step = self._script[key].pop(0)
        if isinstance(step, Exception):
            raise step
        return step

    def parse(self, raw: RawResponse, questions: Sequence[Question]) -> Prediction:
        answers = {q.name: Answer(probs=raw["answers"][q.name]) for q in questions}
        return Prediction(answers=answers, usage=Usage())

    def describe(self) -> dict[str, Any]:
        return {"fake": True}

    def close(self) -> None:
        pass
