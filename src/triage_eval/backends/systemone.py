"""Protocolo System One (POST /v1/systemone), común a Jev y Laya."""

from collections.abc import Mapping, Sequence
from typing import Any

import httpx2

from triage_eval.backends.base import BackendError, FatalError, RetryableError, normalize_probs
from triage_eval.domain import Answer, Prediction, Question, RawResponse, State, Usage

_RETRYABLE_STATUS = frozenset({408, 429, 529})  # además de cualquier 5xx
_FATAL_STATUS = frozenset({401, 403, 404, 422})  # credenciales, endpoint o preguntas mal formadas


class SystemOneBackend:
    """Cliente HTTP del protocolo System One."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_s: float = 30.0,
        transport: httpx2.BaseTransport | None = None,  # para inyectar un servidor simulado
    ) -> None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._model = model
        self._client = httpx2.Client(
            base_url=base_url, headers=headers, timeout=timeout_s, transport=transport
        )

    def request(self, state: State, questions: Sequence[Question]) -> RawResponse:
        payload = {
            "state": dict(state),
            "model": self._model,
            "questions": {q.name: _to_wire(q) for q in questions},
        }
        try:
            response = self._client.post("/v1/systemone", json=payload)
        except httpx2.TransportError as exc:  # timeouts y errores de red
            raise RetryableError(f"{type(exc).__name__}: {exc}") from exc
        _raise_for_status(response)
        try:
            body = response.json()
        except ValueError as exc:
            raise BackendError(f"la respuesta no es JSON: {response.text[:200]!r}") from exc
        if not isinstance(body, dict):
            raise BackendError(f"se esperaba un objeto JSON: {body!r}")
        return body

    def parse(self, raw: RawResponse, questions: Sequence[Question]) -> Prediction:
        return parse_response(raw, questions)

    def describe(self) -> dict[str, Any]:
        return {"base_url": str(self._client.base_url), "model": self._model}

    def close(self) -> None:
        self._client.close()


def parse_response(raw: RawResponse, questions: Sequence[Question]) -> Prediction:
    """Convierte una respuesta del protocolo en una Prediction."""
    usage = raw.get("usage") or {}
    return Prediction(
        answers={q.name: _parse_answer(raw["answers"][q.name], q) for q in questions},
        usage=Usage(
            input_tokens=int(usage.get("input_tokens", 0)),
            output_tokens=int(usage.get("output_tokens", 0)),
        ),
        model=raw.get("model"),
    )


def _to_wire(question: Question) -> dict[str, Any]:
    criteria: dict[str, str] | list[str]
    if question.kind == "choice":
        criteria = dict(question.options)
    else:  # en score los niveles van en orden y se identifican por su posición
        criteria = list(question.options.values())
    return {"type": question.kind, "instructions": question.instructions, "criteria": criteria}


def _parse_answer(answer: Mapping[str, Any], question: Question) -> Answer:
    labels = list(question.options)
    probs: Mapping[str, float] = answer["probabilities"]
    if question.kind == "score":  # los niveles llegan como "0", "1", ...
        probs = {labels[int(level)]: p for level, p in probs.items()}
    confidence = answer.get("confidence")
    return Answer(
        probs=normalize_probs(probs, labels),
        vendor_confidence=None if confidence is None else float(confidence),
    )


def _raise_for_status(response: httpx2.Response) -> None:
    if response.is_success:
        return
    status = response.status_code
    message = f"HTTP {status}: {_error_detail(response)}"
    if status in _RETRYABLE_STATUS or status >= 500:
        raise RetryableError(message)
    if status in _FATAL_STATUS:
        raise FatalError(message)
    raise BackendError(message)  # p. ej. 413: el ticket excede un límite del servidor


def _error_detail(response: httpx2.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text[:300]
    if isinstance(body, dict):
        return str(body.get("detail") or body.get("error") or body)[:300]
    return str(body)[:300]
