"""request() del protocolo System One contra un servidor simulado."""

import json
from collections.abc import Callable
from contextlib import closing
from typing import Any

import httpx2
import pytest

from triage_eval.backends.base import BackendError, FatalError, RetryableError
from triage_eval.backends.systemone import SystemOneBackend
from triage_eval.domain import Question

STATE = {"asunto": "Cobro duplicado", "cuerpo": "Me cobraron dos veces."}

Handler = Callable[[httpx2.Request], httpx2.Response]


def _backend(handler: Handler, api_key: str | None = None) -> SystemOneBackend:
    return SystemOneBackend(
        base_url="http://test",
        model="multilingual",
        api_key=api_key,
        transport=httpx2.MockTransport(handler),
    )


def test_sends_the_ticket_and_questions_in_protocol_format(questions: list[Question]) -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx2.Response(200, json={"answers": {}})

    with closing(_backend(handler, api_key="secreto")) as backend:
        backend.request(STATE, questions)

    assert seen["url"] == "http://test/v1/systemone"
    assert seen["auth"] == "Bearer secreto"
    body = seen["body"]
    assert body["state"] == STATE
    assert body["model"] == "multilingual"
    assert body["questions"]["categoria"] == {
        "type": "choice",
        "instructions": "¿Qué equipo debe atender este ticket?",
        "criteria": {"facturacion": "Cobros", "tecnico": "Errores", "otro": "Otros temas"},
    }
    # en score los niveles se envían como lista ordenada de descripciones
    assert body["questions"]["prioridad"]["criteria"] == ["Puede esperar", "Esta semana", "Hoy"]


def test_does_not_send_authorization_without_api_key(questions: list[Question]) -> None:
    headers: list[str | None] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        headers.append(request.headers.get("authorization"))
        return httpx2.Response(200, json={"answers": {}})

    with closing(_backend(handler)) as backend:
        backend.request(STATE, questions)

    assert headers == [None]


@pytest.mark.parametrize(
    ("status", "error"),
    [
        (429, RetryableError),
        (529, RetryableError),
        (503, RetryableError),
        (401, FatalError),
        (422, FatalError),
        (413, BackendError),
    ],
)
def test_maps_http_errors(questions: list[Question], status: int, error: type[Exception]) -> None:
    with (
        closing(_backend(lambda request: httpx2.Response(status, json={"detail": "x"}))) as backend,
        pytest.raises(error) as info,
    ):
        backend.request(STATE, questions)

    assert type(info.value) is error  # 413 no debe tratarse como reintentable
    assert f"HTTP {status}: x" in str(info.value)


def test_network_errors_are_retryable(questions: list[Question]) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("conexión rechazada", request=request)

    with closing(_backend(handler)) as backend, pytest.raises(RetryableError):
        backend.request(STATE, questions)
