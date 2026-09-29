from pathlib import Path

import pytest
from fakes import FakeBackend

from triage_eval.backends.base import BackendError, FatalError, RetryableError
from triage_eval.domain import Item, Question
from triage_eval.runner import RetryPolicy, evaluate_item, run_evaluation
from triage_eval.storage import RunStore

QUESTIONS = [
    Question(name="categoria", kind="choice", instructions="?", options={"a": "", "b": ""})
]
NO_WAIT = RetryPolicy(max_attempts=3, base_delay_s=0)
OK = {"answers": {"categoria": {"a": 0.8, "b": 0.2}}}


def _item(key: str) -> Item:
    return Item(id=key, state={"texto": key}, gold={"categoria": "a"})


def test_retries_transient_errors() -> None:
    backend = FakeBackend({"T1": [RetryableError("HTTP 429"), OK]})

    record = evaluate_item(backend, _item("T1"), QUESTIONS, NO_WAIT)

    assert record.attempts == 2
    assert record.error is None
    assert record.prediction is not None
    assert record.prediction.answers["categoria"].label == "a"
    assert record.latency_s is not None


def test_gives_up_after_the_last_attempt() -> None:
    backend = FakeBackend({"T1": [RetryableError("HTTP 529")] * 3})

    record = evaluate_item(backend, _item("T1"), QUESTIONS, NO_WAIT)

    assert (record.attempts, record.raw, record.prediction) == (3, None, None)
    assert record.error == "HTTP 529"


def test_does_not_retry_errors_of_a_single_ticket() -> None:
    backend = FakeBackend({"T1": [BackendError("HTTP 413")]})

    record = evaluate_item(backend, _item("T1"), QUESTIONS, NO_WAIT)

    assert record.attempts == 1
    assert record.error == "HTTP 413"


def test_keeps_the_raw_response_when_it_cannot_be_parsed() -> None:
    raw: dict[str, object] = {"answers": {}}  # falta la pregunta
    backend = FakeBackend({"T1": [raw]})

    record = evaluate_item(backend, _item("T1"), QUESTIONS, NO_WAIT)

    assert record.raw == raw
    assert record.prediction is None
    assert record.error is not None and record.error.startswith("respuesta no interpretable")


def test_fatal_errors_stop_the_run() -> None:
    backend = FakeBackend({"T1": [FatalError("HTTP 401")]})

    with pytest.raises(FatalError):
        evaluate_item(backend, _item("T1"), QUESTIONS, NO_WAIT)


def test_resuming_only_requests_tickets_without_response(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    items = [_item("T1"), _item("T2")]
    backend = FakeBackend({"T1": [OK], "T2": [BackendError("servidor caído"), OK]})

    run_evaluation(backend, items, QUESTIONS, store, NO_WAIT)
    run_evaluation(backend, items, QUESTIONS, store, NO_WAIT)

    assert backend.calls == ["T1", "T2", "T2"]
    assert [record.error for record in store.load_records()] == [None, None]
