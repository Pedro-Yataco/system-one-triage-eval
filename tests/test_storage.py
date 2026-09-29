from pathlib import Path

from paths import EXAMPLE_CONFIG

from triage_eval.config import load_config
from triage_eval.domain import Answer, EvalRecord, Prediction, Usage
from triage_eval.storage import RunStore


def _record(item_id: str, *, error: str | None = None) -> EvalRecord:
    if error is not None:
        return EvalRecord(item_id, {"categoria": "facturacion"}, None, None, None, 3, error)
    prediction = Prediction(
        answers={
            "categoria": Answer(probs={"facturacion": 0.7, "otro": 0.3}, vendor_confidence=0.4)
        },
        usage=Usage(input_tokens=120),
        model="multilingual",
    )
    return EvalRecord(
        item_id=item_id,
        gold={"categoria": "facturacion"},
        raw={"answers": {"categoria": {"choice": "facturacion"}}},
        prediction=prediction,
        latency_s=0.05,
        attempts=1,
    )


def test_records_survive_a_round_trip(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    records = [_record("T1"), _record("T2", error="HTTP 413: ticket demasiado largo")]

    for record in records:
        store.append(record)

    assert store.load_records() == records


def test_the_last_record_of_a_ticket_wins(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    store.append(_record("T1", error="timeout"))
    store.append(_record("T2"))
    store.append(_record("T1"))

    records = store.load_records()

    assert [r.item_id for r in records] == ["T1", "T2"]
    assert records[0].error is None
    assert store.completed_ids() == {"T1", "T2"}


def test_saved_config_is_portable_and_reloads_identically(tmp_path: Path) -> None:
    config = load_config(EXAMPLE_CONFIG)
    store = RunStore(tmp_path)

    store.write_config(config)

    assert "path: data/example/tickets.csv" in store.config_path.read_text(encoding="utf-8")
    assert load_config(store.config_path) == config


def test_failed_requests_are_not_completed(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    store.append(_record("T1", error="timeout"))

    assert store.completed_ids() == set()
