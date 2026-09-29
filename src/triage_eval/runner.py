"""Evalúa un backend ticket a ticket: reintentos, latencia y registro de errores."""

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from triage_eval.backends.base import Backend, BackendError, RetryableError
from triage_eval.domain import EvalRecord, Item, Question
from triage_eval.storage import RunStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    base_delay_s: float = 1.0

    def delay_s(self, attempt: int) -> float:
        """Espera antes del siguiente intento: backoff exponencial (1 s, 2 s, 4 s...)."""
        return self.base_delay_s * 2.0 ** (attempt - 1)


def run_evaluation(
    backend: Backend,
    items: Sequence[Item],
    questions: Sequence[Question],
    store: RunStore,
    policy: RetryPolicy,
) -> None:
    """Evalúa los tickets que aún no tienen respuesta en la corrida y guarda cada resultado."""
    done = store.completed_ids()
    pending = [item for item in items if item.id not in done]
    if done:
        logger.info("Reanudando: %d tickets ya evaluados, %d pendientes", len(done), len(pending))
    with logging_redirect_tqdm():
        for item in tqdm(pending, desc="Evaluando", unit="ticket"):
            store.append(evaluate_item(backend, item, questions, policy))


def evaluate_item(
    backend: Backend, item: Item, questions: Sequence[Question], policy: RetryPolicy
) -> EvalRecord:
    """Pide la respuesta al backend (con reintentos) y la interpreta.

    Los errores de un ticket quedan en su registro; solo un FatalError detiene la corrida.
    """
    attempt = 0
    while True:
        attempt += 1
        start = time.perf_counter()
        try:
            raw = backend.request(item.state, questions)
        except RetryableError as exc:
            if attempt >= policy.max_attempts:
                return _failure(item, attempt, exc)
            time.sleep(policy.delay_s(attempt))
        except BackendError as exc:
            return _failure(item, attempt, exc)
        else:
            latency_s = time.perf_counter() - start
            break

    try:
        prediction = backend.parse(raw, questions)
    except Exception as exc:  # respuesta inesperada: se conserva la cruda para re-interpretarla
        logger.warning("Ticket %s: respuesta no interpretable (%r)", item.id, exc)
        return EvalRecord(
            item_id=item.id,
            gold=item.gold,
            raw=raw,
            prediction=None,
            latency_s=latency_s,
            attempts=attempt,
            error=f"respuesta no interpretable: {exc!r}",
        )
    return EvalRecord(
        item_id=item.id,
        gold=item.gold,
        raw=raw,
        prediction=prediction,
        latency_s=latency_s,
        attempts=attempt,
    )


def _failure(item: Item, attempts: int, exc: BackendError) -> EvalRecord:
    logger.warning("Ticket %s: %s", item.id, exc)
    return EvalRecord(
        item_id=item.id,
        gold=item.gold,
        raw=None,
        prediction=None,
        latency_s=None,
        attempts=attempts,
        error=str(exc),
    )
