"""Contrato común de los backends y errores que el runner sabe manejar."""

import math
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from triage_eval.domain import Prediction, Question, RawResponse, State


class Backend(Protocol):
    """Un modelo evaluable. Separa la llamada (I/O) de su interpretación (pura)."""

    def request(self, state: State, questions: Sequence[Question]) -> RawResponse:
        """Envía el ticket y las preguntas al modelo y devuelve la respuesta cruda."""

    def parse(self, raw: RawResponse, questions: Sequence[Question]) -> Prediction:
        """Interpreta una respuesta cruda. No hace I/O, así que se puede re-ejecutar offline."""

    def describe(self) -> dict[str, Any]:
        """Metadatos para el manifest de la corrida (versión, dispositivo, endpoint...)."""

    def close(self) -> None:
        """Libera las conexiones abiertas."""


class BackendError(Exception):
    """Falla al procesar un ticket: se registra en ese ticket y la corrida sigue."""


class RetryableError(BackendError):
    """Falla transitoria (límite de tasa, sobrecarga, timeout): el runner reintenta."""


class FatalError(Exception):
    """Falla que afecta a toda la corrida (servidor caído, credenciales, configuración)."""


def normalize_probs(probs: Mapping[str, float], options: Sequence[str]) -> dict[str, float]:
    """Ajusta una distribución a las opciones de la pregunta.

    Completa con 0 las opciones que falten, rechaza etiquetas desconocidas y valores inválidos,
    y reescala para que sume exactamente 1 (corrige redondeos del proveedor).
    """
    unknown = set(probs) - set(options)
    if unknown:
        raise ValueError(f"etiquetas desconocidas: {sorted(unknown)}")
    values = {label: float(probs.get(label, 0.0)) for label in options}
    if any(not math.isfinite(v) or v < 0 for v in values.values()):
        raise ValueError(f"probabilidades inválidas: {values}")
    total = sum(values.values())
    if total <= 0:
        raise ValueError("todas las probabilidades son 0")
    return {label: v / total for label, v in values.items()}
