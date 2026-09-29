"""Tipos del dominio, compartidos por el dataset, los backends, el runner y las métricas."""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

State = Mapping[str, str]  # campos del ticket que ve el modelo, p. ej. asunto y cuerpo
RawResponse = dict[str, Any]  # respuesta del proveedor tal como llegó (JSON)
QuestionKind = Literal["choice", "score"]


@dataclass(frozen=True)
class Question:
    """Pregunta tipada: se define una vez y se envía igual a todos los backends."""

    name: str
    kind: QuestionKind
    instructions: str
    options: Mapping[str, str]  # etiqueta -> descripción; en "score" el orden es la escala


@dataclass(frozen=True)
class Item:
    """Un ticket etiquetado del dataset."""

    id: str
    state: State
    gold: Mapping[str, str]  # nombre de la pregunta -> etiqueta correcta


@dataclass(frozen=True)
class Answer:
    """Distribución de probabilidad de un modelo sobre las opciones de una pregunta."""

    probs: Mapping[str, float]
    vendor_confidence: float | None = None  # "confidence" del proveedor; cada uno usa su fórmula

    def __post_init__(self) -> None:
        if not self.probs:
            raise ValueError("la distribución está vacía")
        if any(not math.isfinite(p) or p < 0 for p in self.probs.values()):
            raise ValueError(f"probabilidades inválidas: {dict(self.probs)}")
        total = sum(self.probs.values())
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(f"las probabilidades deben sumar 1 (suman {total:.6f})")

    @property
    def label(self) -> str:
        """Opción más probable. Ante un empate gana la primera según el orden de las opciones."""
        return max(self.probs, key=self.probs.__getitem__)

    @property
    def confidence(self) -> float:
        """Probabilidad de la opción elegida (p_max), comparable entre backends."""
        return max(self.probs.values())


@dataclass(frozen=True)
class Usage:
    """Tokens consumidos por una llamada; con ellos se estima el costo."""

    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True)
class Prediction:
    """Lo que respondió un modelo sobre un ticket."""

    answers: Mapping[str, Answer]  # nombre de la pregunta -> respuesta
    usage: Usage
    model: str | None = None  # versión que realmente respondió, según el proveedor


@dataclass(frozen=True)
class EvalRecord:
    """Resultado de evaluar un ticket: una línea de predictions.jsonl."""

    item_id: str
    gold: Mapping[str, str]
    raw: RawResponse | None  # se guarda para re-interpretarla sin volver a llamar al modelo
    prediction: Prediction | None
    latency_s: float | None  # latencia del intento exitoso, medida en el cliente
    attempts: int
    error: str | None = None
