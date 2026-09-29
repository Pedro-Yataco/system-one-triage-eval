"""Resumen de una corrida: arma las secuencias de cada pregunta y calcula sus métricas."""

from collections.abc import Sequence
from typing import Any

from triage_eval.domain import EvalRecord, Question
from triage_eval.metrics.quality import quality_metrics


def compute_metrics(records: Sequence[EvalRecord], questions: Sequence[Question]) -> dict[str, Any]:
    """Métricas de la corrida, por pregunta. Los tickets con error cuentan como fallos."""
    return {
        "n_items": len(records),
        "n_errors": sum(record.error is not None for record in records),
        "questions": {
            q.name: quality_metrics(
                gold=[record.gold[q.name] for record in records],
                predicted=[_predicted_label(record, q) for record in records],
                labels=list(q.options),
            )
            for q in questions
        },
    }


def format_summary(metrics: dict[str, Any]) -> str:
    """Texto breve para la consola."""
    lines = [f"Tickets evaluados: {metrics['n_items']} (con error: {metrics['n_errors']})"]
    for name, m in metrics["questions"].items():
        lines.append(
            f"  {name}: accuracy {m['accuracy']:.3f} | F1 macro {m['macro_f1']:.3f}"
            f" | baseline clase mayoritaria {m['majority_baseline_accuracy']:.3f}"
        )
    return "\n".join(lines)


def _predicted_label(record: EvalRecord, question: Question) -> str | None:
    if record.prediction is None:
        return None
    return record.prediction.answers[question.name].label
