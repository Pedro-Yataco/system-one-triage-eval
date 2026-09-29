"""Calidad de una pregunta: accuracy, F1 macro, métricas por clase y matriz de confusión."""

from collections import Counter
from collections.abc import Sequence
from typing import Any

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

MISSING_LABEL = "(sin respuesta)"  # columna de la matriz para tickets donde el backend falló


def quality_metrics(
    gold: Sequence[str], predicted: Sequence[str | None], labels: Sequence[str]
) -> dict[str, Any]:
    """Métricas de una pregunta. Una predicción faltante (None) cuenta como error.

    El F1 macro promedia solo las clases presentes en el gold: una clase sin ejemplos
    no tiene recall definido y bajaría el promedio sin decir nada del modelo.
    """
    if len(gold) != len(predicted):
        raise ValueError("gold y predicted deben tener el mismo largo")
    if not gold:
        raise ValueError("no hay tickets para evaluar")
    unknown = set(gold) - set(labels)
    if unknown:
        raise ValueError(f"etiquetas gold fuera de las opciones: {sorted(unknown)}")

    y_pred = [label if label is not None else MISSING_LABEL for label in predicted]
    labels = list(labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        gold, y_pred, labels=labels, zero_division=0
    )
    columns = [*labels, MISSING_LABEL] if MISSING_LABEL in y_pred else labels
    matrix = confusion_matrix(gold, y_pred, labels=columns)[: len(labels)]

    return {
        "n": len(gold),
        "n_missing": y_pred.count(MISSING_LABEL),
        "accuracy": float(accuracy_score(gold, y_pred)),
        "macro_f1": float(f1[support > 0].mean()),
        "majority_baseline_accuracy": max(Counter(gold).values()) / len(gold),
        "per_class": {
            label: {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
            for label, p, r, f, s in zip(labels, precision, recall, f1, support, strict=True)
        },
        "confusion_matrix": {"gold": labels, "predicted": columns, "counts": matrix.tolist()},
    }
