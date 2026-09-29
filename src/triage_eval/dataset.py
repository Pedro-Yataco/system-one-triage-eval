"""Carga de datasets etiquetados desde CSV o JSONL."""

import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from triage_eval.config import DatasetConfig, QuestionConfig
from triage_eval.domain import Item

_MAX_PROBLEMS_SHOWN = 10

_Row = tuple[int, dict[str, Any]]  # (número de fila en el archivo, valores)


class DatasetError(Exception):
    """El dataset no se puede leer o no calza con la configuración."""


def load_items(config: DatasetConfig, questions: Sequence[QuestionConfig]) -> list[Item]:
    """Lee el dataset y valida columnas, ids únicos y etiquetas conocidas."""
    rows = _read_rows(config.path)
    if not rows:
        raise DatasetError(f"{config.path} no tiene tickets")
    _check_columns(rows, config, questions)

    items: list[Item] = []
    problems: list[str] = []
    seen_ids: set[str] = set()
    for number, row in rows:
        item_id = _text(row.get(config.id_column))
        if not item_id:
            problems.append(f"fila {number}: `{config.id_column}` vacío")
            continue
        if item_id in seen_ids:
            problems.append(f"fila {number}: id repetido '{item_id}'")
        seen_ids.add(item_id)

        gold = {q.name: _text(row.get(q.gold_column)) for q in questions}
        for q in questions:
            label = gold[q.name]
            if not label:
                problems.append(f"fila {number} ({item_id}): sin etiqueta en `{q.gold_column}`")
            elif label not in q.options:
                options = ", ".join(q.options)
                problems.append(
                    f"fila {number} ({item_id}): '{label}' no es una opción de `{q.name}`"
                    f" ({options})"
                )
        state = {field: _text(row.get(field)) for field in config.state_fields}
        items.append(Item(id=item_id, state=state, gold=gold))

    if problems:
        shown = "\n  ".join(problems[:_MAX_PROBLEMS_SHOWN])
        hidden = len(problems) - _MAX_PROBLEMS_SHOWN
        more = f"\n  ... y {hidden} más" if hidden > 0 else ""
        raise DatasetError(f"{config.path} tiene {len(problems)} problema(s):\n  {shown}{more}")
    return items


def _check_columns(
    rows: list[_Row], config: DatasetConfig, questions: Sequence[QuestionConfig]
) -> None:
    required = [config.id_column, *config.state_fields, *(q.gold_column for q in questions)]
    available = {key for _, row in rows for key in row if isinstance(key, str)}
    missing = [column for column in required if column not in available]
    if missing:
        raise DatasetError(
            f"a {config.path} le faltan las columnas {missing}; tiene {sorted(available)}"
        )


def _read_rows(path: Path) -> list[_Row]:
    suffix = path.suffix.lower()
    if suffix not in {".csv", ".jsonl"}:
        raise DatasetError(f"{path}: formato no soportado (usa .csv o .jsonl)")
    try:
        if suffix == ".csv":
            # utf-8-sig acepta el BOM que agrega Excel al exportar "CSV UTF-8"
            with path.open(encoding="utf-8-sig", newline="") as file:
                return list(enumerate(csv.DictReader(file), start=2))  # la fila 1 es el encabezado
        with path.open(encoding="utf-8") as file:
            return [
                (number, _json_object(line, path, number))
                for number, line in enumerate(file, start=1)
                if line.strip()
            ]
    except UnicodeDecodeError as exc:
        raise DatasetError(f"{path} no está codificado en UTF-8") from exc
    except OSError as exc:
        raise DatasetError(f"no se pudo leer {path}: {exc}") from exc


def _json_object(line: str, path: Path, number: int) -> dict[str, Any]:
    try:
        row = json.loads(line)
    except json.JSONDecodeError as exc:
        raise DatasetError(f"{path}, línea {number}: JSON inválido ({exc.msg})") from exc
    if not isinstance(row, dict):
        raise DatasetError(f"{path}, línea {number}: se esperaba un objeto JSON")
    return row


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()
