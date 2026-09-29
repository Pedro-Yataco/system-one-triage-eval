import json
from pathlib import Path

import pytest

from triage_eval.config import DatasetConfig, QuestionConfig
from triage_eval.dataset import DatasetError, load_items

QUESTIONS = [
    QuestionConfig(
        name="categoria",
        kind="choice",
        gold_column="cat",
        instructions="¿Qué equipo?",
        options={"facturacion": "Cobros", "tecnico": "Errores"},
    )
]


def _config(path: Path) -> DatasetConfig:
    return DatasetConfig(path=path, id_column="id", state_fields=["texto"])


def test_loads_csv_exported_by_excel_with_bom(tmp_path: Path) -> None:
    path = tmp_path / "tickets.csv"
    content = (
        'id,texto,cat\nT1,Me cobraron dos veces,facturacion\nT2,"Error 500, otra vez",tecnico\n'
    )
    path.write_bytes(b"\xef\xbb\xbf" + content.encode("utf-8"))

    items = load_items(_config(path), QUESTIONS)

    assert [item.id for item in items] == ["T1", "T2"]
    assert items[1].state == {"texto": "Error 500, otra vez"}
    assert items[0].gold == {"categoria": "facturacion"}


def test_loads_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "tickets.jsonl"
    rows = [{"id": 1, "texto": "Se cayó la app", "cat": "tecnico"}]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n\n", encoding="utf-8")

    [item] = load_items(_config(path), QUESTIONS)

    assert item.id == "1"
    assert item.gold == {"categoria": "tecnico"}


def test_reports_every_invalid_row(tmp_path: Path) -> None:
    path = tmp_path / "tickets.csv"
    path.write_text(
        "id,texto,cat\nT1,a,facturacion\nT1,b,tecnico\nT2,c,ventas\nT3,d,\n", encoding="utf-8"
    )

    with pytest.raises(DatasetError) as info:
        load_items(_config(path), QUESTIONS)

    message = str(info.value)
    assert "3 problema(s)" in message
    assert "id repetido 'T1'" in message
    assert "'ventas' no es una opción" in message
    assert "sin etiqueta en `cat`" in message


def test_reports_missing_columns(tmp_path: Path) -> None:
    path = tmp_path / "tickets.csv"
    path.write_text("id,body,cat\nT1,hola,facturacion\n", encoding="utf-8")

    with pytest.raises(DatasetError, match=r"faltan las columnas \['texto'\]"):
        load_items(_config(path), QUESTIONS)


def test_rejects_unsupported_formats(tmp_path: Path) -> None:
    path = tmp_path / "tickets.xlsx"
    path.write_bytes(b"")

    with pytest.raises(DatasetError, match="formato no soportado"):
        load_items(_config(path), QUESTIONS)
