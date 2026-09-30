"""Invariantes del dataset sintético de escalamiento (data/escalamiento)."""

import csv
from collections import Counter

import pytest
from paths import ROOT

from triage_eval.config import DatasetConfig, load_config
from triage_eval.dataset import load_items

CONFIG = ROOT / "configs" / "escalamiento.yaml"
DATA = ROOT / "data" / "escalamiento"
LOTES = {"principal": DATA / "principal.csv", "ajuste": DATA / "ajuste.csv"}
ESCALABLES = {"ACTUALIZACION_TICKET", "SEGUIMIENTO_TICKET", "MULTI_GESTION"}
ACCION_POR_SUBTIPO = {
    "NUEVA_SOLICITUD": "clasificar",
    "NOTIFICACION": "descartar",
    "CONFIRMACION": "descartar",
    "CIERRE_CANCELACION": "descartar",
}


def _filas(lote: str) -> list[dict[str, str]]:
    with LOTES[lote].open(encoding="utf-8-sig", newline="") as archivo:
        return list(csv.DictReader(archivo))


@pytest.mark.parametrize("lote", LOTES)
def test_the_harness_loads_both_batches(lote: str) -> None:
    config = load_config(CONFIG)
    dataset = DatasetConfig(path=LOTES[lote], id_column="id", state_fields=["texto"])

    items = load_items(dataset, config.questions)  # valida etiquetas contra las opciones del YAML

    assert len(items) == len(_filas(lote))


@pytest.mark.parametrize("lote", LOTES)
def test_labels_are_consistent(lote: str) -> None:
    for fila in _filas(lote):
        if fila["casuistica"] in ESCALABLES:
            assert (fila["accion"], fila["subtipo"]) == ("escalar", ""), fila["id"]
            assert fila["dificultad"] in {"directa", "indirecta"}, fila["id"]
        else:
            assert fila["casuistica"] == "OTROS", fila["id"]
            assert fila["accion"] == ACCION_POR_SUBTIPO[fila["subtipo"]], fila["id"]
            assert fila["dificultad"] in {"directa", "engañosa"}, fila["id"]
        assert fila["texto"].startswith("Asunto: ") and "\nCuerpo: " in fila["texto"], fila["id"]


def test_batches_do_not_share_ids_or_texts() -> None:
    filas = _filas("principal") + _filas("ajuste")

    assert len({f["id"] for f in filas}) == len(filas)
    assert len({f["texto"] for f in filas}) == len(filas)


@pytest.mark.parametrize(("lote", "total"), [("principal", 300), ("ajuste", 60)])
def test_composition(lote: str, total: int) -> None:
    filas = _filas(lote)
    casuisticas = Counter(f["casuistica"] for f in filas)

    assert len(filas) == total
    assert casuisticas["OTROS"] / total == pytest.approx(0.6)  # 40 % escalar
    if lote == "principal":
        assert min(casuisticas[c] for c in ESCALABLES) >= 30  # base mínima para métricas por clase
