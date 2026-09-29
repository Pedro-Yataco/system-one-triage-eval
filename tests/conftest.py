import pytest

from triage_eval.domain import Question


@pytest.fixture
def questions() -> list[Question]:
    """Una pregunta de cada tipo, con las etiquetas de fixtures/systemone_doc_example.json."""
    return [
        Question(
            name="categoria",
            kind="choice",
            instructions="¿Qué equipo debe atender este ticket?",
            options={"facturacion": "Cobros", "tecnico": "Errores", "otro": "Otros temas"},
        ),
        Question(
            name="prioridad",
            kind="score",
            instructions="¿Qué tan urgente es?",
            options={"baja": "Puede esperar", "media": "Esta semana", "alta": "Hoy"},
        ),
    ]
