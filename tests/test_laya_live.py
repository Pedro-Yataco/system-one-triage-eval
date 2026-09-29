"""Prueba contra laya-serve real: `uv run pytest -m live` con el servidor levantado."""

from contextlib import closing

import pytest
from paths import EXAMPLE_CONFIG

from triage_eval.backends.laya import LayaBackend
from triage_eval.config import load_config

pytestmark = pytest.mark.live


def test_laya_answers_every_example_question() -> None:
    config = load_config(EXAMPLE_CONFIG)
    questions = [q.to_question() for q in config.questions]
    laya = config.backend_config
    state = {"asunto": "Cobro duplicado", "cuerpo": "Me cobraron dos veces la suscripción."}

    with closing(LayaBackend(base_url=laya.base_url, model=laya.model)) as backend:
        health = backend.describe()["health"]
        prediction = backend.parse(backend.request(state, questions), questions)

    assert health["status"] == "ok"
    assert set(prediction.answers) == {q.name for q in questions}
