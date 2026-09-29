"""Laya: modelo open source servido en local con laya-serve."""

import dataclasses
from collections.abc import Sequence
from typing import Any

import httpx2

from triage_eval.backends.base import FatalError
from triage_eval.backends.systemone import SystemOneBackend
from triage_eval.domain import Prediction, Question, RawResponse


class LayaBackend(SystemOneBackend):
    """Habla el mismo protocolo que Jev; además expone /health (dispositivo y checkpoints)."""

    def parse(self, raw: RawResponse, questions: Sequence[Question]) -> Prediction:
        prediction = super().parse(raw, questions)
        # "model" siempre dice "laya-rl-agent"; el checkpoint que respondió viene en "routing"
        checkpoint = (raw.get("routing") or {}).get("model")
        if checkpoint is None:
            return prediction
        return dataclasses.replace(prediction, model=f"{prediction.model}/{checkpoint}")

    def describe(self) -> dict[str, Any]:
        try:
            response = self._client.get("/health")
            response.raise_for_status()
        except httpx2.HTTPError as exc:
            raise FatalError(
                f"laya-serve no responde en {self._client.base_url} ({exc}). "
                "¿Está corriendo? Revisa «Levantar Laya» en el README."
            ) from exc
        return {**super().describe(), "health": response.json()}
