"""Configuración de un experimento: se lee desde YAML y se valida con pydantic."""

from pathlib import Path
from typing import Annotated, Literal, Self

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PlainSerializer,
    ValidationError,
    model_validator,
)

from triage_eval.domain import Question

# Al guardar la config se escribe con "/" para que la copia de la corrida sirva en cualquier SO.
PortablePath = Annotated[Path, PlainSerializer(Path.as_posix, return_type=str, when_used="json")]


class ConfigError(Exception):
    """El archivo de configuración no se puede leer o no es válido."""


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DatasetConfig(_Model):
    path: PortablePath
    id_column: str = "id"
    state_fields: list[str] = Field(min_length=1)  # columnas que ve el modelo


class QuestionConfig(_Model):
    name: str
    kind: Literal["choice", "score"]
    gold_column: str  # columna del dataset con la etiqueta correcta
    instructions: str
    options: dict[str, str] = Field(min_length=2)  # etiqueta -> descripción

    def to_question(self) -> Question:
        return Question(
            name=self.name, kind=self.kind, instructions=self.instructions, options=self.options
        )


class LayaConfig(_Model):
    kind: Literal["laya"]
    base_url: str = "http://127.0.0.1:8000"
    model: Literal["english", "multilingual", "typed-decisions"] = "multilingual"
    api_key_env: str | None = None  # solo si laya-serve se levantó con LAYA_API_KEY
    timeout_s: float = Field(default=30.0, gt=0)


class RunConfig(_Model):
    output_dir: PortablePath = Path("runs")
    limit: int | None = Field(default=None, ge=1)  # evalúa solo los primeros N tickets
    max_attempts: int = Field(default=3, ge=1)  # intentos por ticket ante fallas transitorias


class ExperimentConfig(_Model):
    dataset: DatasetConfig
    questions: list[QuestionConfig] = Field(min_length=1)
    backends: dict[str, LayaConfig] = Field(min_length=1)
    backend: str  # cuál de `backends` se evalúa
    run: RunConfig = RunConfig()

    @model_validator(mode="after")
    def _check_references(self) -> Self:
        if self.backend not in self.backends:
            available = ", ".join(self.backends)
            raise ValueError(f"el backend '{self.backend}' no está en `backends` ({available})")
        names = [question.name for question in self.questions]
        if len(names) != len(set(names)):
            raise ValueError("hay preguntas con el mismo nombre")
        return self

    @property
    def backend_config(self) -> LayaConfig:
        return self.backends[self.backend]


def load_config(
    path: Path, *, backend: str | None = None, limit: int | None = None
) -> ExperimentConfig:
    """Lee y valida el YAML. `backend` y `limit` reemplazan lo definido en el archivo."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"no se pudo leer {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path} debe contener un objeto YAML")
    if backend is not None:
        data["backend"] = backend
    if limit is not None:
        data["run"] = {**(data.get("run") or {}), "limit": limit}
    try:
        return ExperimentConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"{path} no es válido:\n{exc}") from exc
