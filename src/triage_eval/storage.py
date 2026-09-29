"""Carpeta de una corrida: configuración usada, manifest, predicciones y métricas."""

import dataclasses
import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Self

import yaml

from triage_eval.config import ExperimentConfig
from triage_eval.domain import Answer, EvalRecord, Prediction, Usage

CONFIG_FILE = "config.yaml"
MANIFEST_FILE = "manifest.json"
PREDICTIONS_FILE = "predictions.jsonl"
METRICS_FILE = "metrics.json"

_TRACKED_PACKAGES = ("system-one-triage-eval", "httpx2", "numpy", "pydantic", "scikit-learn")


class RunStore:
    """Lee y escribe los archivos de una corrida.

    Las predicciones se agregan de a una línea, así una corrida interrumpida se puede reanudar.
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    @classmethod
    def create(cls, output_dir: Path, backend: str, dataset: Path) -> Self:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = output_dir / f"{stamp}_{backend}_{dataset.stem}"
        path.mkdir(parents=True)
        return cls(path)

    @classmethod
    def open(cls, path: Path) -> Self:
        if not (path / CONFIG_FILE).is_file():
            raise FileNotFoundError(f"{path} no es una carpeta de corrida (falta {CONFIG_FILE})")
        return cls(path)

    @property
    def config_path(self) -> Path:
        return self.path / CONFIG_FILE

    def write_config(self, config: ExperimentConfig) -> None:
        data = config.model_dump(mode="json")
        self.config_path.write_text(
            yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )

    def read_manifest(self) -> dict[str, Any]:
        manifest: dict[str, Any] = json.loads(
            (self.path / MANIFEST_FILE).read_text(encoding="utf-8")
        )
        return manifest

    def write_manifest(self, manifest: dict[str, Any]) -> None:
        _write_json(self.path / MANIFEST_FILE, manifest)

    def write_metrics(self, metrics: dict[str, Any]) -> None:
        _write_json(self.path / METRICS_FILE, metrics)

    def append(self, record: EvalRecord) -> None:
        line = json.dumps(dataclasses.asdict(record), ensure_ascii=False)
        with (self.path / PREDICTIONS_FILE).open("a", encoding="utf-8", newline="\n") as file:
            file.write(line + "\n")

    def load_records(self) -> list[EvalRecord]:
        """Registros de la corrida. Si un ticket se evaluó más de una vez, vale el último."""
        path = self.path / PREDICTIONS_FILE
        if not path.exists():
            return []
        latest: dict[str, EvalRecord] = {}
        with path.open(encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    record = _record_from_dict(json.loads(line))
                    latest[record.item_id] = record
        return list(latest.values())

    def completed_ids(self) -> set[str]:
        """Tickets que ya tienen respuesta del modelo, aunque no se haya podido interpretar."""
        return {record.item_id for record in self.load_records() if record.raw is not None}


def build_manifest(
    backend_name: str, backend_info: dict[str, Any], dataset: Path, n_items: int
) -> dict[str, Any]:
    """Datos para reproducir y auditar la corrida."""
    return {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "backend": {"name": backend_name, **backend_info},
        "dataset": {"path": dataset.as_posix(), "sha256": file_sha256(dataset), "n_items": n_items},
        "git": _git_state(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": _package_versions(),
        },
    }


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_state() -> dict[str, Any] | None:
    try:
        commit = _git("rev-parse", "HEAD")
        dirty = bool(_git("status", "--porcelain"))
    except (OSError, subprocess.CalledProcessError):
        return None  # sin git o fuera de un repositorio
    return {"commit": commit, "dirty": dirty}


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], capture_output=True, text=True, check=True)
    return result.stdout.strip()


def _package_versions() -> dict[str, str]:
    versions = {}
    for name in _TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return versions


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _record_from_dict(data: dict[str, Any]) -> EvalRecord:
    prediction = data["prediction"]
    return EvalRecord(
        item_id=data["item_id"],
        gold=data["gold"],
        raw=data["raw"],
        prediction=None if prediction is None else _prediction_from_dict(prediction),
        latency_s=data["latency_s"],
        attempts=data["attempts"],
        error=data["error"],
    )


def _prediction_from_dict(data: dict[str, Any]) -> Prediction:
    return Prediction(
        answers={name: Answer(**answer) for name, answer in data["answers"].items()},
        usage=Usage(**data["usage"]),
        model=data["model"],
    )
