"""Línea de comandos: `triage-eval run`."""

import argparse
import logging
import sys
from collections.abc import Sequence
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from triage_eval.backends import FatalError, build_backend
from triage_eval.config import ConfigError, ExperimentConfig, load_config
from triage_eval.dataset import DatasetError, load_items
from triage_eval.report import compute_metrics, format_summary
from triage_eval.runner import RetryPolicy, run_evaluation
from triage_eval.storage import RunStore, build_manifest, file_sha256

logger = logging.getLogger("triage_eval")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.resume and (args.backend or args.limit):
        parser.error("--backend y --limit no se combinan con --resume (se usa la config guardada)")

    load_dotenv()  # credenciales desde .env; la librería nunca lee .env por su cuenta
    logging.basicConfig(level=logging.WARNING, format="%(message)s")  # otras librerías: solo avisos
    logger.setLevel(logging.INFO)
    try:
        return _run(args)
    except (ConfigError, DatasetError, FatalError, FileNotFoundError) as exc:
        logger.error("Error: %s", exc)
        return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="triage-eval",
        description="Evalúa modelos de decisión sobre un dataset de tickets etiquetados.",
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMANDO")

    run = commands.add_parser("run", help="evalúa un backend y guarda la corrida")
    source = run.add_mutually_exclusive_group(required=True)
    source.add_argument("-c", "--config", type=Path, help="YAML del experimento")
    source.add_argument(
        "--resume", type=Path, metavar="CARPETA", help="completa una corrida interrumpida"
    )
    run.add_argument("-b", "--backend", help="backend del YAML a evaluar (reemplaza a `backend`)")
    run.add_argument("--limit", type=int, help="evalúa solo los primeros N tickets")
    return parser


def _run(args: argparse.Namespace) -> int:
    store = RunStore.open(args.resume) if args.resume else None
    if store is not None:
        config = load_config(store.config_path)  # se reanuda con la configuración guardada
    else:
        config = load_config(args.config, backend=args.backend, limit=args.limit)

    questions = [q.to_question() for q in config.questions]
    items = load_items(config.dataset, config.questions)[: config.run.limit]

    with closing(build_backend(config.backend_config)) as backend:
        backend_info = backend.describe()  # falla temprano si el backend no está disponible
        if store is None:
            store = _create_run(config, backend_info, len(items))
        else:
            _register_resume(store, config)
        logger.info("Corrida: %s", store.path)

        policy = RetryPolicy(max_attempts=config.run.max_attempts)
        try:
            run_evaluation(backend, items, questions, store, policy)
        except (KeyboardInterrupt, FatalError) as exc:
            reason = "interrumpida" if isinstance(exc, KeyboardInterrupt) else f"detenida: {exc}"
            logger.error(
                "Corrida %s. Para continuar: triage-eval run --resume %s", reason, store.path
            )
            return 1

    metrics = compute_metrics(store.load_records(), questions)
    store.write_metrics(metrics)
    print(format_summary(metrics))
    print(f"Resultados en {store.path}")
    return 0


def _create_run(config: ExperimentConfig, backend_info: dict[str, Any], n_items: int) -> RunStore:
    store = RunStore.create(config.run.output_dir, config.backend, config.dataset.path)
    store.write_config(config)
    store.write_manifest(build_manifest(config.backend, backend_info, config.dataset.path, n_items))
    return store


def _register_resume(store: RunStore, config: ExperimentConfig) -> None:
    manifest = store.read_manifest()
    if manifest["dataset"]["sha256"] != file_sha256(config.dataset.path):
        raise ConfigError(f"{config.dataset.path} cambió desde que se creó la corrida")
    resumed = manifest.setdefault("resumed_at", [])
    resumed.append(datetime.now(UTC).isoformat(timespec="seconds"))
    store.write_manifest(manifest)


if __name__ == "__main__":
    sys.exit(main())
