"""Backends evaluables y su construcción a partir de la configuración."""

import os

from triage_eval.backends.base import Backend, BackendError, FatalError, RetryableError
from triage_eval.backends.laya import LayaBackend
from triage_eval.config import LayaConfig

__all__ = ["Backend", "BackendError", "FatalError", "RetryableError", "build_backend"]


def build_backend(config: LayaConfig) -> Backend:
    """Crea el backend descrito en la configuración."""
    return LayaBackend(
        base_url=config.base_url,
        model=config.model,
        api_key=_read_api_key(config.api_key_env),
        timeout_s=config.timeout_s,
    )


def _read_api_key(env_var: str | None) -> str | None:
    if env_var is None:
        return None
    value = os.environ.get(env_var)
    if not value:
        raise FatalError(f"falta la variable de entorno {env_var} (defínela en .env)")
    return value
