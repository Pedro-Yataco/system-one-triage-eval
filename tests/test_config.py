from pathlib import Path

import pytest
from paths import EXAMPLE_CONFIG

from triage_eval.config import ConfigError, load_config


def test_example_config_is_valid() -> None:
    config = load_config(EXAMPLE_CONFIG)

    assert config.backend_config.kind == "laya"
    assert [q.name for q in config.questions] == ["categoria", "prioridad"]


def test_cli_overrides_are_validated() -> None:
    assert load_config(EXAMPLE_CONFIG, limit=5).run.limit == 5
    with pytest.raises(ConfigError, match="no está en `backends`"):
        load_config(EXAMPLE_CONFIG, backend="jev")


def test_rejects_repeated_question_names(tmp_path: Path) -> None:
    text = EXAMPLE_CONFIG.read_text(encoding="utf-8").replace("name: prioridad", "name: categoria")
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ConfigError, match="mismo nombre"):
        load_config(path)


def test_rejects_unknown_fields(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(EXAMPLE_CONFIG.read_text(encoding="utf-8") + "\nextra: 1\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="extra"):
        load_config(path)
