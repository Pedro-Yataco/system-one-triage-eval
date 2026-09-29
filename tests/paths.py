"""Rutas usadas por los tests."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
EXAMPLE_CONFIG = ROOT / "configs" / "example.yaml"
