"""Configuration loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    if path.suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("Install PyYAML to load YAML configs") from exc
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    if path.suffix == ".json":
        import json

        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    raise ValueError(f"Unsupported config format: {path.suffix}")


def deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged
