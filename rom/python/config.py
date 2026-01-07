from __future__ import annotations

from pathlib import Path
from typing import Any


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "PyYAML is required to read configs. In your venv, try `python -c \"import yaml\"`."
        ) from exc
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping at top-level: {path}")
    return data


def get(dct: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = dct
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur

