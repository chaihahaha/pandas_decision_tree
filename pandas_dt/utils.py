"""Small I/O helpers for fitted models and experiment reports."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

MODEL_FORMAT_VERSION = 1


def save_model(model: Any, path: str | Path, metadata: dict[str, Any] | None = None) -> Path:
    """Serialize a fitted model (and optional metadata) with pickle."""

    path = Path(path)
    payload = {
        "format_version": MODEL_FORMAT_VERSION,
        "model": model,
        "metadata": metadata or {},
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle)
    return path


def load_model(path: str | Path) -> tuple[Any, dict[str, Any]]:
    """Load a model saved by :func:`save_model`.

    Returns ``(model, metadata)``.  Raw pickles from the original script
    (which stored the tree object directly) are accepted as a convenience.
    """

    path = Path(path)
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if isinstance(payload, dict) and "model" in payload and "format_version" in payload:
        if payload["format_version"] != MODEL_FORMAT_VERSION:
            raise ValueError(
                f"unsupported model format version {payload['format_version']}"
            )
        return payload["model"], payload.get("metadata", {})
    return payload, {}


def save_json(data: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)

    def convert(value: Any) -> Any:
        # numpy scalars/arrays are not JSON serialisable.
        try:
            import numpy as np

            if isinstance(value, np.generic):
                return value.item()
            if isinstance(value, np.ndarray):
                return value.tolist()
        except ImportError:  # pragma: no cover
            pass
        raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")

    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=convert) + "\n",
        encoding="utf-8",
    )
    return path
