"""Configuration for the local Sentinel ML service."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def _port_from_env() -> int:
    raw_value = os.getenv("PORT", "8000")
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("PORT must be an integer between 1 and 65535") from exc
    if not 1 <= value <= 65535:
        raise RuntimeError("PORT must be an integer between 1 and 65535")
    return value


PORT = _port_from_env()
HOST = os.getenv("HOST", "127.0.0.1").strip() or "127.0.0.1"
MAX_REQUEST_BYTES = 64 * 1024
ML_SERVICE_API_KEY = os.getenv("ML_SERVICE_API_KEY", "").strip()

_raw_model_path = Path(os.getenv("MODEL_PATH", "artifacts/isolation_forest.pkl"))
MODEL_PATH = (_raw_model_path if _raw_model_path.is_absolute() else BASE_DIR / _raw_model_path).resolve()

SERVICE_NAME = "sentinel-ml-service"
FEATURE_SCHEMA_VERSION = 1
MODEL_VERSION = "iforest-v1"
ARTIFACT_FORMAT_VERSION = 1
