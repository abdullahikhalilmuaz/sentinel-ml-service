"""Flask API for Sentinel continuous authentication (local service)."""
from __future__ import annotations

import hmac
import logging
from typing import Any

from flask import Flask, jsonify, request

from config import (
    FEATURE_SCHEMA_VERSION,
    HOST,
    MAX_REQUEST_BYTES,
    ML_SERVICE_API_KEY,
    PORT,
    SERVICE_NAME,
)
from src.feature_schema import baseline_validation_error, feature_validation_error
from src.inference import ModelUnavailableError, get_model_status, score_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("sentinel.ml")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_REQUEST_BYTES


@app.before_request
def authenticate_internal_routes():
    """Optional shared-secret guard; enable after Node sends the matching header."""
    if request.path.startswith("/internal/") and ML_SERVICE_API_KEY:
        supplied = request.headers.get("X-ML-Service-Key", "")
        if not supplied or not hmac.compare_digest(supplied, ML_SERVICE_API_KEY):
            return jsonify({"status": "error", "message": "Unauthorized internal request"}), 401
    return None


@app.get("/health")
def health():
    # This means the HTTP service is alive, not that a model is ready to score.
    return jsonify({
        "status": "ok",
        "service": SERVICE_NAME,
        "schemaVersion": FEATURE_SCHEMA_VERSION,
    }), 200


@app.get("/internal/model/status")
def model_status():
    return jsonify(get_model_status()), 200


@app.post("/internal/score")
def score():
    data: Any = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"status": "error", "message": "Request body must be a JSON object"}), 400

    schema_version = data.get("schemaVersion", FEATURE_SCHEMA_VERSION)
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        return jsonify({"status": "error", "message": "schemaVersion must be an integer"}), 400
    if schema_version != FEATURE_SCHEMA_VERSION:
        return jsonify({
            "status": "error",
            "message": "Unsupported feature schema version",
            "expectedSchemaVersion": FEATURE_SCHEMA_VERSION,
        }), 400

    session_id = data.get("sessionId")
    if not isinstance(session_id, str) or not session_id.strip() or len(session_id) > 128:
        return jsonify({"status": "error", "message": "sessionId must be a non-empty string of at most 128 characters"}), 400

    features = data.get("features")
    validation_error = feature_validation_error(features)
    if validation_error:
        return jsonify({"status": "error", "message": validation_error}), 400

    baseline = data.get("baseline")
    if baseline is not None:
        baseline_error = baseline_validation_error(baseline)
        if baseline_error:
            return jsonify({"status": "error", "message": baseline_error}), 400

    try:
        result = score_features(features, baseline=baseline)
    except ModelUnavailableError as exc:
        logger.warning("Scoring unavailable: %s", str(exc))
        return jsonify({
            "status": "unavailable",
            "message": "The ML model is not ready. Train or restore the model artifact before scoring.",
        }), 503
    except Exception:
        logger.exception("Unexpected model scoring failure")
        return jsonify({"status": "error", "message": "Model scoring failed"}), 500

    # Do not log session IDs, features, behavioural profiles, or raw telemetry.
    logger.info("Scored behavioural batch (mode=%s, model=%s)", result.get("scoreMode"), result.get("modelVersion"))
    return jsonify(result), 200


@app.errorhandler(413)
def request_too_large(_error):
    return jsonify({"status": "error", "message": "Request body is too large"}), 413


@app.errorhandler(404)
def not_found(_error):
    return jsonify({"status": "error", "message": "Route not found"}), 404


@app.errorhandler(405)
def method_not_allowed(_error):
    return jsonify({"status": "error", "message": "Method not allowed"}), 405


if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=False)
