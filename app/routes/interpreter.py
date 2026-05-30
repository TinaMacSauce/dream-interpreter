from __future__ import annotations

import os
import traceback

from flask import (
    Blueprint,
    jsonify,
    make_response,
    request,
)

from app.services.interpreter_service import run_interpretation


interpreter_bp = Blueprint("interpreter", __name__)


# ============================================================
# Helpers
# ============================================================

def json_ok(payload=None, status_code=200):
    body = {
        "ok": True,
    }

    if payload:
        body.update(payload)

    return jsonify(body), status_code


def json_error(message, status_code=400, error="request_error", details=None):
    body = {
        "ok": False,
        "error": error,
        "message": message,
    }

    if details is not None:
        body["details"] = details

    return jsonify(body), status_code


def empty_options_response():
    return make_response("", 204)


def should_show_trace():
    return (
        os.getenv("FLASK_DEBUG", "0") == "1"
        or os.getenv("SHOW_ERROR_TRACE", "0") == "1"
    )


# ============================================================
# Main Interpreter Endpoint
# ============================================================

@interpreter_bp.route("/interpret", methods=["POST", "OPTIONS"])
def interpret():
    if request.method == "OPTIONS":
        return empty_options_response()

    try:
        return run_interpretation()

    except Exception as exc:
        trace = traceback.format_exc()
        print(trace, flush=True)

        details = {
            "error_type": exc.__class__.__name__,
        }

        if should_show_trace():
            details["trace"] = trace
            details["raw_error"] = str(exc)

        return json_error(
            "Something went wrong while interpreting the dream.",
            status_code=500,
            error="interpretation_failed",
            details=details,
        )