from __future__ import annotations

import os
import traceback

from flask import (
    Blueprint,
    jsonify,
    make_response,
    render_template,
    request,
)

from app.access import (
    has_active_access,
    persist_email_to_session,
    set_buyer_session,
    set_premium_session,
)

from app.config import Config
from app.services.interpreter_service import run_interpretation
from app.utils import normalize_email


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


def safe_return_url():
    raw = request.args.get("return") or Config.RETURN_URL

    allowed_prefixes = (
        "https://interpreter.jamaicantruestories.com",
        "https://jamaicantruestories.com",
        "https://www.jamaicantruestories.com",
        "/",
    )

    if raw.startswith(allowed_prefixes):
        return raw

    return Config.RETURN_URL


def should_show_trace():
    return (
        os.getenv("FLASK_DEBUG", "0") == "1"
        or os.getenv("SHOW_ERROR_TRACE", "0") == "1"
    )


# ============================================================
# Home Page
# ============================================================

@interpreter_bp.route("/", methods=["GET"])
def home():
    return render_template(Config.TEMPLATE_INDEX)


# ============================================================
# Upgrade Page
# ============================================================

@interpreter_bp.route("/upgrade", methods=["GET"])
def upgrade():
    return render_template(Config.TEMPLATE_UPGRADE)


# ============================================================
# Health Check
# ============================================================

@interpreter_bp.route("/health", methods=["GET"])
def health():
    return json_ok({
        "service": "dream-interpreter",
        "mode": "standalone",
    })


# ============================================================
# Access Check
# ============================================================

@interpreter_bp.route("/check-access", methods=["POST", "OPTIONS"])
def check_access():
    if request.method == "OPTIONS":
        return empty_options_response()

    data = request.get_json(silent=True) or {}

    email = normalize_email(data.get("email", ""))

    if not email:
        return json_error(
            "Email is required to check access.",
            status_code=400,
            error="email_required",
        )

    active, details = has_active_access(email)

    if not active:
        return json_error(
            "No active access found for this email.",
            status_code=403,
            error="access_not_found",
        )

    persist_email_to_session(email)

    access_type = details.get("type", "")

    if access_type == "subscription":
        set_premium_session(email)
    else:
        set_buyer_session(email)

    return json_ok({
        "access_type": access_type,
        "return_url": safe_return_url(),
    })


# ============================================================
# Main Interpreter Endpoint
# ============================================================

@interpreter_bp.route("/interpret", methods=["POST", "OPTIONS"])
def interpret():
    if request.method == "OPTIONS":
        return empty_options_response()

    try:
        return run_interpretation()

    except Exception as e:
        trace = traceback.format_exc()
        print(trace, flush=True)

        payload = {
            "message": "Something went wrong while interpreting the dream.",
            "error_type": e.__class__.__name__,
        }

        if should_show_trace():
            payload["trace"] = trace
            payload["raw_error"] = str(e)

        return json_error(
            "Something went wrong while interpreting the dream.",
            status_code=500,
            error="interpretation_failed",
            details=payload,
        )
