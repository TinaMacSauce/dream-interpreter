from __future__ import annotations

from typing import Any, Dict

from flask import (
    Blueprint,
    jsonify,
    make_response,
    request,
)

from app.config import Config
from app.journal import (
    append_dream_journal_entry,
    get_journal_history,
)
from app.utils import (
    normalize_email,
    validate_dream_text,
    validate_email,
)

journal_bp = Blueprint("journal", __name__)


# ============================================================
# HELPERS
# ============================================================

def _json_response(payload: Dict[str, Any], status: int = 200):
    resp = make_response(jsonify(payload), status)

    resp.headers["Cache-Control"] = "no-store"
    resp.headers["Pragma"] = "no-cache"

    return resp


def _extract_json() -> Dict[str, Any]:
    return request.get_json(silent=True, force=False) or {}


def _safe_string(value: Any, max_len: int = 5000) -> str:
    text = str(value or "").strip()

    if len(text) > max_len:
        return text[:max_len]

    return text


def _safe_list(value: Any, max_items: int = 20):
    if not isinstance(value, list):
        return []

    cleaned = []

    for item in value[:max_items]:
        item_text = _safe_string(item, 120)

        if item_text:
            cleaned.append(item_text)

    return cleaned


def _build_entry_payload(data: Dict[str, Any], email: str) -> Dict[str, Any]:
    return {
        "email": email,
        "dream_text": _safe_string(
            data.get("dream_text"),
            Config.MAX_DREAM_LENGTH,
        ),
        "spiritual_meaning": _safe_string(
            data.get("spiritual_meaning"),
            4000,
        ),
        "effects_in_physical_realm": _safe_string(
            data.get("effects_in_physical_realm"),
            4000,
        ),
        "what_to_do": _safe_string(
            data.get("what_to_do"),
            4000,
        ),
        "full_interpretation": _safe_string(
            data.get("full_interpretation"),
            12000,
        ),
        "receipt_id": _safe_string(
            data.get("receipt_id"),
            200,
        ),
        "top_symbols": _safe_list(
            data.get("top_symbols", []),
            max_items=20,
        ),
        "seal_status": _safe_string(
            data.get("seal_status"),
            200,
        ),
        "seal_type": _safe_string(
            data.get("seal_type"),
            200,
        ),
        "seal_risk": _safe_string(
            data.get("seal_risk"),
            200,
        ),
        "engine_mode": _safe_string(
            data.get("engine_mode"),
            100,
        ),
        "access_type": _safe_string(
            data.get("access_type"),
            100,
        ),
        "is_saved": "yes",
        "notes": _safe_string(
            data.get("notes"),
            3000,
        ),
    }


# ============================================================
# SAVE JOURNAL ENTRY
# ============================================================

@journal_bp.route("/journal/save", methods=["POST", "OPTIONS"])
def journal_save():
    if request.method == "OPTIONS":
        return make_response("", 204)

    data = _extract_json()

    email = normalize_email(
        data.get("email") or ""
    )

    dream_text = _safe_string(
        data.get("dream_text"),
        Config.MAX_DREAM_LENGTH,
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if not validate_email(email):
        return _json_response(
            {
                "ok": False,
                "error": "Valid email is required.",
            },
            400,
        )

    validation_error = validate_dream_text(
        dream_text,
        min_length=Config.MIN_DREAM_LENGTH,
        max_length=Config.MAX_DREAM_LENGTH,
    )

    if validation_error:
        return _json_response(
            {
                "ok": False,
                "error": validation_error,
            },
            400,
        )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    try:
        payload = _build_entry_payload(data, email)

        result = append_dream_journal_entry(payload)

        return _json_response(
            {
                "ok": True,
                "saved": True,
                "entry": result,
            }
        )

    except Exception as e:
        return _json_response(
            {
                "ok": False,
                "error": "Failed to save journal entry.",
                "details": str(e),
            },
            500,
        )


# ============================================================
# JOURNAL HISTORY
# ============================================================

@journal_bp.route("/journal/history", methods=["GET", "OPTIONS"])
def journal_history():
    if request.method == "OPTIONS":
        return make_response("", 204)

    email = normalize_email(
        request.args.get("email") or ""
    )

    if not validate_email(email):
        return _json_response(
            {
                "ok": False,
                "error": "Valid email is required.",
            },
            400,
        )

    try:
        history = get_journal_history(email)

        return _json_response(
            {
                "ok": True,
                "email": email,
                "history": history,
                "count": len(history) if isinstance(history, list) else 0,
            }
        )

    except Exception as e:
        return _json_response(
            {
                "ok": False,
                "error": "Failed to load journal history.",
                "details": str(e),
            },
            500,
        )


# ============================================================
# JOURNAL PING
# ============================================================

@journal_bp.route("/journal/ping", methods=["GET"])
def journal_ping():
    return _json_response(
        {
            "ok": True,
            "service": "journal",
            "enabled": True,
            "sheet": Config.SHEET_DREAM_JOURNAL,
        }
    )
