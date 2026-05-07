from __future__ import annotations

import time
from typing import Any, Dict

from flask import Blueprint, jsonify, make_response

from app.access import (
    get_dream_pack_status,
    get_session_email,
    is_premium_session,
)
from app.billing import stripe_config_ok
from app.config import Config
from app.sheets import (
    doctrine_available,
    get_spreadsheet,
)

health_bp = Blueprint("health", __name__)


# ============================================================
# HELPERS
# ============================================================

def _safe_bool(value: Any) -> bool:
    try:
        return bool(value)
    except Exception:
        return False


def _build_response(payload: Dict[str, Any], status: int = 200):
    resp = make_response(jsonify(payload), status)

    resp.headers["Cache-Control"] = "no-store"
    resp.headers["Pragma"] = "no-cache"

    return resp


def _system_snapshot() -> Dict[str, Any]:
    spreadsheet_ok = False
    doctrine_ok = False
    journal_ok = False
    spreadsheet_error = ""

    try:
        get_spreadsheet()

        spreadsheet_ok = True
        doctrine_ok = doctrine_available()
        journal_ok = True

    except Exception as e:
        spreadsheet_error = str(e)

    session_email = get_session_email()

    return {
        "timestamp": int(time.time()),
        "service": "dream-interpreter",
        "status": "healthy" if spreadsheet_ok else "degraded",
        "spreadsheet_connected": spreadsheet_ok,
        "spreadsheet_error": spreadsheet_error,
        "doctrine_sheets_available": doctrine_ok,
        "dream_journal_available": journal_ok,
        "stripe_configured": _safe_bool(stripe_config_ok()),
        "doctrine_mode_enabled": _safe_bool(Config.DOCTRINE_MODE),
        "narration_enabled": _safe_bool(Config.NARRATION_ENABLED),
        "ai_narration_enabled": _safe_bool(Config.AI_NARRATION_ENABLED),
        "free_quota": int(Config.FREE_TRIES),
        "dream_pack_enabled": bool(Config.PRICE_DREAM_PACK),
        "return_url": Config.RETURN_URL,
        "worksheet_name": Config.WORKSHEET_NAME,
        "spreadsheet_id_present": bool(Config.SPREADSHEET_ID),
        "allowed_origins": Config.ALLOWED_ORIGINS,
        "doctrine_sheet_names": Config.DOCTRINE_SHEET_NAMES,
        "dream_journal_sheet": Config.SHEET_DREAM_JOURNAL,
        "template_index": Config.TEMPLATE_INDEX,
        "template_upgrade": Config.TEMPLATE_UPGRADE,
        "session": {
            "premium": is_premium_session(),
            "email": session_email,
            "dream_pack": get_dream_pack_status(session_email),
        },
        "engine": {
            "priority_order": [
                "action",
                "subject",
                "place",
                "ending",
            ],
            "matching_mode": "strict_word_boundary_with_longest_phrase_priority_and_overlap_guard",
            "layers": [
                "base_symbol_logic",
                "behavior_logic",
                "state_logic",
                "location_logic",
                "relationship_logic",
                "ending_seal_logic",
                "override_logic",
                "template_logic",
            ],
        },
        "limits": {
            "max_dream_length": Config.MAX_DREAM_LENGTH,
            "min_dream_length": Config.MIN_DREAM_LENGTH,
            "base_match_top_k": Config.BASE_MATCH_TOP_K,
            "max_rule_hits_per_layer": Config.MAX_RULE_HITS_PER_LAYER,
            "cache_ttl_seconds": Config.CACHE_TTL_SECONDS,
        },
    }


# ============================================================
# HEALTH
# ============================================================

@health_bp.route("/health", methods=["GET"])
def health():
    snapshot = _system_snapshot()

    status_code = 200

    if snapshot["status"] != "healthy":
        status_code = 503

    return _build_response(snapshot, status=status_code)


# ============================================================
# HEALTHZ
# ============================================================

@health_bp.route("/healthz", methods=["GET"])
def healthz():
    return health()


# ============================================================
# READY
# ============================================================

@health_bp.route("/ready", methods=["GET"])
def ready():
    snapshot = _system_snapshot()

    ready_state = (
        snapshot["spreadsheet_connected"]
        and snapshot["doctrine_mode_enabled"]
    )

    return _build_response(
        {
            "ready": ready_state,
            "status": snapshot["status"],
            "service": snapshot["service"],
        },
        status=200 if ready_state else 503,
    )


# ============================================================
# LIVE
# ============================================================

@health_bp.route("/live", methods=["GET"])
def live():
    return _build_response(
        {
            "alive": True,
            "service": "dream-interpreter",
        },
        status=200,
    )
