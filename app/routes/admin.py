from __future__ import annotations

import time
from typing import Any, Dict

from flask import (
    Blueprint,
    jsonify,
    make_response,
    request,
)

from app.admin import (
    admin_upsert_to_sheet,
    require_admin,
)
from app.config import Config

admin_bp = Blueprint("admin", __name__)


# ============================================================
# HELPERS
# ============================================================

def _json_response(payload: Dict[str, Any], status: int = 200):
    resp = make_response(jsonify(payload), status)

    resp.headers["Cache-Control"] = "no-store"
    resp.headers["Pragma"] = "no-cache"

    return resp


def _auth_failed():
    return _json_response(
        {
            "ok": False,
            "error": "Forbidden",
        },
        403,
    )


def _safe_json():
    return request.get_json(silent=True, force=False) or {}


def _sanitize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    cleaned: Dict[str, Any] = {}

    for key, value in (payload or {}).items():
        key_clean = str(key).strip()[:120]

        if isinstance(value, str):
            cleaned[key_clean] = value.strip()[:10000]

        elif isinstance(value, list):
            cleaned[key_clean] = value[:100]

        elif isinstance(value, dict):
            cleaned[key_clean] = value

        else:
            cleaned[key_clean] = value

    return cleaned


def _debug_enabled() -> bool:
    return bool(Config.DEBUG_MATCH)


# ============================================================
# ADMIN UPSERT
# ============================================================

@admin_bp.route("/admin/upsert", methods=["POST", "OPTIONS"])
def admin_upsert():
    if request.method == "OPTIONS":
        return make_response("", 204)

    auth_fail = require_admin()

    if auth_fail:
        return _auth_failed()

    payload = _sanitize_payload(_safe_json())

    if not payload:
        return _json_response(
            {
                "ok": False,
                "error": "Missing payload.",
            },
            400,
        )

    try:
        result = admin_upsert_to_sheet(payload)

        return _json_response(
            {
                "ok": True,
                "result": result,
                "timestamp": int(time.time()),
            }
        )

    except Exception as e:
        return _json_response(
            {
                "ok": False,
                "error": "Admin upsert failed.",
                "details": str(e),
            },
            500,
        )


# ============================================================
# DEBUG CONFIG
# ============================================================

@admin_bp.route("/debug/config", methods=["GET"])
def debug_config():
    if not _debug_enabled():
        return _json_response(
            {
                "error": "Debug disabled",
            },
            403,
        )

    auth_fail = require_admin()

    if auth_fail:
        return _auth_failed()

    return _json_response(
        {
            "service": "dream-interpreter",
            "timestamp": int(time.time()),

            # ------------------------------------------------
            # CORE
            # ------------------------------------------------

            "spreadsheet_id_present": bool(Config.SPREADSHEET_ID),
            "worksheet_name": Config.WORKSHEET_NAME,
            "cache_ttl_seconds": Config.CACHE_TTL_SECONDS,
            "allowed_origins": Config.ALLOWED_ORIGINS,

            # ------------------------------------------------
            # SECURITY
            # ------------------------------------------------

            "admin_configured": bool(Config.ADMIN_KEY),
            "session_cookie_samesite": Config.SESSION_COOKIE_SAMESITE,
            "session_cookie_secure": Config.SESSION_COOKIE_SECURE,
            "return_url": Config.RETURN_URL,

            # ------------------------------------------------
            # ACCESS
            # ------------------------------------------------

            "free_quota": Config.FREE_TRIES,
            "shadow_window_hours": Config.SHADOW_WINDOW_HOURS,

            # ------------------------------------------------
            # STRIPE
            # ------------------------------------------------

            "stripe_configured": bool(Config.STRIPE_SECRET_KEY),
            "stripe_has_price": bool(Config.DEFAULT_STRIPE_PRICE_ID),
            "stripe_has_webhook": bool(Config.STRIPE_WEBHOOK_SECRET),
            "dream_pack_enabled": bool(Config.PRICE_DREAM_PACK),
            "dream_pack_uses": Config.DREAM_PACK_USES,
            "dream_pack_hours": Config.DREAM_PACK_HOURS,

            # ------------------------------------------------
            # FILES
            # ------------------------------------------------

            "counts_file": Config.COUNTS_FILE,
            "subscribers_file": Config.SUBSCRIBERS_FILE,

            # ------------------------------------------------
            # DOCTRINE
            # ------------------------------------------------

            "doctrine_mode_enabled": Config.DOCTRINE_MODE,
            "doctrine_sheet_names": Config.DOCTRINE_SHEET_NAMES,
            "relationship_rules_sheet": Config.SHEET_RELATIONSHIP_RULES,

            # ------------------------------------------------
            # NARRATION
            # ------------------------------------------------

            "narration_enabled": Config.NARRATION_ENABLED,
            "narration_mode": Config.NARRATION_MODE,
            "ai_narration_enabled": Config.AI_NARRATION_ENABLED,
            "ai_narration_provider": Config.AI_NARRATION_PROVIDER,
            "ai_narration_model": Config.AI_NARRATION_MODEL,

            # ------------------------------------------------
            # LIMITS
            # ------------------------------------------------

            "max_dream_length": Config.MAX_DREAM_LENGTH,
            "min_dream_length": Config.MIN_DREAM_LENGTH,
            "base_match_top_k": Config.BASE_MATCH_TOP_K,
            "max_rule_hits_per_layer": Config.MAX_RULE_HITS_PER_LAYER,

            # ------------------------------------------------
            # TEMPLATES
            # ------------------------------------------------

            "template_index": Config.TEMPLATE_INDEX,
            "template_upgrade": Config.TEMPLATE_UPGRADE,
        }
    )


# ============================================================
# ADMIN PING
# ============================================================

@admin_bp.route("/admin/ping", methods=["GET"])
def admin_ping():
    auth_fail = require_admin()

    if auth_fail:
        return _auth_failed()

    return _json_response(
        {
            "ok": True,
            "admin": True,
            "service": "dream-interpreter",
            "timestamp": int(time.time()),
        }
    )
