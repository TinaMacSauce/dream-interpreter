from __future__ import annotations

import secrets
import time
from typing import Any, Dict

from flask import (
    Blueprint,
    jsonify,
    make_response,
    request,
)

from app.access import (
    get_dream_pack_status,
    mark_dream_pack_purchase,
    set_buyer_session,
)
from app.admin import (
    admin_upsert_to_sheet,
    require_admin,
)
from app.config import Config
from app.utils import normalize_email, validate_email

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


def _bounded_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default

    return max(minimum, min(maximum, parsed))


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
# TEMPORARY QA ACCESS
# ============================================================

QA_EMAIL_DOMAIN = "qa.jamaicantruestories.com"
QA_DEFAULT_USES = 25
QA_MAX_USES = 50
QA_DEFAULT_HOURS = 2
QA_MAX_HOURS = 6


@admin_bp.route("/admin/qa-grant", methods=["POST", "OPTIONS"])
def admin_qa_grant():
    """
    Issue a short-lived, non-billable test allowance for live regression QA.

    The grant reuses the existing Dream Pack access machinery so the normal
    interpretation path, doctrine engine, narration, and post-success access
    deduction are exercised exactly as they are for production requests.
    Only reserved QA aliases are accepted, and every grant is bounded by both
    use count and expiry time.
    """
    if request.method == "OPTIONS":
        return make_response("", 204)

    auth_fail = require_admin()

    if auth_fail:
        return _auth_failed()

    payload = _sanitize_payload(_safe_json())
    email = normalize_email(payload.get("email") or "")

    if not validate_email(email):
        return _json_response(
            {
                "ok": False,
                "error": "A valid QA email is required.",
            },
            400,
        )

    if not email.endswith(f"@{QA_EMAIL_DOMAIN}"):
        return _json_response(
            {
                "ok": False,
                "error": (
                    "QA grants are restricted to reserved "
                    f"@{QA_EMAIL_DOMAIN} aliases."
                ),
            },
            400,
        )

    uses = _bounded_int(
        payload.get("uses"),
        default=QA_DEFAULT_USES,
        minimum=1,
        maximum=QA_MAX_USES,
    )
    hours = _bounded_int(
        payload.get("hours"),
        default=QA_DEFAULT_HOURS,
        minimum=1,
        maximum=QA_MAX_HOURS,
    )

    grant_id = (
        f"qa:{int(time.time())}:"
        f"{secrets.token_hex(6)}"
    )

    granted = mark_dream_pack_purchase(
        email=email,
        uses=uses,
        hours=hours,
        stripe_checkout_session_id=grant_id,
    )

    if not granted:
        return _json_response(
            {
                "ok": False,
                "error": "QA access grant could not be created.",
            },
            409,
        )

    # Bind the reserved QA alias to the current browser/session so the tester
    # can immediately exercise the normal /interpret route without repeatedly
    # supplying an email address.
    set_buyer_session(email)
    status = get_dream_pack_status(email)

    return _json_response(
        {
            "ok": True,
            "qa": True,
            "access_type": "temporary_qa",
            "email": email,
            "uses_remaining": status.get("uses_remaining", 0),
            "expires_at": status.get("expires_at", ""),
            "hard_limits": {
                "max_uses_per_grant": QA_MAX_USES,
                "max_hours_per_grant": QA_MAX_HOURS,
            },
            "timestamp": int(time.time()),
        }
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