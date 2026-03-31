from flask import Blueprint, jsonify, request

from app.admin import admin_upsert_to_sheet, require_admin
from app.config import Config

admin_bp = Blueprint("admin", __name__)


@admin_bp.route("/admin/upsert", methods=["POST", "OPTIONS"])
def admin_upsert():
    if request.method == "OPTIONS":
        from flask import make_response
        return make_response("", 204)

    auth_fail = require_admin()
    if auth_fail:
        return jsonify({"ok": False, "error": "Forbidden"}), 403

    payload = request.get_json(silent=True) or {}
    try:
        result = admin_upsert_to_sheet(payload)
        return jsonify(result)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@admin_bp.route("/debug/config", methods=["GET"])
def debug_config():
    if not Config.DEBUG_MATCH:
        return jsonify({"error": "Debug disabled"}), 403

    return jsonify(
        {
            "spreadsheet_id": Config.SPREADSHEET_ID,
            "worksheet_name": Config.WORKSHEET_NAME,
            "cache_ttl_seconds": Config.CACHE_TTL_SECONDS,
            "allowed_origins": Config.ALLOWED_ORIGINS,
            "admin_configured": bool(Config.ADMIN_KEY),
            "free_quota": Config.FREE_TRIES,
            "shadow_window_hours": Config.SHADOW_WINDOW_HOURS,
            "stripe_has_key": bool(Config.STRIPE_SECRET_KEY),
            "stripe_has_price": bool(Config.DEFAULT_STRIPE_PRICE_ID),
            "webhook_configured": bool(Config.STRIPE_WEBHOOK_SECRET),
            "price_dream_pack_set": bool(Config.PRICE_DREAM_PACK),
            "dream_pack_uses": Config.DREAM_PACK_USES,
            "dream_pack_hours": Config.DREAM_PACK_HOURS,
            "counts_file": Config.COUNTS_FILE,
            "subscribers_file": Config.SUBSCRIBERS_FILE,
            "cookie_samesite": Config.SESSION_COOKIE_SAMESITE,
            "return_url": Config.RETURN_URL,
            "doctrine_sheet_names": Config.DOCTRINE_SHEET_NAMES,
            "relationship_rules_sheet": Config.SHEET_RELATIONSHIP_RULES,
            "template_index": Config.TEMPLATE_INDEX,
            "template_upgrade": Config.TEMPLATE_UPGRADE,
        }
    )
