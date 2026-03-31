from flask import Blueprint, jsonify, make_response, request

from app.access import get_dream_pack_status, get_session_email, is_premium_session

tracking_bp = Blueprint("tracking", __name__)


@tracking_bp.route("/track", methods=["POST", "OPTIONS"])
def track():
    if request.method == "OPTIONS":
        return make_response("", 204)

    payload = request.get_json(silent=True) or {}
    event_name = payload.get("event") or payload.get("event_type") or "unknown"

    return jsonify(
        {
            "ok": True,
            "event": event_name,
            "session_email": get_session_email(),
            "session_premium": is_premium_session(),
            "dream_pack": get_dream_pack_status(get_session_email()),
        }
    )
