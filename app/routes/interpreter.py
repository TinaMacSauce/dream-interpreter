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
    return jsonify({
        "ok": True,
        "service": "dream-interpreter",
        "mode": "standalone",
    })


# ============================================================
# Access Check
# ============================================================
@interpreter_bp.route("/check-access", methods=["POST", "OPTIONS"])
def check_access():

    if request.method == "OPTIONS":
        return make_response("", 204)

    data = request.get_json(silent=True) or {}

    email = normalize_email(
        data.get("email", "")
    )

    if not email:
        return jsonify({
            "ok": False,
            "error": "Email required.",
        }), 400

    active, details = has_active_access(email)

    if not active:
        return jsonify({
            "ok": False,
            "error": "No active access found.",
        }), 403

    persist_email_to_session(email)

    access_type = details.get("type", "")

    if access_type == "subscription":
        set_premium_session(email)
    else:
        set_buyer_session(email)

    return jsonify({
        "ok": True,
        "access_type": access_type,
        "return_url": Config.RETURN_URL,
    })


# ============================================================
# Main Interpreter Endpoint
# ============================================================
@interpreter_bp.route("/interpret", methods=["POST", "OPTIONS"])
def interpret():

    if request.method == "OPTIONS":
        return make_response("", 204)

    return run_interpretation()
