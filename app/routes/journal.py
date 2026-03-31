from flask import Blueprint, jsonify, make_response, request

from app.journal import append_dream_journal_entry, get_journal_history
from app.utils import normalize_email, validate_email

journal_bp = Blueprint("journal", __name__)


@journal_bp.route("/journal/save", methods=["POST", "OPTIONS"])
def journal_save():
    if request.method == "OPTIONS":
        return make_response("", 204)

    data = request.get_json(silent=True) or {}
    email = normalize_email(data.get("email") or "")
    dream_text = (data.get("dream_text") or "").strip()

    if not validate_email(email):
        return jsonify({"ok": False, "error": "Valid email is required."}), 400
    if not dream_text:
        return jsonify({"ok": False, "error": "Dream text is required."}), 400

    try:
        result = append_dream_journal_entry(
            {
                "email": email,
                "dream_text": dream_text,
                "spiritual_meaning": data.get("spiritual_meaning", ""),
                "effects_in_physical_realm": data.get("effects_in_physical_realm", ""),
                "what_to_do": data.get("what_to_do", ""),
                "full_interpretation": data.get("full_interpretation", ""),
                "receipt_id": data.get("receipt_id", ""),
                "top_symbols": data.get("top_symbols", []),
                "seal_status": data.get("seal_status", ""),
                "seal_type": data.get("seal_type", ""),
                "seal_risk": data.get("seal_risk", ""),
                "engine_mode": data.get("engine_mode", ""),
                "access_type": data.get("access_type", ""),
                "is_saved": "yes",
                "notes": data.get("notes", ""),
            }
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@journal_bp.route("/journal/history", methods=["GET", "OPTIONS"])
def journal_history():
    if request.method == "OPTIONS":
        return make_response("", 204)

    email = normalize_email(request.args.get("email") or "")
    if not validate_email(email):
        return jsonify({"ok": False, "error": "Valid email is required."}), 400

    try:
        return jsonify(get_journal_history(email))
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500
