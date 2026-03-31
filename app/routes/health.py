from flask import Blueprint, jsonify

from app.access import get_dream_pack_status, get_session_email, is_premium_session
from app.billing import stripe_config_ok
from app.config import Config
from app.sheets import doctrine_available, get_spreadsheet

health_bp = Blueprint("health", __name__)


@health_bp.route("/health", methods=["GET"])
def health():
    doctrine_ok = False
    journal_ok = False
    self_healing_ready = False

    try:
        get_spreadsheet()
        self_healing_ready = True
        doctrine_ok = doctrine_available()
        journal_ok = True
    except Exception:
        doctrine_ok = False
        journal_ok = False
        self_healing_ready = False

    return jsonify(
        {
            "ok": True,
            "service": "dream-interpreter",
            "sheet": Config.WORKSHEET_NAME,
            "has_spreadsheet_id": bool(Config.SPREADSHEET_ID),
            "allowed_origins": Config.ALLOWED_ORIGINS,
            "match_mode": "strict_word_boundary_with_longest_phrase_priority_and_overlap_guard",
            "brain_layers": [
                "base_symbol_logic",
                "behavior_logic",
                "state_logic",
                "location_logic",
                "relationship_logic",
                "ending_seal_logic",
                "override_logic",
                "template_logic",
            ],
            "doctrine_mode_enabled": Config.DOCTRINE_MODE,
            "doctrine_sheets_available": doctrine_ok,
            "doctrine_sheet_names": Config.DOCTRINE_SHEET_NAMES,
            "dream_journal_sheet": Config.SHEET_DREAM_JOURNAL,
            "dream_journal_available": journal_ok,
            "self_healing_worksheets_ready": self_healing_ready,
            "free_quota": Config.FREE_TRIES,
            "stripe_configured": bool(stripe_config_ok()),
            "price_dream_pack_set": bool(Config.PRICE_DREAM_PACK),
            "return_url": Config.RETURN_URL,
            "session_premium": is_premium_session(),
            "session_email": get_session_email(),
            "session_dream_pack": get_dream_pack_status(get_session_email()),
            "template_index": Config.TEMPLATE_INDEX,
            "template_upgrade": Config.TEMPLATE_UPGRADE,
        }
    )


@health_bp.route("/healthz", methods=["GET"])
def healthz():
    return health()
