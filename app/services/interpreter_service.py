import secrets
from typing import Any, Dict

from flask import jsonify, make_response, request

from app.access import (
    consume_dream_pack_use,
    free_tries_remaining_after_this,
    get_client_ip,
    get_cookie_tries_used,
    get_dream_pack_status,
    get_session_email,
    persist_email_to_session,
    shadow_count,
    shadow_increment,
)
from app.billing import has_active_access
from app.config import Config
from app.fields import get_symbol_cell
from app.interpretation import build_doctrine_interpretation, build_legacy_interpretation
from app.matching import match_base_symbols_doctrine, match_symbols_legacy
from app.overrides import apply_override_rules
from app.rules import detect_rule_hits
from app.seal import compute_doctrine_seal, compute_seal_from_symbol_count
from app.sheets import doctrine_available, load_doctrine_sheets, load_legacy_rows
from app.utils import normalize_email, safe_debug_payload_preview, validate_dream_text


def run_interpretation():
    data = request.get_json(silent=True) or {}
    dream = (data.get("dream") or data.get("text") or "").strip()

    if Config.DEBUG_MATCH:
        print("RAW JSON RECEIVED:", safe_debug_payload_preview(data), flush=True)
        print("RAW DREAM RECEIVED:", repr(dream), flush=True)

    validation_error = validate_dream_text(
        dream,
        min_length=Config.MIN_DREAM_LENGTH,
        max_length=Config.MAX_DREAM_LENGTH,
    )
    if validation_error:
        return jsonify({"error": validation_error}), 400

    request_email = normalize_email(
        data.get("email")
        or request.args.get("email")
        or request.form.get("email")
        or get_session_email()
        or ""
    )
    if request_email:
        persist_email_to_session(request_email)

    session_email = get_session_email()
    access_ok, access_meta = has_active_access(session_email)
    access_type = access_meta.get("type", "")
    is_paid = access_ok and access_type == "subscription"

    dream_pack_status_before = get_dream_pack_status(session_email)
    has_active_dream_pack = access_ok and access_type == "dream_pack"

    if not access_ok:
        ip = get_client_ip()
        cookie_used = get_cookie_tries_used()
        ip_used = shadow_count(ip)
        effective_used = max(cookie_used, ip_used)

        if effective_used >= Config.FREE_TRIES:
            return jsonify(
                {
                    "blocked": True,
                    "reason": "free_limit_reached",
                    "message": f"You’ve used your {Config.FREE_TRIES} free tries.",
                    "free_uses_left": 0,
                    "access": "blocked",
                    "is_paid": False,
                    "email": session_email,
                    "dream_pack_available": bool(Config.PRICE_DREAM_PACK),
                }
            ), 402

    free_uses_left = 0
    dream_pack_status_after = dream_pack_status_before

    if not access_ok:
        ip = get_client_ip()
        effective_used = max(get_cookie_tries_used(), shadow_count(ip))
        shadow_increment(ip)
        free_uses_left = free_tries_remaining_after_this(effective_used)
    elif has_active_dream_pack:
        dream_pack_status_after = consume_dream_pack_use(session_email)

    doctrine_active = Config.DOCTRINE_MODE and doctrine_available()

    if doctrine_active:
        try:
            sheets = load_doctrine_sheets()
            base_rows = sheets.get(Config.SHEET_BASE_SYMBOLS, [])
            behavior_rows = sheets.get(Config.SHEET_BEHAVIOR_RULES, [])
            state_rows = sheets.get(Config.SHEET_SIZE_STATE_RULES, [])
            location_rows = sheets.get(Config.SHEET_LOCATION_RULES, [])
            relationship_rows = sheets.get(Config.SHEET_RELATIONSHIP_RULES, [])
            override_rows = sheets.get(Config.SHEET_OVERRIDE_RULES, [])
            template_rows = sheets.get(Config.SHEET_OUTPUT_TEMPLATES, [])

            base_matches = match_base_symbols_doctrine(
                dream,
                base_rows,
                top_k=Config.BASE_MATCH_TOP_K,
            )
            behaviors = detect_rule_hits(
                dream,
                behavior_rows,
                "behavior",
                max_hits=Config.MAX_RULE_HITS_PER_LAYER,
            )
            states = detect_rule_hits(
                dream,
                state_rows,
                "state",
                max_hits=Config.MAX_RULE_HITS_PER_LAYER,
            )
            locations = detect_rule_hits(
                dream,
                location_rows,
                "location",
                max_hits=Config.MAX_RULE_HITS_PER_LAYER,
            )
            relationships = detect_rule_hits(
                dream,
                relationship_rows,
                "relationship",
                max_hits=Config.MAX_RULE_HITS_PER_LAYER,
            )

            override_hit = apply_override_rules(
                base_matches,
                behaviors,
                states,
                locations,
                relationships,
                dream,
                override_rows,
            )

            doctrine_seal = compute_doctrine_seal(
                dream,
                base_matches,
                behaviors,
                states,
                locations,
                relationships,
                override_hit,
            )

            receipt_id = f"JTS-{secrets.token_hex(4).upper()}"

            built = build_doctrine_interpretation(
                dream,
                base_matches,
                behaviors,
                states,
                locations,
                relationships,
                override_hit,
                template_rows,
                doctrine_seal,
                narrative_max_symbols=Config.NARRATIVE_MAX_SYMBOLS,
            )

            payload: Dict[str, Any] = {
                "engine_mode": "doctrine",
                "access": "paid" if is_paid else ("dream_pack" if has_active_dream_pack else "free"),
                "is_paid": bool(is_paid),
                "email": session_email,
                "free_uses_left": free_uses_left,
                "dream_pack": dream_pack_status_after,
                "seal": doctrine_seal,
                "brain": {
                    "behaviors": [b["name"] for b in behaviors],
                    "states": [s["name"] for s in states],
                    "locations": [l["name"] for l in locations],
                    "relationships": [r["name"] for r in relationships],
                    "override_applied": built["override_applied"],
                    "override_name": built["override_name"],
                    "template_type": built.get("template_type", "default"),
                },
                "interpretation": built["interpretation"],
                "full_interpretation": built["full_interpretation"],
                "receipt": {
                    "id": receipt_id,
                    "top_symbols": built["top_symbols"],
                    "share_phrase": "I decoded my dream on Jamaican True Stories.",
                },
            }

            resp = make_response(jsonify(payload))

        except Exception as e:
            return jsonify({"error": "Doctrine engine failed", "details": str(e)}), 500

    else:
        try:
            legacy_rows = load_legacy_rows()
        except Exception as e:
            return jsonify({"error": "Sheet load failed", "details": str(e)}), 500

        matches = match_symbols_legacy(
            dream,
            legacy_rows,
            top_k=Config.BASE_MATCH_TOP_K,
        )
        receipt_id = f"JTS-{secrets.token_hex(4).upper()}"
        seal = compute_seal_from_symbol_count(len(matches))
        built_legacy = build_legacy_interpretation(
            matches,
            narrative_max_symbols=Config.NARRATIVE_MAX_SYMBOLS,
        )

        payload = {
            "engine_mode": "legacy",
            "access": "paid" if is_paid else ("dream_pack" if has_active_dream_pack else "free"),
            "is_paid": bool(is_paid),
            "email": session_email,
            "free_uses_left": free_uses_left,
            "dream_pack": dream_pack_status_after,
            "seal": seal,
            "brain": {},
            "interpretation": built_legacy,
            "full_interpretation": "\n\n".join(
                [
                    "This dream is revealing symbolic meaning through the matched symbols.",
                    built_legacy["spiritual_meaning"],
                    built_legacy["effects_in_physical_realm"],
                    built_legacy["what_to_do"],
                ]
            ),
            "receipt": {
                "id": receipt_id,
                "top_symbols": [
                    get_symbol_cell(row)
                    for row, _score, _hit in matches
                    if get_symbol_cell(row)
                ],
                "share_phrase": "I decoded my dream on Jamaican True Stories.",
            },
        }

        resp = make_response(jsonify(payload))

    if not access_ok:
        resp.set_cookie(
            Config.COOKIE_NAME,
            str(get_cookie_tries_used() + 1),
            max_age=Config.COOKIE_MAX_AGE,
            samesite=Config.SESSION_COOKIE_SAMESITE,
            secure=Config.SESSION_COOKIE_SECURE,
        )
    else:
        resp.set_cookie(
            Config.COOKIE_NAME,
            "0",
            max_age=0,
            samesite=Config.SESSION_COOKIE_SAMESITE,
            secure=Config.SESSION_COOKIE_SECURE,
        )

    return resp
