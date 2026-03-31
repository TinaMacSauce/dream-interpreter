import secrets
from typing import Any, Dict, Tuple

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


def _access_label(is_paid: bool, has_active_dream_pack: bool) -> str:
    if is_paid:
        return "paid"
    if has_active_dream_pack:
        return "dream_pack"
    return "free"


def _build_receipt(top_symbols) -> Dict[str, Any]:
    return {
        "id": f"JTS-{secrets.token_hex(4).upper()}",
        "top_symbols": top_symbols or [],
        "share_phrase": "I decoded my dream on Jamaican True Stories.",
    }


def _get_request_email(data: Dict[str, Any]) -> str:
    return normalize_email(
        data.get("email")
        or request.args.get("email")
        or request.form.get("email")
        or get_session_email()
        or ""
    )


def _check_and_apply_access(session_email: str) -> Tuple[bool, Dict[str, Any], bool, bool, int, Dict[str, Any]]:
    access_ok, access_meta = has_active_access(session_email)
    access_type = access_meta.get("type", "")
    is_paid = access_ok and access_type == "subscription"
    has_active_dream_pack = access_ok and access_type == "dream_pack"

    dream_pack_status_before = get_dream_pack_status(session_email)
    dream_pack_status_after = dream_pack_status_before
    free_uses_left = 0

    if not access_ok:
        ip = get_client_ip()
        cookie_used = get_cookie_tries_used()
        ip_used = shadow_count(ip)
        effective_used = max(cookie_used, ip_used)

        if effective_used >= Config.FREE_TRIES:
            raise PermissionError("free_limit_reached")

        shadow_increment(ip)
        free_uses_left = free_tries_remaining_after_this(effective_used)

    elif has_active_dream_pack:
        dream_pack_status_after = consume_dream_pack_use(session_email)

    return access_ok, access_meta, is_paid, has_active_dream_pack, free_uses_left, dream_pack_status_after


def _build_doctrine_payload(
    dream: str,
    session_email: str,
    is_paid: bool,
    has_active_dream_pack: bool,
    free_uses_left: int,
    dream_pack_status_after: Dict[str, Any],
) -> Dict[str, Any]:
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
        "access": _access_label(is_paid, has_active_dream_pack),
        "is_paid": bool(is_paid),
        "email": session_email,
        "free_uses_left": free_uses_left,
        "dream_pack": dream_pack_status_after,
        "seal": doctrine_seal,
        "brain": {
            "behaviors": [b.get("name", "") for b in behaviors if b.get("name")],
            "states": [s.get("name", "") for s in states if s.get("name")],
            "locations": [l.get("name", "") for l in locations if l.get("name")],
            "relationships": [r.get("name", "") for r in relationships if r.get("name")],
            "override_applied": built.get("override_applied", False),
            "override_name": built.get("override_name", ""),
            "template_type": built.get("template_type", "default"),
        },
        "interpretation": built["interpretation"],
        "full_interpretation": built["full_interpretation"],
        "receipt": _build_receipt(built.get("top_symbols", [])),
        "doctrine_facts": built.get("doctrine_facts", {}),
    }

    if Config.DEBUG_MATCH:
        payload["debug"] = {
            "base_match_count": len(base_matches),
            "behavior_count": len(behaviors),
            "state_count": len(states),
            "location_count": len(locations),
            "relationship_count": len(relationships),
        }

    return payload


def _build_legacy_payload(
    dream: str,
    session_email: str,
    is_paid: bool,
    has_active_dream_pack: bool,
    free_uses_left: int,
    dream_pack_status_after: Dict[str, Any],
) -> Dict[str, Any]:
    legacy_rows = load_legacy_rows()

    matches = match_symbols_legacy(
        dream,
        legacy_rows,
        top_k=Config.BASE_MATCH_TOP_K,
    )

    seal = compute_seal_from_symbol_count(len(matches))
    built_legacy = build_legacy_interpretation(
        matches,
        narrative_max_symbols=Config.NARRATIVE_MAX_SYMBOLS,
    )

    top_symbols = [
        get_symbol_cell(row)
        for row, _score, _hit in matches
        if get_symbol_cell(row)
    ]

    payload: Dict[str, Any] = {
        "engine_mode": "legacy",
        "access": _access_label(is_paid, has_active_dream_pack),
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
        "receipt": _build_receipt(top_symbols),
        "doctrine_facts": {},
    }

    if Config.DEBUG_MATCH:
        payload["debug"] = {
            "legacy_match_count": len(matches),
        }

    return payload


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

    request_email = _get_request_email(data)
    if request_email:
        persist_email_to_session(request_email)

    session_email = get_session_email()

    try:
        access_ok, access_meta, is_paid, has_active_dream_pack, free_uses_left, dream_pack_status_after = _check_and_apply_access(
            session_email
        )
    except PermissionError:
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
    except Exception as e:
        return jsonify({"error": "Access check failed", "details": str(e)}), 500

    doctrine_active = Config.DOCTRINE_MODE and doctrine_available()

    try:
        if doctrine_active:
            payload = _build_doctrine_payload(
                dream=dream,
                session_email=session_email,
                is_paid=is_paid,
                has_active_dream_pack=has_active_dream_pack,
                free_uses_left=free_uses_left,
                dream_pack_status_after=dream_pack_status_after,
            )
        else:
            payload = _build_legacy_payload(
                dream=dream,
                session_email=session_email,
                is_paid=is_paid,
                has_active_dream_pack=has_active_dream_pack,
                free_uses_left=free_uses_left,
                dream_pack_status_after=dream_pack_status_after,
            )
    except Exception as e:
        engine_name = "Doctrine engine failed" if doctrine_active else "Legacy engine failed"
        return jsonify({"error": engine_name, "details": str(e)}), 500

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
