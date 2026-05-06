import re
import secrets
from typing import Any, Dict, List, Optional, Tuple

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
from app.services.narration_service import build_narration_result
from app.sheets import doctrine_available, load_doctrine_sheets, load_legacy_rows
from app.utils import normalize_email, safe_debug_payload_preview, validate_dream_text


# ---------------------------------------------------------
# ACCESS / RECEIPT HELPERS
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# GENERAL NORMALIZATION HELPERS
# ---------------------------------------------------------

def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _clean_lower(value: Any) -> str:
    return _clean_text(value).lower()


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value or default))
    except Exception:
        return default


def _row_name(row: Dict[str, Any]) -> str:
    return (
        _clean_text(row.get("name"))
        or _clean_text(row.get("behavior_name"))
        or _clean_text(row.get("location_name"))
        or _clean_text(row.get("state_name"))
        or _clean_text(row.get("relationship_name"))
        or _clean_text(row.get("ending_name"))
        or _clean_text(row.get("symbol"))
        or _clean_text(row.get("condition"))
        or _clean_text(row.get("input"))
    )


def _row_priority(row: Dict[str, Any]) -> int:
    return _safe_int(row.get("priority"), 0)


def _sort_hits_by_priority(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        hits or [],
        key=lambda row: (_row_priority(row), len(_row_name(row))),
        reverse=True,
    )


def _first_hit(hits: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    ordered = _sort_hits_by_priority(hits)
    return ordered[0] if ordered else None


def _split_keywords(value: Any) -> List[str]:
    raw = _clean_text(value)
    if not raw:
        return []
    return [part.strip().lower() for part in raw.split(",") if part.strip()]


def _phrase_exists(dream: str, phrase: str) -> bool:
    phrase = _clean_lower(phrase)
    if not phrase:
        return False

    dream_l = _clean_lower(dream)

    # Multi-word phrase: exact phrase match.
    if " " in phrase:
        return phrase in dream_l

    # Single word: word-boundary match to avoid weak collisions.
    return re.search(rf"\b{re.escape(phrase)}\b", dream_l) is not None


# ---------------------------------------------------------
# EVENT LOGIC HELPERS
# Priority order:
# ACTION first, SUBJECT second, PLACE third, ENDING last.
# ---------------------------------------------------------

def _detect_endings(dream: str, ending_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    EndingRules is a new optional Google Sheet.
    Columns should include:
    ending_name, keywords, outcome_meaning, physical_modifier,
    action_modifier, priority, active
    """
    if not ending_rows:
        return []

    hits: List[Dict[str, Any]] = []

    for row in ending_rows:
        if _clean_lower(row.get("active")) not in ("yes", "true", "1", "active"):
            continue

        keywords = _split_keywords(row.get("keywords"))
        name = _clean_lower(row.get("ending_name"))

        possible_phrases = keywords + ([name] if name else [])

        for phrase in possible_phrases:
            if _phrase_exists(dream, phrase):
                hit = dict(row)
                hit["name"] = _clean_text(row.get("ending_name")) or phrase
                hit["matched_keyword"] = phrase
                hit["layer"] = "ending"
                hits.append(hit)
                break

    return _sort_hits_by_priority(hits)


def _detect_old_place_cluster(dream: str) -> Optional[Dict[str, Any]]:
    """
    Safety net for your doctrine:
    old school / old house / old places = backwardness, stagnation, regression.
    This helps even if the Google Sheet row is missing or not loaded yet.
    """
    old_place_keywords = [
        "old school",
        "old house",
        "old neighborhood",
        "old job",
        "old workplace",
        "former workplace",
        "childhood home",
        "old classroom",
        "primary school",
        "old church",
        "old yard",
        "old place",
        "old places",
    ]

    for phrase in old_place_keywords:
        if _phrase_exists(dream, phrase):
            return {
                "name": "old_place",
                "location_name": "old_place",
                "keywords": ", ".join(old_place_keywords),
                "life_area_meaning": "backwardness, stagnation, regression, or being pulled back into old cycles",
                "physical_area_meaning": "delay, repeated patterns, unfinished past issues, or blocked progress",
                "action_modifier": "pray to break backward cycles and move forward",
                "priority": 100,
                "active": "yes",
                "matched_keyword": phrase,
                "layer": "location",
            }

    return None


def _ensure_old_place_location(dream: str, locations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    old_place = _detect_old_place_cluster(dream)
    if not old_place:
        return locations

    existing_names = {_clean_lower(_row_name(row)) for row in locations}
    if "old_place" not in existing_names and "old place" not in existing_names:
        return _sort_hits_by_priority([old_place] + (locations or []))

    return _sort_hits_by_priority(locations)


def _match_base_symbols_with_context(
    dream: str,
    base_rows: List[Dict[str, Any]],
    behaviors: List[Dict[str, Any]],
    states: List[Dict[str, Any]],
    locations: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
):
    """
    Some older versions of match_base_symbols_doctrine only accept:
    dream, base_rows, top_k.

    Some newer versions may accept contextual layers.
    This helper tries the better context-aware call first,
    then safely falls back to the older call.
    """
    try:
        return match_base_symbols_doctrine(
            dream,
            base_rows,
            top_k=Config.BASE_MATCH_TOP_K,
            behaviors=behaviors,
            states=states,
            locations=locations,
            relationships=relationships,
        )
    except TypeError:
        return match_base_symbols_doctrine(
            dream,
            base_rows,
            top_k=Config.BASE_MATCH_TOP_K,
        )


def _base_match_symbol_name(match: Any) -> str:
    try:
        row, _score, _hit = match
        return get_symbol_cell(row) or row.get("symbol", "") or row.get("input", "")
    except Exception:
        return ""


def _build_event_context(
    dream: str,
    base_matches,
    behaviors: List[Dict[str, Any]],
    locations: List[Dict[str, Any]],
    states: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
    endings: List[Dict[str, Any]],
) -> Dict[str, Any]:
    primary_action = _first_hit(behaviors)
    primary_place = _first_hit(locations)
    primary_state = _first_hit(states)
    primary_relationship = _first_hit(relationships)
    primary_ending = _first_hit(endings)

    subjects = []
    for match in base_matches or []:
        symbol = _base_match_symbol_name(match)
        if symbol:
            subjects.append(symbol)

    primary_subject = subjects[0] if subjects else ""

    return {
        "priority_order": ["action", "subject", "place", "ending"],
        "primary_action": {
            "name": _row_name(primary_action) if primary_action else "",
            "meaning": _clean_text(
                (primary_action or {}).get("meaning_modifier")
                or (primary_action or {}).get("behavior_effect")
                or (primary_action or {}).get("spiritual_meaning")
                or (primary_action or {}).get("effect")
            ),
            "action": _clean_text(
                (primary_action or {}).get("action_modifier")
                or (primary_action or {}).get("base_action")
                or (primary_action or {}).get("what_to_do")
            ),
            "priority": _row_priority(primary_action or {}),
        },
        "primary_subject": primary_subject,
        "subjects": subjects[: Config.NARRATIVE_MAX_SYMBOLS],
        "primary_place": {
            "name": _row_name(primary_place) if primary_place else "",
            "meaning": _clean_text(
                (primary_place or {}).get("life_area_meaning")
                or (primary_place or {}).get("location_effect")
                or (primary_place or {}).get("spiritual_meaning")
                or (primary_place or {}).get("effect")
            ),
            "physical_area": _clean_text(
                (primary_place or {}).get("physical_area_meaning")
                or (primary_place or {}).get("physical_effects")
            ),
            "action": _clean_text(
                (primary_place or {}).get("action_modifier")
                or (primary_place or {}).get("base_action")
            ),
            "priority": _row_priority(primary_place or {}),
        },
        "primary_state": {
            "name": _row_name(primary_state) if primary_state else "",
            "meaning": _clean_text(
                (primary_state or {}).get("state_effect")
                or (primary_state or {}).get("spiritual_meaning")
                or (primary_state or {}).get("effect")
            ),
            "priority": _row_priority(primary_state or {}),
        },
        "primary_relationship": {
            "name": _row_name(primary_relationship) if primary_relationship else "",
            "meaning": _clean_text(
                (primary_relationship or {}).get("relationship_effect")
                or (primary_relationship or {}).get("spiritual_meaning")
                or (primary_relationship or {}).get("effect")
            ),
            "priority": _row_priority(primary_relationship or {}),
        },
        "primary_ending": {
            "name": _row_name(primary_ending) if primary_ending else "",
            "meaning": _clean_text((primary_ending or {}).get("outcome_meaning")),
            "physical_modifier": _clean_text((primary_ending or {}).get("physical_modifier")),
            "action": _clean_text((primary_ending or {}).get("action_modifier")),
            "matched_keyword": _clean_text((primary_ending or {}).get("matched_keyword")),
            "priority": _row_priority(primary_ending or {}),
        },
    }


def _build_event_summary(event_context: Dict[str, Any]) -> str:
    """
    Human-readable event summary.
    This does not replace the full doctrine interpretation yet.
    It gives the frontend and narration layer a cleaner human logic summary.
    """
    action = event_context.get("primary_action", {})
    subject = event_context.get("primary_subject", "")
    place = event_context.get("primary_place", {})
    ending = event_context.get("primary_ending", {})

    lines: List[str] = []

    if action.get("name"):
        action_line = f"The strongest part of this dream is the action: {action['name']}."
        if action.get("meaning"):
            action_line += f" This points to {action['meaning']}."
        lines.append(action_line)

    if subject:
        lines.append(
            f"The main subject involved is {subject}, which adds detail to what the action is affecting or revealing."
        )

    if place.get("name"):
        place_line = f"The place adds context: {place['name']}."
        if place.get("meaning"):
            place_line += f" This points to {place['meaning']}."
        lines.append(place_line)

    if ending.get("name"):
        ending_line = f"The ending matters because it shows the outcome: {ending['name']}."
        if ending.get("meaning"):
            ending_line += f" This suggests {ending['meaning']}."
        lines.append(ending_line)

    if not lines:
        return ""

    return " ".join(lines)


# ---------------------------------------------------------
# OVERRIDE SAFETY
# ---------------------------------------------------------

def _override_condition_exactly_matches(dream: str, override_hit: Dict[str, Any]) -> bool:
    if not override_hit:
        return False

    # Hard override still allowed, but only if your override system marks it that way.
    if bool(override_hit.get("is_hard_override", False)):
        return True

    condition = _clean_text(override_hit.get("condition"))
    if not condition:
        return False

    # If condition has comma-separated pieces, require at least one full phrase.
    phrases = _split_keywords(condition) if "," in condition else [condition.lower()]

    for phrase in phrases:
        if phrase and _phrase_exists(dream, phrase):
            return True

    return False


def _apply_safe_override(
    base_matches,
    behaviors: List[Dict[str, Any]],
    states: List[Dict[str, Any]],
    locations: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
    dream: str,
    override_rows: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Overrides should not dominate the dream from weak keyword collisions.
    They only survive if the override condition actually appears in the dream
    or the rule is explicitly marked as hard override.
    """
    override_hit = apply_override_rules(
        base_matches,
        behaviors,
        states,
        locations,
        relationships,
        dream,
        override_rows,
    )

    if not override_hit:
        return None

    if _override_condition_exactly_matches(dream, override_hit):
        return override_hit

    return None


def _extract_override_meta(override_hit: Dict[str, Any]) -> Dict[str, Any]:
    if not override_hit:
        return {
            "applied": False,
            "name": "",
            "priority": 0,
            "condition": "",
            "target_mode": "",
            "is_hard_override": False,
        }

    return {
        "applied": True,
        "name": override_hit.get("override_name", "") or "",
        "priority": _safe_int(override_hit.get("priority"), 0),
        "condition": override_hit.get("condition", "") or "",
        "target_mode": override_hit.get("target_mode", "") or "",
        "is_hard_override": bool(override_hit.get("is_hard_override", False)),
    }


# ---------------------------------------------------------
# DOCTRINE PAYLOAD
# ---------------------------------------------------------

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

    ending_sheet_name = getattr(Config, "SHEET_ENDING_RULES", "EndingRules")
    ending_rows = sheets.get(ending_sheet_name, []) or sheets.get("EndingRules", [])

    # 1. ACTION FIRST
    behaviors = detect_rule_hits(
        dream,
        behavior_rows,
        "behavior",
        max_hits=Config.MAX_RULE_HITS_PER_LAYER,
    )
    behaviors = _sort_hits_by_priority(behaviors)

    # 2. PLACE / STATE / RELATIONSHIP CONTEXT
    states = detect_rule_hits(
        dream,
        state_rows,
        "state",
        max_hits=Config.MAX_RULE_HITS_PER_LAYER,
    )
    states = _sort_hits_by_priority(states)

    locations = detect_rule_hits(
        dream,
        location_rows,
        "location",
        max_hits=Config.MAX_RULE_HITS_PER_LAYER,
    )
    locations = _ensure_old_place_location(dream, _sort_hits_by_priority(locations))

    relationships = detect_rule_hits(
        dream,
        relationship_rows,
        "relationship",
        max_hits=Config.MAX_RULE_HITS_PER_LAYER,
    )
    relationships = _sort_hits_by_priority(relationships)

    # 3. ENDING LAST, BUT IT CHANGES THE FINAL WEIGHT
    endings = _detect_endings(dream, ending_rows)

    # 4. SUBJECT SECOND IN MEANING, BUT MATCHED AFTER ACTION CONTEXT
    base_matches = _match_base_symbols_with_context(
        dream=dream,
        base_rows=base_rows,
        behaviors=behaviors,
        states=states,
        locations=locations,
        relationships=relationships,
    )

    # 5. OVERRIDE ONLY IF SAFE / EXACT
    override_hit = _apply_safe_override(
        base_matches=base_matches,
        behaviors=behaviors,
        states=states,
        locations=locations,
        relationships=relationships,
        dream=dream,
        override_rows=override_rows,
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

    event_context = _build_event_context(
        dream=dream,
        base_matches=base_matches,
        behaviors=behaviors,
        locations=locations,
        states=states,
        relationships=relationships,
        endings=endings,
    )
    event_summary = _build_event_summary(event_context)

    doctrine_facts = built.get("doctrine_facts", {})
    doctrine_facts["event_context"] = event_context
    doctrine_facts["event_summary"] = event_summary

    narration = build_narration_result(
        doctrine_facts=doctrine_facts,
        interpretation=built.get("interpretation", {}),
        ai_enabled=Config.AI_NARRATION_ENABLED,
    )

    override_meta = _extract_override_meta(override_hit)
    engine_mode = "override" if override_meta["applied"] else "doctrine_event"

    payload: Dict[str, Any] = {
        "engine_mode": engine_mode,
        "access": _access_label(is_paid, has_active_dream_pack),
        "is_paid": bool(is_paid),
        "email": session_email,
        "free_uses_left": free_uses_left,
        "dream_pack": dream_pack_status_after,
        "seal": doctrine_seal,
        "brain": {
            "priority_order": ["action", "subject", "place", "ending"],
            "primary_action": event_context["primary_action"],
            "primary_subject": event_context["primary_subject"],
            "primary_place": event_context["primary_place"],
            "primary_ending": event_context["primary_ending"],
            "behaviors": [b.get("name", "") or _row_name(b) for b in behaviors if _row_name(b)],
            "states": [s.get("name", "") or _row_name(s) for s in states if _row_name(s)],
            "locations": [l.get("name", "") or _row_name(l) for l in locations if _row_name(l)],
            "relationships": [r.get("name", "") or _row_name(r) for r in relationships if _row_name(r)],
            "endings": [e.get("name", "") or _row_name(e) for e in endings if _row_name(e)],
            "override_applied": override_meta["applied"],
            "override_name": override_meta["name"],
            "override_priority": override_meta["priority"],
            "override_condition": override_meta["condition"],
            "override_target_mode": override_meta["target_mode"],
            "is_hard_override": override_meta["is_hard_override"],
            "template_type": built.get("template_type", "default"),
        },
        "interpretation": built["interpretation"],
        "full_interpretation": built["full_interpretation"],
        "event_summary": event_summary,
        "receipt": _build_receipt(built.get("top_symbols", [])),
        "doctrine_facts": doctrine_facts,
        "narration": narration,
    }

    if Config.DEBUG_MATCH:
        payload["debug"] = {
            "dream_preview": dream[:300],
            "base_match_count": len(base_matches),
            "behavior_count": len(behaviors),
            "state_count": len(states),
            "location_count": len(locations),
            "relationship_count": len(relationships),
            "ending_count": len(endings),
            "base_matches": [
                {
                    "symbol": row.get("symbol", "") or row.get("input", ""),
                    "score": score,
                    "match_type": (hit or {}).get("type", ""),
                    "token": (hit or {}).get("token", ""),
                }
                for row, score, hit in base_matches
            ],
            "behaviors": [b.get("name", "") or _row_name(b) for b in behaviors],
            "states": [s.get("name", "") or _row_name(s) for s in states],
            "locations": [l.get("name", "") or _row_name(l) for l in locations],
            "relationships": [r.get("name", "") or _row_name(r) for r in relationships],
            "endings": [e.get("name", "") or _row_name(e) for e in endings],
            "event_context": event_context,
            "event_summary": event_summary,
            "override": override_meta,
            "seal": doctrine_seal,
        }

    return payload


# ---------------------------------------------------------
# LEGACY PAYLOAD
# ---------------------------------------------------------

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
        "brain": {
            "priority_order": ["symbol"],
            "behaviors": [],
            "states": [],
            "locations": [],
            "relationships": [],
            "endings": [],
            "override_applied": False,
            "override_name": "",
            "override_priority": 0,
            "override_condition": "",
            "override_target_mode": "",
            "is_hard_override": False,
            "template_type": "",
        },
        "interpretation": built_legacy,
        "full_interpretation": "\n\n".join(
            [
                "This dream is revealing symbolic meaning through the matched symbols.",
                built_legacy["spiritual_meaning"],
                built_legacy["effects_in_physical_realm"],
                built_legacy["what_to_do"],
            ]
        ),
        "event_summary": "",
        "receipt": _build_receipt(top_symbols),
        "doctrine_facts": {},
        "narration": {
            "mode": "legacy_fallback",
            "enabled": False,
            "used_ai": False,
            "readable_summary": "",
            "prompt_payload": {},
        },
    }

    if Config.DEBUG_MATCH:
        payload["debug"] = {
            "dream_preview": dream[:300],
            "legacy_match_count": len(matches),
            "legacy_matches": [
                {
                    "symbol": get_symbol_cell(row),
                    "score": score,
                    "match_type": (hit or {}).get("type", "") if hit else "",
                    "token": (hit or {}).get("token", "") if hit else "",
                }
                for row, score, hit in matches
            ],
            "seal": seal,
        }

    return payload


# ---------------------------------------------------------
# MAIN ROUTE SERVICE
# ---------------------------------------------------------

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
