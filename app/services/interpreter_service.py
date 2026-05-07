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
from app.interpretation import (
    build_doctrine_interpretation,
    build_legacy_interpretation,
)
from app.matching import (
    match_base_symbols_doctrine,
    match_symbols_legacy,
)
from app.overrides import apply_override_rules
from app.rules import detect_rule_hits
from app.seal import (
    compute_doctrine_seal,
    compute_seal_from_symbol_count,
)
from app.services.narration_service import build_narration_result
from app.sheets import (
    doctrine_available,
    load_doctrine_sheets,
    load_legacy_rows,
)
from app.utils import (
    normalize_email,
    safe_debug_payload_preview,
    validate_dream_text,
)


# =========================================================
# ACCESS HELPERS
# =========================================================

def _access_label(is_paid: bool, has_dream_pack: bool) -> str:
    if is_paid:
        return "paid"

    if has_dream_pack:
        return "dream_pack"

    return "free"


def _build_receipt(top_symbols: List[str]) -> Dict[str, Any]:
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


# =========================================================
# TEXT HELPERS
# =========================================================

def _clean(value: Any) -> str:
    return str(value or "").strip()


def _lower(value: Any) -> str:
    return _clean(value).lower()


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value or default))
    except Exception:
        return default


def _split_keywords(value: Any) -> List[str]:
    raw = _clean(value)

    if not raw:
        return []

    return [
        x.strip().lower()
        for x in raw.split(",")
        if x.strip()
    ]


def _phrase_exists(text: str, phrase: str) -> bool:
    text_l = _lower(text)
    phrase_l = _lower(phrase)

    if not phrase_l:
        return False

    if " " in phrase_l:
        return phrase_l in text_l

    return re.search(
        rf"\b{re.escape(phrase_l)}\b",
        text_l,
    ) is not None


# =========================================================
# RULE HELPERS
# =========================================================

def _row_name(item: Any) -> str:
    if not item:
        return ""

    if isinstance(item, str):
        return item.strip()

    if not isinstance(item, dict):
        return str(item).strip()

    row = item.get("row") if isinstance(item.get("row"), dict) else item

    return (
        _clean(item.get("name"))
        or _clean(row.get("behavior_name"))
        or _clean(row.get("location_name"))
        or _clean(row.get("relationship_name"))
        or _clean(row.get("state_name"))
        or _clean(row.get("ending_name"))
        or _clean(row.get("symbol"))
        or _clean(row.get("override_name"))
        or _clean(row.get("input"))
    )


def _row_priority(item: Any) -> int:
    if not isinstance(item, dict):
        return 0

    if item.get("priority"):
        return _safe_int(item.get("priority"))

    row = item.get("row")

    if isinstance(row, dict):
        return _safe_int(row.get("priority"))

    return 0


def _normalize_hits(
    items: List[Any],
    kind: str,
) -> List[Dict[str, Any]]:

    normalized = []

    for item in items or []:

        if isinstance(item, dict):

            row = item.get("row") if isinstance(item.get("row"), dict) else item

            normalized.append(
                {
                    **item,
                    "row": row,
                    "name": _row_name(item),
                    "kind": kind,
                    "priority": _row_priority(item),
                }
            )

        else:

            normalized.append(
                {
                    "name": str(item),
                    "row": {},
                    "kind": kind,
                    "priority": 0,
                    "score": 0,
                }
            )

    return normalized


def _sort_hits(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        items or [],
        key=lambda x: (
            _row_priority(x),
            _safe_int(x.get("score", 0)),
            len(_row_name(x)),
        ),
        reverse=True,
    )


def _first_hit(items: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    ordered = _sort_hits(items)
    return ordered[0] if ordered else None


def _safe_names(items: List[Dict[str, Any]]) -> List[str]:
    output = []

    for item in items or []:
        name = _row_name(item)

        if name:
            output.append(name)

    return output


# =========================================================
# OLD PLACE DOCTRINE
# =========================================================

OLD_PLACE_KEYWORDS = [
    "old school",
    "old house",
    "old neighborhood",
    "old workplace",
    "former workplace",
    "old church",
    "old classroom",
    "childhood home",
    "old yard",
    "old place",
]


def _detect_old_place(dream: str):

    for phrase in OLD_PLACE_KEYWORDS:

        if _phrase_exists(dream, phrase):

            row = {
                "location_name": "old_place",
                "life_area_meaning": (
                    "backwardness, stagnation, "
                    "regression, or old cycles"
                ),
                "physical_area_meaning": (
                    "delay, repeated patterns, "
                    "unfinished past issues"
                ),
                "action_modifier": (
                    "pray against backwardness "
                    "and move forward"
                ),
                "priority": 100,
                "matched_keyword": phrase,
            }

            return {
                "name": "old_place",
                "row": row,
                "kind": "location",
                "priority": 100,
            }

    return None


# =========================================================
# ENDING DETECTION
# =========================================================

def _detect_endings(
    dream: str,
    ending_rows: List[Dict[str, Any]],
):

    hits = []

    for row in ending_rows or []:

        if not isinstance(row, dict):
            continue

        active = _lower(row.get("active"))

        if active and active not in {
            "1",
            "true",
            "yes",
            "on",
            "active",
        }:
            continue

        keywords = _split_keywords(row.get("keywords"))

        if row.get("ending_name"):
            keywords.append(_lower(row.get("ending_name")))

        for keyword in keywords:

            if _phrase_exists(dream, keyword):

                hits.append(
                    {
                        "name": _clean(
                            row.get("ending_name")
                        ) or keyword,
                        "row": row,
                        "kind": "ending",
                        "priority": _row_priority(row),
                    }
                )

                break

    return _sort_hits(_normalize_hits(hits, "ending"))


# =========================================================
# EVENT BRAIN
# =========================================================

def _base_symbol_name(match: Any) -> str:
    try:
        row, _score, _hit = match

        if not isinstance(row, dict):
            return ""

        return (
            get_symbol_cell(row)
            or row.get("symbol", "")
            or row.get("input", "")
        )

    except Exception:
        return ""


def _build_event_context(
    base_matches,
    behaviors,
    locations,
    states,
    relationships,
    endings,
):

    primary_action = _first_hit(behaviors)
    primary_place = _first_hit(locations)
    primary_state = _first_hit(states)
    primary_relationship = _first_hit(relationships)
    primary_ending = _first_hit(endings)

    subjects = []

    for match in base_matches or []:
        symbol = _base_symbol_name(match)

        if symbol:
            subjects.append(symbol)

    primary_subject = subjects[0] if subjects else ""

    return {
        "priority_order": [
            "action",
            "subject",
            "place",
            "ending",
        ],

        "primary_action": {
            "name": _row_name(primary_action),
        },

        "primary_subject": primary_subject,

        "primary_place": {
            "name": _row_name(primary_place),
        },

        "primary_state": {
            "name": _row_name(primary_state),
        },

        "primary_relationship": {
            "name": _row_name(primary_relationship),
        },

        "primary_ending": {
            "name": _row_name(primary_ending),
        },

        "subjects": subjects,
    }


# =========================================================
# MAIN INTERPRETATION
# =========================================================

def run_interpretation():

    data = request.get_json(silent=True) or {}

    dream = (
        data.get("dream")
        or data.get("text")
        or ""
    ).strip()

    if Config.DEBUG_MATCH:
        print(
            "RAW DREAM:",
            safe_debug_payload_preview(dream),
            flush=True,
        )

    validation_error = validate_dream_text(
        dream,
        min_length=Config.MIN_DREAM_LENGTH,
        max_length=Config.MAX_DREAM_LENGTH,
    )

    if validation_error:
        return jsonify({
            "error": validation_error
        }), 400

    request_email = _get_request_email(data)

    if request_email:
        persist_email_to_session(request_email)

    session_email = get_session_email()

    # =====================================================
    # ACCESS
    # =====================================================

    try:

        access_ok, access_meta = has_active_access(
            session_email
        )

        access_type = access_meta.get("type", "")

        is_paid = (
            access_ok
            and access_type == "subscription"
        )

        has_dream_pack = (
            access_ok
            and access_type == "dream_pack"
        )

        free_uses_left = 0

        dream_pack_status = get_dream_pack_status(
            session_email
        )

        if not access_ok:

            ip = get_client_ip()

            cookie_used = get_cookie_tries_used()
            ip_used = shadow_count(ip)

            effective_used = max(
                cookie_used,
                ip_used,
            )

            if effective_used >= Config.FREE_TRIES:

                return jsonify({
                    "blocked": True,
                    "reason": "free_limit_reached",
                    "message": (
                        f"You’ve used your "
                        f"{Config.FREE_TRIES} free tries."
                    ),
                    "free_uses_left": 0,
                    "access": "blocked",
                    "email": session_email,
                }), 402

            shadow_increment(ip)

            free_uses_left = (
                free_tries_remaining_after_this(
                    effective_used
                )
            )

        elif has_dream_pack:

            dream_pack_status = consume_dream_pack_use(
                session_email
            )

    except Exception as e:

        return jsonify({
            "error": "Access system failed",
            "details": str(e),
        }), 500

    # =====================================================
    # ENGINE MODE
    # =====================================================

    doctrine_active = (
        Config.DOCTRINE_MODE
        and doctrine_available()
    )

    try:

        # =================================================
        # DOCTRINE MODE
        # =================================================

        if doctrine_active:

            sheets = load_doctrine_sheets()

            base_rows = sheets.get(
                Config.SHEET_BASE_SYMBOLS,
                [],
            )

            behavior_rows = sheets.get(
                Config.SHEET_BEHAVIOR_RULES,
                [],
            )

            location_rows = sheets.get(
                Config.SHEET_LOCATION_RULES,
                [],
            )

            state_rows = sheets.get(
                Config.SHEET_SIZE_STATE_RULES,
                [],
            )

            relationship_rows = sheets.get(
                Config.SHEET_RELATIONSHIP_RULES,
                [],
            )

            override_rows = sheets.get(
                Config.SHEET_OVERRIDE_RULES,
                [],
            )

            template_rows = sheets.get(
                Config.SHEET_OUTPUT_TEMPLATES,
                [],
            )

            ending_rows = sheets.get(
                Config.SHEET_ENDING_RULES,
                [],
            )

            # =============================================
            # ACTION FIRST
            # =============================================

            behaviors = detect_rule_hits(
                dream,
                behavior_rows,
                "behavior",
                max_hits=Config.MAX_RULE_HITS_PER_LAYER,
            )

            behaviors = _sort_hits(
                _normalize_hits(
                    behaviors,
                    "behavior",
                )
            )

            # =============================================
            # STATES
            # =============================================

            states = detect_rule_hits(
                dream,
                state_rows,
                "state",
                max_hits=Config.MAX_RULE_HITS_PER_LAYER,
            )

            states = _sort_hits(
                _normalize_hits(states, "state")
            )

            # =============================================
            # LOCATIONS
            # =============================================

            locations = detect_rule_hits(
                dream,
                location_rows,
                "location",
                max_hits=Config.MAX_RULE_HITS_PER_LAYER,
            )

            locations = _sort_hits(
                _normalize_hits(
                    locations,
                    "location",
                )
            )

            old_place = _detect_old_place(dream)

            if old_place:
                locations.insert(0, old_place)

            locations = _sort_hits(locations)

            # =============================================
            # RELATIONSHIPS
            # =============================================

            relationships = detect_rule_hits(
                dream,
                relationship_rows,
                "relationship",
                max_hits=Config.MAX_RULE_HITS_PER_LAYER,
            )

            relationships = _sort_hits(
                _normalize_hits(
                    relationships,
                    "relationship",
                )
            )

            # =============================================
            # ENDINGS
            # =============================================

            endings = _detect_endings(
                dream,
                ending_rows,
            )

            # =============================================
            # BASE MATCHES
            # =============================================

            try:

                base_matches = (
                    match_base_symbols_doctrine(
                        dream,
                        base_rows,
                        top_k=Config.BASE_MATCH_TOP_K,
                        behaviors=behaviors,
                        states=states,
                        locations=locations,
                        relationships=relationships,
                    )
                )

            except TypeError:

                base_matches = (
                    match_base_symbols_doctrine(
                        dream,
                        base_rows,
                        top_k=Config.BASE_MATCH_TOP_K,
                    )
                )

            # =============================================
            # OVERRIDES
            # =============================================

            override_hit = apply_override_rules(
                base_matches,
                behaviors,
                states,
                locations,
                relationships,
                dream,
                override_rows,
            )

            # =============================================
            # SEAL
            # =============================================

            doctrine_seal = compute_doctrine_seal(
                dream,
                base_matches,
                behaviors,
                states,
                locations,
                relationships,
                override_hit,
            )

            # =============================================
            # BUILD INTERPRETATION
            # =============================================

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

            # =============================================
            # EVENT CONTEXT
            # =============================================

            event_context = _build_event_context(
                base_matches=base_matches,
                behaviors=behaviors,
                locations=locations,
                states=states,
                relationships=relationships,
                endings=endings,
            )

            doctrine_facts = (
                built.get("doctrine_facts", {})
                or {}
            )

            doctrine_facts["event_context"] = (
                event_context
            )

            narration = build_narration_result(
                doctrine_facts=doctrine_facts,
                interpretation=built.get(
                    "interpretation",
                    {},
                ),
                ai_enabled=Config.AI_NARRATION_ENABLED,
            )

            payload = {
                "engine_mode": "doctrine_event",

                "access": _access_label(
                    is_paid,
                    has_dream_pack,
                ),

                "is_paid": bool(is_paid),

                "email": session_email,

                "free_uses_left": free_uses_left,

                "dream_pack": dream_pack_status,

                "seal": doctrine_seal,

                "brain": {
                    "priority_order": [
                        "action",
                        "subject",
                        "place",
                        "ending",
                    ],

                    "primary_action":
                        event_context["primary_action"],

                    "primary_subject":
                        event_context["primary_subject"],

                    "primary_place":
                        event_context["primary_place"],

                    "primary_ending":
                        event_context["primary_ending"],

                    "behaviors":
                        _safe_names(behaviors),

                    "states":
                        _safe_names(states),

                    "locations":
                        _safe_names(locations),

                    "relationships":
                        _safe_names(relationships),

                    "endings":
                        _safe_names(endings),
                },

                "interpretation":
                    built["interpretation"],

                "full_interpretation":
                    built["full_interpretation"],

                "receipt":
                    _build_receipt(
                        built.get(
                            "top_symbols",
                            [],
                        )
                    ),

                "doctrine_facts":
                    doctrine_facts,

                "narration":
                    narration,
            }

        # =================================================
        # LEGACY FALLBACK
        # =================================================

        else:

            legacy_rows = load_legacy_rows()

            matches = match_symbols_legacy(
                dream,
                legacy_rows,
                top_k=Config.BASE_MATCH_TOP_K,
            )

            seal = compute_seal_from_symbol_count(
                len(matches)
            )

            built = build_legacy_interpretation(
                matches,
                narrative_max_symbols=(
                    Config.NARRATIVE_MAX_SYMBOLS
                ),
            )

            payload = {
                "engine_mode": "legacy",

                "access": _access_label(
                    is_paid,
                    has_dream_pack,
                ),

                "is_paid": bool(is_paid),

                "email": session_email,

                "free_uses_left": free_uses_left,

                "dream_pack": dream_pack_status,

                "seal": seal,

                "brain": {
                    "priority_order": ["symbol"],
                },

                "interpretation": built,

                "full_interpretation":
                    "\n\n".join([
                        built[
                            "spiritual_meaning"
                        ],
                        built[
                            "effects_in_physical_realm"
                        ],
                        built["what_to_do"],
                    ]),

                "receipt": _build_receipt(
                    [
                        get_symbol_cell(row)
                        for row, _s, _h in matches
                        if get_symbol_cell(row)
                    ]
                ),

                "doctrine_facts": {},

                "narration": {
                    "mode": "legacy",
                    "enabled": False,
                    "used_ai": False,
                    "readable_summary": "",
                },
            }

    except Exception as e:

        return jsonify({
            "error": "Interpreter engine failed",
            "details": str(e),
        }), 500

    # =====================================================
    # RESPONSE
    # =====================================================

    response = make_response(
        jsonify(payload)
    )

    if not access_ok:

        response.set_cookie(
            Config.COOKIE_NAME,
            str(
                get_cookie_tries_used() + 1
            ),
            max_age=Config.COOKIE_MAX_AGE,
            samesite=Config.SESSION_COOKIE_SAMESITE,
            secure=Config.SESSION_COOKIE_SECURE,
        )

    else:

        response.set_cookie(
            Config.COOKIE_NAME,
            "0",
            max_age=0,
            samesite=Config.SESSION_COOKIE_SAMESITE,
            secure=Config.SESSION_COOKIE_SECURE,
        )

    return response
