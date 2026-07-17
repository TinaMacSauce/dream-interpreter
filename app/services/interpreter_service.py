import os
import re
import secrets
import time
from typing import Any, Dict, List, Optional

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
from app.fields import (
    get_behavior_action_modifier,
    get_behavior_meaning_modifier,
    get_behavior_physical_modifier,
    get_location_action_modifier,
    get_location_life_area_meaning,
    get_location_physical_area_meaning,
    get_relationship_action_modifier,
    get_relationship_meaning_modifier,
    get_relationship_physical_modifier,
    get_state_action_modifier,
    get_state_meaning_modifier,
    get_state_physical_modifier,
    get_symbol_cell,
)
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
from app.services.admob_ssv_service import (
    consume_verified_reward,
    reward_is_available,
)
from app.services.narration_service import build_narration_result
from app.sheets import (
    doctrine_available,
    load_doctrine_sheets,
    load_legacy_rows,
)
from app.utils import (
    normalize_email,
    validate_dream_text,
)


# =========================================================
# ACCESS HELPERS
# =========================================================

def _access_label(
    is_paid: bool,
    has_dream_pack: bool,
    has_rewarded_access: bool = False,
) -> str:
    if is_paid:
        return "paid"
    if has_dream_pack:
        return "dream_pack"
    if has_rewarded_access:
        return "rewarded_ad"
    return "free"


def _env_flag(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default)).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _has_valid_test_reward(data: Dict[str, Any]) -> bool:
    """Accept a client reward only while the server is explicitly in test mode.

    This is intentionally limited to Google test ads. Before live ads are enabled,
    replace this test grant with verified AdMob server-side verification (SSV).
    """
    if not _env_flag("REWARDED_AD_TEST_MODE"):
        return False

    reward = data.get("rewarded_ad")
    if not isinstance(reward, dict):
        return False

    if reward.get("mode") != "test" or reward.get("earned") is not True:
        return False

    if _safe_int(reward.get("amount"), 0) < 1:
        return False

    attempt_id = _clean(reward.get("attempt_id"))
    if len(attempt_id) < 12 or len(attempt_id) > 128:
        return False

    earned_at_ms = _safe_int(reward.get("earned_at"), 0)
    now_ms = int(time.time() * 1000)

    # Reject stale or implausibly future-dated test rewards.
    if earned_at_ms <= 0 or abs(now_ms - earned_at_ms) > 5 * 60 * 1000:
        return False

    return True


def _get_ssv_reward_request(
    data: Dict[str, Any],
) -> Optional[Dict[str, str]]:
    """Return a structurally valid verified-reward request from the app."""
    reward = data.get("rewarded_ad")

    if not isinstance(reward, dict):
        return None

    if _clean(reward.get("mode")).lower() != "ssv":
        return None

    reward_id = _clean(reward.get("reward_id"))
    user_id = _clean(reward.get("user_id"))

    if len(reward_id) < 12 or len(reward_id) > 128:
        return None

    if len(user_id) < 6 or len(user_id) > 128:
        return None

    return {
        "reward_id": reward_id,
        "user_id": user_id,
    }


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
    return [x.strip().lower() for x in raw.split(",") if x.strip()]


def _phrase_exists(text: str, phrase: str) -> bool:
    text_l = _lower(text)
    phrase_l = _lower(phrase)

    if not phrase_l:
        return False

    if " " in phrase_l:
        return phrase_l in text_l

    return re.search(rf"\b{re.escape(phrase_l)}\b", text_l) is not None


# =========================================================
# RULE HELPERS
# =========================================================

def _hit_row(item: Any) -> Dict[str, Any]:
    if isinstance(item, dict):
        row = item.get("row")
        if isinstance(row, dict):
            return row
        return item
    return {}


def _row_name(item: Any) -> str:
    if not item:
        return ""

    if isinstance(item, str):
        return item.strip()

    if not isinstance(item, dict):
        return str(item).strip()

    row = _hit_row(item)

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


def _normalize_hits(items: List[Any], kind: str) -> List[Dict[str, Any]]:
    normalized = []

    for item in items or []:
        if isinstance(item, dict):
            row = _hit_row(item)

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
# FIELD FALLBACK HELPERS
# =========================================================

def _row_get(row: Dict[str, Any], *keys: str) -> str:
    if not isinstance(row, dict):
        return ""

    for key in keys:
        if key in row and _clean(row.get(key)):
            return _clean(row.get(key))

    lowered = {str(k).strip().lower(): v for k, v in row.items()}

    for key in keys:
        key_l = key.strip().lower()
        if key_l in lowered and _clean(lowered.get(key_l)):
            return _clean(lowered.get(key_l))

    return ""


def _ending_meaning(row: Dict[str, Any]) -> str:
    return _row_get(
        row,
        "outcome_meaning",
        "outcome meaning",
        "meaning_modifier",
        "meaning modifier",
        "spiritual_meaning",
        "spiritual meaning",
        "meaning",
        "effect",
        "effects",
    )


def _ending_physical(row: Dict[str, Any]) -> str:
    return _row_get(
        row,
        "physical_modifier",
        "physical modifier",
        "physical_effect",
        "physical effect",
        "physical effects",
        "effects_in_physical_realm",
        "effects in the physical realm",
        "effect",
        "effects",
    )


def _ending_action(row: Dict[str, Any]) -> str:
    return _row_get(
        row,
        "action_modifier",
        "action modifier",
        "what_to_do",
        "what to do",
        "base_action",
        "base action",
        "action",
        "actions",
    )


# =========================================================
# OLD PLACE DOCTRINE
# =========================================================

OLD_PLACE_KEYWORDS = [
    "old school",
    "old high school",
    "old primary school",
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
                "life_area_meaning": "backwardness, stagnation, regression, or old cycles",
                "physical_area_meaning": "delay, repeated patterns, unfinished past issues",
                "action_modifier": "pray against backwardness and move forward",
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

def _detect_endings(dream: str, ending_rows: List[Dict[str, Any]]):
    hits = []

    for row in ending_rows or []:
        if not isinstance(row, dict):
            continue

        active = _lower(row.get("active"))

        if active and active not in {"1", "true", "yes", "on", "active"}:
            continue

        keywords = _split_keywords(row.get("keywords"))

        if row.get("ending_name"):
            keywords.append(_lower(row.get("ending_name")))

        for keyword in keywords:
            if _phrase_exists(dream, keyword):
                hits.append(
                    {
                        "name": _clean(row.get("ending_name")) or keyword,
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


def _event_piece(
    hit: Optional[Dict[str, Any]],
    *,
    kind: str,
) -> Dict[str, str]:
    row = _hit_row(hit)

    name = _row_name(hit)

    if not hit and not row:
        return {
            "name": "",
            "meaning": "",
            "physical_area": "",
            "action": "",
        }

    if kind == "behavior":
        return {
            "name": name,
            "meaning": _clean(get_behavior_meaning_modifier(row)),
            "physical_area": _clean(get_behavior_physical_modifier(row)),
            "action": _clean(get_behavior_action_modifier(row)),
        }

    if kind == "location":
        return {
            "name": name,
            "meaning": _clean(get_location_life_area_meaning(row)),
            "physical_area": _clean(get_location_physical_area_meaning(row)),
            "action": _clean(get_location_action_modifier(row)),
        }

    if kind == "state":
        return {
            "name": name,
            "meaning": _clean(get_state_meaning_modifier(row)),
            "physical_area": _clean(get_state_physical_modifier(row)),
            "action": _clean(get_state_action_modifier(row)),
        }

    if kind == "relationship":
        return {
            "name": name,
            "meaning": _clean(get_relationship_meaning_modifier(row)),
            "physical_area": _clean(get_relationship_physical_modifier(row)),
            "action": _clean(get_relationship_action_modifier(row)),
        }

    if kind == "ending":
        return {
            "name": name,
            "meaning": _clean(_ending_meaning(row)),
            "physical_area": _clean(_ending_physical(row)),
            "action": _clean(_ending_action(row)),
        }

    return {
        "name": name,
        "meaning": "",
        "physical_area": "",
        "action": "",
    }


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
        "priority_order": ["action", "subject", "place", "ending"],
        "primary_action": _event_piece(primary_action, kind="behavior"),
        "primary_subject": primary_subject,
        "primary_place": _event_piece(primary_place, kind="location"),
        "primary_state": _event_piece(primary_state, kind="state"),
        "primary_relationship": _event_piece(primary_relationship, kind="relationship"),
        "primary_ending": _event_piece(primary_ending, kind="ending"),
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
        print("RAW DREAM:", dream[:500], flush=True)

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

    # =====================================================
    # ACCESS
    # =====================================================

    try:
        access_ok, access_meta = has_active_access(session_email)

        access_type = access_meta.get("type", "")

        is_paid = access_ok and access_type == "subscription"
        has_dream_pack = access_ok and access_type == "dream_pack"

        has_test_rewarded_access = _has_valid_test_reward(data)
        ssv_reward_request = _get_ssv_reward_request(data)
        has_ssv_rewarded_access = False

        if not access_ok and ssv_reward_request:
            has_ssv_rewarded_access = reward_is_available(
                reward_id=ssv_reward_request["reward_id"],
                user_id=ssv_reward_request["user_id"],
            )

        has_rewarded_access = (
            has_test_rewarded_access
            or has_ssv_rewarded_access
        )
        used_free_try = False

        free_uses_left = 0
        dream_pack_status = get_dream_pack_status(session_email)

        # A verified rewarded ad grants exactly one decode without consuming a
        # free try. The legacy client-side test grant remains available only
        # while REWARDED_AD_TEST_MODE is explicitly enabled on Render.
        if not access_ok and not has_rewarded_access:
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
                        "email": session_email,
                    }
                ), 402

            shadow_increment(ip)
            used_free_try = True
            free_uses_left = free_tries_remaining_after_this(effective_used)

        elif has_dream_pack:
            dream_pack_status = consume_dream_pack_use(session_email)

    except Exception:
        import traceback

        trace = traceback.format_exc()

        print("\n========== ACCESS TRACEBACK ==========", flush=True)
        print(trace, flush=True)
        print("======================================\n", flush=True)

        return jsonify(
            {
                "error": "Access system failed. Please try again."
            }
        ), 500

    # =====================================================
    # ENGINE MODE
    # =====================================================

    doctrine_active = Config.DOCTRINE_MODE and doctrine_available()

    try:
        # =================================================
        # DOCTRINE MODE
        # =================================================

        if doctrine_active:
            sheets = load_doctrine_sheets()

            base_rows = sheets.get(Config.SHEET_BASE_SYMBOLS, [])
            behavior_rows = sheets.get(Config.SHEET_BEHAVIOR_RULES, [])
            location_rows = sheets.get(Config.SHEET_LOCATION_RULES, [])
            state_rows = sheets.get(Config.SHEET_SIZE_STATE_RULES, [])
            relationship_rows = sheets.get(Config.SHEET_RELATIONSHIP_RULES, [])
            override_rows = sheets.get(Config.SHEET_OVERRIDE_RULES, [])
            template_rows = sheets.get(Config.SHEET_OUTPUT_TEMPLATES, [])
            ending_rows = sheets.get(Config.SHEET_ENDING_RULES, [])

            # ACTION FIRST
            behaviors = detect_rule_hits(
                dream,
                behavior_rows,
                "behavior",
                max_hits=Config.MAX_RULE_HITS_PER_LAYER,
            )
            behaviors = _sort_hits(_normalize_hits(behaviors, "behavior"))

            # STATES
            states = detect_rule_hits(
                dream,
                state_rows,
                "state",
                max_hits=Config.MAX_RULE_HITS_PER_LAYER,
            )
            states = _sort_hits(_normalize_hits(states, "state"))

            # LOCATIONS
            locations = detect_rule_hits(
                dream,
                location_rows,
                "location",
                max_hits=Config.MAX_RULE_HITS_PER_LAYER,
            )
            locations = _sort_hits(_normalize_hits(locations, "location"))

            old_place = _detect_old_place(dream)
            if old_place:
                locations.insert(0, old_place)

            locations = _sort_hits(locations)

            # RELATIONSHIPS
            relationships = detect_rule_hits(
                dream,
                relationship_rows,
                "relationship",
                max_hits=Config.MAX_RULE_HITS_PER_LAYER,
            )
            relationships = _sort_hits(_normalize_hits(relationships, "relationship"))

            # ENDINGS
            endings = _detect_endings(dream, ending_rows)

            # BASE MATCHES
            try:
                base_matches = match_base_symbols_doctrine(
                    dream,
                    base_rows,
                    top_k=Config.BASE_MATCH_TOP_K,
                    behaviors=behaviors,
                    states=states,
                    locations=locations,
                    relationships=relationships,
                )

            except TypeError:
                base_matches = match_base_symbols_doctrine(
                    dream,
                    base_rows,
                    top_k=Config.BASE_MATCH_TOP_K,
                )

            # OVERRIDES
            override_hit = apply_override_rules(
                base_matches,
                behaviors,
                states,
                locations,
                relationships,
                dream,
                override_rows,
            )

            # SEAL
            doctrine_seal = compute_doctrine_seal(
                dream,
                base_matches,
                behaviors,
                states,
                locations,
                relationships,
                override_hit,
            )

            # BUILD INTERPRETATION
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

            # EVENT CONTEXT
            event_context = _build_event_context(
                base_matches=base_matches,
                behaviors=behaviors,
                locations=locations,
                states=states,
                relationships=relationships,
                endings=endings,
            )

            doctrine_facts = built.get("doctrine_facts", {}) or {}
            doctrine_facts["event_context"] = event_context

            narration = build_narration_result(
                doctrine_facts=doctrine_facts,
                interpretation=built.get("interpretation", {}),
                ai_enabled=Config.AI_NARRATION_ENABLED,
            )

            payload = {
                "engine_mode": "doctrine_event",
                "access": _access_label(is_paid, has_dream_pack, has_rewarded_access),
                "is_rewarded": bool(has_rewarded_access),
                "is_paid": bool(is_paid),
                "email": session_email,
                "free_uses_left": free_uses_left,
                "dream_pack": dream_pack_status,
                "seal": doctrine_seal,
                "brain": {
                    "priority_order": ["action", "subject", "place", "ending"],
                    "primary_action": event_context["primary_action"],
                    "primary_subject": event_context["primary_subject"],
                    "primary_place": event_context["primary_place"],
                    "primary_state": event_context["primary_state"],
                    "primary_relationship": event_context["primary_relationship"],
                    "primary_ending": event_context["primary_ending"],
                    "behaviors": _safe_names(behaviors),
                    "states": _safe_names(states),
                    "locations": _safe_names(locations),
                    "relationships": _safe_names(relationships),
                    "endings": _safe_names(endings),
                },
                "interpretation": built["interpretation"],
                "full_interpretation": built["full_interpretation"],
                "receipt": _build_receipt(built.get("top_symbols", [])),
                "doctrine_facts": doctrine_facts,
                "narration": narration,
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

            seal = compute_seal_from_symbol_count(len(matches))

            built = build_legacy_interpretation(
                matches,
                narrative_max_symbols=Config.NARRATIVE_MAX_SYMBOLS,
            )

            payload = {
                "engine_mode": "legacy",
                "access": _access_label(is_paid, has_dream_pack, has_rewarded_access),
                "is_rewarded": bool(has_rewarded_access),
                "is_paid": bool(is_paid),
                "email": session_email,
                "free_uses_left": free_uses_left,
                "dream_pack": dream_pack_status,
                "seal": seal,
                "brain": {
                    "priority_order": ["symbol"],
                },
                "interpretation": built,
                "full_interpretation": "\n\n".join(
                    [
                        built["spiritual_meaning"],
                        built["effects_in_physical_realm"],
                        built["what_to_do"],
                    ]
                ),
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

    except Exception:
        import traceback

        trace = traceback.format_exc()

        print("\n========== INTERPRETER ENGINE TRACEBACK ==========", flush=True)
        print(trace, flush=True)
        print("==================================================\n", flush=True)

        return jsonify(
            {
                "error": "Interpreter engine failed. Please try again."
            }
        ), 500

    # =====================================================
    # RESPONSE
    # =====================================================

    # Consume the SSV reward only after a successful interpretation. This makes
    # each verified Google transaction usable for one dream decode only.
    if (
        not access_ok
        and has_ssv_rewarded_access
        and ssv_reward_request
    ):
        reward_consumed = consume_verified_reward(
            reward_id=ssv_reward_request["reward_id"],
            user_id=ssv_reward_request["user_id"],
        )

        if not reward_consumed:
            return jsonify(
                {
                    "blocked": True,
                    "reason": "reward_unavailable",
                    "message": "This sponsored reward has already been used or is no longer available.",
                    "access": "blocked",
                    "email": session_email,
                }
            ), 409

    response = make_response(jsonify(payload))

    if used_free_try:
        response.set_cookie(
            Config.COOKIE_NAME,
            str(get_cookie_tries_used() + 1),
            max_age=Config.COOKIE_MAX_AGE,
            samesite=Config.SESSION_COOKIE_SAMESITE,
            secure=Config.SESSION_COOKIE_SECURE,
        )

    elif access_ok:
        response.set_cookie(
            Config.COOKIE_NAME,
            "0",
            max_age=0,
            samesite=Config.SESSION_COOKIE_SAMESITE,
            secure=Config.SESSION_COOKIE_SECURE,
        )

    return response
