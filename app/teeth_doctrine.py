from typing import Any, Dict, List

from app.rules import _affirmative_teeth_fallout_token
from app.teeth_context import extract_teeth_context
from app.utils import normalize_text


def build_teeth_doctrine_context(dream: str) -> Dict[str, Any]:
    """Map approved Teeth fallout facts into doctrine-safe structured output.

    This layer intentionally does not assign meanings to upper/lower/front/back
    positions and does not infer blood-relative status. It only activates
    fallout doctrine when the existing guarded Teeth-fallout detector confirms
    that the event actually occurred, protecting negated, hypothetical, and
    near-miss language.
    """
    context = extract_teeth_context(dream)
    actual_fallout = bool(_affirmative_teeth_fallout_token(normalize_text(dream)))

    result: Dict[str, Any] = {
        "active_fallout": False,
        "owner": context.get("owner", "unknown"),
        "owner_relationship": context.get("owner_relationship", ""),
        "count": context.get("count", "unknown"),
        "pain": context.get("pain", "unknown"),
        "positions": list(context.get("positions", [])),
        "relationship_scope": "",
        "warning_count": "",
        "proximity": "",
    }

    if not context.get("has_teeth") or not actual_fallout:
        return result

    result["active_fallout"] = True

    # Approved rule: the dreamer's own teeth concern a relative or close friend.
    if context.get("owner") == "dreamer":
        result["relationship_scope"] = "relative_or_close_friend"

    # Approved rule: one fallen tooth = one-person warning; multiple = multiple people.
    if context.get("count") == "one":
        result["warning_count"] = "one_person"
    elif context.get("count") == "multiple":
        result["warning_count"] = "multiple_people"

    # Approved pain/proximity modifier. No blood-relative inference is made.
    if context.get("pain") == "painful":
        result["proximity"] = "very_close_or_close_relative"
    elif context.get("pain") == "painless":
        result["proximity"] = "friend_acquaintance_or_more_distant"

    return result


def build_teeth_narration_facts(dream: str) -> Dict[str, Any]:
    """Return deterministic, doctrine-safe narration material for Teeth fallout.

    These strings are deliberately narrow. They express only the user's approved
    Caribbean Teeth rules and leave positional details descriptive rather than
    assigning an unverified kinship class. The caller can merge these facts into
    the broader narration pipeline without reproducing doctrine logic there.
    """
    doctrine = build_teeth_doctrine_context(dream)

    result: Dict[str, Any] = {
        "active": bool(doctrine.get("active_fallout")),
        "lead": "",
        "details": [],
        "relationship_scope": doctrine.get("relationship_scope", ""),
        "warning_count": doctrine.get("warning_count", ""),
        "proximity": doctrine.get("proximity", ""),
        "positions": list(doctrine.get("positions", [])),
    }

    if not result["active"]:
        return result

    details: List[str] = []

    warning_count = doctrine.get("warning_count")
    if warning_count == "one_person":
        result["lead"] = "In Caribbean dream tradition, one fallen tooth is treated as a warning concerning one person."
    elif warning_count == "multiple_people":
        result["lead"] = "In Caribbean dream tradition, multiple fallen teeth are treated as warnings concerning multiple people."
    else:
        result["lead"] = "In Caribbean dream tradition, teeth falling out are treated as a serious warning."

    if doctrine.get("relationship_scope") == "relative_or_close_friend":
        details.append("Because these were your own teeth, the warning concerns a relative or close friend.")

    proximity = doctrine.get("proximity")
    if proximity == "very_close_or_close_relative":
        details.append("Pain with the tooth loss points to someone very close or a close relative.")
    elif proximity == "friend_acquaintance_or_more_distant":
        details.append("Painless tooth loss points toward a friend, acquaintance, or someone more distant.")

    result["details"] = details
    return result
