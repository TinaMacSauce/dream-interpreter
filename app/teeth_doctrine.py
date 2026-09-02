from typing import Any, Dict, List

from app.rules import _affirmative_teeth_fallout_token
from app.teeth_context import extract_teeth_context
from app.utils import normalize_text


def build_teeth_doctrine_context(dream: str) -> Dict[str, Any]:
    """Map approved Teeth facts into doctrine-safe structured output.

    This layer intentionally does not assign meanings to upper/lower/front/back
    positions and does not infer blood-relative status. Fallout doctrine only
    activates when the guarded Teeth-fallout detector confirms that the event
    actually occurred, protecting negated, hypothetical, and near-miss language.
    """
    context = extract_teeth_context(dream)
    actual_fallout = bool(_affirmative_teeth_fallout_token(normalize_text(dream)))

    loose_warning = bool(context.get("loose_or_wobbly")) and not actual_fallout
    bleeding_gums_warning = (
        bool(context.get("gum_bleeding"))
        and not actual_fallout
        and not bool(context.get("bleeding_physical_cause"))
    )

    warning_kind = ""
    if actual_fallout:
        warning_kind = "tooth_loss"
    elif loose_warning:
        warning_kind = "loose_sickness"
    elif bleeding_gums_warning:
        warning_kind = "bleeding_gums"

    result: Dict[str, Any] = {
        "active_warning": bool(warning_kind),
        "warning_kind": warning_kind,
        "active_fallout": bool(actual_fallout),
        "owner": context.get("owner", "unknown"),
        "owner_relationship": context.get("owner_relationship", ""),
        "count": context.get("count", "unknown"),
        "pain": context.get("pain", "unknown"),
        "positions": list(context.get("positions", [])),
        "relationship_scope": "",
        "warning_count": "",
        "proximity": "",
        "emotional_intensity": "",
        "loose_warning": loose_warning,
        "bleeding_gums_warning": bleeding_gums_warning,
        "blood_on_fallen_tooth": bool(context.get("blood_on_tooth")) and bool(actual_fallout),
        "severity_modifier": "",
        "bleeding_physical_cause": bool(context.get("bleeding_physical_cause")),
        "restorative_state": bool(context.get("restorative_state")),
    }

    if not context.get("has_teeth_cluster"):
        return result

    if not actual_fallout:
        return result

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
        result["emotional_intensity"] = "heightened"
    elif context.get("pain") == "painless":
        result["proximity"] = "friend_acquaintance_or_more_distant"

    # Approved blood rule: blood on the fallen tooth changes severity/intensity
    # only. It must not change person, relationship, count, actor, or outcome.
    if result["blood_on_fallen_tooth"]:
        result["severity_modifier"] = "increased"

    return result


def build_teeth_narration_facts(dream: str) -> Dict[str, Any]:
    """Return deterministic, doctrine-safe narration material for Teeth dreams."""
    doctrine = build_teeth_doctrine_context(dream)

    result: Dict[str, Any] = {
        "active": bool(doctrine.get("active_warning")),
        "warning_kind": doctrine.get("warning_kind", ""),
        "lead": "",
        "details": [],
        "relationship_scope": doctrine.get("relationship_scope", ""),
        "warning_count": doctrine.get("warning_count", ""),
        "proximity": doctrine.get("proximity", ""),
        "emotional_intensity": doctrine.get("emotional_intensity", ""),
        "severity_modifier": doctrine.get("severity_modifier", ""),
        "positions": list(doctrine.get("positions", [])),
        "restorative_state": bool(doctrine.get("restorative_state")),
        "bleeding_physical_cause": bool(doctrine.get("bleeding_physical_cause")),
    }

    if not result["active"]:
        return result

    details: List[str] = []
    warning_kind = doctrine.get("warning_kind")

    if warning_kind == "tooth_loss":
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

        if doctrine.get("emotional_intensity") == "heightened":
            details.append("Pain also increases the emotional intensity of the warning.")

        if doctrine.get("severity_modifier") == "increased":
            details.append(
                "Blood on the fallen tooth increases the severity or intensity of the underlying warning; "
                "it does not determine who is involved, how many people are involved, or the outcome."
            )

    elif warning_kind == "loose_sickness":
        result["lead"] = (
            "In Jamaican True Stories doctrine, a loose or wobbly tooth is treated as a sickness warning, "
            "not a medical diagnosis or a guarantee that someone will become ill."
        )
        if doctrine.get("restorative_state"):
            details.append(
                "The tooth later becoming firm again is preserved as a restorative ending, but no cancellation meaning "
                "is asserted without an approved doctrine rule."
            )

    elif warning_kind == "bleeding_gums":
        result["lead"] = (
            "In Jamaican True Stories doctrine, standalone bleeding gums are treated as a warning that a bad omen may "
            "be approaching, not as a guaranteed outcome."
        )

    result["details"] = details
    return result
