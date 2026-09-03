from typing import Any, Dict, List

from app.release_info import DOCTRINE_REGISTRY, TEETH_DOCTRINE_VERSION
from app.rules import _affirmative_teeth_fallout_token
from app.teeth_context import TEETH_CONTEXT_VERSION, extract_teeth_context
from app.teeth_provenance import build_teeth_rule_provenance
from app.teeth_registry import (
    get_teeth_registry_snapshot,
    public_registry_metadata,
    rule_id_for,
    unresolved_rule_id_for,
)
from app.utils import normalize_text


def _event_status(context: Dict[str, Any], actual_fallout: bool) -> str:
    """Return a factual event class without assigning an unapproved meaning."""
    if actual_fallout and context.get("returned_same_tooth_firm"):
        return "returned_same_tooth_firm"
    if actual_fallout:
        return "completed_loss"
    if context.get("near_miss_loss"):
        return "near_miss_loss"
    if context.get("hypothetical_loss"):
        return "hypothetical_loss"
    if context.get("loose_or_wobbly") and context.get("restorative_state"):
        return "loose_restored"
    if context.get("loose_or_wobbly"):
        return "loose_only"
    if context.get("broken_or_cracked"):
        return "broken_or_cracked"
    if context.get("rotten_or_decayed"):
        return "rotten_or_decayed"
    if context.get("gum_bleeding"):
        return "bleeding_gums"
    if context.get("gold_teeth"):
        return "gold_without_loss"
    if context.get("has_teeth"):
        return "teeth_present"
    if context.get("has_teeth_cluster"):
        return "gums_only"
    return ""


def _pending_distinctions(
    context: Dict[str, Any],
    actual_fallout: bool,
) -> List[str]:
    pending: List[str] = []
    if context.get("replacement_growth"):
        pending.append("replacement_growth_meaning")
    if actual_fallout and context.get("returned_same_tooth_firm"):
        pending.append("returned_same_tooth_consequence")
    if context.get("positions"):
        pending.append("tooth_position_mapping")
    if actual_fallout and (context.get("broken_or_cracked") or context.get("rotten_or_decayed")):
        pending.append("tooth_state_with_fallout_precedence")
    if context.get("subject_class") in {"non_human", "artificial"}:
        pending.append("non_human_or_artificial_teeth")
    return pending


def _unresolved_rule_ids(
    pending: List[str],
    registry: Dict[str, Any],
) -> List[str]:
    mapping = {
        "replacement_growth_meaning": "pending_replacement_growth",
        "returned_same_tooth_consequence": "pending_returned_same_tooth",
        "tooth_position_mapping": "pending_position_mapping",
        "tooth_state_with_fallout_precedence": "pending_state_fallout_precedence",
        "non_human_or_artificial_teeth": "pending_non_human",
    }
    return [
        rule_id
        for item in pending
        if item in mapping
        for rule_id in [unresolved_rule_id_for(registry, mapping[item])]
        if rule_id
    ]


def build_teeth_doctrine_context(dream: str) -> Dict[str, Any]:
    """Map approved Teeth facts into versioned, doctrine-safe output."""
    registry = get_teeth_registry_snapshot()
    registry_verified = registry.get("verified") is True
    context = extract_teeth_context(dream)
    phrase_fallout = bool(_affirmative_teeth_fallout_token(normalize_text(dream)))
    actual_fallout = phrase_fallout or bool(context.get("explicit_pull_removal"))
    supported_subject = bool(
        registry_verified
        and context.get("subject_class") == "human_or_unspecified"
    )
    terminal_return = bool(actual_fallout and context.get("returned_same_tooth_firm"))

    loose_warning = bool(context.get("loose_or_wobbly")) and not actual_fallout
    broken_warning = bool(context.get("broken_or_cracked")) and not actual_fallout
    rotten_warning = bool(context.get("rotten_or_decayed")) and not actual_fallout
    bleeding_gums_warning = (
        bool(context.get("gum_bleeding"))
        and not actual_fallout
        and not bool(context.get("bleeding_physical_cause"))
    )

    warning_kind = ""
    if supported_subject and not terminal_return:
        if actual_fallout:
            warning_kind = "tooth_loss"
        elif loose_warning:
            warning_kind = "loose_sickness"
        elif broken_warning:
            warning_kind = "broken_sickness"
        elif rotten_warning:
            warning_kind = "rotten_sickness"
        elif bleeding_gums_warning:
            warning_kind = "bleeding_gums"

    active_warning = bool(warning_kind)
    active_doctrine = bool(
        supported_subject
        and (active_warning or terminal_return or context.get("gold_teeth"))
    )
    pending = _pending_distinctions(context, actual_fallout)

    result: Dict[str, Any] = {
        "active_doctrine": active_doctrine,
        "active_warning": active_warning,
        "warning_kind": warning_kind,
        "event_status": _event_status(context, actual_fallout),
        "active_fallout": bool(actual_fallout),
        "owner": context.get("owner", "unknown"),
        "owner_relationship": context.get("owner_relationship", ""),
        "subject_scope": "",
        "subject_class": context.get("subject_class", "human_or_unspecified"),
        "count": context.get("count", "unknown"),
        "pain": context.get("pain", "unknown"),
        "positions": list(context.get("positions", [])),
        "relationship_scope": "",
        "warning_count": "",
        "proximity": "",
        "emotional_intensity": "",
        "loose_warning": loose_warning,
        "broken_warning": broken_warning,
        "rotten_warning": rotten_warning,
        "bleeding_gums_warning": bleeding_gums_warning,
        "blood_on_fallen_tooth": bool(context.get("blood_on_tooth")) and bool(actual_fallout),
        "severity_modifier": "",
        "favorable_modifier": "outwardly_favorable" if context.get("gold_teeth") else "",
        "salience_modifier": "increased" if context.get("repetition") else "",
        "removal_actor": context.get("removal_actor", ""),
        "pull_modifier": "",
        "terminal_ending": "same_tooth_returned_firm" if terminal_return else "",
        "ending_precedence": terminal_return,
        "outcome_resolution": "unresolved" if terminal_return else "",
        "bleeding_physical_cause": bool(context.get("bleeding_physical_cause")),
        "restorative_state": bool(context.get("restorative_state")),
        "broken_or_cracked": bool(context.get("broken_or_cracked")),
        "rotten_or_decayed": bool(context.get("rotten_or_decayed")),
        "gold_teeth": bool(context.get("gold_teeth")),
        "near_miss_loss": bool(context.get("near_miss_loss")),
        "hypothetical_loss": bool(context.get("hypothetical_loss")),
        "replacement_growth": bool(context.get("replacement_growth")),
        "repetition": bool(context.get("repetition")),
        "pending_distinctions": pending,
        "unresolved_rule_ids": _unresolved_rule_ids(pending, registry),
        "applied_rule_ids": [],
        "structural_rule_ids": [],
        "doctrine_version": registry.get("doctrine_version") or TEETH_DOCTRINE_VERSION,
        "context_version": TEETH_CONTEXT_VERSION,
        "doctrine_source": DOCTRINE_REGISTRY,
        "doctrine_registry": public_registry_metadata(registry),
    }

    def finalize() -> Dict[str, Any]:
        result["rule_provenance"] = build_teeth_rule_provenance(
            dream=dream,
            context=context,
            registry=registry,
            applied_rule_ids=result["applied_rule_ids"],
            unresolved_rule_ids=result["unresolved_rule_ids"],
            active_warning=bool(result["active_warning"]),
            warning_count=str(result["warning_count"] or ""),
        )
        return result

    if not context.get("has_teeth_cluster") or not supported_subject:
        return finalize()

    applied: List[str] = []

    def apply_rule(implementation_key: str) -> None:
        rule_id = rule_id_for(registry, implementation_key)
        if rule_id:
            applied.append(rule_id)

    if terminal_return:
        # Keep the experienced loss provenance while sealing its warning at the
        # genuine later ending. The approved terminal rule is structural; the
        # returned-same-tooth consequence remains unresolved and inactive.
        if context.get("owner") == "dreamer":
            apply_rule("own_fallout")
        elif context.get("owner") == "other":
            apply_rule("other_fallout")
        if context.get("count") == "one":
            apply_rule("one_fallout")
        elif context.get("count") == "multiple":
            apply_rule("multiple_fallout")
        result["structural_rule_ids"] = [
            rule_id
            for rule_id in [rule_id_for(registry, "terminal_ending")]
            if rule_id
        ]
        result["applied_rule_ids"] = applied
        return finalize()

    if actual_fallout:
        if context.get("owner") == "dreamer":
            result["relationship_scope"] = "relative_or_close_friend"
            result["subject_scope"] = "relative_close_friend_or_relationship_circle"
            apply_rule("own_fallout")
        elif context.get("owner") == "other":
            result["subject_scope"] = context.get("owner_relationship") or "other_person"
            apply_rule("other_fallout")

        if context.get("count") == "one":
            result["warning_count"] = "one_person"
            apply_rule("one_fallout")
        elif context.get("count") == "multiple":
            result["warning_count"] = "multiple_people"
            apply_rule("multiple_fallout")

        if context.get("pain") == "painful":
            result["proximity"] = "very_close_or_close_relative"
            result["emotional_intensity"] = "heightened"
            apply_rule("pain")
        elif context.get("pain") == "painless":
            result["proximity"] = "friend_acquaintance_or_more_distant"
            apply_rule("painless")

        if result["blood_on_fallen_tooth"]:
            result["severity_modifier"] = "increased"
            apply_rule("tooth_blood")

        if context.get("removal_actor") == "self":
            result["pull_modifier"] = "self_participation"
            apply_rule("self_pull")
        elif context.get("removal_actor") == "other":
            result["pull_modifier"] = "external_interference"
            apply_rule("external_pull")

    elif warning_kind == "loose_sickness":
        apply_rule("loose")
    elif warning_kind == "broken_sickness":
        apply_rule("broken")
    elif warning_kind == "rotten_sickness":
        apply_rule("rotten")
    elif warning_kind == "bleeding_gums":
        apply_rule("gum_blood")

    if context.get("gold_teeth"):
        apply_rule("gold")
    if context.get("repetition") and active_doctrine:
        apply_rule("repetition")

    result["applied_rule_ids"] = applied
    return finalize()


def build_teeth_narration_facts(dream: str) -> Dict[str, Any]:
    """Return deterministic narration material from approved Teeth doctrine."""
    doctrine = build_teeth_doctrine_context(dream)
    passthrough_keys = (
        "relationship_scope", "subject_scope", "subject_class", "warning_count",
        "proximity", "emotional_intensity", "severity_modifier", "favorable_modifier",
        "salience_modifier", "removal_actor", "pull_modifier", "terminal_ending",
        "ending_precedence", "outcome_resolution", "restorative_state",
        "bleeding_physical_cause", "broken_or_cracked", "rotten_or_decayed",
        "gold_teeth", "near_miss_loss", "hypothetical_loss", "replacement_growth",
        "repetition", "doctrine_version", "context_version", "doctrine_source",
        "doctrine_registry",
        "rule_provenance",
        "structural_rule_ids",
    )
    result: Dict[str, Any] = {
        "active": bool(doctrine.get("active_doctrine")),
        "active_warning": bool(doctrine.get("active_warning")),
        "warning_kind": doctrine.get("warning_kind", ""),
        "event_status": doctrine.get("event_status", ""),
        "lead": "",
        "details": [],
        **{key: doctrine.get(key) for key in passthrough_keys},
        "positions": list(doctrine.get("positions", [])),
        "pending_distinctions": list(doctrine.get("pending_distinctions", [])),
        "unresolved_rule_ids": list(doctrine.get("unresolved_rule_ids", [])),
        "applied_rule_ids": list(doctrine.get("applied_rule_ids", [])),
    }

    if not result["active"]:
        return result

    details: List[str] = []

    if doctrine.get("terminal_ending") == "same_tooth_returned_firm":
        result["lead"] = (
            "The same tooth returning firmly to the same place is the dream's final state. "
            "Its exact cultural consequence is still unresolved, so no outcome is asserted."
        )
        details.append(
            "The earlier loss is preserved as context, while the genuine terminal ending controls the final-state narration."
        )
    elif doctrine.get("warning_kind") == "tooth_loss":
        count = doctrine.get("warning_count")
        if count == "one_person":
            result["lead"] = (
                "In Jamaican and wider Caribbean dream tradition, one fallen tooth is treated as a "
                "death-associated omen warning concerning one person, not as a prediction or guarantee."
            )
        elif count == "multiple_people":
            result["lead"] = (
                "In Jamaican and wider Caribbean dream tradition, multiple fallen teeth are treated as "
                "death-associated omen warnings concerning multiple people, without exact arithmetic or certainty."
            )
        else:
            result["lead"] = (
                "In Jamaican and wider Caribbean dream tradition, teeth falling out are treated as a "
                "death-associated omen warning, not as a prediction or guarantee."
            )

        if doctrine.get("relationship_scope") == "relative_or_close_friend":
            details.append(
                "Because these were your own teeth, the warning concerns a relative or close friend, or someone in your relationship circle."
            )
        elif doctrine.get("subject_scope"):
            relationship = doctrine.get("owner_relationship")
            if relationship:
                details.append(f"Because the tooth belonged to your {relationship}, the warning concerns that person.")
            else:
                details.append("Because the teeth belonged to another person, the warning concerns that person.")

        if doctrine.get("proximity") == "very_close_or_close_relative":
            details.append("Pain points to someone very close or a close relative, with stronger emotional intensity.")
        elif doctrine.get("proximity") == "friend_acquaintance_or_more_distant":
            details.append("Painless loss points toward a friend, acquaintance, or someone more distant; it does not increase certainty.")

        if doctrine.get("severity_modifier") == "increased":
            details.append(
                "Blood on the fallen tooth increases the severity or intensity through emotional depth only; "
                "it does not determine who is involved, the count, or the outcome."
            )

        if doctrine.get("pull_modifier") == "self_participation":
            details.append(
                "Pulling your own tooth adds self-participation to the underlying warning, without assigning blame."
            )
        elif doctrine.get("pull_modifier") == "external_interference":
            details.append(
                "Another person pulling your tooth adds external interference, without identifying a real-world culprit or proving intent."
            )

    elif doctrine.get("warning_kind") in {"loose_sickness", "broken_sickness", "rotten_sickness"}:
        labels = {
            "loose_sickness": "loose or wobbly tooth",
            "broken_sickness": "broken or cracked tooth",
            "rotten_sickness": "rotten or decayed tooth",
        }
        label = labels[doctrine.get("warning_kind")]
        result["lead"] = (
            f"In Jamaican True Stories doctrine, a {label} is treated as a sickness warning, "
            "not a medical diagnosis or a guarantee of illness."
        )
        if doctrine.get("restorative_state"):
            details.append(
                "The tooth later becoming firm again is preserved as a restorative ending, but no cancellation meaning is asserted without an approved rule."
            )
    elif doctrine.get("warning_kind") == "bleeding_gums":
        result["lead"] = (
            "In Jamaican True Stories doctrine, standalone bleeding gums are treated as a pre-warning "
            "that a bad omen may be approaching, not as a guaranteed outcome."
        )
    elif doctrine.get("gold_teeth"):
        result["lead"] = (
            "In Jamaican True Stories doctrine, a gold tooth is treated as favorable in outward appearance. "
            "That modifier does not erase an underlying tooth state."
        )

    if doctrine.get("gold_teeth") and doctrine.get("warning_kind"):
        details.append("The gold appearance is favorable outwardly, but it does not erase the underlying warning state.")
    if doctrine.get("repetition"):
        details.append("Repetition increases salience only; it does not prove or strengthen an outcome.")

    result["details"] = details
    return result
