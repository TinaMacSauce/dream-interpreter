from typing import Any, Dict, List, Tuple

from app.teeth_doctrine import build_teeth_narration_facts
from app.utils import normalize_text


TEETH_ENDING_NAMES = {"tooth", "teeth", "molar", "molars"}


def _live_teeth_lead(teeth: Dict[str, Any]) -> str:
    """Build a narration-first sentence from approved Teeth doctrine only."""
    lead = str(teeth.get("lead", "") or "").strip()
    details = [str(item).strip() for item in teeth.get("details", []) if str(item).strip()]
    return " ".join([lead, *details]).strip()


def build_teeth_output_summary(teeth: Dict[str, Any]) -> str:
    """Return concise user-facing Teeth doctrine text without generic repeats."""
    if teeth.get("active") is not True:
        return ""

    parts: List[str] = []
    seen = set()
    for item in [teeth.get("lead", ""), *(teeth.get("details") or [])]:
        text = str(item or "").strip()
        key = normalize_text(text)
        if not text or not key or key in seen:
            continue
        seen.add(key)
        parts.append(text)

    return "\n".join(parts)


def build_teeth_output_assessment(teeth: Dict[str, Any]) -> Dict[str, Any]:
    """Separate doctrine warning facts from non-predictive rule confidence."""
    registry = teeth.get("doctrine_registry")
    registry = registry if isinstance(registry, dict) else {}
    applied_rule_ids = list(teeth.get("applied_rule_ids") or [])
    active = teeth.get("active") is True
    rule_bound = bool(active and registry.get("verified") is True and applied_rule_ids)

    return {
        "warning_present": bool(teeth.get("active_warning")),
        "warning_severity": (
            "heightened"
            if teeth.get("severity_modifier") == "increased"
            else "not_scaled"
        ),
        "interpretation_confidence": (
            "approved_rule_match" if rule_bound else "withheld"
        ),
        "predictive_certainty": "none",
        "registry_verified": bool(registry.get("verified") is True),
        "applied_rule_ids": applied_rule_ids,
    }


def bind_teeth_output_contract(
    *,
    doctrine_facts: Dict[str, Any],
    seal: Dict[str, Any],
    interpretation: Dict[str, str],
    full_interpretation: str,
) -> Tuple[Dict[str, Any], Dict[str, str], str]:
    """Bind active Teeth output to approved narration and unambiguous labels.

    The generic seal ``risk`` field describes mixed structural heuristics and
    must not follow a serious Teeth warning into user-visible output. Active
    Teeth responses therefore expose warning and rule-match facts separately,
    while retaining no predictive certainty claim.
    """
    teeth = doctrine_facts.get("teeth_narration")
    teeth = teeth if isinstance(teeth, dict) else {}
    if teeth.get("active") is not True:
        return dict(seal or {}), dict(interpretation or {}), full_interpretation

    summary = build_teeth_output_summary(teeth)
    assessment = build_teeth_output_assessment(teeth)

    bound_seal = dict(seal or {})
    bound_seal["risk"] = ""
    bound_seal["risk_label"] = ""
    bound_seal["legacy_risk_suppressed"] = True
    bound_seal["warning_assessment"] = assessment

    bound_interpretation = dict(interpretation or {})
    if summary:
        bound_interpretation["spiritual_meaning"] = summary
    bound_interpretation["effects_in_physical_realm"] = (
        "This is tradition-based guidance, not a prediction or guarantee of "
        "death, illness, or another serious outcome."
    )

    full_parts = [
        bound_interpretation.get("spiritual_meaning", ""),
        bound_interpretation.get("effects_in_physical_realm", ""),
        bound_interpretation.get("what_to_do", ""),
    ]
    bound_full = "\n\n".join(
        str(item).strip() for item in full_parts if str(item or "").strip()
    )
    return bound_seal, bound_interpretation, bound_full


def _sanitize_event_context_for_teeth(event_context: Any) -> Dict[str, Any]:
    """Prevent generic event parsing from overriding verified Teeth fallout facts.

    Teeth fallout is already bound deterministically before narration. The generic
    primary action is therefore removed so it cannot replace the count-specific
    Teeth lead. A dental token by itself is never accepted as a later ending.
    Other context, including genuine non-dental endings, is preserved.
    """
    context = dict(event_context) if isinstance(event_context, dict) else {}
    context["primary_action"] = {}

    ending = context.get("primary_ending")
    if isinstance(ending, dict):
        ending_name = normalize_text(ending.get("name", ""))
    else:
        ending_name = normalize_text(ending or "")

    if ending_name in TEETH_ENDING_NAMES:
        context["primary_ending"] = {}

    return context


def attach_teeth_narration_facts(
    dream: str,
    doctrine_facts: Dict[str, Any] | None,
) -> Dict[str, Any]:
    """Attach approved Teeth facts and bind them into the live narration path.

    Active Teeth doctrine outranks generic fallback narration. Fallout receives
    the stricter event sanitizer because count-specific loss doctrine has already
    been bound. Loose-tooth and standalone bleeding-gum warnings keep unrelated
    event context intact while replacing generic risk/state wording with the
    approved warning language.
    """
    output: Dict[str, Any] = dict(doctrine_facts or {})
    teeth = build_teeth_narration_facts(dream)
    output["teeth_narration"] = teeth

    if not teeth.get("active"):
        return output

    output["lead_message"] = _live_teeth_lead(teeth)
    output["risk"] = ""
    output["relationship_meaning"] = ""

    warning_kind = teeth.get("warning_kind", "")
    if warning_kind == "tooth_loss" or teeth.get("ending_precedence"):
        output["behavior_meaning"] = ""
        output["state_meaning"] = ""
        output["event_context"] = _sanitize_event_context_for_teeth(output.get("event_context"))
    elif warning_kind in {
        "loose_sickness", "broken_sickness", "rotten_sickness", "bleeding_gums"
    } or teeth.get("gold_teeth"):
        output["state_meaning"] = ""

    return output
