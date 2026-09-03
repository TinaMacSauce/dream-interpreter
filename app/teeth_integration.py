from typing import Any, Dict

from app.teeth_doctrine import build_teeth_narration_facts
from app.utils import normalize_text


TEETH_ENDING_NAMES = {"tooth", "teeth", "molar", "molars"}


def _live_teeth_lead(teeth: Dict[str, Any]) -> str:
    """Build a narration-first sentence from approved Teeth doctrine only."""
    lead = str(teeth.get("lead", "") or "").strip()
    details = [str(item).strip() for item in teeth.get("details", []) if str(item).strip()]
    return " ".join([lead, *details]).strip()


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
