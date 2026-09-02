from typing import Any, Dict

from app.teeth_doctrine import build_teeth_narration_facts
from app.utils import normalize_text


TEETH_ENDING_NAMES = {"tooth", "teeth", "molar", "molars"}


def _live_teeth_lead(teeth: Dict[str, Any]) -> str:
    """Build a narration-first sentence from approved Teeth doctrine only."""
    warning_count = teeth.get("warning_count", "")
    if warning_count == "one_person":
        lead = "This dream is read in Jamaican True Stories doctrine as a warning concerning one person."
    elif warning_count == "multiple_people":
        lead = "This dream is read in Jamaican True Stories doctrine as a warning concerning multiple people."
    else:
        lead = "This dream is read in Jamaican True Stories doctrine as a serious teeth-loss warning."

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

    Active Teeth fallout must outrank generic symbol/action fallback narration.
    This bridge therefore supplies the doctrine-specific lead, suppresses the
    legacy generic risk label for this warning class, and removes only the
    generic action plus any fabricated dental-token ending. Unsupported
    positional kinship meanings remain excluded.
    """
    output: Dict[str, Any] = dict(doctrine_facts or {})
    teeth = build_teeth_narration_facts(dream)
    output["teeth_narration"] = teeth

    if not teeth.get("active"):
        return output

    output["lead_message"] = _live_teeth_lead(teeth)
    output["behavior_meaning"] = ""
    output["relationship_meaning"] = ""
    output["risk"] = ""
    output["event_context"] = _sanitize_event_context_for_teeth(output.get("event_context"))
    return output
