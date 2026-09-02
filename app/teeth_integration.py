from typing import Any, Dict

from app.teeth_doctrine import build_teeth_narration_facts


def attach_teeth_narration_facts(
    dream: str,
    doctrine_facts: Dict[str, Any] | None,
) -> Dict[str, Any]:
    """Attach doctrine-safe Teeth facts without mutating unrelated doctrine data.

    This is the narrow bridge for the live interpreter path. It deliberately
    carries only the already-approved Teeth narration structure and leaves
    unsupported positional kinship meanings out of the doctrine payload.
    """
    output: Dict[str, Any] = dict(doctrine_facts or {})
    teeth = build_teeth_narration_facts(dream)

    output["teeth_narration"] = teeth
    return output
