from __future__ import annotations

from typing import Any, Dict, Tuple

from app.snake_doctrine import build_snake_narration_facts


def build_snake_output_summary(snake: Dict[str, Any]) -> str:
    if snake.get("active") is not True:
        return ""
    return str(snake.get("narration_text") or "").strip()


def attach_snake_narration_facts(
    dream: str,
    doctrine_facts: Dict[str, Any] | None,
) -> Dict[str, Any]:
    output = dict(doctrine_facts or {})
    snake = build_snake_narration_facts(dream)
    output["snake_narration"] = snake
    if snake.get("active") is True:
        output["lead_message"] = build_snake_output_summary(snake)
        output["risk"] = ""
        output["relationship_meaning"] = ""
    return output


def bind_snake_output_contract(
    *,
    doctrine_facts: Dict[str, Any],
    seal: Dict[str, Any],
    interpretation: Dict[str, str],
    full_interpretation: str,
) -> Tuple[Dict[str, Any], Dict[str, str], str]:
    snake = doctrine_facts.get("snake_narration")
    snake = snake if isinstance(snake, dict) else {}
    # Mixed Teeth/Snake arbitration is not approved; preserve the existing,
    # specialized Teeth contract when both symbols appear.
    teeth = doctrine_facts.get("teeth_narration")
    if snake.get("active") is not True or (
        isinstance(teeth, dict) and teeth.get("active") is True
    ):
        return dict(seal or {}), dict(interpretation or {}), full_interpretation

    bound_seal = dict(seal or {})
    bound_seal["risk"] = ""
    bound_seal["risk_label"] = ""
    bound_seal["legacy_risk_suppressed"] = True
    bound_seal["warning_assessment"] = {
        "warning_present": True,
        "interpretation_confidence": "approved_rule_match",
        "predictive_certainty": "none",
        "registry_verified": bool(
            (snake.get("doctrine_registry") or {}).get("verified") is True
        ),
        "applied_rule_ids": list(snake.get("applied_rule_ids") or []),
    }

    bound_interpretation = dict(interpretation or {})
    bound_interpretation["spiritual_meaning"] = build_snake_output_summary(snake)
    bound_interpretation["effects_in_physical_realm"] = (
        "This is tradition-based spiritual guidance, not proof of an enemy, "
        "culprit, curse, illness, causation, or a future event."
    )
    bound_interpretation["what_to_do"] = " ".join(
        item for item in (
            str(snake.get("response_guidance") or "").strip(),
            str(snake.get("best_practice_guidance") or "").strip(),
            "These faith practices are not scientifically proven prevention and do not guarantee an outcome.",
        ) if item
    )
    bound_full = "\n\n".join(
        str(bound_interpretation.get(key) or "").strip()
        for key in ("spiritual_meaning", "effects_in_physical_realm", "what_to_do")
        if str(bound_interpretation.get(key) or "").strip()
    )
    return bound_seal, bound_interpretation, bound_full
