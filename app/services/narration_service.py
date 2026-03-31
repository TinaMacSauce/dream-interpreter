from typing import Any, Dict, List

from app.config import Config


def _clean(value: str) -> str:
    return (value or "").strip()


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_list(items: List[str], max_items: int = 3) -> List[str]:
    out: List[str] = []
    seen = set()

    for item in items or []:
        item = _clean(item)
        if not item:
            continue

        key = item.lower()
        if key in seen:
            continue

        seen.add(key)
        out.append(item)

        if len(out) >= max_items:
            break

    return out


def _human_join(items: List[str]) -> str:
    items = [x for x in items if x]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _sentence(text: str) -> str:
    text = " ".join((text or "").split()).strip()
    if not text:
        return ""
    text = text[:1].upper() + text[1:]
    if text[-1] not in ".!?":
        text += "."
    return text


def _should_skip_placeholder(value: str, placeholders: List[str]) -> bool:
    value_n = value.strip().lower()
    return any(p in value_n for p in placeholders)


def build_doctrine_bound_summary(
    doctrine_facts: Dict[str, Any],
    interpretation: Dict[str, str],
) -> str:
    """
    Deterministic narration layer.

    This function does NOT invent doctrine.
    It only rewrites doctrine facts already produced by the backend.
    """
    max_symbols = max(1, Config.NARRATION_MAX_SYMBOLS)

    lead_message = _safe_text(doctrine_facts.get("lead_message"))
    top_symbols = _normalize_list(doctrine_facts.get("top_symbols", []), max_items=max_symbols)
    behavior_meaning = _safe_text(doctrine_facts.get("behavior_meaning"))
    state_meaning = _safe_text(doctrine_facts.get("state_meaning"))
    location_meaning = _safe_text(doctrine_facts.get("location_meaning"))
    relationship_meaning = _safe_text(doctrine_facts.get("relationship_meaning"))
    seal_type = _safe_text(doctrine_facts.get("seal_type"))
    seal_message = _safe_text(doctrine_facts.get("seal_message"))
    risk = _safe_text(doctrine_facts.get("risk"))

    parts: List[str] = []

    if lead_message:
        lead_lower = lead_message.lower()
        if lead_lower.startswith(("a ", "an ", "the ")):
            parts.append(_sentence(f"This dream points to {lead_message}"))
        elif "suggests" in lead_lower or "reveals" in lead_lower or "indicates" in lead_lower:
            parts.append(_sentence(f"This dream shows that {lead_message}"))
        else:
            parts.append(_sentence(f"This dream points to {lead_message}"))

    if top_symbols:
        if len(top_symbols) == 1:
            parts.append(_sentence(f"The main symbol in the dream is {top_symbols[0]}"))
        else:
            parts.append(_sentence(f"The main symbols in the dream are {_human_join(top_symbols)}"))

    support_clauses: List[str] = []

    if behavior_meaning and not _should_skip_placeholder(
        behavior_meaning,
        ["active pattern in the dream"],
    ):
        support_clauses.append(f"the actions show {behavior_meaning}")

    if state_meaning and not _should_skip_placeholder(
        state_meaning,
        ["condition attached to the message"],
    ):
        support_clauses.append(f"the condition of what appeared suggests {state_meaning}")

    if location_meaning and not _should_skip_placeholder(
        location_meaning,
        ["area of life being touched"],
    ):
        support_clauses.append(f"the setting connects this to {location_meaning}")

    if relationship_meaning and not _should_skip_placeholder(
        relationship_meaning,
        ["people dimension of the dream"],
    ):
        support_clauses.append(f"the people involved highlight {relationship_meaning}")

    if support_clauses:
        parts.append(_sentence(_human_join(support_clauses)))

    if seal_type:
        if seal_type.lower() == "symbol confirmed":
            parts.append(_sentence("The ending confirms a major symbol from the dream"))
        else:
            parts.append(_sentence(f"The ending seals this as {seal_type.lower()}"))

    if seal_message:
        parts.append(_sentence(seal_message))

    if risk:
        parts.append(_sentence(f"The level of concern attached to this dream appears to be {risk.lower()}"))

    if not parts:
        fallback = _safe_text((interpretation or {}).get("spiritual_meaning"))
        if fallback:
            return _sentence(fallback)
        return ""

    return "\n".join([p for p in parts if p]).strip()


def build_ai_prompt_payload(
    doctrine_facts: Dict[str, Any],
    interpretation: Dict[str, str],
) -> Dict[str, Any]:
    """
    Structured payload for a future AI narration layer.
    This keeps the prompt grounded in doctrine facts only.
    """
    max_input_chars = max(500, Config.AI_NARRATION_MAX_INPUT_CHARS)

    instruction = (
        "Rewrite the supplied doctrine findings into natural, clear, spiritually serious language. "
        "Do not add new meanings, symbols, warnings, or instructions. "
        "Do not contradict the doctrine facts. "
        "Do not speak with absolute certainty beyond what is supplied."
    )

    if Config.AI_NARRATION_STRICT_DOCTRINE:
        instruction += " Stay strictly within the provided doctrine facts."

    payload = {
        "instruction": instruction,
        "doctrine_facts": doctrine_facts or {},
        "interpretation": interpretation or {},
        "constraints": {
            "provider": Config.AI_NARRATION_PROVIDER,
            "model": Config.AI_NARRATION_MODEL,
            "temperature": Config.AI_NARRATION_TEMPERATURE,
            "max_output_chars": Config.AI_NARRATION_MAX_OUTPUT_CHARS,
            "strict_doctrine": Config.AI_NARRATION_STRICT_DOCTRINE,
        },
    }

    payload_str = str(payload)
    if len(payload_str) > max_input_chars:
        return {
            "instruction": instruction,
            "doctrine_facts": doctrine_facts or {},
            "constraints": payload["constraints"],
            "trimmed": True,
        }

    return payload


def _build_disabled_result() -> Dict[str, Any]:
    return {
        "mode": "disabled",
        "enabled": False,
        "used_ai": False,
        "readable_summary": "",
        "prompt_payload": {} if not Config.NARRATION_INCLUDE_PROMPT_PAYLOAD else {},
    }


def build_narration_result(
    doctrine_facts: Dict[str, Any],
    interpretation: Dict[str, str],
    ai_enabled: bool = False,
) -> Dict[str, Any]:
    """
    Main narration entry point.

    Current behavior:
    - honors Config.NARRATION_ENABLED
    - produces deterministic doctrine-bound narration
    - optionally includes prompt payload for future AI narration
    - is ready for future AI enhancement without changing response shape
    """
    if not Config.NARRATION_ENABLED:
        return _build_disabled_result()

    deterministic_summary = build_doctrine_bound_summary(
        doctrine_facts=doctrine_facts,
        interpretation=interpretation,
    )

    result: Dict[str, Any] = {
        "mode": Config.NARRATION_MODE or "deterministic",
        "enabled": True,
        "used_ai": False,
        "readable_summary": deterministic_summary,
        "prompt_payload": {},
    }

    if Config.NARRATION_INCLUDE_PROMPT_PAYLOAD:
        result["prompt_payload"] = build_ai_prompt_payload(doctrine_facts, interpretation)

    # Future AI upgrade point:
    #
    # if ai_enabled and Config.AI_NARRATION_ENABLED:
    #     try:
    #         ai_text = call_your_model_here(...)
    #         if ai_text:
    #             result["mode"] = "ai_enhanced"
    #             result["used_ai"] = True
    #             result["readable_summary"] = ai_text.strip()
    #     except Exception:
    #         pass

    return result
