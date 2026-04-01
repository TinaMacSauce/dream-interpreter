from typing import Any, Dict, List

from app.config import Config
from app.utils import compress_phrase_list, human_join, normalize_text


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

        key = normalize_text(item)
        if not key or key in seen:
            continue

        seen.add(key)
        out.append(item)

        if len(out) >= max_items:
            break

    return out


def _sentence(text: str) -> str:
    text = " ".join((text or "").split()).strip()
    if not text:
        return ""
    text = text[:1].upper() + text[1:]
    if text[-1] not in ".!?":
        text += "."
    return text


def _should_skip_placeholder(value: str, placeholders: List[str]) -> bool:
    value_n = normalize_text(value)
    return any(normalize_text(p) in value_n for p in placeholders)


def _dedupe_sentences(parts: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()

    for part in parts:
        part = _clean(part)
        if not part:
            continue

        key = normalize_text(part)
        if not key or key in seen:
            continue

        duplicate = False
        for existing in out:
            ex_key = normalize_text(existing)
            if key == ex_key:
                duplicate = True
                break
            if key in ex_key or ex_key in key:
                duplicate = True
                break

        if duplicate:
            continue

        seen.add(key)
        out.append(part)

    return out


def _phrase_to_natural_clause(text: str) -> str:
    """
    Converts literal doctrine fragments into smoother summary language
    without inventing new doctrine.
    """
    text = _clean(text)
    if not text:
        return ""

    lower = normalize_text(text)

    replacements = [
        (" suggests ", " points to "),
        (" indicate ", " points to "),
        (" indicates ", " points to "),
        (" reveals ", " points to "),
        (" enemy and pursuit", " enemy pursuit"),
        (" active spiritual pursuit", " active spiritual pressure"),
    ]

    out = text
    for old, new in replacements:
        out = out.replace(old, new)

    return out.strip()


def _build_lead_sentence(lead_message: str) -> str:
    lead_message = _phrase_to_natural_clause(lead_message)
    if not lead_message:
        return ""

    lead_lower = normalize_text(lead_message)

    if " points to " in lead_lower:
        return _sentence(f"This dream points to {lead_message.split(' points to ', 1)[1]}")

    if lead_lower.startswith(("a ", "an ", "the ")):
        return _sentence(f"This dream points to {lead_message}")

    return _sentence(f"This dream points to {lead_message}")


def _build_symbols_sentence(top_symbols: List[str]) -> str:
    top_symbols = _normalize_list(top_symbols, max_items=max(1, Config.NARRATION_MAX_SYMBOLS))
    if not top_symbols:
        return ""
    if len(top_symbols) == 1:
        return _sentence(f"The main symbol in the dream is {top_symbols[0]}")
    return _sentence(f"The main symbols in the dream are {human_join(top_symbols)}")


def _build_support_sentence(
    behavior_meaning: str,
    state_meaning: str,
    location_meaning: str,
    relationship_meaning: str,
) -> str:
    support_clauses: List[str] = []

    if behavior_meaning and not _should_skip_placeholder(
        behavior_meaning,
        ["active pattern in the dream"],
    ):
        support_clauses.append(f"the actions show {_phrase_to_natural_clause(behavior_meaning)}")

    if state_meaning and not _should_skip_placeholder(
        state_meaning,
        ["condition attached to the message"],
    ):
        support_clauses.append(f"the condition of what appeared points to {_phrase_to_natural_clause(state_meaning)}")

    if location_meaning and not _should_skip_placeholder(
        location_meaning,
        ["area of life being touched"],
    ):
        support_clauses.append(f"the setting connects this to {_phrase_to_natural_clause(location_meaning)}")

    if relationship_meaning and not _should_skip_placeholder(
        relationship_meaning,
        ["people dimension of the dream"],
    ):
        support_clauses.append(f"the people involved point to {_phrase_to_natural_clause(relationship_meaning)}")

    support_clauses = compress_phrase_list(support_clauses)
    if not support_clauses:
        return ""

    return _sentence(human_join(support_clauses))


def _build_seal_sentence(seal_type: str, seal_message: str) -> str:
    seal_type = _clean(seal_type)
    seal_message = _clean(seal_message)

    if not seal_type and not seal_message:
        return ""

    seal_type_n = normalize_text(seal_type)
    seal_message_n = normalize_text(seal_message)

    if seal_type_n == "symbol confirmed":
        return _sentence("The ending confirms a major symbol from the dream")

    if seal_type:
        generic = _sentence(f"The ending seals this as {seal_type.lower()}")
        if seal_message:
            # If the message already says basically the same thing, keep only one.
            if "confirm" in seal_message_n and "symbol" in seal_message_n and seal_type_n in {"symbol confirmed", "confirmed"}:
                return generic
            if normalize_text(generic) in seal_message_n or seal_message_n in normalize_text(generic):
                return generic
        return generic

    return _sentence(seal_message)


def _build_risk_sentence(risk: str) -> str:
    risk = _clean(risk)
    if not risk:
        return ""
    return _sentence(f"The level of concern attached to this dream appears to be {risk.lower()}")


def _fallback_from_interpretation(interpretation: Dict[str, str]) -> str:
    spiritual = _safe_text((interpretation or {}).get("spiritual_meaning"))
    if not spiritual:
        return ""
    return _sentence(spiritual)


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

    lead_sentence = _build_lead_sentence(lead_message)
    symbols_sentence = _build_symbols_sentence(top_symbols)
    support_sentence = _build_support_sentence(
        behavior_meaning=behavior_meaning,
        state_meaning=state_meaning,
        location_meaning=location_meaning,
        relationship_meaning=relationship_meaning,
    )
    seal_sentence = _build_seal_sentence(seal_type, seal_message)
    risk_sentence = _build_risk_sentence(risk)

    for item in [
        lead_sentence,
        symbols_sentence,
        support_sentence,
        seal_sentence,
        risk_sentence,
    ]:
        if item:
            parts.append(item)

    parts = _dedupe_sentences(parts)

    if not parts:
        fallback = _fallback_from_interpretation(interpretation)
        if fallback:
            return fallback
        return ""

    return "\n".join(parts).strip()


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
        "prompt_payload": {},
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
