from typing import Any, Dict, List, Optional

from app.config import Config
from app.utils import compress_phrase_list, human_join, normalize_text


def _clean(value: str) -> str:
    return (value or "").strip()


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _sentence(text: str) -> str:
    text = " ".join((text or "").split()).strip()
    if not text:
        return ""
    text = text[:1].upper() + text[1:]
    if text[-1] not in ".!?":
        text += "."
    return text


def _semantic_key(text: str) -> str:
    text_n = normalize_text(text)
    replacements = {
        "repeated or confirmed": "confirmed",
        "confirms": "confirmed",
        "major symbol from the dream": "major symbol",
        "the ending": "ending",
        "suggests": "points to",
        "indicates": "points to",
        "reveals": "points to",
        "active spiritual pursuit": "active spiritual pressure",
    }
    out = text_n
    for old, new in replacements.items():
        out = out.replace(old, new)
    out = " ".join(out.split())
    return out


def _normalize_list(items: List[str], max_items: int = 3) -> List[str]:
    out: List[str] = []
    seen = set()

    for item in items or []:
        item = _clean(item)
        if not item:
            continue

        key = _semantic_key(item)
        if not key or key in seen:
            continue

        seen.add(key)
        out.append(item)

        if len(out) >= max_items:
            break

    return out


def _dedupe_sentences(parts: List[str]) -> List[str]:
    out: List[str] = []

    for part in parts:
        part = _clean(part)
        if not part:
            continue

        key = _semantic_key(part)
        if not key:
            continue

        duplicate = False
        for existing in out:
            ex_key = _semantic_key(existing)

            if key == ex_key:
                duplicate = True
                break
            if key in ex_key:
                duplicate = True
                break
            if ex_key in key and len(ex_key.split()) >= 4:
                duplicate = True
                break

            if "ending" in key and "ending" in ex_key and "major symbol" in key and "major symbol" in ex_key:
                duplicate = True
                break

        if duplicate:
            continue

        out.append(part)

    return out


def _should_skip_placeholder(value: str, placeholders: List[str]) -> bool:
    value_n = normalize_text(value)
    return any(normalize_text(p) in value_n for p in placeholders)


def _phrase_to_natural_clause(text: str) -> str:
    text = _clean(text)
    if not text:
        return ""

    out = text

    direct_replacements = [
        (" suggests ", " points to "),
        (" indicate ", " points to "),
        (" indicates ", " points to "),
        (" reveals ", " points to "),
    ]
    for old, new in direct_replacements:
        out = out.replace(old, new)

    return " ".join(out.split()).strip()


def _extract_symbol_meaning_pairs(top_symbols: List[str], lead_message: str) -> List[Dict[str, str]]:
    """
    Reads simple doctrine phrases like:
    'snake suggests enemy and chasing suggests pursuit'
    and turns them into structured pairs.
    """
    pairs: List[Dict[str, str]] = []
    lead = _clean(lead_message)
    if not lead:
        return pairs

    chunks = [c.strip() for c in lead.split(" and ") if c.strip()]
    normalized_symbols = [normalize_text(x) for x in top_symbols if x]

    for chunk in chunks:
        chunk_n = normalize_text(chunk)
        if " suggests " in chunk_n:
            parts = chunk.split(" suggests ", 1)
        elif " points to " in chunk_n:
            parts = chunk.split(" points to ", 1)
        else:
            continue

        if len(parts) != 2:
            continue

        left = _clean(parts[0])
        right = _clean(parts[1])

        if not left or not right:
            continue

        if normalized_symbols and normalize_text(left) not in normalized_symbols:
            # still allow it, but prefer actual top symbols
            pass

        pairs.append({"symbol": left, "meaning": right})

    return pairs


def _build_smoother_lead_from_pairs(
    pairs: List[Dict[str, str]],
    behavior_meaning: str,
) -> str:
    """
    Makes the summary sound less robotic without adding doctrine.
    Example:
    snake -> enemy
    chasing -> pursuit
    =>
    This dream points to enemy pursuit.
    """
    if not pairs:
        return ""

    if len(pairs) == 1:
        meaning = _clean(pairs[0].get("meaning", ""))
        if meaning:
            return _sentence(f"This dream points to {meaning}")
        return ""

    symbol_to_meaning = {normalize_text(p["symbol"]): _clean(p["meaning"]) for p in pairs}
    behavior_n = normalize_text(behavior_meaning)

    # Common natural compression: main symbol + pursuit/chase behavior
    if "snake" in symbol_to_meaning and ("pursuit" in behavior_n or "chasing" in symbol_to_meaning):
        base_meaning = symbol_to_meaning.get("snake", "")
        if base_meaning and "enemy" in normalize_text(base_meaning):
            return _sentence("This dream points to enemy pursuit")

    ordered_meanings = []
    seen = set()
    for pair in pairs:
        meaning = _clean(pair.get("meaning", ""))
        key = _semantic_key(meaning)
        if meaning and key and key not in seen:
            seen.add(key)
            ordered_meanings.append(meaning)

    if not ordered_meanings:
        return ""

    if len(ordered_meanings) == 1:
        return _sentence(f"This dream points to {ordered_meanings[0]}")

    return _sentence(f"This dream points to {human_join(ordered_meanings)}")


def _build_lead_sentence(
    lead_message: str,
    top_symbols: List[str],
    behavior_meaning: str,
) -> str:
    lead_message = _phrase_to_natural_clause(lead_message)
    if not lead_message:
        return ""

    pairs = _extract_symbol_meaning_pairs(top_symbols, lead_message)
    smoother = _build_smoother_lead_from_pairs(pairs, behavior_meaning)
    if smoother:
        return smoother

    lead_lower = normalize_text(lead_message)

    if " points to " in lead_lower:
        right = lead_message.split(" points to ", 1)[1].strip()
        if right:
            return _sentence(f"This dream points to {right}")

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
    seal_message_n = _semantic_key(seal_message)

    if seal_type_n == "symbol confirmed":
        return _sentence("The ending confirms a major symbol from the dream")

    if seal_type:
        generic = _sentence(f"The ending seals this as {seal_type.lower()}")
        generic_n = _semantic_key(generic)

        if seal_message:
            if generic_n == seal_message_n:
                return generic
            if generic_n in seal_message_n or seal_message_n in generic_n:
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

    lead_sentence = _build_lead_sentence(
        lead_message=lead_message,
        top_symbols=top_symbols,
        behavior_meaning=behavior_meaning,
    )
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
