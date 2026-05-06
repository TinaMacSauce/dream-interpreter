from typing import Any, Dict, List

from app.config import Config
from app.utils import compress_phrase_list, human_join, normalize_text


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _safe_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


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
        "suggests": "points to",
        "indicates": "points to",
        "reveals": "points to",
        "shows": "points to",
        "the behavior shows": "the action points to",
        "the setting connects this to": "the place points to",
        "active spiritual pursuit": "active spiritual pressure",
    }
    out = text_n
    for old, new in replacements.items():
        out = out.replace(old, new)
    return " ".join(out.split())


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
            existing_key = _semantic_key(existing)

            if key == existing_key:
                duplicate = True
                break

            if key in existing_key:
                duplicate = True
                break

            if existing_key in key and len(existing_key.split()) >= 4:
                duplicate = True
                break

        if not duplicate:
            out.append(part)

    return out


def _phrase_to_natural_clause(text: str) -> str:
    text = _clean(text)
    if not text:
        return ""

    replacements = [
        (" suggests ", " points to "),
        (" indicate ", " point to "),
        (" indicates ", " points to "),
        (" reveals ", " points to "),
        (" shows ", " points to "),
    ]

    out = text
    for old, new in replacements:
        out = out.replace(old, new)

    return " ".join(out.split()).strip()


def _should_skip_placeholder(value: str, placeholders: List[str]) -> bool:
    value_n = normalize_text(value)
    return any(normalize_text(p) in value_n for p in placeholders)


def _is_attack_or_impersonation_text(text: str) -> bool:
    text_n = normalize_text(text)
    trigger_terms = [
        "spiritual attack",
        "warfare",
        "impersonation",
        "familiar person",
        "familiar spirit",
        "appearing in the form",
    ]
    return any(term in text_n for term in trigger_terms)


def _is_emotion_reversal_text(text: str) -> bool:
    text_n = normalize_text(text)
    return "joy is near" in text_n or "sadness is near" in text_n


def _get_event_context(doctrine_facts: Dict[str, Any]) -> Dict[str, Any]:
    return doctrine_facts.get("event_context", {}) or {}


def _get_event_summary(doctrine_facts: Dict[str, Any]) -> str:
    return _safe_text(doctrine_facts.get("event_summary"))


def _event_value(event_context: Dict[str, Any], section: str, field: str) -> str:
    block = event_context.get(section, {}) or {}
    return _safe_text(block.get(field))


def _build_event_lead_sentence(
    lead_message: str,
    event_context: Dict[str, Any],
    behavior_meaning: str,
    relationship_meaning: str,
) -> str:
    """
    Human-first lead sentence.

    Priority:
    1. Protected doctrine warnings stay explicit.
    2. Primary action leads the interpretation.
    3. Subject/place/ending support the action.
    4. Fallback to doctrine lead message.
    """
    lead_message = _safe_text(lead_message)

    if lead_message and _is_attack_or_impersonation_text(lead_message):
        return _sentence(lead_message)

    if lead_message and _is_emotion_reversal_text(lead_message):
        return _sentence(lead_message)

    action_name = _event_value(event_context, "primary_action", "name")
    action_meaning = _event_value(event_context, "primary_action", "meaning")
    subject = _event_value(event_context, "primary_subject", "")
    place_name = _event_value(event_context, "primary_place", "name")
    place_meaning = _event_value(event_context, "primary_place", "meaning")
    ending_name = _event_value(event_context, "primary_ending", "name")
    ending_meaning = _event_value(event_context, "primary_ending", "meaning")

    if action_name:
        if subject and place_meaning:
            return _sentence(
                f"This dream centers on {action_name}, with {subject} involved, and the place connects it to {place_meaning}"
            )

        if subject:
            return _sentence(
                f"This dream centers on {action_name}, and {subject} shows what the action is affecting or revealing"
            )

        if place_meaning:
            return _sentence(
                f"This dream centers on {action_name}, and the place connects it to {place_meaning}"
            )

        if action_meaning:
            return _sentence(f"This dream centers on {action_name}, which points to {action_meaning}")

        return _sentence(f"This dream centers on {action_name}")

    if lead_message:
        lead_n = normalize_text(lead_message)

        if relationship_meaning:
            rel_n = normalize_text(relationship_meaning)
            relationship_terms = [
                "this concerns your mother",
                "this concerns your father",
                "this concerns that person",
                "this concerns your partner",
                "this concerns that child",
                "this concerns that family member",
            ]
            if any(term in rel_n for term in relationship_terms):
                return _sentence(_phrase_to_natural_clause(lead_message))

        if " points to " in lead_n:
            right = lead_message.split(" points to ", 1)[1].strip()
            if right:
                return _sentence(f"This dream points to {right}")

        if " suggests " in lead_n:
            right = lead_message.split(" suggests ", 1)[1].strip()
            if right:
                return _sentence(f"This dream points to {right}")

        return _sentence(f"This dream points to {lead_message}")

    if ending_name and ending_meaning:
        return _sentence(f"The ending of this dream points to {ending_meaning}")

    return ""


def _build_action_sentence(event_context: Dict[str, Any], behavior_meaning: str) -> str:
    action_name = _event_value(event_context, "primary_action", "name")
    action_meaning = _event_value(event_context, "primary_action", "meaning")

    meaning = action_meaning or behavior_meaning
    if not action_name and not meaning:
        return ""

    if action_name and meaning:
        return _sentence(f"The action is the strongest part: {action_name} points to {_phrase_to_natural_clause(meaning)}")

    if action_name:
        return _sentence(f"The action is the strongest part: {action_name}")

    return _sentence(f"The action in the dream points to {_phrase_to_natural_clause(meaning)}")


def _build_subject_sentence(event_context: Dict[str, Any], top_symbols: List[str]) -> str:
    subject = _event_value(event_context, "primary_subject", "")
    subjects = event_context.get("subjects", []) or top_symbols or []
    subjects = _normalize_list(subjects, max_items=max(1, Config.NARRATION_MAX_SYMBOLS))

    if subject:
        return _sentence(f"The main subject is {subject}, so it adds detail to what the action is touching")

    if len(subjects) == 1:
        return _sentence(f"The main subject in the dream is {subjects[0]}")

    if len(subjects) > 1:
        return _sentence(f"The main subjects in the dream are {human_join(subjects)}")

    return ""


def _build_place_sentence(event_context: Dict[str, Any], location_meaning: str) -> str:
    place_name = _event_value(event_context, "primary_place", "name")
    place_meaning = _event_value(event_context, "primary_place", "meaning")
    place_physical = _event_value(event_context, "primary_place", "physical_area")

    meaning = place_meaning or location_meaning

    if not place_name and not meaning:
        return ""

    if place_name and meaning:
        text = f"The place matters: {place_name} points to {_phrase_to_natural_clause(meaning)}"
        if place_physical:
            text += f", especially around {place_physical}"
        return _sentence(text)

    if place_name:
        return _sentence(f"The place matters: {place_name} gives context to where this issue is showing up")

    return _sentence(f"The setting connects this dream to {_phrase_to_natural_clause(meaning)}")


def _build_state_relationship_sentence(
    event_context: Dict[str, Any],
    state_meaning: str,
    relationship_meaning: str,
) -> str:
    state_name = _event_value(event_context, "primary_state", "name")
    state_event_meaning = _event_value(event_context, "primary_state", "meaning")
    relationship_name = _event_value(event_context, "primary_relationship", "name")
    relationship_event_meaning = _event_value(event_context, "primary_relationship", "meaning")

    clauses: List[str] = []

    final_state_meaning = state_event_meaning or state_meaning
    if final_state_meaning and not _should_skip_placeholder(final_state_meaning, ["condition attached to the message"]):
        if state_name:
            clauses.append(f"{state_name} points to {_phrase_to_natural_clause(final_state_meaning)}")
        else:
            clauses.append(f"the condition points to {_phrase_to_natural_clause(final_state_meaning)}")

    final_relationship_meaning = relationship_event_meaning or relationship_meaning
    if final_relationship_meaning and not _should_skip_placeholder(final_relationship_meaning, ["people dimension of the dream"]):
        if relationship_name:
            clauses.append(f"{relationship_name} points to {_phrase_to_natural_clause(final_relationship_meaning)}")
        else:
            clauses.append(f"the people involved point to {_phrase_to_natural_clause(final_relationship_meaning)}")

    clauses = compress_phrase_list(clauses)
    if not clauses:
        return ""

    return _sentence(human_join(clauses))


def _build_ending_sentence(event_context: Dict[str, Any], seal_type: str, seal_message: str) -> str:
    ending_name = _event_value(event_context, "primary_ending", "name")
    ending_meaning = _event_value(event_context, "primary_ending", "meaning")
    ending_action = _event_value(event_context, "primary_ending", "action")

    if ending_name and ending_meaning:
        text = f"The ending is important: {ending_name} points to {ending_meaning}"
        if ending_action:
            text += f", so the response is to {ending_action}"
        return _sentence(text)

    if ending_name:
        return _sentence(f"The ending is important because it shows the outcome: {ending_name}")

    seal_type = _safe_text(seal_type)
    seal_message = _safe_text(seal_message)

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


def _build_guidance_sentence(
    event_context: Dict[str, Any],
    interpretation: Dict[str, str],
) -> str:
    action_guidance = _event_value(event_context, "primary_action", "action")
    place_guidance = _event_value(event_context, "primary_place", "action")
    ending_guidance = _event_value(event_context, "primary_ending", "action")
    doctrine_guidance = _safe_text((interpretation or {}).get("what_to_do"))

    guidance_items = _normalize_list(
        [ending_guidance, action_guidance, place_guidance, doctrine_guidance],
        max_items=2,
    )

    if not guidance_items:
        return ""

    if len(guidance_items) == 1:
        return _sentence(f"The response is to {guidance_items[0]}")

    return _sentence(f"The response is to {human_join(guidance_items)}")


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

    Locked rule:
    This function may clarify wording, but it must not invent new doctrine.
    It uses backend doctrine facts and event_context only.
    """
    max_symbols = max(1, Config.NARRATION_MAX_SYMBOLS)

    doctrine_facts = doctrine_facts or {}
    interpretation = interpretation or {}

    event_context = _get_event_context(doctrine_facts)

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

    lead_sentence = _build_event_lead_sentence(
        lead_message=lead_message,
        event_context=event_context,
        behavior_meaning=behavior_meaning,
        relationship_meaning=relationship_meaning,
    )

    action_sentence = _build_action_sentence(
        event_context=event_context,
        behavior_meaning=behavior_meaning,
    )

    subject_sentence = _build_subject_sentence(
        event_context=event_context,
        top_symbols=top_symbols,
    )

    place_sentence = _build_place_sentence(
        event_context=event_context,
        location_meaning=location_meaning,
    )

    state_relationship_sentence = _build_state_relationship_sentence(
        event_context=event_context,
        state_meaning=state_meaning,
        relationship_meaning=relationship_meaning,
    )

    ending_sentence = _build_ending_sentence(
        event_context=event_context,
        seal_type=seal_type,
        seal_message=seal_message,
    )

    guidance_sentence = _build_guidance_sentence(
        event_context=event_context,
        interpretation=interpretation,
    )

    risk_sentence = _build_risk_sentence(risk)

    for item in [
        lead_sentence,
        action_sentence,
        subject_sentence,
        place_sentence,
        state_relationship_sentence,
        ending_sentence,
        guidance_sentence,
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
    Structured payload for future AI narration.
    It must stay grounded in doctrine facts only.
    """
    max_input_chars = max(500, Config.AI_NARRATION_MAX_INPUT_CHARS)

    instruction = (
        "Rewrite the supplied doctrine findings into natural, clear, spiritually serious language. "
        "Follow this order: action first, subject second, place third, ending last. "
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
    - keeps response shape stable
    """
    if not Config.NARRATION_ENABLED:
        return _build_disabled_result()

    deterministic_summary = build_doctrine_bound_summary(
        doctrine_facts=doctrine_facts,
        interpretation=interpretation,
    )

    result: Dict[str, Any] = {
        "mode": Config.NARRATION_MODE or "deterministic_event",
        "enabled": True,
        "used_ai": False,
        "readable_summary": deterministic_summary,
        "prompt_payload": {},
    }

    if Config.NARRATION_INCLUDE_PROMPT_PAYLOAD:
        result["prompt_payload"] = build_ai_prompt_payload(doctrine_facts, interpretation)

    return result
