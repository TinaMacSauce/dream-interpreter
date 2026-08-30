import re
from typing import Any, Dict, List, Set

from app.config import Config
from app.utils import compress_phrase_list, human_join, normalize_text


WHITESPACE_RE = re.compile(r"\s+")
SENTENCE_END_RE = re.compile(r"[.!?]\s*$")

PHRASE_REPLACEMENTS = {
    "suggests": "points to",
    "indicates": "points to",
    "reveals": "points to",
    "shows": "points to",
    "confirms": "points to",
    "indicate": "point to",
    "the behavior shows": "the action points to",
    "the setting connects this to": "the place points to",
    "active spiritual pursuit": "active spiritual pressure",
    "old mindset,stagnant pattern": "old mindset or stagnant pattern",
    "old mindset, stagnant pattern": "old mindset or stagnant pattern",
}

FULL_SENTENCE_PREFIXES = (
    "the ending", "the action", "the place", "the setting", "the condition",
    "the people", "the main subject", "this dream",
)


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _safe_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _display_text(value: Any) -> str:
    text = _safe_text(value)
    if not text:
        return ""
    text = text.replace("_", " ").replace("-", " ")
    return WHITESPACE_RE.sub(" ", text).strip()


def _sentence(text: str) -> str:
    text = WHITESPACE_RE.sub(" ", (text or "")).strip()
    if not text:
        return ""
    text = _display_text(text)
    if not text:
        return ""
    text = text[:1].upper() + text[1:]
    if not SENTENCE_END_RE.search(text):
        text += "."
    return text


def _semantic_key(text: str) -> str:
    text_n = normalize_text(_display_text(text))
    for old, new in PHRASE_REPLACEMENTS.items():
        text_n = text_n.replace(old, new)
    return WHITESPACE_RE.sub(" ", text_n).strip()


def _normalize_list(items: Any, max_items: int = 3, display: bool = True) -> List[str]:
    if isinstance(items, str):
        items = [items]
    out: List[str] = []
    seen: Set[str] = set()
    for item in items or []:
        if isinstance(item, dict):
            item = item.get("name") or item.get("symbol") or item.get("value") or ""
        processed_item = _display_text(item) if display else _clean(item)
        if not processed_item:
            continue
        key = _semantic_key(processed_item)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(processed_item)
        if len(out) >= max_items:
            break
    return out


def _dedupe_sentences(parts: List[str]) -> List[str]:
    out: List[str] = []
    place_overlap_terms = ["backwardness", "stagnation", "regression", "old cycles"]
    action_overlap_terms = ["chased", "escaping", "sadness", "crying"]
    for part in parts:
        part = _display_text(_clean(part))
        if not part:
            continue
        key = _semantic_key(part)
        if not key:
            continue
        duplicate = False
        for existing in out:
            existing_key = _semantic_key(existing)
            if key == existing_key or key in existing_key or existing_key in key:
                duplicate = True
                break
            if "ending" in key and "ending" in existing_key:
                duplicate = True
                break
            if "place" in key and "place" in existing_key:
                if any(term in key for term in place_overlap_terms) and any(term in existing_key for term in place_overlap_terms):
                    duplicate = True
                    break
            if "action" in key and "action" in existing_key:
                if any(term in key for term in action_overlap_terms) and any(term in existing_key for term in action_overlap_terms):
                    duplicate = True
                    break
        if not duplicate:
            out.append(part)
    return out


def _phrase_to_natural_clause(text: str) -> str:
    text = _clean(text)
    if not text:
        return ""
    out = f" {text} "
    for old, new in PHRASE_REPLACEMENTS.items():
        out = out.replace(f" {old} ", f" {new} ")
    out = re.sub(r",(?=\S)", ", ", out)
    return WHITESPACE_RE.sub(" ", _display_text(out)).strip()


def _ending_outcome_clause(ending_name: str, ending_meaning: str) -> str:
    ending_name_clean = _display_text(ending_name)
    ending_meaning_clean = _phrase_to_natural_clause(ending_meaning)
    if not ending_name_clean or not ending_meaning_clean:
        return ""
    ending_n = normalize_text(ending_name_clean)
    escape_endings = {"escape", "escaped", "escaping", "got away", "got away safely", "made it out", "came out"}
    if ending_n in escape_endings:
        return f"the successful escape points to {ending_meaning_clean}"
    return f"the {ending_name_clean} ending points to {ending_meaning_clean}"


def _is_full_sentence_fragment(text: str) -> bool:
    return normalize_text(text).startswith(FULL_SENTENCE_PREFIXES)


def _lead_to_sentence(right: str) -> str:
    right_clean = _phrase_to_natural_clause(right)
    if not right_clean:
        return ""
    if _is_full_sentence_fragment(right_clean):
        return _sentence(right_clean)
    return _sentence(f"This dream points to {right_clean}")


def _should_skip_placeholder(value: str, placeholders: List[str]) -> bool:
    value_n = normalize_text(value)
    return any(normalize_text(p) in value_n for p in placeholders)


def _is_attack_or_impersonation_text(text: str) -> bool:
    text_n = normalize_text(text)
    return any(term in text_n for term in (
        "spiritual attack", "warfare", "impersonation", "familiar person",
        "familiar spirit", "appearing in the form",
    ))


def _is_emotion_reversal_text(text: str) -> bool:
    text_n = normalize_text(text)
    return "joy is near" in text_n or "sadness is near" in text_n


def _get_event_context(doctrine_facts: Dict[str, Any]) -> Dict[str, Any]:
    context = (doctrine_facts or {}).get("event_context")
    return context if isinstance(context, dict) else {}


def _get_event_summary(doctrine_facts: Dict[str, Any]) -> str:
    return _safe_text((doctrine_facts or {}).get("event_summary"))


def _event_value(event_context: Dict[str, Any], section: str, field: str = "") -> str:
    if not isinstance(event_context, dict):
        return ""
    value = event_context.get(section, "")
    if value is None:
        return ""
    if isinstance(value, str):
        if field in ("", "name", "value"):
            return _safe_text(value)
        return ""
    if isinstance(value, dict):
        if not field:
            return _safe_text(value.get("name") or value.get("value") or "")
        return _safe_text(value.get(field))
    return _safe_text(value)


def _build_event_lead_sentence(lead_message: str, event_context: Dict[str, Any], behavior_meaning: str, relationship_meaning: str) -> str:
    lead_message = _safe_text(lead_message)
    ending_name = _display_text(_event_value(event_context, "primary_ending", "name"))
    ending_meaning = _phrase_to_natural_clause(_event_value(event_context, "primary_ending", "meaning"))
    if not lead_message:
        return ""
    if _is_attack_or_impersonation_text(lead_message):
        attack_clause = _phrase_to_natural_clause(lead_message)
        ending_clause = _ending_outcome_clause(ending_name, ending_meaning)
        if ending_clause:
            return _sentence(f"{attack_clause}, but {ending_clause}")
        return _sentence(attack_clause)
    if _is_emotion_reversal_text(lead_message):
        return _sentence(_phrase_to_natural_clause(lead_message))
    action_name = _display_text(_event_value(event_context, "primary_action", "name"))
    action_meaning = _phrase_to_natural_clause(_event_value(event_context, "primary_action", "meaning"))
    subject = _display_text(_event_value(event_context, "primary_subject", "name"))
    place_meaning = _phrase_to_natural_clause(_event_value(event_context, "primary_place", "meaning"))
    if action_name:
        if subject and place_meaning:
            return _sentence(f"This dream centers on {action_name}, with {subject} involved, and the place connects it to {place_meaning}")
        if subject:
            return _sentence(f"This dream centers on {action_name}, and {subject} shows what the action is affecting or revealing")
        if place_meaning:
            return _sentence(f"This dream centers on {action_name}, and the place connects it to {place_meaning}")
        if action_meaning:
            return _sentence(f"This dream centers on {action_name}, which points to {action_meaning}")
        return _sentence(f"This dream centers on {action_name}")
    lead_n = normalize_text(lead_message)
    if relationship_meaning:
        rel_n = normalize_text(relationship_meaning)
        relationship_terms = (
            "this concerns your mother", "this concerns your father", "this concerns that person",
            "this concerns your partner", "this concerns that child", "this concerns that family member",
        )
        if any(term in rel_n for term in relationship_terms):
            return _sentence(_phrase_to_natural_clause(lead_message))
    if " points to " in lead_n:
        return _lead_to_sentence(lead_message.split(" points to ", 1)[1])
    if " suggests " in lead_n:
        return _lead_to_sentence(lead_message.split(" suggests ", 1)[1])
    if _is_full_sentence_fragment(lead_message):
        return _sentence(_phrase_to_natural_clause(lead_message))
    return _sentence(f"This dream points to {_phrase_to_natural_clause(lead_message)}")


def _build_action_sentence(event_context: Dict[str, Any], behavior_meaning: str) -> str:
    action_name = _display_text(_event_value(event_context, "primary_action", "name"))
    action_meaning = _phrase_to_natural_clause(_event_value(event_context, "primary_action", "meaning"))
    meaning = action_meaning or _phrase_to_natural_clause(behavior_meaning)
    if not action_name and not meaning:
        return ""
    if action_name and meaning:
        return _sentence(f"The action is the strongest part: {action_name} points to {meaning}")
    if action_name:
        return _sentence(f"The action is the strongest part: {action_name}")
    return _sentence(f"The action in the dream points to {meaning}")


def _build_subject_sentence(event_context: Dict[str, Any], top_symbols: List[str]) -> str:
    subject = _display_text(_event_value(event_context, "primary_subject", "name"))
    if subject:
        return _sentence(f"The main subject is {subject}, so it adds detail to what the action is touching")
    subjects_list = event_context.get("subjects", []) if isinstance(event_context, dict) else []
    subjects = _normalize_list(subjects_list or top_symbols or [], max_items=max(1, Config.NARRATION_MAX_SYMBOLS))
    if len(subjects) == 1:
        return _sentence(f"The main subject in the dream is {subjects[0]}")
    if len(subjects) > 1:
        return _sentence(f"The main subjects in the dream are {human_join(subjects)}")
    return ""


def _build_place_sentence(event_context: Dict[str, Any], location_meaning: str) -> str:
    place_name = _display_text(_event_value(event_context, "primary_place", "name"))
    place_meaning = _phrase_to_natural_clause(_event_value(event_context, "primary_place", "meaning"))
    place_physical = _phrase_to_natural_clause(_event_value(event_context, "primary_place", "physical_area"))
    meaning = place_meaning or _phrase_to_natural_clause(location_meaning)
    if not place_name and not meaning:
        return ""
    if place_name and meaning:
        text = f"The place matters: {place_name} points to {meaning}"
        if place_physical:
            text += f", especially around {place_physical}"
        return _sentence(text)
    if place_name:
        return _sentence(f"The place matters: {place_name} gives context to where this issue is showing up")
    return _sentence(f"The setting connects this dream to {meaning}")


def _build_state_relationship_sentence(event_context: Dict[str, Any], state_meaning: str, relationship_meaning: str) -> str:
    state_name = _display_text(_event_value(event_context, "primary_state", "name"))
    state_event_meaning = _phrase_to_natural_clause(_event_value(event_context, "primary_state", "meaning"))
    relationship_name = _display_text(_event_value(event_context, "primary_relationship", "name"))
    relationship_event_meaning = _phrase_to_natural_clause(_event_value(event_context, "primary_relationship", "meaning"))
    clauses: List[str] = []
    final_state_meaning = state_event_meaning or _phrase_to_natural_clause(state_meaning)
    if final_state_meaning and not _should_skip_placeholder(final_state_meaning, ["condition attached to the message"]):
        clauses.append(f"{state_name} points to {final_state_meaning}" if state_name else f"the condition points to {final_state_meaning}")
    final_relationship_meaning = relationship_event_meaning or _phrase_to_natural_clause(relationship_meaning)
    if final_relationship_meaning and not _should_skip_placeholder(final_relationship_meaning, ["people dimension of the dream"]):
        clauses.append(f"{relationship_name} points to {final_relationship_meaning}" if relationship_name else f"the people involved point to {final_relationship_meaning}")
    compressed_clauses = compress_phrase_list(clauses)
    if not compressed_clauses:
        return ""
    return _sentence(human_join(compressed_clauses))


def _build_ending_sentence(event_context: Dict[str, Any], seal_type: str, seal_message: str, lead_sentence: str = "") -> str:
    ending_name = _display_text(_event_value(event_context, "primary_ending", "name"))
    ending_meaning = _phrase_to_natural_clause(_event_value(event_context, "primary_ending", "meaning"))
    ending_action = _phrase_to_natural_clause(_event_value(event_context, "primary_ending", "action"))
    lead_key = _semantic_key(lead_sentence)
    ending_key = _semantic_key(ending_meaning)
    if ending_name and ending_meaning:
        if ending_key and ending_key in lead_key:
            return ""
        ending_clause = _ending_outcome_clause(ending_name, ending_meaning)
        text = f"The ending is important: {ending_clause}"
        if ending_action:
            text += f", so the response is to {ending_action}"
        return _sentence(text)
    if ending_name:
        if normalize_text(ending_name) in lead_key:
            return ""
        return _sentence(f"The ending is important because it shows the outcome: {ending_name}")
    seal_type_clean = _display_text(seal_type)
    seal_message_clean = _phrase_to_natural_clause(seal_message)
    if normalize_text(seal_type_clean) == "symbol confirmed":
        return _sentence("The ending confirms a major symbol from the dream")
    if seal_message_clean:
        if _semantic_key(seal_message_clean) in lead_key:
            return ""
        return _sentence(seal_message_clean)
    return ""


def _build_guidance_sentence(event_context: Dict[str, Any], interpretation: Dict[str, str]) -> str:
    action_guidance = _phrase_to_natural_clause(_event_value(event_context, "primary_action", "action"))
    place_guidance = _phrase_to_natural_clause(_event_value(event_context, "primary_place", "action"))
    ending_guidance = _phrase_to_natural_clause(_event_value(event_context, "primary_ending", "action"))
    doctrine_guidance = _phrase_to_natural_clause(_safe_text((interpretation or {}).get("what_to_do")))
    guidance_items = _normalize_list([ending_guidance, action_guidance, place_guidance, doctrine_guidance], max_items=4, display=False)
    if not guidance_items:
        return ""
    selected: List[str] = []
    for item in guidance_items:
        item_key = _semantic_key(item)
        if any(item_key in _semantic_key(existing) or _semantic_key(existing) in item_key for existing in selected):
            continue
        selected.append(item)
        if len(selected) >= 2:
            break
    if not selected:
        return ""
    if len(selected) == 1:
        return _sentence(f"The response is to {selected[0]}")
    return _sentence(f"The response is to {human_join(selected)}")


def _build_risk_sentence(risk: str) -> str:
    risk_text = _display_text(risk)
    if not risk_text:
        return ""
    return _sentence(f"The level of concern attached to this dream appears to be {risk_text.lower()}")


def _fallback_from_interpretation(interpretation: Dict[str, str]) -> str:
    spiritual = _phrase_to_natural_clause(_safe_text((interpretation or {}).get("spiritual_meaning")))
    if not spiritual:
        return ""
    return _sentence(spiritual)


def build_doctrine_bound_summary(doctrine_facts: Dict[str, Any], interpretation: Dict[str, str]) -> str:
    facts = doctrine_facts or {}
    interp = interpretation or {}
    event_context = _get_event_context(facts)
    max_symbols = max(1, Config.NARRATION_MAX_SYMBOLS)
    lead_sentence = _build_event_lead_sentence(
        _safe_text(facts.get("lead_message")), event_context,
        _safe_text(facts.get("behavior_meaning")), _safe_text(facts.get("relationship_meaning")),
    )
    parts = [
        lead_sentence,
        _build_action_sentence(event_context, _safe_text(facts.get("behavior_meaning"))),
        _build_subject_sentence(event_context, _normalize_list(facts.get("top_symbols", []), max_items=max_symbols)),
        _build_place_sentence(event_context, _safe_text(facts.get("location_meaning"))),
        _build_state_relationship_sentence(event_context, _safe_text(facts.get("state_meaning")), _safe_text(facts.get("relationship_meaning"))),
        _build_ending_sentence(event_context, _safe_text(facts.get("seal_type")), _safe_text(facts.get("seal_message")), lead_sentence),
        _build_guidance_sentence(event_context, interp),
        _build_risk_sentence(_safe_text(facts.get("risk"))),
    ]
    parts = _dedupe_sentences([part for part in parts if part])
    if not parts:
        return _fallback_from_interpretation(interp)
    return "\n".join(parts).strip()


def build_ai_prompt_payload(doctrine_facts: Dict[str, Any], interpretation: Dict[str, str]) -> Dict[str, Any]:
    max_input_chars = max(500, Config.AI_NARRATION_MAX_INPUT_CHARS)
    instruction = (
        "Rewrite the supplied doctrine findings into natural, clear, spiritually serious language. "
        "Follow this order: action first, subject second, place third, then use the ending to qualify and finalize the outcome. "
        "If the ending changes or resolves the threat, make that resolution explicit and do not let the earlier threat read as the final outcome. "
        "Do not repeat an ending outcome or response that has already been stated. "
        "Do not add new meanings, symbols, warnings, or instructions. "
        "Do not contradict the doctrine facts. "
        "Do not speak with absolute certainty beyond what is supplied."
    )
    if Config.AI_NARRATION_STRICT_DOCTRINE:
        instruction += " Stay strictly within the provided doctrine facts."
    constraints = {
        "provider": Config.AI_NARRATION_PROVIDER,
        "model": Config.AI_NARRATION_MODEL,
        "temperature": Config.AI_NARRATION_TEMPERATURE,
        "max_output_chars": Config.AI_NARRATION_MAX_OUTPUT_CHARS,
        "strict_doctrine": Config.AI_NARRATION_STRICT_DOCTRINE,
    }
    payload = {"instruction": instruction, "doctrine_facts": doctrine_facts or {}, "interpretation": interpretation or {}, "constraints": constraints}
    if len(str(payload)) > max_input_chars:
        return {"instruction": instruction, "doctrine_facts": doctrine_facts or {}, "constraints": constraints, "trimmed": True}
    return payload


def _build_disabled_result() -> Dict[str, Any]:
    return {"mode": "disabled", "enabled": False, "used_ai": False, "readable_summary": "", "prompt_payload": {}}


def build_narration_result(doctrine_facts: Dict[str, Any], interpretation: Dict[str, str], ai_enabled: bool = False) -> Dict[str, Any]:
    if not Config.NARRATION_ENABLED:
        return _build_disabled_result()
    deterministic_summary = build_doctrine_bound_summary(doctrine_facts=doctrine_facts, interpretation=interpretation)
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
