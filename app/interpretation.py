from typing import Any, Dict, List, Optional, Tuple

from app.fields import (
    get_base_symbol_action,
    get_base_symbol_effects,
    get_base_symbol_input,
    get_base_symbol_meaning,
    get_behavior_action_modifier,
    get_behavior_meaning_modifier,
    get_behavior_physical_modifier,
    get_location_action_modifier,
    get_location_life_area_meaning,
    get_location_physical_area_meaning,
    get_output_template,
    get_relationship_action_modifier,
    get_relationship_meaning_modifier,
    get_relationship_physical_modifier,
    get_spiritual_meaning_cell,
    get_effects_cell,
    get_state_action_modifier,
    get_state_meaning_modifier,
    get_state_physical_modifier,
    get_symbol_cell,
    get_what_to_do_cell,
)
from app.seal import dream_has_escape_cue
from app.utils import (
    clean_sentence,
    compress_phrase_list,
    human_join,
    normalize_action_phrase,
    normalize_effect_phrase,
    normalize_text,
    sentence,
    strip_trailing_punct,
)


def _hit_row(hit: Any) -> Dict[str, Any]:
    if isinstance(hit, dict):
        row = hit.get("row")
        if isinstance(row, dict):
            return row
        return hit
    return {}


def _safe_score(hit: Any, key: str) -> int:
    try:
        if isinstance(hit, dict):
            return int(hit.get(key, 0) or 0)
    except Exception:
        pass
    return 0


def _pretty_action_name(name: str) -> str:
    n = normalize_text(name)
    mapping = {
        "chasing": "being chased",
        "chased": "being chased",
        "being chased": "being chased",
        "attacking": "being attacked",
        "being attacked": "being attacked",
        "biting": "being bitten",
        "being bitten": "being bitten",
    }
    return mapping.get(n, name)


def _hit_name(hit: Any) -> str:
    if isinstance(hit, str):
        return _pretty_action_name(strip_trailing_punct(hit))

    if not isinstance(hit, dict):
        return ""

    name = hit.get("name", "")
    if name:
        return _pretty_action_name(strip_trailing_punct(name))

    row = _hit_row(hit)
    return _pretty_action_name(
        strip_trailing_punct(
            row.get("behavior_name")
            or row.get("location_name")
            or row.get("state_name")
            or row.get("relationship_name")
            or row.get("ending_name")
            or row.get("symbol")
            or row.get("input")
            or ""
        )
    )


def _is_placeholder(text: str) -> bool:
    t = normalize_text(text)
    placeholders = [
        "the active pattern in the dream",
        "the condition attached to the message",
        "the area of life being touched",
        "the people dimension of the dream",
        "the main subject",
    ]
    return any(p in t for p in placeholders)


def _is_full_sentence_fragment(text: str) -> bool:
    t = normalize_text(text)
    return t.startswith(
        (
            "this dream",
            "the ending",
            "the action",
            "the place",
            "the setting",
            "the condition",
            "the people",
            "the main subject",
        )
    )


def _clean_output_phrase(text: str) -> str:
    text = strip_trailing_punct(text)
    if not text:
        return ""

    replacements = {
        "state is from a previous season": "a previous season or old pattern is involved",
        "old points to state is from a previous season": "this connects to a previous season or old pattern",
        "old mindset,stagnant pattern": "old mindset or stagnant pattern",
        "old mindset, stagnant pattern": "old mindset or stagnant pattern",
        "dog, school represents": "the dog and school point to",
        "dog, school points to": "the dog and school point to",
        "chasing points to active spiritual pursuit": "being chased points to enemy pursuit or spiritual pursuit",
        "being chased points to active spiritual pursuit": "being chased points to enemy pursuit or spiritual pursuit",
        "the ending seals this as layered": "the ending gives this dream a layered meaning",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = clean_sentence(text)
    return strip_trailing_punct(text)


def clean_debug_like_phrase(text: str) -> str:
    text = _clean_output_phrase(text)
    if not text:
        return ""

    lower = text.lower()
    banned_starts = [
        "logic layer",
        "behavior detected",
        "state detected",
        "location detected",
        "relationship detected",
    ]

    for banned in banned_starts:
        if lower.startswith(banned):
            return ""

    if _is_placeholder(text):
        return ""

    return text


def _semantic_key(text: str) -> str:
    text_n = normalize_text(text)
    replacements = {
        "suggests": "points",
        "reveals": "points",
        "indicates": "points",
        "shows": "points",
        "the actions show": "action points",
        "the behavior shows": "action points",
        "the setting connects": "place connects",
        "the ending confirms": "ending confirms",
        "the ending seals": "ending gives",
        "active spiritual pursuit": "spiritual pursuit",
        "enemy pursuit": "spiritual pursuit",
        "being chased": "chasing",
    }

    out = text_n
    for old, new in replacements.items():
        out = out.replace(old, new)

    return " ".join(out.split())


def _is_duplicate_or_subsumed(candidate: str, existing_parts: List[str]) -> bool:
    cand_n = _semantic_key(candidate)
    if not cand_n:
        return True

    for existing in existing_parts:
        ex_n = _semantic_key(existing)
        if not ex_n:
            continue
        if cand_n == ex_n:
            return True
        if cand_n in ex_n:
            return True
        if ex_n in cand_n and len(ex_n.split()) >= 4:
            return True

    return False


def _strong_unique_parts(parts: List[str], max_items: Optional[int] = None) -> List[str]:
    out: List[str] = []

    for part in parts:
        part = clean_debug_like_phrase(part)
        if not part:
            continue
        if _is_duplicate_or_subsumed(part, out):
            continue

        out.append(part)

        if max_items is not None and len(out) >= max_items:
            break

    return out


def merge_natural_paragraphs(parts: List[str]) -> str:
    cleaned: List[str] = []

    for part in parts:
        part = clean_debug_like_phrase(part)
        if not part:
            continue
        part = sentence(part)
        if _is_duplicate_or_subsumed(part, cleaned):
            continue
        cleaned.append(part)

    return "\n\n".join(cleaned).strip()


def unique_text_parts(parts: List[str], max_items: Optional[int] = None) -> List[str]:
    return _strong_unique_parts(parts, max_items=max_items)


def _primary_behavior(behaviors: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not behaviors:
        return None
    return sorted(
        behaviors,
        key=lambda x: (_safe_score(x, "score"), _safe_score(x, "priority"), _safe_score(x, "token_len")),
        reverse=True,
    )[0]


def _primary_location(locations: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not locations:
        return None
    return sorted(
        locations,
        key=lambda x: (_safe_score(x, "score"), _safe_score(x, "priority"), _safe_score(x, "token_len")),
        reverse=True,
    )[0]


def _primary_state(states: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not states:
        return None
    return sorted(
        states,
        key=lambda x: (_safe_score(x, "score"), _safe_score(x, "priority"), _safe_score(x, "token_len")),
        reverse=True,
    )[0]


def _primary_relationship(relationships: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not relationships:
        return None
    return sorted(
        relationships,
        key=lambda x: (_safe_score(x, "score"), _safe_score(x, "priority"), _safe_score(x, "token_len")),
        reverse=True,
    )[0]


def compact_rule_meaning_clause(hits: List[Dict[str, Any]], getter, max_items: int = 2) -> str:
    parts: List[str] = []

    for hit in hits or []:
        try:
            part = _clean_output_phrase(getter(_hit_row(hit)))
        except Exception:
            part = ""
        if part and not _is_placeholder(part):
            parts.append(part)

    parts = _strong_unique_parts(parts, max_items=max_items)
    return human_join(parts)


def build_base_symbol_clause(base_matches, narrative_max_symbols: int) -> str:
    clauses: List[str] = []

    for row, _score, _hit in (base_matches or [])[:narrative_max_symbols]:
        symbol = _clean_output_phrase(get_base_symbol_input(row))
        meaning = _clean_output_phrase(get_base_symbol_meaning(row))

        if symbol and meaning:
            clauses.append(f"{symbol} points to {meaning}")
        elif meaning:
            clauses.append(meaning)
        elif symbol:
            clauses.append(symbol)

    clauses = _strong_unique_parts(clauses, max_items=2)
    return human_join(clauses)


def _top_symbol_names(base_matches, narrative_max_symbols: int) -> List[str]:
    symbols = [
        _clean_output_phrase(get_base_symbol_input(row))
        for row, _score, _hit in base_matches or []
        if get_base_symbol_input(row)
    ]
    return _strong_unique_parts(symbols, max_items=max(narrative_max_symbols, 3))


def compact_spiritual_meaning(
    base_matches: List[Tuple[Dict[str, Any], int, Dict[str, Any]]],
    override_hit: Optional[Dict[str, Any]],
    seal: Dict[str, Any],
    narrative_max_symbols: int,
) -> str:
    if override_hit and _clean_output_phrase(override_hit.get("spiritual", "")):
        return _clean_output_phrase(override_hit.get("spiritual", ""))

    symbol_clause = build_base_symbol_clause(base_matches, narrative_max_symbols)
    return symbol_clause or _clean_output_phrase(seal.get("message", "")) or "the main symbolic message in the dream"


def build_primary_focus(
    dream: str,
    base_matches,
    behaviors,
    states,
    locations,
    relationships,
    override_hit,
    seal,
    narrative_max_symbols: int,
) -> Dict[str, str]:
    primary_action = _primary_behavior(behaviors)
    primary_place = _primary_location(locations)
    primary_state = _primary_state(states)
    primary_relationship = _primary_relationship(relationships)

    action_name = _hit_name(primary_action) if primary_action else ""
    action_meaning = (
        _clean_output_phrase(get_behavior_meaning_modifier(_hit_row(primary_action)))
        if primary_action
        else ""
    )

    subject_clause = build_base_symbol_clause(base_matches, narrative_max_symbols)

    place_meaning = (
        _clean_output_phrase(get_location_life_area_meaning(_hit_row(primary_place)))
        if primary_place
        else ""
    )

    state_meaning = (
        _clean_output_phrase(get_state_meaning_modifier(_hit_row(primary_state)))
        if primary_state
        else ""
    )

    relationship_meaning = (
        _clean_output_phrase(get_relationship_meaning_modifier(_hit_row(primary_relationship)))
        if primary_relationship
        else ""
    )

    if override_hit and _clean_output_phrase(override_hit.get("spiritual", "")):
        lead = _clean_output_phrase(override_hit.get("spiritual", ""))
        mode = "override"
    elif action_name and action_meaning:
        lead = f"{action_name} points to {action_meaning}"
        mode = "event"
    elif action_meaning:
        lead = action_meaning
        mode = "event"
    elif subject_clause:
        lead = subject_clause
        mode = "symbol"
    else:
        lead = _clean_output_phrase(seal.get("message", "")) or "the main spiritual message in the dream"
        mode = "seal"

    return {
        "mode": mode,
        "lead": _clean_output_phrase(lead),
        "behavior": action_meaning,
        "behavior_name": action_name,
        "state": state_meaning,
        "location": place_meaning,
        "location_name": _hit_name(primary_place) if primary_place else "",
        "relationship": relationship_meaning,
        "relationship_name": _hit_name(primary_relationship) if primary_relationship else "",
        "subject": subject_clause,
    }


def _dream_escaped(dream: str) -> bool:
    return dream_has_escape_cue(dream)


def build_event_scenario(
    dream: str,
    base_matches,
    behaviors: List[Dict[str, Any]],
    states: List[Dict[str, Any]],
    locations: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
    narrative_max_symbols: int,
) -> Dict[str, str]:
    primary_action = _primary_behavior(behaviors)
    primary_place = _primary_location(locations)
    primary_state = _primary_state(states)
    primary_relationship = _primary_relationship(relationships)

    action_name = _hit_name(primary_action) if primary_action else ""
    action_meaning = (
        _clean_output_phrase(get_behavior_meaning_modifier(_hit_row(primary_action)))
        if primary_action
        else compact_rule_meaning_clause(behaviors, get_behavior_meaning_modifier, max_items=1)
    )

    subject_clause = build_base_symbol_clause(base_matches, narrative_max_symbols)

    place_name = _hit_name(primary_place) if primary_place else ""
    place_meaning = (
        _clean_output_phrase(get_location_life_area_meaning(_hit_row(primary_place)))
        if primary_place
        else compact_rule_meaning_clause(locations, get_location_life_area_meaning, max_items=1)
    )

    state_meaning = (
        _clean_output_phrase(get_state_meaning_modifier(_hit_row(primary_state)))
        if primary_state
        else compact_rule_meaning_clause(states, get_state_meaning_modifier, max_items=1)
    )

    relationship_meaning = (
        _clean_output_phrase(get_relationship_meaning_modifier(_hit_row(primary_relationship)))
        if primary_relationship
        else compact_rule_meaning_clause(relationships, get_relationship_meaning_modifier, max_items=1)
    )

    behavior_names = {normalize_text(_hit_name(x)) for x in behaviors or []}
    location_names = {normalize_text(_hit_name(x)) for x in locations or []}

    if _dream_escaped(dream):
        seal_type = "Deliverance"
    elif {"being chased", "chased", "chasing", "being attacked", "being bitten", "fighting", "stabbing"} & behavior_names:
        seal_type = "Warfare"
    elif {"escaping", "crossing", "finding"} & behavior_names:
        seal_type = "Breakthrough"
    elif {"old_place", "old place", "old school", "old house", "old neighborhood", "old job"} & location_names:
        seal_type = "Backwardness"
    elif place_meaning and any(x in normalize_text(place_meaning) for x in ["backwardness", "stagnation", "regression"]):
        seal_type = "Backwardness"
    elif relationship_meaning:
        seal_type = "Relational"
    else:
        seal_type = ""

    lead_parts: List[str] = []
    support_parts: List[str] = []

    if action_name and action_meaning:
        lead_parts.append(f"{action_name} points to {action_meaning}")
    elif action_meaning:
        lead_parts.append(action_meaning)

    if subject_clause:
        support_parts.append(f"The main subject adds detail: {subject_clause}")

    if place_meaning:
        if place_name:
            support_parts.append(f"The place, {place_name}, connects this to {place_meaning}")
        else:
            support_parts.append(f"The place connects this to {place_meaning}")

    if state_meaning and not _is_placeholder(state_meaning):
        support_parts.append(f"The condition points to {state_meaning}")

    if relationship_meaning and not _is_placeholder(relationship_meaning):
        support_parts.append(f"The people involved point to {relationship_meaning}")

    lead = human_join(_strong_unique_parts(lead_parts, max_items=1))
    support = " ".join(_strong_unique_parts(support_parts, max_items=4))

    if not lead and subject_clause:
        lead = subject_clause

    return {
        "lead": _clean_output_phrase(lead),
        "support": _clean_output_phrase(support),
        "seal_type": _clean_output_phrase(seal_type),
        "action_name": action_name,
        "place_name": place_name,
    }


def build_core_message(
    dream: str,
    base_matches,
    behaviors,
    states,
    locations,
    relationships,
    override_hit,
    seal,
    narrative_max_symbols: int,
) -> Tuple[str, Dict[str, str], Dict[str, str]]:
    focus = build_primary_focus(
        dream,
        base_matches,
        behaviors,
        states,
        locations,
        relationships,
        override_hit,
        seal,
        narrative_max_symbols,
    )

    event_scenario = build_event_scenario(
        dream,
        base_matches,
        behaviors,
        states,
        locations,
        relationships,
        narrative_max_symbols,
    )

    seal_type = _clean_output_phrase(seal.get("type", ""))
    seal_message = _clean_output_phrase(seal.get("message", ""))

    parts: List[str] = []
    lead = _clean_output_phrase(focus.get("lead", "")) or _clean_output_phrase(event_scenario.get("lead", ""))

    if lead:
        if _is_full_sentence_fragment(lead):
            parts.append(sentence(lead))
        else:
            parts.append(sentence(f"This dream points to {lead}"))

    support = _clean_output_phrase(event_scenario.get("support", ""))
    if support and normalize_text(support) not in normalize_text(lead):
        parts.append(sentence(support))

    event_seal_type = _clean_output_phrase(event_scenario.get("seal_type", ""))
    if event_seal_type:
        seal_type = event_seal_type

    if seal_type:
        seal_n = normalize_text(seal_type)
        if seal_n in {"symbol confirmed", "confirmed"}:
            parts.append(sentence("The ending confirms a major symbol from the dream"))
        elif seal_n == "backwardness":
            parts.append(sentence("The place gives this dream a backwardness, stagnation, or regression theme"))
        elif seal_n == "deliverance":
            parts.append(sentence("The ending points to escape, protection, or deliverance"))
        else:
            parts.append(sentence(f"The ending gives this dream a {seal_type.lower()} meaning"))

    if dream_has_escape_cue(dream) and not any(
        x in normalize_text(" ".join(parts))
        for x in ["release", "escape", "escaped", "came out", "regain control", "way out", "deliverance", "protection"]
    ):
        parts.append(sentence("There is a way of escape, so the situation is serious but not final"))

    if seal_message:
        seal_message_n = normalize_text(seal_message)
        if all(seal_message_n not in normalize_text(p) for p in parts):
            parts.append(sentence(seal_message))

    return merge_natural_paragraphs(parts), focus, event_scenario


def build_layered_support_paragraph(
    behaviors: List[Dict[str, Any]],
    states: List[Dict[str, Any]],
    locations: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
) -> str:
    behavior_parts = compress_phrase_list(
        [
            _clean_output_phrase(get_behavior_meaning_modifier(_hit_row(x)))
            for x in behaviors or []
            if get_behavior_meaning_modifier(_hit_row(x))
        ]
    )[:1]

    location_parts = compress_phrase_list(
        [
            _clean_output_phrase(get_location_life_area_meaning(_hit_row(x)))
            for x in locations or []
            if get_location_life_area_meaning(_hit_row(x))
        ]
    )[:1]

    lines: List[str] = []

    if behavior_parts:
        lines.append(sentence(f"The action layer points to {human_join(behavior_parts)}"))

    if location_parts:
        lines.append(sentence(f"The place connects this to {human_join(location_parts)}"))

    return merge_natural_paragraphs(lines)


def _collapse_effects(effect_parts: List[str], max_items: int = 3) -> List[str]:
    cleaned = compress_phrase_list([normalize_effect_phrase(_clean_output_phrase(x)) for x in effect_parts if x])
    cleaned.sort(key=lambda x: (-len(normalize_text(x).split()), -len(x)))

    out: List[str] = []
    for item in cleaned:
        if _is_placeholder(item):
            continue

        item_n = _semantic_key(item)
        skip = False

        for existing in out:
            existing_n = _semantic_key(existing)
            if item_n == existing_n or item_n in existing_n or existing_n in item_n:
                skip = True
                break
            if "relationship matters" in item_n and "relationship" not in existing_n:
                skip = True
                break

        if skip:
            continue

        out.append(item)
        if len(out) >= max_items:
            break

    out.sort(key=lambda x: (len(normalize_text(x).split()), len(x)))
    return out


def build_real_world_impact_paragraph(
    base_matches,
    behaviors,
    states,
    locations,
    relationships,
    override_hit,
    narrative_max_symbols: int,
) -> str:
    effect_parts: List[str] = []

    effect_parts.extend([get_behavior_physical_modifier(_hit_row(x)) for x in behaviors or []])

    for row, _score, _hit in (base_matches or [])[:narrative_max_symbols]:
        effect_parts.append(get_base_symbol_effects(row))

    effect_parts.extend([get_location_physical_area_meaning(_hit_row(x)) for x in locations or []])
    effect_parts.extend([get_state_physical_modifier(_hit_row(x)) for x in states or []])

    if relationships:
        effect_parts.extend([get_relationship_physical_modifier(_hit_row(x)) for x in relationships or []])

    if override_hit:
        effect_parts.insert(0, (override_hit or {}).get("physical", ""))

    cleaned_effects = _collapse_effects(effect_parts, max_items=3)

    if not cleaned_effects:
        return ""

    return sentence(f"This can show up as {human_join(cleaned_effects)}")


def _collapse_actions(action_parts: List[str], max_items: int = 2) -> List[str]:
    cleaned = compress_phrase_list([normalize_action_phrase(_clean_output_phrase(x)) for x in action_parts if x])
    cleaned.sort(key=lambda x: (-len(normalize_text(x).split()), -len(x)))

    out: List[str] = []
    for item in cleaned:
        if _is_placeholder(item):
            continue

        item_n = _semantic_key(item)
        skip = False

        for existing in out:
            existing_n = _semantic_key(existing)
            if item_n == existing_n or item_n in existing_n or existing_n in item_n:
                skip = True
                break
            if "pray" in item_n and "pray" in existing_n:
                skip = True
                break
            if "stay covered" in item_n and "stay covered" in existing_n:
                skip = True
                break

        if skip:
            continue

        out.append(item)
        if len(out) >= max_items:
            break

    return out


def build_action_guidance_paragraph(
    base_matches,
    behaviors,
    states,
    locations,
    relationships,
    override_hit,
    narrative_max_symbols: int,
) -> str:
    action_parts: List[str] = []

    action_parts.extend([get_behavior_action_modifier(_hit_row(x)) for x in behaviors or []])

    for row, _score, _hit in (base_matches or [])[:narrative_max_symbols]:
        action_parts.append(get_base_symbol_action(row))

    action_parts.extend([get_location_action_modifier(_hit_row(x)) for x in locations or []])
    action_parts.extend([get_state_action_modifier(_hit_row(x)) for x in states or []])

    if relationships:
        action_parts.extend([get_relationship_action_modifier(_hit_row(x)) for x in relationships or []])

    if override_hit:
        action_parts.insert(0, (override_hit or {}).get("action", ""))

    cleaned_actions = _collapse_actions(action_parts, max_items=2)

    if not cleaned_actions:
        return "Pray for clarity and protection."

    if len(cleaned_actions) == 1:
        return sentence(cleaned_actions[0])

    return sentence(f"{cleaned_actions[0]}, and {cleaned_actions[1]}")


def build_final_summary_paragraph(interpretation: Dict[str, str], seal: Dict[str, Any]) -> str:
    seal_type = _clean_output_phrase(seal.get("type", ""))
    risk = _clean_output_phrase(seal.get("risk", ""))

    if seal_type and risk:
        return sentence(f"This is a {seal_type.lower()} message with {risk.lower()} risk")

    if seal_type:
        return sentence(f"This dream carries a {seal_type.lower()} message")

    return ""


def render_template_text(template_text: str, context: Dict[str, str]) -> str:
    text = template_text or ""

    for key, value in context.items():
        text = text.replace("{" + key + "}", _clean_output_phrase(value or ""))

    text = _clean_output_phrase(text)
    return sentence(text) if text else ""


def choose_template_type(
    override_hit: Optional[Dict[str, Any]],
    behaviors: List[Dict[str, Any]],
    states: List[Dict[str, Any]],
    locations: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
    interpretation: Dict[str, str],
    seal: Dict[str, Any],
) -> str:
    if override_hit:
        name = normalize_text(override_hit.get("override_name", ""))
        spiritual = normalize_text(override_hit.get("spiritual", ""))
        combined = f"{name} {spiritual}"

        if any(x in combined for x in ["death", "grave", "teeth falling out", "death omen"]):
            return "death_omen"
        if any(x in combined for x in ["monitor", "watch", "spy"]):
            return "monitoring"
        if any(x in combined for x in ["escape", "deliver", "freedom", "victory"]):
            return "deliverance"
        if any(x in combined for x in ["clean", "wash", "purif", "release"]):
            return "cleansing"
        if any(x in combined for x in ["promotion", "elevation", "lifted", "favor"]):
            return "promotion"
        if any(x in combined for x in ["warning", "danger", "attack", "disgrace"]):
            return "warning"

    behavior_names = {normalize_text(_hit_name(x)) for x in behaviors or []}
    location_names = {normalize_text(_hit_name(x)) for x in locations or []}
    relationship_names = {normalize_text(_hit_name(x)) for x in relationships or []}
    interp_text = normalize_text(" ".join(interpretation.values()))
    seal_type = normalize_text(seal.get("type", ""))

    if "death omen" in seal_type:
        return "death_omen"
    if {"being attacked", "being chased", "chased", "chasing", "fighting"} & behavior_names:
        return "warfare"
    if {"escaping", "crossing", "finding"} & behavior_names:
        return "breakthrough"
    if {"old_place", "old place", "old school", "old house", "old neighborhood", "old job"} & location_names:
        return "warning"
    if relationship_names:
        return "relational"
    if any(x in interp_text for x in ["monitoring spirit", "being watched", "spiritual spy"]):
        return "monitoring"
    if any(x in interp_text for x in ["cleansing", "washed", "release"]):
        return "cleansing"

    return "default"


def build_doctrine_interpretation(
    dream: str,
    base_matches,
    behaviors,
    states,
    locations,
    relationships,
    override_hit,
    templates,
    seal,
    narrative_max_symbols: int,
) -> Dict[str, Any]:
    top_symbols = _top_symbol_names(base_matches, narrative_max_symbols)

    core_message, focus, event_scenario = build_core_message(
        dream,
        base_matches,
        behaviors,
        states,
        locations,
        relationships,
        override_hit,
        seal,
        narrative_max_symbols,
    )

    support_message = build_layered_support_paragraph(behaviors, states, locations, relationships)

    physical_message = build_real_world_impact_paragraph(
        base_matches,
        behaviors,
        states,
        locations,
        relationships,
        override_hit,
        narrative_max_symbols,
    )

    action_message = build_action_guidance_paragraph(
        base_matches,
        behaviors,
        states,
        locations,
        relationships,
        override_hit,
        narrative_max_symbols,
    )

    interpretation = {
        "spiritual_meaning": core_message or "No clear spiritual meaning was generated.",
        "effects_in_physical_realm": physical_message or "",
        "what_to_do": action_message or "Pray for clarity and protection.",
    }

    summary_message = build_final_summary_paragraph(interpretation, seal)
    effective_seal_type = event_scenario.get("seal_type") or seal.get("type", "a layered message")

    template_type = choose_template_type(
        override_hit,
        behaviors,
        states,
        locations,
        relationships,
        interpretation,
        {**seal, "type": effective_seal_type},
    )

    opening_tpl = get_output_template(
        templates,
        "opening",
        "This dream is revealing a spiritual condition that requires discernment, not panic.",
    )

    closing_tpl = get_output_template(
        templates,
        "closing",
        "Dreams expose what needs attention so you can respond with wisdom, prayer, and obedience.",
    )

    compact_symbol = ", ".join([x for x in top_symbols[:2] if x]).strip()
    compact_meaning = (
        focus.get("lead")
        or event_scenario.get("lead")
        or compact_spiritual_meaning(base_matches, override_hit, seal, narrative_max_symbols)
        or "the main spiritual message here"
    )
    compact_behavior = (
        focus.get("behavior")
        or compact_rule_meaning_clause(behaviors, get_behavior_meaning_modifier, max_items=1)
    )
    compact_state = focus.get("state") or compact_rule_meaning_clause(states, get_state_meaning_modifier, max_items=1)
    compact_location = focus.get("location") or compact_rule_meaning_clause(locations, get_location_life_area_meaning, max_items=1)
    compact_relationship = focus.get("relationship") or compact_rule_meaning_clause(relationships, get_relationship_meaning_modifier, max_items=1)

    human_main_parts: List[str] = []

    if compact_meaning:
        human_main_parts.append(compact_meaning)

    if compact_symbol:
        human_main_parts.append(f"The main subject involved is {compact_symbol}")

    if compact_behavior:
        human_main_parts.append(f"The action points to {compact_behavior}")

    if compact_location:
        human_main_parts.append(f"The place connects this to {compact_location}")

    if compact_state and not _is_placeholder(compact_state):
        human_main_parts.append(f"The condition adds {compact_state}")

    if compact_relationship and not _is_placeholder(compact_relationship):
        human_main_parts.append(f"The people involved point to {compact_relationship}")

    if effective_seal_type:
        seal_phrase = str(effective_seal_type).lower()
        if normalize_text(seal_phrase) == "deliverance":
            human_main_parts.append("The ending points to escape, protection, or deliverance")
        else:
            human_main_parts.append(f"The ending gives this dream a {seal_phrase} meaning")

    rendered_main = merge_natural_paragraphs(human_main_parts)

    doctrine_facts = {
        "lead_message": _clean_output_phrase(compact_meaning),
        "top_symbols": top_symbols,
        "behavior_meaning": _clean_output_phrase(compact_behavior),
        "state_meaning": _clean_output_phrase(compact_state),
        "location_meaning": _clean_output_phrase(compact_location),
        "relationship_meaning": _clean_output_phrase(compact_relationship),
        "seal_type": _clean_output_phrase(effective_seal_type),
        "seal_message": _clean_output_phrase(seal.get("message", "")),
        "risk": _clean_output_phrase(seal.get("risk", "")),
        "override_applied": bool(override_hit),
        "override_name": (override_hit or {}).get("override_name", ""),
        "template_type": template_type,
    }

    full_parts = [
        sentence(opening_tpl),
        rendered_main,
        support_message,
        interpretation["effects_in_physical_realm"],
        interpretation["what_to_do"],
        summary_message,
        sentence(closing_tpl),
    ]

    full_interpretation = merge_natural_paragraphs(full_parts)

    return {
        "interpretation": interpretation,
        "full_interpretation": full_interpretation,
        "top_symbols": top_symbols,
        "override_applied": bool(override_hit),
        "override_name": (override_hit or {}).get("override_name", ""),
        "template_type": template_type,
        "doctrine_facts": doctrine_facts,
    }


def build_legacy_interpretation(matches, narrative_max_symbols: int) -> Dict[str, str]:
    spiritual_parts: List[str] = []
    physical_parts: List[str] = []
    action_parts: List[str] = []

    for row, _score, _hit in (matches or [])[:narrative_max_symbols]:
        symbol = _clean_output_phrase(get_symbol_cell(row))
        base_meaning = _clean_output_phrase(get_spiritual_meaning_cell(row))
        base_effects = _clean_output_phrase(get_effects_cell(row))
        base_action = _clean_output_phrase(get_what_to_do_cell(row))

        if symbol and base_meaning:
            spiritual_parts.append(f"{symbol} points to {base_meaning}")
        elif symbol:
            spiritual_parts.append(symbol)
        elif base_meaning:
            spiritual_parts.append(base_meaning)

        if base_effects:
            physical_parts.append(normalize_effect_phrase(base_effects))
        if base_action:
            action_parts.append(normalize_action_phrase(base_action))

    spiritual_parts = compress_phrase_list(spiritual_parts)[:narrative_max_symbols]
    physical_parts = compress_phrase_list(physical_parts)[:3]
    action_parts = compress_phrase_list(action_parts)[:2]

    spiritual_text = (
        sentence(f"This dream points to {human_join(spiritual_parts)}")
        if spiritual_parts
        else "No clear spiritual meaning was generated."
    )

    physical_text = (
        sentence(f"This can show up as {human_join(physical_parts)}")
        if physical_parts
        else ""
    )

    if action_parts:
        primary = action_parts[0]
        remainder = compress_phrase_list(action_parts[1:])
        if remainder:
            action_text = sentence(f"{primary}, and {human_join(remainder)}")
        else:
            action_text = sentence(primary)
    else:
        action_text = "Pray for clarity and protection."

    return {
        "spiritual_meaning": spiritual_text,
        "effects_in_physical_realm": physical_text,
        "what_to_do": action_text,
    }
