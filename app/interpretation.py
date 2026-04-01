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


def clean_debug_like_phrase(text: str) -> str:
    text = clean_sentence(text)
    if not text:
        return ""

    banned_starts = [
        "logic layer",
        "behavior detected",
        "state detected",
        "location detected",
        "relationship detected",
    ]
    lower = text.lower()
    for banned in banned_starts:
        if lower.startswith(banned):
            return ""
    return text


def _semantic_key(text: str) -> str:
    text_n = normalize_text(text)
    replacements = {
        "repeated or confirmed": "confirmed",
        "confirms": "confirmed",
        "confirmed": "confirmed",
        "major symbol from the dream": "major symbol",
        "the ending": "ending",
        "ongoing conflict": "conflict",
        "spiritually protected": "protected",
        "stay alert and spiritually protected": "stay alert protected",
        "pray and stay alert": "pray stay alert",
        "points to": "points",
        "suggests": "points",
        "reveals": "points",
        "indicates": "points",
    }
    out = text_n
    for old, new in replacements.items():
        out = out.replace(old, new)
    out = " ".join(out.split())
    return out


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

        if "ending" in cand_n and "ending" in ex_n:
            if "major symbol" in cand_n and "major symbol" in ex_n:
                return True

    return False


def _strong_unique_parts(parts: List[str], max_items: Optional[int] = None) -> List[str]:
    out: List[str] = []

    for part in parts:
        part = strip_trailing_punct(part)
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
        part = clean_sentence(part)
        if not part:
            continue

        if _is_duplicate_or_subsumed(part, cleaned):
            continue

        cleaned.append(part)

    return "\n".join(cleaned).strip()


def unique_text_parts(parts: List[str], max_items: Optional[int] = None) -> List[str]:
    return _strong_unique_parts(parts, max_items=max_items)


def compact_rule_meaning_clause(hits: List[Dict[str, Any]], getter, max_items: int = 2) -> str:
    parts: List[str] = []

    for hit in hits:
        try:
            part = strip_trailing_punct(getter(hit["row"]))
        except Exception:
            part = ""
        if part:
            parts.append(part)

    parts = _strong_unique_parts(parts, max_items=max_items)
    return human_join(parts)


def compact_spiritual_meaning(
    base_matches: List[Tuple[Dict[str, Any], int, Dict[str, Any]]],
    override_hit: Optional[Dict[str, Any]],
    seal: Dict[str, Any],
    narrative_max_symbols: int,
) -> str:
    """
    LOCKED DOCTRINE PRIORITY:
    override > base symbol doctrine > seal fallback

    Do not let behavior/location abstractions replace the symbol doctrine lead.
    """
    if override_hit and strip_trailing_punct(override_hit.get("spiritual", "")):
        return strip_trailing_punct(override_hit.get("spiritual", ""))

    parts: List[str] = []

    for row, _score, _hit in base_matches[:narrative_max_symbols]:
        symbol = strip_trailing_punct(get_base_symbol_input(row))
        meaning = strip_trailing_punct(get_base_symbol_meaning(row))

        if symbol and meaning:
            parts.append(f"{symbol} suggests {meaning}")
        elif meaning:
            parts.append(meaning)
        elif symbol:
            parts.append(symbol)

    parts = _strong_unique_parts(parts, max_items=narrative_max_symbols)
    joined = human_join(parts)

    return joined or strip_trailing_punct(seal.get("message", "")) or "the main symbolic message in the dream"


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
    return {
        "mode": "symbol",
        "lead": compact_spiritual_meaning(base_matches, override_hit, seal, narrative_max_symbols),
        "behavior": compact_rule_meaning_clause(behaviors, get_behavior_meaning_modifier, max_items=2),
        "state": compact_rule_meaning_clause(states, get_state_meaning_modifier, max_items=1),
        "location": compact_rule_meaning_clause(locations, get_location_life_area_meaning, max_items=1),
        "relationship": compact_rule_meaning_clause(relationships, get_relationship_meaning_modifier, max_items=1),
    }


def build_base_symbol_clause(
    base_matches,
    narrative_max_symbols: int,
) -> str:
    symbol_clauses: List[str] = []

    for row, _score, _hit in base_matches[:narrative_max_symbols]:
        symbol = strip_trailing_punct(get_base_symbol_input(row))
        meaning = strip_trailing_punct(get_base_symbol_meaning(row))

        if symbol and meaning:
            symbol_clauses.append(f"{symbol} suggests {meaning}")
        elif meaning:
            symbol_clauses.append(meaning)
        elif symbol:
            symbol_clauses.append(symbol)

    symbol_clauses = _strong_unique_parts(symbol_clauses, max_items=2)
    return human_join(symbol_clauses)


def build_event_scenario(
    dream: str,
    base_matches,
    behaviors: List[Dict[str, Any]],
    states: List[Dict[str, Any]],
    locations: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
    narrative_max_symbols: int,
) -> Dict[str, str]:
    """
    This is SUPPORT only.
    It must not replace the doctrine lead unless there is no usable lead.
    """
    behavior_names = {normalize_text(x.get("name", "")) for x in behaviors}
    state_names = {normalize_text(x.get("name", "")) for x in states}
    location_names = {normalize_text(x.get("name", "")) for x in locations}
    relationship_names = {normalize_text(x.get("name", "")) for x in relationships}

    lead = ""
    support_parts: List[str] = []
    seal_type = ""

    base_clause = build_base_symbol_clause(base_matches, narrative_max_symbols)
    behavior_clause = compact_rule_meaning_clause(behaviors, get_behavior_meaning_modifier, max_items=2)
    state_clause = compact_rule_meaning_clause(states, get_state_meaning_modifier, max_items=1)
    location_clause = compact_rule_meaning_clause(locations, get_location_life_area_meaning, max_items=1)
    relationship_clause = compact_rule_meaning_clause(relationships, get_relationship_meaning_modifier, max_items=1)

    if {"being attacked", "being bitten", "being chased", "fighting"} & behavior_names:
        lead = "active spiritual warfare or pressure against you"
        seal_type = "Warfare"
    elif {"escaping", "crossing", "finding"} & behavior_names:
        lead = "a path of movement, release, or transition"
        seal_type = "Breakthrough"
    elif {"dirty", "murky", "broken", "bleeding", "dark"} & state_names:
        lead = "warning signs of contamination, instability, or damage"
        seal_type = "Warning"
    elif {"graveyard", "prison", "darkness"} & location_names:
        lead = "a heavy or restrictive spiritual condition"
        seal_type = "Bound"
    elif relationship_names:
        lead = "a relationship-centered spiritual message"
        seal_type = "Relational"
    elif base_clause:
        lead = base_clause

    if base_clause and normalize_text(base_clause) not in normalize_text(lead):
        support_parts.append(f"The main symbols suggest {base_clause}")
    if behavior_clause:
        support_parts.append(f"The actions show {behavior_clause}")
    if state_clause:
        support_parts.append(f"The condition of what appeared points to {state_clause}")
    if location_clause:
        support_parts.append(f"The setting connects this message to {location_clause}")
    if relationship_clause:
        support_parts.append(f"The people involved point to {relationship_clause}")

    support_parts = _strong_unique_parts(support_parts, max_items=4)

    return {
        "lead": strip_trailing_punct(lead),
        "support": " ".join([strip_trailing_punct(x) for x in support_parts if x]).strip(),
        "seal_type": strip_trailing_punct(seal_type),
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

    seal_type = strip_trailing_punct(seal.get("type", ""))
    seal_message = strip_trailing_punct(seal.get("message", ""))

    parts: List[str] = []

    # LOCKED: lead comes from doctrine symbol focus first, not event abstraction first.
    lead = strip_trailing_punct(focus.get("lead", "")) or strip_trailing_punct(event_scenario.get("lead", ""))

    if lead:
        lead_n = normalize_text(lead)
        if "suggests" in lead_n or "indicates" in lead_n:
            parts.append(sentence(f"This dream reveals that {lead}"))
        else:
            parts.append(sentence(f"This dream points to {lead}"))

    if event_scenario.get("support"):
        parts.append(sentence(event_scenario["support"]))

    event_seal_type = strip_trailing_punct(event_scenario.get("seal_type", ""))
    if event_seal_type:
        seal_type = event_seal_type

    ending_line = ""
    if seal_type:
        if normalize_text(seal_type) in {"symbol confirmed", "confirmed"}:
            ending_line = sentence("The ending confirms a major symbol from the dream")
        else:
            ending_line = sentence(f"The ending confirms this as {seal_type.lower()}")

    if ending_line:
        parts.append(ending_line)

    if dream_has_escape_cue(dream) and not any(
        x in normalize_text(" ".join(parts)) for x in ["release", "escape", "came out", "regain control", "way out"]
    ):
        parts.append(sentence("The ending shows there is a path of escape, so the warning is serious but not without a way through"))

    if seal_message:
        parts.append(sentence(seal_message))

    return merge_natural_paragraphs(parts), focus, event_scenario


def build_layered_support_paragraph(
    behaviors: List[Dict[str, Any]],
    states: List[Dict[str, Any]],
    locations: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
) -> str:
    behavior_parts = compress_phrase_list(
        [get_behavior_meaning_modifier(x["row"]) for x in behaviors if get_behavior_meaning_modifier(x["row"])]
    )[:2]
    state_parts = compress_phrase_list(
        [get_state_meaning_modifier(x["row"]) for x in states if get_state_meaning_modifier(x["row"])]
    )[:1]
    location_parts = compress_phrase_list(
        [get_location_life_area_meaning(x["row"]) for x in locations if get_location_life_area_meaning(x["row"])]
    )[:1]
    relationship_parts = compress_phrase_list(
        [get_relationship_meaning_modifier(x["row"]) for x in relationships if get_relationship_meaning_modifier(x["row"])]
    )[:1]

    lines: List[str] = []

    if behavior_parts:
        lines.append(sentence(f"The actions in the dream show {human_join(behavior_parts)}"))
    if state_parts:
        lines.append(sentence(f"The condition of what appeared suggests {human_join(state_parts)}"))
    if location_parts:
        lines.append(sentence(f"The setting connects the message to {human_join(location_parts)}"))
    if relationship_parts:
        lines.append(sentence(f"The people involved highlight {human_join(relationship_parts)}"))

    return merge_natural_paragraphs(lines)


def _collapse_effects(effect_parts: List[str], max_items: int = 4) -> List[str]:
    cleaned = compress_phrase_list([normalize_effect_phrase(x) for x in effect_parts if x])
    cleaned.sort(key=lambda x: (-len(normalize_text(x).split()), -len(x)))

    out: List[str] = []
    for item in cleaned:
        item_n = _semantic_key(item)
        skip = False

        for existing in out:
            existing_n = _semantic_key(existing)

            if item_n == existing_n:
                skip = True
                break
            if item_n in existing_n:
                skip = True
                break
            if existing_n in item_n and len(existing_n.split()) >= 1:
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

    for row, _score, _hit in base_matches[:narrative_max_symbols]:
        effect_parts.append(get_base_symbol_effects(row))

    effect_parts.extend([get_behavior_physical_modifier(x["row"]) for x in behaviors])
    effect_parts.extend([get_state_physical_modifier(x["row"]) for x in states])
    effect_parts.extend([get_location_physical_area_meaning(x["row"]) for x in locations])
    effect_parts.extend([get_relationship_physical_modifier(x["row"]) for x in relationships])

    if override_hit:
        effect_parts.append((override_hit or {}).get("physical", ""))

    cleaned_effects = _collapse_effects(effect_parts, max_items=4)

    if not cleaned_effects:
        return "No clear physical effects were generated."

    return sentence(f"In practical terms, this may show up as {human_join(cleaned_effects)}")


def _collapse_actions(action_parts: List[str], max_items: int = 2) -> List[str]:
    cleaned = compress_phrase_list([normalize_action_phrase(x) for x in action_parts if x])
    cleaned.sort(key=lambda x: (-len(normalize_text(x).split()), -len(x)))

    out: List[str] = []
    for item in cleaned:
        item_n = _semantic_key(item)
        skip = False

        for existing in out:
            existing_n = _semantic_key(existing)

            if item_n == existing_n:
                skip = True
                break
            if item_n in existing_n:
                skip = True
                break
            if "stay alert" in item_n and "stay alert" in existing_n:
                skip = True
                break
            if "pray" in item_n and "pray" in existing_n and ("stay alert" in item_n or "stay alert" in existing_n):
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

    for row, _score, _hit in base_matches[:narrative_max_symbols]:
        action_parts.append(get_base_symbol_action(row))

    action_parts.extend([get_behavior_action_modifier(x["row"]) for x in behaviors])
    action_parts.extend([get_state_action_modifier(x["row"]) for x in states])
    action_parts.extend([get_location_action_modifier(x["row"]) for x in locations])
    action_parts.extend([get_relationship_action_modifier(x["row"]) for x in relationships])

    if override_hit:
        action_parts.append((override_hit or {}).get("action", ""))

    cleaned_actions = _collapse_actions(action_parts, max_items=2)

    if not cleaned_actions:
        return "Pray for wisdom and confirmation."

    if len(cleaned_actions) == 1:
        return sentence(cleaned_actions[0])

    return sentence(f"{cleaned_actions[0]}, and {cleaned_actions[1]}")


def build_final_summary_paragraph(interpretation: Dict[str, str], seal: Dict[str, Any]) -> str:
    seal_type = strip_trailing_punct(seal.get("type", ""))
    risk = strip_trailing_punct(seal.get("risk", ""))

    parts: List[str] = []

    if seal_type and risk:
        parts.append(sentence(f"Overall, this is a {seal_type.lower()} message with {risk.lower()} risk"))
    elif seal_type:
        parts.append(sentence(f"Overall, this dream carries a {seal_type.lower()} message"))

    parts.append(
        sentence(
            "Do not react in panic. Take the message seriously, stay spiritually grounded, and respond with discipline"
        )
    )

    return merge_natural_paragraphs(parts)


def render_template_text(template_text: str, context: Dict[str, str]) -> str:
    text = template_text or ""
    for key, value in context.items():
        text = text.replace("{" + key + "}", strip_trailing_punct(value or ""))
    return sentence(text)


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

    behavior_names = {normalize_text(x["name"]) for x in behaviors}
    state_names = {normalize_text(x["name"]) for x in states}
    location_names = {normalize_text(x["name"]) for x in locations}
    relationship_names = {normalize_text(x["name"]) for x in relationships}
    interp_text = normalize_text(" ".join(interpretation.values()))
    seal_type = normalize_text(seal.get("type", ""))

    if "death omen" in seal_type:
        return "death_omen"
    if {"being attacked", "being chased", "fighting"} & behavior_names:
        return "warfare"
    if {"escaping", "crossing", "finding"} & behavior_names:
        return "breakthrough"
    if {"dirty", "murky", "broken", "bleeding", "dark"} & state_names:
        return "warning"
    if {"graveyard", "prison", "darkness"} & location_names:
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
    # Preserve matcher order exactly.
    top_symbols = [
        get_base_symbol_input(row)
        for row, _score, _hit in base_matches
        if get_base_symbol_input(row)
    ]
    top_symbols = _strong_unique_parts(top_symbols, max_items=max(narrative_max_symbols, 3))

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
        "effects_in_physical_realm": physical_message or "No clear physical effects were generated.",
        "what_to_do": action_message or "Pray for wisdom and confirmation.",
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
    main_tpl = get_output_template(
        templates,
        template_type,
        "{symbol} carries the central meaning of {meaning}. The dream actions support {behavior_effect}. "
        "The condition of what appeared points to {state_effect}. The setting connects this to {location_effect}. "
        "The people involved point to {relationship_effect}. The ending seals the dream as {seal_type}.",
    )

    compact_symbol = ", ".join([x for x in top_symbols[:2] if x]).strip() or "This dream"
    compact_meaning = (
        focus.get("lead")
        or event_scenario.get("lead")
        or compact_spiritual_meaning(base_matches, override_hit, seal, narrative_max_symbols)
        or "the main spiritual message here"
    )
    compact_behavior = (
        focus.get("behavior")
        or compact_rule_meaning_clause(behaviors, get_behavior_meaning_modifier, max_items=2)
        or "the active pattern in the dream"
    )
    compact_state = (
        focus.get("state")
        or compact_rule_meaning_clause(states, get_state_meaning_modifier, max_items=1)
        or "the condition attached to the message"
    )
    compact_location = (
        focus.get("location")
        or compact_rule_meaning_clause(locations, get_location_life_area_meaning, max_items=1)
        or "the area of life being touched"
    )
    compact_relationship = (
        focus.get("relationship")
        or compact_rule_meaning_clause(relationships, get_relationship_meaning_modifier, max_items=1)
        or "the people dimension of the dream"
    )

    context = {
        "symbol": compact_symbol,
        "meaning": compact_meaning,
        "behavior_effect": compact_behavior,
        "state_effect": compact_state,
        "location_effect": compact_location,
        "relationship_effect": compact_relationship,
        "seal_type": effective_seal_type,
    }

    rendered_main = render_template_text(main_tpl, context)

    doctrine_facts = {
        "lead_message": compact_meaning,
        "top_symbols": top_symbols,
        "behavior_meaning": compact_behavior,
        "state_meaning": compact_state,
        "location_meaning": compact_location,
        "relationship_meaning": compact_relationship,
        "seal_type": effective_seal_type,
        "seal_message": strip_trailing_punct(seal.get("message", "")),
        "risk": strip_trailing_punct(seal.get("risk", "")),
        "override_applied": bool(override_hit),
        "override_name": (override_hit or {}).get("override_name", ""),
        "template_type": template_type,
    }

    # LOCKED: full interpretation stays doctrine-assembled, not narration-led.
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

    for row, _score, _hit in matches[:narrative_max_symbols]:
        symbol = strip_trailing_punct(get_symbol_cell(row))
        base_meaning = strip_trailing_punct(get_spiritual_meaning_cell(row))
        base_effects = strip_trailing_punct(get_effects_cell(row))
        base_action = strip_trailing_punct(get_what_to_do_cell(row))

        if symbol and base_meaning:
            spiritual_parts.append(f"{symbol} suggests {base_meaning}")
        elif symbol:
            spiritual_parts.append(symbol)
        elif base_meaning:
            spiritual_parts.append(base_meaning)

        if base_effects:
            physical_parts.append(normalize_effect_phrase(base_effects))
        if base_action:
            action_parts.append(normalize_action_phrase(base_action))

    spiritual_parts = compress_phrase_list(spiritual_parts)[:narrative_max_symbols]
    physical_parts = compress_phrase_list(physical_parts)[:4]
    action_parts = compress_phrase_list(action_parts)[:3]

    spiritual_text = (
        sentence(f"This dream points to {human_join(spiritual_parts)}")
        if spiritual_parts
        else "No clear spiritual meaning was generated."
    )

    physical_text = (
        sentence(f"In practical terms, this may show up as {human_join(physical_parts)}")
        if physical_parts
        else "No clear physical effects were generated."
    )

    if action_parts:
        primary = action_parts[0]
        remainder = compress_phrase_list(action_parts[1:])
        if remainder:
            action_text = sentence(f"{primary}, and {human_join(remainder)}")
        else:
            action_text = sentence(primary)
    else:
        action_text = "Pray for wisdom and confirmation."

    return {
        "spiritual_meaning": spiritual_text,
        "effects_in_physical_realm": physical_text,
        "what_to_do": action_text,
    }
