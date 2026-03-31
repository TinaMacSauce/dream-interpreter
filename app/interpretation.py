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
from app.seal import dream_has_escape_cue


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


def merge_natural_paragraphs(parts: List[str]) -> str:
    cleaned = []
    seen = set()

    for part in parts:
        part = clean_debug_like_phrase(part)
        part = clean_sentence(part)
        if not part:
            continue

        key = part.lower()
        if key in seen:
            continue

        seen.add(key)
        cleaned.append(part)

    return "\n".join(cleaned).strip()


def compact_rule_meaning_clause(hits: List[Dict[str, Any]], getter, max_items: int = 2) -> str:
    parts: List[str] = []

    for hit in hits[:max_items]:
        try:
            part = strip_trailing_punct(getter(hit["row"]))
        except Exception:
            part = ""
        if part:
            parts.append(part)

    return human_join(parts)


def compact_spiritual_meaning(
    base_matches: List[Tuple[Dict[str, Any], int, Dict[str, Any]]],
    override_hit: Optional[Dict[str, Any]],
    seal: Dict[str, Any],
    narrative_max_symbols: int,
) -> str:
    if override_hit and strip_trailing_punct(override_hit.get("spiritual", "")):
        return strip_trailing_punct(override_hit.get("spiritual", ""))

    parts: List[str] = []
    for row, _score, _hit in base_matches[:narrative_max_symbols]:
        symbol = strip_trailing_punct(get_base_symbol_input(row))
        meaning = strip_trailing_punct(get_base_symbol_meaning(row))

        if symbol and meaning:
            parts.append(f"{symbol} points to {meaning}")
        elif meaning:
            parts.append(meaning)
        elif symbol:
            parts.append(symbol)

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
):
    return {
        "mode": "symbol",
        "lead": compact_spiritual_meaning(base_matches, override_hit, seal, narrative_max_symbols),
        "behavior": compact_rule_meaning_clause(behaviors, get_behavior_meaning_modifier, max_items=2),
        "state": compact_rule_meaning_clause(states, get_state_meaning_modifier, max_items=1),
        "location": compact_rule_meaning_clause(locations, get_location_life_area_meaning, max_items=1),
        "relationship": compact_rule_meaning_clause(relationships, get_relationship_meaning_modifier, max_items=1),
    }


def build_event_scenario(
    dream: str,
    behaviors: List[Dict[str, Any]],
    states: List[Dict[str, Any]],
    locations: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
) -> Dict[str, str]:
    behavior_names = {normalize_text(x.get("name", "")) for x in behaviors}
    state_names = {normalize_text(x.get("name", "")) for x in states}
    location_names = {normalize_text(x.get("name", "")) for x in locations}
    relationship_names = {normalize_text(x.get("name", "")) for x in relationships}

    lead = ""
    support_parts: List[str] = []
    seal_type = ""

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

    behavior_clause = compact_rule_meaning_clause(behaviors, get_behavior_meaning_modifier, max_items=2)
    state_clause = compact_rule_meaning_clause(states, get_state_meaning_modifier, max_items=1)
    location_clause = compact_rule_meaning_clause(locations, get_location_life_area_meaning, max_items=1)
    relationship_clause = compact_rule_meaning_clause(relationships, get_relationship_meaning_modifier, max_items=1)

    if behavior_clause:
        support_parts.append(f"The actions reinforce {behavior_clause}")
    if state_clause:
        support_parts.append(f"The condition points to {state_clause}")
    if location_clause:
        support_parts.append(f"The setting ties this to {location_clause}")
    if relationship_clause:
        support_parts.append(f"The people involved point to {relationship_clause}")

    return {
        "lead": lead,
        "support": " ".join([strip_trailing_punct(x) for x in support_parts if x]).strip(),
        "seal_type": seal_type,
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
    event_scenario = build_event_scenario(dream, behaviors, states, locations, relationships)
    seal_type = strip_trailing_punct(seal.get("type", ""))
    seal_message = strip_trailing_punct(seal.get("message", ""))

    parts = []

    if event_scenario.get("lead"):
        parts.append(sentence(f"This dream points to {event_scenario['lead']}"))
    elif focus.get("lead"):
        parts.append(sentence(f"This dream points to {focus['lead']}"))

    if event_scenario.get("support"):
        parts.append(sentence(event_scenario["support"]))

    event_seal_type = strip_trailing_punct(event_scenario.get("seal_type", ""))
    if event_seal_type:
        seal_type = event_seal_type

    if seal_type:
        parts.append(sentence(f"The ending confirms this as {seal_type.lower()}"))

    if dream_has_escape_cue(dream) and not any(
        x in normalize_text(" ".join(parts)) for x in ["release", "escape", "came out", "regain control"]
    ):
        parts.append(sentence("The ending shows that you came out of the danger, so this is a warning with a path of escape"))

    if seal_message and normalize_text(seal_message) not in normalize_text(" ".join(parts)):
        parts.append(sentence(seal_message))

    return merge_natural_paragraphs(parts), focus, event_scenario


def build_layered_support_paragraph(
    behaviors: List[Dict[str, Any]],
    states: List[Dict[str, Any]],
    locations: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
) -> str:
    behavior_parts = compress_phrase_list([get_behavior_meaning_modifier(x["row"]) for x in behaviors])
    state_parts = compress_phrase_list([get_state_meaning_modifier(x["row"]) for x in states])
    location_parts = compress_phrase_list([get_location_life_area_meaning(x["row"]) for x in locations])
    relationship_parts = compress_phrase_list([get_relationship_meaning_modifier(x["row"]) for x in relationships])

    lines = []
    if behavior_parts:
        lines.append(sentence(f"The actions in the dream suggest {human_join(behavior_parts)}"))
    if state_parts:
        lines.append(sentence(f"The condition of what appeared points to {human_join(state_parts)}"))
    if location_parts:
        lines.append(sentence(f"The setting connects this message to {human_join(location_parts)}"))
    if relationship_parts:
        lines.append(sentence(f"The people involved point to {human_join(relationship_parts)}"))

    return merge_natural_paragraphs(lines)


def build_real_world_impact_paragraph(
    base_matches,
    behaviors,
    states,
    locations,
    relationships,
    override_hit,
    narrative_max_symbols: int,
) -> str:
    effect_parts = []

    for row, _score, _hit in base_matches[:narrative_max_symbols]:
        effect_parts.append(get_base_symbol_effects(row))

    effect_parts.extend([get_behavior_physical_modifier(x["row"]) for x in behaviors])
    effect_parts.extend([get_state_physical_modifier(x["row"]) for x in states])
    effect_parts.extend([get_location_physical_area_meaning(x["row"]) for x in locations])
    effect_parts.extend([get_relationship_physical_modifier(x["row"]) for x in relationships])

    if override_hit:
        effect_parts.append((override_hit or {}).get("physical", ""))

    effect_parts = compress_phrase_list([normalize_effect_phrase(x) for x in effect_parts if x])

    if not effect_parts:
        return "No clear physical effects were generated."

    return sentence(f"In practical terms, this may show up as {human_join(effect_parts)}")


def build_action_guidance_paragraph(
    base_matches,
    behaviors,
    states,
    locations,
    relationships,
    override_hit,
    narrative_max_symbols: int,
) -> str:
    action_parts = []

    for row, _score, _hit in base_matches[:narrative_max_symbols]:
        action_parts.append(get_base_symbol_action(row))

    action_parts.extend([get_behavior_action_modifier(x["row"]) for x in behaviors])
    action_parts.extend([get_state_action_modifier(x["row"]) for x in states])
    action_parts.extend([get_location_action_modifier(x["row"]) for x in locations])
    action_parts.extend([get_relationship_action_modifier(x["row"]) for x in relationships])

    if override_hit:
        action_parts.append((override_hit or {}).get("action", ""))

    action_parts = compress_phrase_list([normalize_action_phrase(x) for x in action_parts if x])

    if not action_parts:
        return "Pray for wisdom and confirmation."

    return sentence(" ".join(action_parts).strip())


def build_final_summary_paragraph(interpretation: Dict[str, str], seal: Dict[str, Any]) -> str:
    seal_type = strip_trailing_punct(seal.get("type", ""))
    risk = strip_trailing_punct(seal.get("risk", ""))

    parts = []
    if seal_type and risk:
        parts.append(sentence(f"Overall, this is a {seal_type.lower()} message with {risk.lower()} risk"))

    parts.append(sentence("Do not react in panic. Take the message seriously, stay spiritually grounded, and respond with discipline"))
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
    top_symbols = [get_base_symbol_input(row) for row, _score, _hit in base_matches]

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
    summary_message = build_final_summary_paragraph(
        {
            "spiritual_meaning": core_message,
            "effects_in_physical_realm": physical_message,
            "what_to_do": action_message,
        },
        seal,
    )

    interpretation = {
        "spiritual_meaning": core_message or "No clear spiritual meaning was generated.",
        "effects_in_physical_realm": physical_message or "No clear physical effects were generated.",
        "what_to_do": action_message or "Pray for wisdom and confirmation.",
    }

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
    compact_meaning = event_scenario.get("lead") or focus.get("lead") or compact_spiritual_meaning(base_matches, override_hit, seal, narrative_max_symbols) or "the main spiritual message here"
    compact_behavior = focus.get("behavior") or compact_rule_meaning_clause(behaviors, get_behavior_meaning_modifier, max_items=2) or "the active pattern in the dream"
    compact_state = focus.get("state") or compact_rule_meaning_clause(states, get_state_meaning_modifier, max_items=1) or "the condition attached to the message"
    compact_location = focus.get("location") or compact_rule_meaning_clause(locations, get_location_life_area_meaning, max_items=1) or "the area of life being touched"
    compact_relationship = focus.get("relationship") or compact_rule_meaning_clause(relationships, get_relationship_meaning_modifier, max_items=1) or "the people dimension of the dream"

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
            spiritual_parts.append(f"{symbol} points to {base_meaning}")
        elif symbol:
            spiritual_parts.append(symbol)

        if base_effects:
            physical_parts.append(normalize_effect_phrase(base_effects))
        if base_action:
            action_parts.append(normalize_action_phrase(base_action))

    spiritual_text = sentence(human_join(spiritual_parts)) if spiritual_parts else "No clear spiritual meaning was generated."
    physical_text = sentence(f"In practical terms, this may show up as {human_join(physical_parts)}") if physical_parts else "No clear physical effects were generated."
    action_text = sentence(" ".join(compress_phrase_list(action_parts))) if action_parts else "Pray for wisdom and confirmation."

    return {
        "spiritual_meaning": spiritual_text,
        "effects_in_physical_realm": physical_text,
        "what_to_do": action_text,
    }
