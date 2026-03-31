from typing import Any, Dict, List

from app.utils import normalize_yes_no, row_get


def row_is_active(row: Dict[str, Any]) -> bool:
    active = row_get(row, "active")
    if not active:
        return True
    return normalize_yes_no(active)


def get_symbol_cell(row: Dict[str, Any]) -> str:
    return row_get(
        row,
        "input",
        "symbol",
        "symbols",
        "input symbol",
        "dream symbol",
        "symbol name",
    )


def get_spiritual_meaning_cell(row: Dict[str, Any]) -> str:
    return row_get(
        row,
        "spiritual meaning",
        "spiritual_meaning",
        "spiritual",
        "meaning",
        "base spiritual meaning",
        "base_spiritual_meaning",
        "final spiritual meaning",
        "final_spiritual_meaning",
    )


def get_effects_cell(row: Dict[str, Any]) -> str:
    return row_get(
        row,
        "effects in the physical realm",
        "effects_in_the_physical_realm",
        "physical effects",
        "physical_effects",
        "effects",
        "base physical effects",
        "base_physical_effects",
        "final physical effects",
        "final_physical_effects",
    )


def get_what_to_do_cell(row: Dict[str, Any]) -> str:
    return row_get(
        row,
        "what to do",
        "what_to_do",
        "action",
        "actions",
        "base action",
        "base_action",
        "final action",
        "final_action",
    )


def get_keywords_cell(row: Dict[str, Any]) -> str:
    return row_get(row, "keywords", "keyword", "tags")


def get_priority_cell(row: Dict[str, Any], default: int = 0) -> int:
    raw = row_get(row, "priority")
    if not raw:
        return default
    try:
        return int(float(raw))
    except Exception:
        return default


def get_base_symbol_input(row: Dict[str, Any]) -> str:
    return row_get(
        row,
        "symbol",
        "input",
        "input symbol",
        "dream symbol",
    )


def get_base_symbol_category(row: Dict[str, Any]) -> str:
    return row_get(row, "category") or "unknown"


def get_base_symbol_meaning(row: Dict[str, Any]) -> str:
    return row_get(
        row,
        "base_spiritual_meaning",
        "base spiritual meaning",
        "spiritual meaning",
        "meaning",
    )


def get_base_symbol_effects(row: Dict[str, Any]) -> str:
    return row_get(
        row,
        "base_physical_effects",
        "base physical effects",
        "physical effects",
        "effects",
    )


def get_base_symbol_action(row: Dict[str, Any]) -> str:
    return row_get(
        row,
        "base_action",
        "base action",
        "action",
        "actions",
    )


def get_rule_name(row: Dict[str, Any], *keys: str) -> str:
    return row_get(row, *keys)


def get_rule_keywords(row: Dict[str, Any]) -> List[str]:
    import re

    from app.utils import normalize_text

    raw = row_get(row, "keywords", "keyword", "tags")
    if not raw:
        return []

    parts = re.split(r"[,|;]+", raw)
    out: List[str] = []
    seen = set()

    for part in parts:
        part = normalize_text(part)
        if not part:
            continue
        if part in seen:
            continue
        seen.add(part)
        out.append(part)

    return out


def get_behavior_name(row: Dict[str, Any]) -> str:
    return get_rule_name(row, "behavior_name", "behavior")


def get_behavior_meaning_modifier(row: Dict[str, Any]) -> str:
    return row_get(
        row,
        "meaning_modifier",
        "meaning modifier",
        "effect",
        "effects",
    )


def get_behavior_physical_modifier(row: Dict[str, Any]) -> str:
    return row_get(
        row,
        "physical_modifier",
        "physical modifier",
        "physical_effect",
        "physical effects",
        "effect",
        "effects",
    )


def get_behavior_action_modifier(row: Dict[str, Any]) -> str:
    return row_get(
        row,
        "action_modifier",
        "action modifier",
        "action",
        "actions",
    )


def get_state_name(row: Dict[str, Any]) -> str:
    return get_rule_name(row, "state_name", "state")


def get_state_meaning_modifier(row: Dict[str, Any]) -> str:
    return row_get(
        row,
        "meaning_modifier",
        "meaning modifier",
        "effect",
        "effects",
    )


def get_state_physical_modifier(row: Dict[str, Any]) -> str:
    return row_get(
        row,
        "physical_modifier",
        "physical modifier",
        "physical_effect",
        "physical effects",
        "effect",
        "effects",
    )


def get_state_action_modifier(row: Dict[str, Any]) -> str:
    return row_get(
        row,
        "action_modifier",
        "action modifier",
        "action",
        "actions",
    )


def get_location_name(row: Dict[str, Any]) -> str:
    return get_rule_name(row, "location_name", "location")


def get_location_life_area_meaning(row: Dict[str, Any]) -> str:
    return row_get(
        row,
        "life_area_meaning",
        "life area meaning",
        "effect",
        "effects",
    )


def get_location_physical_area_meaning(row: Dict[str, Any]) -> str:
    return row_get(
        row,
        "physical_area_meaning",
        "physical area meaning",
        "effect",
        "effects",
    )


def get_location_action_modifier(row: Dict[str, Any]) -> str:
    return row_get(
        row,
        "action_modifier",
        "action modifier",
        "action",
        "actions",
    )


def get_relationship_name(row: Dict[str, Any]) -> str:
    return get_rule_name(row, "relationship_name", "relationship")


def get_relationship_meaning_modifier(row: Dict[str, Any]) -> str:
    return row_get(
        row,
        "meaning_modifier",
        "meaning modifier",
        "effect",
        "effects",
    )


def get_relationship_physical_modifier(row: Dict[str, Any]) -> str:
    return row_get(
        row,
        "physical_modifier",
        "physical modifier",
        "physical_effect",
        "physical effects",
        "effect",
        "effects",
    )


def get_relationship_action_modifier(row: Dict[str, Any]) -> str:
    return row_get(
        row,
        "action_modifier",
        "action modifier",
        "action",
        "actions",
    )


def get_override_name(row: Dict[str, Any]) -> str:
    return row_get(row, "override_name", "override name", "name")


def get_override_spiritual(row: Dict[str, Any]) -> str:
    return row_get(
        row,
        "final_spiritual_meaning",
        "final spiritual meaning",
        "spiritual meaning",
        "meaning",
    )


def get_override_physical(row: Dict[str, Any]) -> str:
    return row_get(
        row,
        "final_physical_effects",
        "final physical effects",
        "physical effects",
        "effects",
    )


def get_override_action(row: Dict[str, Any]) -> str:
    return row_get(
        row,
        "final_action",
        "final action",
        "action",
        "actions",
    )


def get_output_template(rows: List[Dict[str, Any]], template_type: str, fallback: str) -> str:
    from app.utils import normalize_text

    wanted = normalize_text(template_type)

    for row in rows:
        if not row_is_active(row):
            continue

        row_type = normalize_text(row_get(row, "template_type"))
        if row_type == wanted:
            text = row_get(row, "template_text")
            if text:
                return text

    return fallback
