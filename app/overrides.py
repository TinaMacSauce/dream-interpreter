from typing import Any, Dict, List, Optional, Set, Tuple

from app.fields import (
    behavior_is_attack,
    behavior_supports_impersonation,
    get_base_symbol_category,
    get_base_symbol_input,
    get_behavior_name,
    get_override_action,
    get_override_condition,
    get_override_name,
    get_override_physical,
    get_override_spiritual,
    get_override_target_mode,
    get_relationship_name,
    override_is_hard,
    relationship_impersonation_when_attacking,
    relationship_literal_when_peaceful,
    row_is_active,
)
from app.utils import (
    clean_sentence,
    contains_phrase,
    extract_dream_ending_text,
    normalize_text,
    row_get,
)


def _priority_from_row(row: Dict[str, Any]) -> int:
    try:
        return int(str(row_get(row, "priority", "priority_weight", "weight") or "0").strip())
    except Exception:
        return 0


def _normalized_set(values: List[str]) -> Set[str]:
    out: Set[str] = set()
    for value in values:
        v = normalize_text(value)
        if v:
            out.add(v)
    return out


def _base_symbol_names(
    base_matches: List[Tuple[Dict[str, Any], int, Dict[str, Any]]]
) -> List[str]:
    out: List[str] = []
    for row, _score, _hit in base_matches:
        symbol = get_base_symbol_input(row)
        if symbol:
            out.append(normalize_text(symbol))
    return out


def _base_symbol_categories(
    base_matches: List[Tuple[Dict[str, Any], int, Dict[str, Any]]]
) -> List[str]:
    out: List[str] = []
    for row, _score, _hit in base_matches:
        category = normalize_text(get_base_symbol_category(row)) or "unknown"
        out.append(category)
    return out


def _behavior_names(behaviors: List[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for hit in behaviors:
        row = hit.get("row", {})
        name = hit.get("name") or get_behavior_name(row)
        name_n = normalize_text(name)
        if name_n:
            out.append(name_n)
    return out


def _state_names(states: List[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for hit in states:
        name_n = normalize_text(hit.get("name", ""))
        if name_n:
            out.append(name_n)
    return out


def _location_names(locations: List[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for hit in locations:
        name_n = normalize_text(hit.get("name", ""))
        if name_n:
            out.append(name_n)
    return out


def _relationship_names(relationships: List[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for hit in relationships:
        row = hit.get("row", {})
        name = hit.get("name") or get_relationship_name(row)
        name_n = normalize_text(name)
        if name_n:
            out.append(name_n)
    return out


def _has_attack_behavior(behaviors: List[Dict[str, Any]]) -> bool:
    hard_attack_names = {
        "being attacked",
        "attacking",
        "being bitten",
        "biting",
        "being chased",
        "chasing",
        "fighting",
        "stabbing",
        "shooting",
        "strangling",
        "threatening",
    }

    for hit in behaviors:
        row = hit.get("row", {})
        name = normalize_text(hit.get("name") or get_behavior_name(row))

        if name in hard_attack_names:
            return True
        if row and behavior_is_attack(row):
            return True
        if row and behavior_supports_impersonation(row):
            return True

    return False


def _relationship_impersonation_enabled(relationships: List[Dict[str, Any]]) -> bool:
    for hit in relationships:
        row = hit.get("row", {})
        if row and relationship_impersonation_when_attacking(row):
            return True
    return False


def _relationship_literal_enabled(relationships: List[Dict[str, Any]]) -> bool:
    for hit in relationships:
        row = hit.get("row", {})
        if row and relationship_literal_when_peaceful(row):
            return True
    return False


def override_context(
    dream: str,
    base_matches,
    behaviors,
    states,
    locations,
    relationships,
) -> Dict[str, Any]:
    symbol_names = _base_symbol_names(base_matches)
    categories = _base_symbol_categories(base_matches)
    behavior_names = _behavior_names(behaviors)
    state_names = _state_names(states)
    location_names = _location_names(locations)
    relationship_names = _relationship_names(relationships)

    return {
        "symbol": _normalized_set(symbol_names),
        "category": _normalized_set(categories),
        "behavior": _normalized_set(behavior_names),
        "state": _normalized_set(state_names),
        "location": _normalized_set(location_names),
        "relationship": _normalized_set(relationship_names),
        "text": normalize_text(dream),
        "ending": normalize_text(extract_dream_ending_text(dream)),
        "has_attack_behavior": _has_attack_behavior(behaviors),
        "relationship_impersonation_enabled": _relationship_impersonation_enabled(relationships),
        "relationship_literal_enabled": _relationship_literal_enabled(relationships),
    }


def parse_override_groups(condition: str) -> List[List[str]]:
    import re

    groups: List[List[str]] = []
    for raw_group in re.split(r"\|\|", condition or ""):
        tokens = [
            clean_sentence(x).strip()
            for x in re.split(r"\+", raw_group)
            if clean_sentence(x).strip()
        ]
        if tokens:
            groups.append(tokens)
    return groups


def token_matches_context(token: str, ctx: Dict[str, Any]) -> Tuple[bool, int]:
    raw = (token or "").strip()
    if not raw:
        return False, 0

    negative = raw.startswith("!")
    if negative:
        raw = raw[1:].strip()

    raw_norm = normalize_text(raw)

    field = ""
    value = ""
    op = ""

    if "!=" in raw:
        field, value = raw.split("!=", 1)
        op = "!="
    elif "=" in raw:
        field, value = raw.split("=", 1)
        op = "="

    if field and op:
        field_n = normalize_text(field)
        value_n = normalize_text(value)

        if field_n in {"symbol", "category", "behavior", "state", "location", "relationship"}:
            exists = value_n in ctx.get(field_n, set())
        elif field_n in {"text", "ending"}:
            exists = contains_phrase(ctx.get(field_n, ""), value_n)
        elif field_n in {
            "has_attack_behavior",
            "relationship_impersonation_enabled",
            "relationship_literal_enabled",
        }:
            truthy = value_n in {"1", "true", "yes", "on"}
            exists = bool(ctx.get(field_n, False)) == truthy
        else:
            exists = False

        matched = (not exists) if op == "!=" else exists
        if negative:
            matched = not matched

        specificity = 6 if field_n in {
            "symbol",
            "category",
            "behavior",
            "state",
            "location",
            "relationship",
            "has_attack_behavior",
            "relationship_impersonation_enabled",
            "relationship_literal_enabled",
        } else 3
        return matched, specificity

    exists_plain = False

    for field_name in ["symbol", "category", "behavior", "state", "location", "relationship"]:
        if raw_norm in ctx.get(field_name, set()):
            exists_plain = True
            break

    if not exists_plain:
        if contains_phrase(ctx.get("text", ""), raw_norm):
            exists_plain = True
        elif contains_phrase(ctx.get("ending", ""), raw_norm):
            exists_plain = True

    matched_plain = (not exists_plain) if negative else exists_plain
    return matched_plain, 2


def _build_locked_doctrine_override(
    dream: str,
    base_matches,
    behaviors,
    states,
    locations,
    relationships,
) -> Optional[Dict[str, Any]]:
    """
    Hard-locked doctrine overrides.

    Priority:
    1. dead people / familiar spirit logic
    2. eating in dreams
    3. family-member or familiar-person attack impersonation
    4. literal family/friend emotion reversals
    """
    symbol_names = set(_base_symbol_names(base_matches))
    relationship_names = set(_relationship_names(relationships))
    behavior_names = set(_behavior_names(behaviors))
    dream_text = normalize_text(dream)

    dead_terms = {"dead person", "dead people", "deceased", "corpse"}

    # 1. Dead people / deceased visit logic
    if dead_terms & symbol_names:
        if any(x in dream_text for x in ["recently deceased", "recent dead", "just died", "newly dead"]):
            return {
                "override_name": "recent_dead_return",
                "spiritual": "this may point to a returning familiar spirit and should not be accepted",
                "physical": "spiritual deception, heaviness, or death-linked pressure",
                "action": "tell them they are dead, reject the encounter, and pray against any familiar spirit",
                "priority": 1000,
                "row": None,
                "condition": "hard_locked_dead_person_recent",
                "target_mode": "spiritual_warning",
                "is_hard_override": True,
            }

        return {
            "override_name": "dead_person_rule",
            "spiritual": "this dream carries a serious warning around death or spiritual danger",
            "physical": "spiritual heaviness, danger, or death-linked pressure",
            "action": "do not accept anything from the dead and cover yourself in prayer",
            "priority": 1000,
            "row": None,
            "condition": "hard_locked_dead_person",
            "target_mode": "spiritual_warning",
            "is_hard_override": True,
        }

    # 2. Eating in dreams
    if "eating" in behavior_names or contains_phrase(dream_text, "eating"):
        return {
            "override_name": "eating_rule",
            "spiritual": "this may point to something spiritually harmful trying to take root",
            "physical": "spiritual contamination, heaviness, or negative influence",
            "action": "pray against it immediately and reject what is not of God",
            "priority": 950,
            "row": None,
            "condition": "hard_locked_eating",
            "target_mode": "spiritual_warning",
            "is_hard_override": True,
        }

    # 3. Familiar person attacking = impersonation / attack
    if relationship_names and _has_attack_behavior(behaviors):
        if _relationship_impersonation_enabled(relationships) or relationship_names:
            return {
                "override_name": "familiar_attack_rule",
                "spiritual": "this is a spiritual attack appearing in the form of a familiar person",
                "physical": "spiritual warfare, pressure, or hostile interference",
                "action": "pray against the attack and cancel it immediately instead of causing confusion with the person seen",
                "priority": 900,
                "row": None,
                "condition": "hard_locked_familiar_attack",
                "target_mode": "impersonation",
                "is_hard_override": True,
            }

    # 4. Literal person rules for peaceful appearance
    if relationship_names and (_relationship_literal_enabled(relationships) or not _has_attack_behavior(behaviors)):
        target = next(iter(sorted(relationship_names)))

        if "crying" in behavior_names:
            return {
                "override_name": "relationship_crying_rule",
                "spiritual": f"this dream is about {target}, and joy is near",
                "physical": f"a positive turn, breakthrough, or joyful event connected to {target}",
                "action": f"pray over {target} and prepare for good news",
                "priority": 700,
                "row": None,
                "condition": "hard_locked_relationship_crying",
                "target_mode": "literal_person",
                "is_hard_override": False,
            }

        if {"happy", "smiling", "laughing"} & behavior_names:
            return {
                "override_name": "relationship_laughing_rule",
                "spiritual": f"this dream is about {target}, and sadness is near",
                "physical": f"an emotional reversal, disappointment, or sorrow connected to {target}",
                "action": f"pray for {target} and stay spiritually alert",
                "priority": 690,
                "row": None,
                "condition": "hard_locked_relationship_laughing",
                "target_mode": "literal_person",
                "is_hard_override": False,
            }

    return None


def apply_override_rules(
    base_matches,
    behaviors,
    states,
    locations,
    relationships,
    dream: str,
    override_rows: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    # 1. Locked doctrine overrides come first
    locked = _build_locked_doctrine_override(
        dream=dream,
        base_matches=base_matches,
        behaviors=behaviors,
        states=states,
        locations=locations,
        relationships=relationships,
    )
    if locked:
        return locked

    # 2. Sheet-driven overrides
    ctx = override_context(dream, base_matches, behaviors, states, locations, relationships)

    best: Optional[Dict[str, Any]] = None
    best_score = -10**9

    for row in override_rows:
        if not row_is_active(row):
            continue

        condition = get_override_condition(row)
        if not condition:
            continue

        groups = parse_override_groups(condition)
        if not groups:
            continue

        matched_any_group = False
        group_score = -10**9

        for group in groups:
            all_ok = True
            specificity = 0

            for token in group:
                ok, spec = token_matches_context(token, ctx)
                if not ok:
                    all_ok = False
                    break
                specificity += spec

            if all_ok:
                matched_any_group = True
                group_score = max(group_score, specificity + len(group))

        if not matched_any_group:
            continue

        priority = _priority_from_row(row)
        hard_bonus = 10000 if override_is_hard(row) else 0
        score = hard_bonus + (priority * 100) + group_score

        if score > best_score:
            best_score = score
            best = row

    if not best:
        return None

    priority = _priority_from_row(best)

    return {
        "override_name": get_override_name(best) or row_get(best, "condition"),
        "spiritual": get_override_spiritual(best) or row_get(best, "override_effect", "effect", "effects"),
        "physical": get_override_physical(best),
        "action": get_override_action(best),
        "priority": priority,
        "row": best,
        "condition": get_override_condition(best),
        "target_mode": get_override_target_mode(best),
        "is_hard_override": override_is_hard(best),
    }
