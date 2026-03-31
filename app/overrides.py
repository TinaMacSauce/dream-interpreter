from typing import Any, Dict, List, Optional, Tuple

from app.fields import get_base_symbol_category, get_base_symbol_input, row_get, row_is_active
from app.utils import clean_sentence, contains_phrase, extract_dream_ending_text, normalize_text


def override_context(
    dream: str,
    base_matches,
    behaviors,
    states,
    locations,
    relationships,
) -> Dict[str, Any]:
    symbol_names = [
        normalize_text(get_base_symbol_input(row))
        for row, _score, _hit in base_matches
        if get_base_symbol_input(row)
    ]
    categories = [
        normalize_text(get_base_symbol_category(row)) or "unknown"
        for row, _score, _hit in base_matches
    ]
    behavior_names = [normalize_text(x["name"]) for x in behaviors if x.get("name")]
    state_names = [normalize_text(x["name"]) for x in states if x.get("name")]
    location_names = [normalize_text(x["name"]) for x in locations if x.get("name")]
    relationship_names = [normalize_text(x["name"]) for x in relationships if x.get("name")]

    return {
        "symbol": set(symbol_names),
        "category": set(categories),
        "behavior": set(behavior_names),
        "state": set(state_names),
        "location": set(location_names),
        "relationship": set(relationship_names),
        "text": normalize_text(dream),
        "ending": normalize_text(extract_dream_ending_text(dream)),
    }


def parse_override_groups(condition: str) -> List[List[str]]:
    import re

    groups = []
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
        else:
            exists = False

        matched = (not exists) if op == "!=" else exists
        if negative:
            matched = not matched

        specificity = 5 if field_n in {
            "symbol",
            "category",
            "behavior",
            "state",
            "location",
            "relationship",
        } else 3
        return matched, specificity

    exists_plain = False
    for field_name in ["symbol", "category", "behavior", "state", "location", "relationship"]:
        if raw_norm in ctx.get(field_name, set()):
            exists_plain = True
            break

    if not exists_plain:
        if contains_phrase(ctx.get("text", ""), raw_norm) or contains_phrase(ctx.get("ending", ""), raw_norm):
            exists_plain = True

    matched_plain = not exists_plain if negative else exists_plain
    return matched_plain, 2


def apply_override_rules(
    base_matches,
    behaviors,
    states,
    locations,
    relationships,
    dream: str,
    override_rows: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    ctx = override_context(dream, base_matches, behaviors, states, locations, relationships)

    best: Optional[Dict[str, Any]] = None
    best_score = -10**9

    for row in override_rows:
        if not row_is_active(row):
            continue

        condition = row_get(row, "condition")
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

        try:
            priority = int(str(row_get(row, "priority") or "0").strip())
        except Exception:
            priority = 0

        score = (priority * 100) + group_score

        if score > best_score:
            best_score = score
            best = row

    if not best:
        return None

    try:
        pr = int(str(row_get(best, "priority") or "0").strip())
    except Exception:
        pr = 0

    return {
        "override_name": row_get(best, "override_name", "condition"),
        "spiritual": row_get(best, "final_spiritual_meaning", "override_effect", "effect", "effects"),
        "physical": row_get(best, "final_physical_effects"),
        "action": row_get(best, "final_action"),
        "priority": pr,
        "row": best,
        "condition": row_get(best, "condition"),
    }
