from typing import Any, Dict, List

from app.fields import (
    get_behavior_name,
    get_location_name,
    get_relationship_name,
    get_rule_keywords,
    get_state_name,
    row_is_active,
)
from app.utils import contains_phrase, extract_dream_ending_text, normalize_text


def dedupe_rule_hits_by_best_row(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best_by_name: Dict[str, Dict[str, Any]] = {}

    for hit in hits:
        key = normalize_text(hit.get("name", ""))
        if not key:
            continue

        existing = best_by_name.get(key)
        if not existing:
            best_by_name[key] = hit
            continue

        existing_tuple = (existing.get("token_len", 0), existing.get("priority", 0))
        current_tuple = (hit.get("token_len", 0), hit.get("priority", 0))

        if current_tuple > existing_tuple:
            best_by_name[key] = hit

    out = list(best_by_name.values())
    out.sort(key=lambda x: (-x["token_len"], -x["priority"], x["name"].lower()))
    return out


def select_non_conflicting_rule_hits(
    hits: List[Dict[str, Any]],
    max_hits: int,
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    seen_tokens = set()

    for hit in hits:
        token = normalize_text(hit.get("matched_token", ""))
        if token and token in seen_tokens:
            continue

        selected.append(hit)
        if token:
            seen_tokens.add(token)

        if len(selected) >= max_hits:
            break

    return selected


def detect_rule_hits(
    dream: str,
    rows: List[Dict[str, Any]],
    kind: str,
    max_hits: int,
    priority_field: str = "priority",
) -> List[Dict[str, Any]]:
    dream_norm = normalize_text(dream)
    ending_text = extract_dream_ending_text(dream)
    ending_norm = normalize_text(ending_text)

    hits: List[Dict[str, Any]] = []

    for row in rows:
        if not row_is_active(row):
            continue

        if kind == "behavior":
            name = get_behavior_name(row)
        elif kind == "state":
            name = get_state_name(row)
        elif kind == "location":
            name = get_location_name(row)
        elif kind == "relationship":
            name = get_relationship_name(row)
        else:
            name = ""

        if not name:
            continue

        rule_keywords = get_rule_keywords(row)
        if not rule_keywords:
            fallback = normalize_text(name)
            if fallback:
                rule_keywords = [fallback]
            else:
                continue

        matched_token = ""
        token_len = 0
        matched_in_ending = False

        for kw in sorted(rule_keywords, key=len, reverse=True):
            if contains_phrase(dream_norm, kw):
                matched_token = kw
                token_len = len(kw)
                matched_in_ending = contains_phrase(ending_norm, kw)
                break

        if not matched_token:
            continue

        try:
            priority_val = int(str(row.get(priority_field, "") or "0").strip())
        except Exception:
            priority_val = 0

        if matched_in_ending:
            priority_val += 3

        hits.append(
            {
                "name": name,
                "row": row,
                "matched_token": matched_token,
                "token_len": token_len,
                "priority": priority_val,
                "kind": kind,
                "matched_in_ending": matched_in_ending,
            }
        )

    hits = dedupe_rule_hits_by_best_row(hits)
    hits = select_non_conflicting_rule_hits(hits, max_hits)
    return hits
