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


def _priority_int(row: Dict[str, Any], field: str = "priority") -> int:
    try:
        return int(str(row.get(field, "") or "0").strip())
    except Exception:
        return 0


def _kind_weight(kind: str) -> int:
    """
    Gives stable preference when two hits are otherwise close.
    This does NOT decide doctrine by itself, it only helps ordering.
    """
    weights = {
        "behavior": 40,
        "state": 30,
        "location": 20,
        "relationship": 35,
    }
    return weights.get(kind, 0)


def _name_weight(kind: str, name: str) -> int:
    """
    Strong doctrine-sensitive boosts.
    These are ranking hints only.
    """
    n = normalize_text(name)

    if kind == "behavior":
        if n in {"being attacked", "attacking", "being chased", "being bitten", "fighting"}:
            return 30
        if n in {"escaping", "crossing", "finding"}:
            return 18
        if n in {"crying", "laughing", "happy", "smiling", "sad"}:
            return 16
        return 0

    if kind == "state":
        if n in {"dirty", "murky", "broken", "bleeding", "dark"}:
            return 18
        if n in {"clear", "clean"}:
            return 14
        if n in {"heavy", "stuck", "unable to move"}:
            return 16
        return 0

    if kind == "location":
        if n in {"graveyard", "prison", "darkness"}:
            return 20
        if n in {"road", "path", "crossroad"}:
            return 14
        if n in {"house", "home"}:
            return 12
        if n in {"water", "river", "sea"}:
            return 14
        return 0

    if kind == "relationship":
        if n in {
            "mother",
            "father",
            "child",
            "son",
            "daughter",
            "husband",
            "wife",
            "spouse",
            "friend",
            "relative",
            "family member",
            "sister",
            "brother",
            "aunt",
            "uncle",
            "grandmother",
            "grandfather",
        }:
            return 22
        return 0

    return 0


def _token_quality_weight(token: str) -> int:
    """
    Longer, more exact doctrine phrases should beat tiny fragments.
    """
    token_n = normalize_text(token)
    if not token_n:
        return 0

    words = [w for w in token_n.split() if w]
    if len(words) >= 4:
        return 18
    if len(words) == 3:
        return 12
    if len(words) == 2:
        return 7
    return 0


def _score_hit(hit: Dict[str, Any]) -> int:
    """
    Unified score for de-duplication and sorting.
    """
    name = hit.get("name", "")
    kind = hit.get("kind", "")
    token = hit.get("matched_token", "")
    base_priority = int(hit.get("priority", 0))
    token_len = int(hit.get("token_len", 0))
    matched_in_ending = bool(hit.get("matched_in_ending", False))

    score = 0
    score += base_priority * 10
    score += _kind_weight(kind)
    score += _name_weight(kind, name)
    score += _token_quality_weight(token)
    score += token_len

    if matched_in_ending:
        score += 12

    return score


def dedupe_rule_hits_by_best_row(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Keep only the strongest row per normalized rule name.
    """
    best_by_name: Dict[str, Dict[str, Any]] = {}

    for hit in hits:
        key = normalize_text(hit.get("name", ""))
        if not key:
            continue

        existing = best_by_name.get(key)
        if not existing:
            best_by_name[key] = hit
            continue

        existing_score = _score_hit(existing)
        current_score = _score_hit(hit)

        if current_score > existing_score:
            best_by_name[key] = hit
            continue

        if current_score == existing_score:
            existing_tuple = (
                int(existing.get("token_len", 0)),
                int(existing.get("priority", 0)),
            )
            current_tuple = (
                int(hit.get("token_len", 0)),
                int(hit.get("priority", 0)),
            )
            if current_tuple > existing_tuple:
                best_by_name[key] = hit

    out = list(best_by_name.values())
    out.sort(
        key=lambda x: (
            -_score_hit(x),
            -int(x.get("token_len", 0)),
            -int(x.get("priority", 0)),
            x.get("name", "").lower(),
        )
    )
    return out


def _tokens_conflict(a: str, b: str) -> bool:
    """
    Prevent weaker overlapping tokens from crowding stronger ones.
    """
    a_n = normalize_text(a)
    b_n = normalize_text(b)

    if not a_n or not b_n:
        return False

    if a_n == b_n:
        return True
    if a_n in b_n:
        return True
    if b_n in a_n:
        return True

    return False


def select_non_conflicting_rule_hits(
    hits: List[Dict[str, Any]],
    max_hits: int,
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    seen_tokens: List[str] = []

    for hit in hits:
        token = normalize_text(hit.get("matched_token", ""))
        if token:
            blocked = False
            for seen in seen_tokens:
                if _tokens_conflict(token, seen):
                    blocked = True
                    break
            if blocked:
                continue

        selected.append(hit)

        if token:
            seen_tokens.append(token)

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

        for kw in sorted(rule_keywords, key=lambda x: (-len(normalize_text(x)), x)):
            kw_n = normalize_text(kw)
            if not kw_n:
                continue

            if contains_phrase(dream_norm, kw_n):
                matched_token = kw_n
                token_len = len(kw_n)
                matched_in_ending = contains_phrase(ending_norm, kw_n)
                break

        if not matched_token:
            continue

        priority_val = _priority_int(row, priority_field)

        if matched_in_ending:
            priority_val += 3

        hit = {
            "name": name,
            "row": row,
            "matched_token": matched_token,
            "token_len": token_len,
            "priority": priority_val,
            "kind": kind,
            "matched_in_ending": matched_in_ending,
        }
        hit["score"] = _score_hit(hit)

        hits.append(hit)

    hits = dedupe_rule_hits_by_best_row(hits)
    hits.sort(
        key=lambda x: (
            -int(x.get("score", 0)),
            -int(x.get("token_len", 0)),
            -int(x.get("priority", 0)),
            x.get("name", "").lower(),
        )
    )
    hits = select_non_conflicting_rule_hits(hits, max_hits)
    return hits
