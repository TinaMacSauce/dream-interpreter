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


def _get_ending_name(row: Dict[str, Any]) -> str:
    return str(row.get("ending_name", "") or row.get("name", "") or "").strip()


def _kind_weight(kind: str) -> int:
    """
    Enforces doctrine hierarchy:
    ACTION first, SUBJECT second, PLACE third, ENDING finalizes outcome.
    """
    weights = {
        "behavior": 80,
        "ending": 70,
        "relationship": 45,
        "state": 35,
        "location": 30,
    }
    return weights.get(kind, 0)


def _name_weight(kind: str, name: str) -> int:
    n = normalize_text(name)

    if kind == "behavior":
        if n in {"being chased", "chased", "chasing"}:
            return 60
        if n in {"being attacked", "attacking", "being bitten", "fighting", "stabbing"}:
            return 50
        if n in {"escaping", "running", "hiding", "falling", "driving", "eating", "drinking"}:
            return 40
        if n in {"crossing", "finding", "opening", "closing", "entering", "leaving"}:
            return 25
        return 10

    if kind == "ending":
        if n in {"caught", "trapped", "captured", "grabbed"}:
            return 55
        if n in {"escaped", "got away", "survived", "made it out"}:
            return 55
        if n in {"fought_back", "fought back", "won", "defeated it"}:
            return 50
        if n in {"continued", "never ended", "no ending", "still running"}:
            return 45
        if n in {"helped", "rescued"}:
            return 45
        return 25

    if kind == "location":
        if n in {"old_place", "old place", "old school", "old house", "old neighborhood", "old job"}:
            return 45
        if n in {"graveyard", "prison", "darkness"}:
            return 30
        if n in {"road", "path", "crossroad"}:
            return 22
        if n in {"house", "home"}:
            return 18
        if n in {"water", "river", "sea", "ocean"}:
            return 18
        return 0

    if kind == "state":
        if n in {"dirty", "murky", "broken", "bleeding", "dark"}:
            return 25
        if n in {"heavy", "stuck", "unable to move"}:
            return 22
        if n in {"clear", "clean"}:
            return 18
        return 0

    if kind == "relationship":
        if n in {
            "mother", "father", "child", "son", "daughter", "husband", "wife",
            "spouse", "friend", "relative", "family member", "sister", "brother",
            "aunt", "uncle", "grandmother", "grandfather",
        }:
            return 28
        return 0

    return 0


def _token_quality_weight(token: str) -> int:
    token_n = normalize_text(token)
    if not token_n:
        return 0

    words = [w for w in token_n.split() if w]
    if len(words) >= 5:
        return 30
    if len(words) == 4:
        return 24
    if len(words) == 3:
        return 18
    if len(words) == 2:
        return 10
    return 0


def _score_hit(hit: Dict[str, Any]) -> int:
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
        score += 20

    return score


def dedupe_rule_hits_by_best_row(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best_by_name: Dict[str, Dict[str, Any]] = {}

    for hit in hits:
        key = normalize_text(hit.get("name", ""))
        if not key:
            continue

        existing = best_by_name.get(key)
        if not existing or _score_hit(hit) > _score_hit(existing):
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
    a_n = normalize_text(a)
    b_n = normalize_text(b)

    if not a_n or not b_n:
        return False

    if a_n == b_n:
        return True

    a_words = a_n.split()
    b_words = b_n.split()

    if len(a_words) == 1 and a_n in b_words:
        return True
    if len(b_words) == 1 and b_n in a_words:
        return True

    if len(a_words) > 1 and len(b_words) > 1:
        if a_n in b_n or b_n in a_n:
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
            blocked = any(_tokens_conflict(token, seen) for seen in seen_tokens)
            if blocked:
                continue

        selected.append(hit)

        if token:
            seen_tokens.append(token)

        if len(selected) >= max_hits:
            break

    return selected


def _get_rule_name(row: Dict[str, Any], kind: str) -> str:
    if kind == "behavior":
        return get_behavior_name(row)
    if kind == "state":
        return get_state_name(row)
    if kind == "location":
        return get_location_name(row)
    if kind == "relationship":
        return get_relationship_name(row)
    if kind == "ending":
        return _get_ending_name(row)
    return ""


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

        name = _get_rule_name(row, kind)
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

        if kind == "behavior":
            priority_val += 8

        if kind == "ending":
            priority_val += 6

        if matched_in_ending:
            priority_val += 5

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

    return select_non_conflicting_rule_hits(hits, max_hits)
