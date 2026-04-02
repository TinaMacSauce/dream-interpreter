from typing import Any, Dict, List, Optional, Tuple

from app.fields import (
    get_base_symbol_category,
    get_base_symbol_input,
    get_keywords_cell,
    get_rule_keywords,
    get_symbol_cell,
    row_is_active,
)
from app.utils import (
    contains_phrase,
    extract_dream_ending_text,
    find_phrase_spans,
    is_span_blocked,
    normalize_text,
    tokenize_words,
)


MatchHit = Dict[str, Any]
MatchTuple = Tuple[Dict[str, Any], int, MatchHit]


def symbol_length_penalty(symbol: str) -> int:
    words = [w for w in normalize_text(symbol).split() if w]
    n = len(words)
    if n <= 2:
        return 0
    if n == 3:
        return 2
    if n == 4:
        return 6
    if n == 5:
        return 12
    return 18


def _normalize_keyword_list(raw_keywords: str) -> List[str]:
    from app.utils import normalize_text as norm
    import re

    keywords: List[str] = []
    seen = set()

    if not raw_keywords:
        return keywords

    for part in re.split(r"[,|;]+", raw_keywords):
        part = norm(part)
        if not part or part in seen:
            continue
        seen.add(part)
        keywords.append(part)

    return keywords


def _row_identity_key(row: Dict[str, Any]) -> str:
    symbol = normalize_text(get_base_symbol_input(row))
    if symbol:
        return symbol
    return normalize_text(get_symbol_cell(row))


def _is_action_like_base_symbol(row: Dict[str, Any]) -> bool:
    """
    Action-like rows should usually support interpretation,
    not dominate the main symbol list.
    """
    symbol = normalize_text(get_base_symbol_input(row))
    category = normalize_text(get_base_symbol_category(row))

    action_like_words = {
        "chasing",
        "running",
        "crying",
        "fighting",
        "biting",
        "watching",
        "following",
        "walking",
        "standing",
        "speaking",
        "looking",
        "looking at",
        "escaping",
        "crossing",
        "falling",
        "flying",
        "hiding",
        "attacking",
        "smiling",
        "laughing",
        "eating",
    }

    if category in {"movement", "action", "behavior"}:
        return True

    if symbol in action_like_words:
        return True

    return False


def _relationship_bonus(row: Dict[str, Any]) -> int:
    category = normalize_text(get_base_symbol_category(row))
    symbol = normalize_text(get_base_symbol_input(row))

    family_people = {
        "mother",
        "father",
        "sister",
        "brother",
        "child",
        "son",
        "daughter",
        "husband",
        "wife",
        "spouse",
        "grandmother",
        "grandfather",
        "friend",
        "relative",
        "cousin",
        "aunt",
        "uncle",
        "family",
    }

    if category == "person":
        return 8
    if symbol in family_people:
        return 10
    return 0


def _death_omen_bonus(row: Dict[str, Any]) -> int:
    symbol = normalize_text(get_base_symbol_input(row))
    category = normalize_text(get_base_symbol_category(row))

    if symbol in {"teeth", "teeth falling out", "falling", "dead person", "dead people"}:
        return 14

    if category in {"death", "graveyard", "ending"}:
        return 10

    return 0


def _threat_symbol_bonus(row: Dict[str, Any]) -> int:
    """
    Concrete hostile symbols should stay ahead of reaction-type symbols.
    """
    symbol = normalize_text(get_base_symbol_input(row))
    category = normalize_text(get_base_symbol_category(row))

    threat_symbols = {
        "snake",
        "serpent",
        "dog",
        "cat",
        "duppy",
        "demon",
        "enemy",
        "thief",
        "attacker",
        "dead person",
        "dead people",
    }

    if symbol in threat_symbols:
        return 12

    if category in {"animal", "person", "death"}:
        return 4

    return 0


def _prefer_literal_symbol_over_action(row: Dict[str, Any], score: int) -> int:
    """
    Concrete symbols should usually lead.
    Action-like rows can still match, but should rarely dominate.
    """
    if _is_action_like_base_symbol(row):
        score -= 28
    return score


def category_priority(category: str) -> int:
    """
    Higher means more likely to be a core symbol users expect to see first.
    """
    c = normalize_text(category or "") or "unknown"
    priorities = {
        "ending": 42,
        "death": 38,
        "graveyard": 36,
        "body": 34,
        "person": 32,
        "animal": 30,
        "water": 24,
        "nature": 22,
        "object": 20,
        "location": 18,
        "emotion": 14,
        "movement": 8,
        "action": 8,
        "behavior": 8,
    }
    return priorities.get(c, 12)


def score_base_candidate(row: Dict[str, Any], match_type: str, symbol_raw: str) -> int:
    if match_type == "symbol_phrase":
        base = 120
    elif match_type == "symbol":
        base = 112
    elif match_type == "keyword_phrase":
        base = 104
    else:
        base = 96

    category_bonus = category_priority(get_base_symbol_category(row))
    length_penalty = symbol_length_penalty(symbol_raw)

    score = base + category_bonus - length_penalty
    score = _prefer_literal_symbol_over_action(row, score)
    score += _relationship_bonus(row)
    score += _death_omen_bonus(row)
    score += _threat_symbol_bonus(row)

    return score


def category_conflict_penalty(
    row: Dict[str, Any],
    already_selected: List[MatchTuple],
) -> int:
    current_cat = normalize_text(get_base_symbol_category(row))
    if current_cat == "unknown":
        return 0

    penalty = 0
    for existing_row, _score, _hit in already_selected:
        ex_cat = normalize_text(get_base_symbol_category(existing_row))
        if ex_cat == current_cat:
            penalty += 6

    return penalty


def ending_bonus_for_symbol(row: Dict[str, Any], ending_text: str) -> int:
    symbol = get_base_symbol_input(row)
    if not symbol:
        return 0

    ending_norm = normalize_text(ending_text)
    symbol_n = normalize_text(symbol)

    if symbol_n and contains_phrase(ending_norm, symbol_n):
        return 18

    for kw in get_rule_keywords(row):
        if contains_phrase(ending_norm, kw):
            return 10

    return 0


def _candidate_sort_key(item: MatchTuple):
    row, score, hit = item
    category = normalize_text(get_base_symbol_category(row))
    symbol = normalize_text(get_base_symbol_input(row))
    token_len = int((hit or {}).get("token_len", 0))
    token_chars = int((hit or {}).get("token_chars", 0))
    category_rank = category_priority(category)
    action_like_penalty = 1 if _is_action_like_base_symbol(row) else 0

    return (
        action_like_penalty,   # concrete before action-like
        -category_rank,        # stronger symbolic categories first
        -score,                # then raw score
        -token_len,            # then word count in matched token
        -token_chars,          # then character length
        -len(symbol),
    )


def _selected_output_sort_key(item: MatchTuple):
    row, score, _hit = item
    category = normalize_text(get_base_symbol_category(row))
    symbol = normalize_text(get_base_symbol_input(row))
    action_like_penalty = 1 if _is_action_like_base_symbol(row) else 0

    return (
        action_like_penalty,
        -category_priority(category),
        -score,
        -len(symbol.split()),
        -len(symbol),
    )


def _select_non_overlapping_candidates(
    candidates: List[MatchTuple],
    top_k: int,
) -> List[MatchTuple]:
    selected: List[MatchTuple] = []
    used_spans: List[Tuple[int, int]] = []
    seen_symbols = set()

    for row, score, hit in candidates:
        span = hit.get("span")
        if not span:
            continue

        row_key = _row_identity_key(row)
        if not row_key or row_key in seen_symbols:
            continue
        if is_span_blocked(span, used_spans):
            continue

        score_after_conflict = score - category_conflict_penalty(row, selected)
        selected.append((row, score_after_conflict, hit))
        used_spans.append(span)
        seen_symbols.add(row_key)

        if len(selected) >= top_k:
            break

    return selected


def _drop_action_symbols_when_concrete_exists(selected: List[MatchTuple]) -> List[MatchTuple]:
    """
    Locked doctrine behavior:
    if at least one concrete symbol exists, action-like base symbols
    should not occupy the limited top symbol slots.
    """
    if not selected:
        return selected

    concrete = [item for item in selected if not _is_action_like_base_symbol(item[0])]
    action_like = [item for item in selected if _is_action_like_base_symbol(item[0])]

    if not concrete:
        return selected

    # Keep concrete symbols only. Action layers should be expressed by behavior rules,
    # not by dominating top base symbols.
    return concrete if concrete else action_like


def _build_candidate_hit(
    dream_norm: str,
    row: Dict[str, Any],
    symbol_raw: str,
    ending_text: str,
) -> Optional[MatchTuple]:
    symbol = normalize_text(symbol_raw)
    if not symbol:
        return None

    keywords = get_rule_keywords(row)
    local_candidates: List[Tuple[str, str]] = []

    if len(tokenize_words(symbol)) > 1:
        local_candidates.append(("symbol_phrase", symbol))
    else:
        local_candidates.append(("symbol", symbol))

    for kw in keywords:
        if len(tokenize_words(kw)) > 1:
            local_candidates.append(("keyword_phrase", kw))
        else:
            local_candidates.append(("keyword", kw))

    # Longest / fullest tokens first so "black dog" beats "dog"
    local_candidates.sort(key=lambda x: (-len(x[1].split()), -len(x[1])))

    for match_type, token in local_candidates:
        spans = find_phrase_spans(dream_norm, token)
        if not spans:
            continue

        ending_bonus = ending_bonus_for_symbol(row, ending_text)
        score = score_base_candidate(row, match_type, symbol_raw) + ending_bonus

        return (
            row,
            score,
            {
                "type": match_type,
                "token": token,
                "span": spans[0],
                "token_len": len(token.split()),
                "token_chars": len(token),
                "ending_bonus": ending_bonus,
            },
        )

    return None


def score_row_strict(
    dream_norm: str,
    row: Dict[str, Any],
    used_spans: Optional[List[Tuple[int, int]]] = None,
) -> Tuple[int, Optional[MatchHit]]:
    """
    Legacy-compatible scorer.
    """
    symbol_raw = get_symbol_cell(row)
    if not symbol_raw:
        return 0, None

    symbol = normalize_text(symbol_raw)
    if not symbol:
        return 0, None

    raw_keywords = get_keywords_cell(row)
    keywords = _normalize_keyword_list(raw_keywords)

    candidates: List[Tuple[str, str, int]] = []

    if len(tokenize_words(symbol)) > 1:
        candidates.append(("symbol_phrase", symbol, 100 - symbol_length_penalty(symbol_raw)))
    else:
        candidates.append(("symbol", symbol, 100))

    for kw in keywords:
        if len(tokenize_words(kw)) > 1:
            candidates.append(("keyword_phrase", kw, 94))
        else:
            candidates.append(("keyword", kw, 90))

    candidates.sort(key=lambda x: (-len(x[1].split()), -len(x[1]), -x[2]))

    for match_type, token, score in candidates:
        spans = find_phrase_spans(dream_norm, token)
        if not spans:
            continue

        chosen_span = None
        for sp in spans:
            if not used_spans or not is_span_blocked(sp, used_spans):
                chosen_span = sp
                break

        if chosen_span:
            adjusted_score = score
            if _is_action_like_base_symbol(row):
                adjusted_score -= 28

            return adjusted_score, {
                "type": match_type,
                "token": token,
                "span": chosen_span,
                "token_len": len(token.split()),
                "token_chars": len(token),
            }

    return 0, None


def match_symbols_legacy(
    dream: str,
    rows: List[Dict[str, Any]],
    top_k: int = 3,
) -> List[Tuple[Dict[str, Any], int, Optional[MatchHit]]]:
    dream_norm = normalize_text(dream)
    if not dream_norm:
        return []

    candidates: List[Tuple[Dict[str, Any], int, Optional[MatchHit]]] = []

    for row in rows:
        if not row_is_active(row):
            continue

        score, hit = score_row_strict(dream_norm, row, used_spans=[])
        if score > 0 and hit:
            candidates.append((row, score, hit))

    candidates.sort(key=lambda item: _candidate_sort_key(item))  # type: ignore[arg-type]

    used_spans: List[Tuple[int, int]] = []
    seen_symbols = set()
    out: List[Tuple[Dict[str, Any], int, Optional[MatchHit]]] = []

    for row, score, hit in candidates:
        if not hit:
            continue

        span = hit.get("span")
        if not span:
            continue

        row_key = _row_identity_key(row)
        if not row_key or row_key in seen_symbols:
            continue
        if is_span_blocked(span, used_spans):
            continue

        used_spans.append(span)
        seen_symbols.add(row_key)
        out.append((row, score, hit))

        if len(out) >= top_k:
            break

    # Lock action-like rows out if concrete rows exist.
    concrete = [item for item in out if not _is_action_like_base_symbol(item[0])]
    if concrete:
        out = concrete[:top_k]

    out.sort(key=lambda item: _selected_output_sort_key(item))  # type: ignore[arg-type]
    return out


def match_base_symbols_doctrine(
    dream: str,
    base_rows: List[Dict[str, Any]],
    top_k: int = 3,
) -> List[MatchTuple]:
    dream_norm = normalize_text(dream)
    ending_text = extract_dream_ending_text(dream)

    if not dream_norm:
        return []

    candidates: List[MatchTuple] = []

    for row in base_rows:
        if not row_is_active(row):
            continue

        symbol_raw = get_base_symbol_input(row)
        if not symbol_raw:
            continue

        candidate = _build_candidate_hit(
            dream_norm=dream_norm,
            row=row,
            symbol_raw=symbol_raw,
            ending_text=ending_text,
        )
        if candidate:
            candidates.append(candidate)

    candidates.sort(key=_candidate_sort_key)

    selected = _select_non_overlapping_candidates(candidates, top_k=top_k)
    selected = _drop_action_symbols_when_concrete_exists(selected)
    selected = selected[:top_k]
    selected.sort(key=_selected_output_sort_key)

    return selected
