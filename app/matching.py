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


def _is_action_like_base_symbol(row: Dict[str, Any]) -> bool:
    """
    Base symbols like 'chasing', 'running', 'crying', etc. should not usually
    outrank concrete symbols like snake, dog, mother, house, teeth, river.
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
    }

    if category in {"movement", "action", "behavior"}:
        return True

    if symbol in action_like_words:
        return True

    return False


def score_row_strict(
    dream_norm: str,
    row: Dict[str, Any],
    used_spans: Optional[List[Tuple[int, int]]] = None,
) -> Tuple[int, Optional[Dict[str, Any]]]:
    symbol_raw = get_symbol_cell(row)
    if not symbol_raw:
        return 0, None

    symbol = normalize_text(symbol_raw)
    if not symbol:
        return 0, None

    from app.utils import normalize_text as norm
    import re

    raw_keywords = get_keywords_cell(row)
    keywords = []
    if raw_keywords:
        for part in re.split(r"[,|;]+", raw_keywords):
            part = norm(part)
            if part:
                keywords.append(part)

    candidates: List[Tuple[str, str, int]] = []

    if symbol:
        if len(tokenize_words(symbol)) > 1:
            candidates.append(("symbol_phrase", symbol, 100 - symbol_length_penalty(symbol_raw)))
        else:
            candidates.append(("symbol", symbol, 100))

    for kw in keywords:
        if len(tokenize_words(kw)) > 1:
            candidates.append(("keyword_phrase", kw, 94))
        else:
            candidates.append(("keyword", kw, 90))

    candidates.sort(key=lambda x: (-len(x[1]), -x[2]))

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
            return score, {
                "type": match_type,
                "token": token,
                "span": chosen_span,
                "token_len": len(token),
            }

    return 0, None


def match_symbols_legacy(
    dream: str,
    rows: List[Dict[str, Any]],
    top_k: int = 3,
) -> List[Tuple[Dict[str, Any], int, Optional[Dict[str, Any]]]]:
    dream_norm = normalize_text(dream)
    if not dream_norm:
        return []

    candidates: List[Tuple[Dict[str, Any], int, Optional[Dict[str, Any]]]] = []
    for row in rows:
        if not row_is_active(row):
            continue
        score, hit = score_row_strict(dream_norm, row, used_spans=[])
        if score > 0 and hit:
            candidates.append((row, score, hit))

    def sort_key(item):
        row, score, hit = item
        sym = get_symbol_cell(row)
        token_len = int((hit or {}).get("token_len", 0))
        sym_len = len(normalize_text(sym))
        category = normalize_text(get_base_symbol_category(row))
        action_like_penalty = 1 if _is_action_like_base_symbol(row) else 0
        return (
            action_like_penalty,
            -category_priority(category),
            -score,
            -token_len,
            -sym_len,
        )

    candidates.sort(key=sort_key)

    used_spans: List[Tuple[int, int]] = []
    seen_symbols = set()
    out: List[Tuple[Dict[str, Any], int, Optional[Dict[str, Any]]]] = []

    for row, score, hit in candidates:
        if not hit:
            continue

        span = hit.get("span")
        if not span:
            continue

        sym = get_symbol_cell(row)
        sym_key = normalize_text(sym)

        if not sym_key or sym_key in seen_symbols:
            continue
        if is_span_blocked(span, used_spans):
            continue

        used_spans.append(span)
        seen_symbols.add(sym_key)
        out.append((row, score, hit))

        if len(out) >= top_k:
            break

    return out


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

    # Action-like base symbols should normally support interpretation,
    # not dominate it.
    if _is_action_like_base_symbol(row):
        score -= 18

    return score


def category_conflict_penalty(
    row: Dict[str, Any],
    already_selected: List[Tuple[Dict[str, Any], int, Dict[str, Any]]],
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


def _candidate_sort_key(item):
    row, score, hit = item
    category = normalize_text(get_base_symbol_category(row))
    symbol = normalize_text(get_base_symbol_input(row))
    token_len = int((hit or {}).get("token_len", 0))
    category_rank = category_priority(category)
    action_like_penalty = 1 if _is_action_like_base_symbol(row) else 0

    return (
        action_like_penalty,                 # concrete symbols before action-like symbols
        -category_rank,                      # stronger symbolic categories first
        -score,                              # then raw score
        -len(symbol.split()),                # slightly prefer fuller concrete symbols
        -token_len,
        -len(symbol),
    )


def _selected_output_sort_key(item):
    row, score, hit = item
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


def match_base_symbols_doctrine(
    dream: str,
    base_rows: List[Dict[str, Any]],
    top_k: int = 3,
) -> List[Tuple[Dict[str, Any], int, Dict[str, Any]]]:
    dream_norm = normalize_text(dream)
    ending_text = extract_dream_ending_text(dream)

    if not dream_norm:
        return []

    candidates: List[Tuple[Dict[str, Any], int, Dict[str, Any]]] = []

    for row in base_rows:
        if not row_is_active(row):
            continue

        symbol_raw = get_base_symbol_input(row)
        if not symbol_raw:
            continue

        symbol = normalize_text(symbol_raw)
        if not symbol:
            continue

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

        local_candidates.sort(key=lambda x: -len(x[1]))

        for match_type, token in local_candidates:
            spans = find_phrase_spans(dream_norm, token)
            if not spans:
                continue

            ending_bonus = ending_bonus_for_symbol(row, ending_text)
            score = score_base_candidate(row, match_type, symbol_raw) + ending_bonus

            candidates.append(
                (
                    row,
                    score,
                    {
                        "type": match_type,
                        "token": token,
                        "span": spans[0],
                        "token_len": len(token),
                        "ending_bonus": ending_bonus,
                    },
                )
            )
            break

    candidates.sort(key=_candidate_sort_key)

    selected: List[Tuple[Dict[str, Any], int, Dict[str, Any]]] = []
    used_spans: List[Tuple[int, int]] = []
    seen_symbols = set()

    for row, score, hit in candidates:
        span = hit.get("span")
        if not span:
            continue

        sym_key = normalize_text(get_base_symbol_input(row))
        if not sym_key or sym_key in seen_symbols:
            continue
        if is_span_blocked(span, used_spans):
            continue

        score_after_conflict = score - category_conflict_penalty(row, selected)
        selected.append((row, score_after_conflict, hit))
        used_spans.append(span)
        seen_symbols.add(sym_key)

        if len(selected) >= top_k:
            break

    # Final output order should look natural to users:
    # core concrete symbol first, action-like support symbol later.
    selected.sort(key=_selected_output_sort_key)
    return selected
