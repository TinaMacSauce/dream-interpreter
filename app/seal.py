from typing import Any, Dict, List, Optional, Tuple

from app.fields import get_base_symbol_input
from app.utils import contains_any_phrase, contains_phrase, extract_dream_ending_text, normalize_text


def dream_has_escape_cue(dream: str) -> bool:
    cues = [
        "escape",
        "escaped",
        "got away",
        "came out",
        "ran out",
        "left safely",
        "crossed over",
        "was free",
        "freed",
        "rescue",
        "rescued",
        "delivered",
        "survived",
        "made it out",
    ]
    return contains_any_phrase(dream, cues)


def _normalized_names(items: List[Dict[str, Any]]) -> set:
    return {normalize_text(x.get("name", "")) for x in items if x.get("name")}


def _base_symbol_names(
    base_matches: List[Tuple[Dict[str, Any], int, Dict[str, Any]]]
) -> set:
    out = set()
    for row, _score, _hit in base_matches:
        symbol = normalize_text(get_base_symbol_input(row))
        if symbol:
            out.add(symbol)
    return out


def _override_text(override_hit: Optional[Dict[str, Any]]) -> str:
    if not override_hit:
        return ""
    return normalize_text(
        " ".join(
            [
                str(override_hit.get("override_name", "") or ""),
                str(override_hit.get("spiritual", "") or ""),
                str(override_hit.get("physical", "") or ""),
                str(override_hit.get("action", "") or ""),
            ]
        )
    )


def _has_death_omen_signal(
    base_symbol_names: set,
    override_text: str,
) -> bool:
    death_terms = {
        "teeth",
        "teeth falling out",
        "falling",
        "dead person",
        "dead people",
        "deceased",
        "corpse",
    }

    if death_terms & base_symbol_names:
        return True

    if any(term in override_text for term in ["death omen", "death", "grave"]):
        return True

    return False


def _has_dead_person_signal(base_symbol_names: set, override_text: str) -> bool:
    if {"dead person", "dead people", "deceased", "corpse"} & base_symbol_names:
        return True

    if any(term in override_text for term in ["dead person", "deceased", "familiar spirit"]):
        return True

    return False


def _has_attack_signal(behavior_names: set, override_text: str) -> bool:
    attack_terms = {
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

    if attack_terms & behavior_names:
        return True

    if any(term in override_text for term in ["attack", "warfare", "spiritual warfare"]):
        return True

    return False


def _has_relationship_focus(relationship_names: set) -> bool:
    focus_terms = {
        "mother",
        "father",
        "child",
        "son",
        "daughter",
        "spouse",
        "husband",
        "wife",
        "friend",
        "relative",
        "family",
        "family member",
        "sister",
        "brother",
        "aunt",
        "uncle",
        "grandmother",
        "grandfather",
    }
    return bool(focus_terms & relationship_names)


def compute_doctrine_seal(
    dream: str,
    base_matches: List[Tuple[Dict[str, Any], int, Dict[str, Any]]],
    behaviors: List[Dict[str, Any]],
    states: List[Dict[str, Any]],
    locations: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
    override_hit: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    LOCKED SEAL HIERARCHY

    Priority:
    1. hard override / death omen / dead person
    2. attack / warfare
    3. escape / breakthrough
    4. state corruption
    5. restrictive location
    6. relationship focus
    7. ending symbol confirmation
    8. generic symbol-count fallback
    """
    ending_text = extract_dream_ending_text(dream)
    ending_norm = normalize_text(ending_text)

    behavior_names = _normalized_names(behaviors)
    state_names = _normalized_names(states)
    location_names = _normalized_names(locations)
    relationship_names = _normalized_names(relationships)
    base_symbol_names = _base_symbol_names(base_matches)

    ending_hits: List[str] = []
    for row, _score, _hit in base_matches:
        symbol = get_base_symbol_input(row)
        if symbol and contains_phrase(ending_norm, symbol):
            ending_hits.append(symbol)

    override_text = _override_text(override_hit)

    status = "Processing"
    seal_type = "Layered"
    risk = "Medium"
    message = "The ending suggests the dream should be taken seriously in prayer."

    # 1) Hard death / dead-person level warnings
    if _has_dead_person_signal(base_symbol_names, override_text):
        status = "Sealed"
        seal_type = "Death Warning"
        risk = "High"
        message = "The dream carries a serious warning and should be handled carefully in prayer."

    elif _has_death_omen_signal(base_symbol_names, override_text):
        status = "Sealed"
        seal_type = "Death Omen"
        risk = "High"
        message = "The ending points to a sealed warning that should be taken seriously in prayer."

    # 2) Warfare before everything else relational/state/location
    elif _has_attack_signal(behavior_names, override_text):
        status = "Contested"
        seal_type = "Warfare"
        risk = "High"
        message = "The ending shows active contention around the message and calls for prayer."

    # 3) Escape / breakthrough
    elif {"escaping", "crossing", "finding"} & behavior_names or dream_has_escape_cue(dream):
        status = "Open"
        seal_type = "Breakthrough"
        risk = "Low"
        message = "The ending shows movement toward escape, transition, or release."

    # 4) State warning
    elif {"dirty", "murky", "broken", "bleeding", "dark", "stuck", "heavy"} & state_names:
        status = "Warning"
        seal_type = "Corrupted"
        risk = "High"
        message = "The ending carries warning signs of confusion, damage, or obstruction."

    # 5) Restrictive locations
    elif {"graveyard", "prison", "darkness"} & location_names:
        status = "Warning"
        seal_type = "Bound"
        risk = "High"
        message = "The ending places the message in a heavy or restrictive spiritual environment."

    # 6) Relationship focus
    elif _has_relationship_focus(relationship_names):
        status = "Focused"
        seal_type = "Relational"
        risk = "Medium"
        message = "The dream ending keeps the message centered on a specific person or relationship."

    # 7) Ending repeats a major symbol
    elif ending_hits:
        status = "Sealed"
        seal_type = "Symbol Confirmed"
        risk = "Medium"
        message = "The ending repeated or confirmed a major symbol from the dream."

    # 8) Generic fallback
    elif len(base_matches) == 1:
        status = "Live"
        seal_type = "Focused"
        risk = "Low"
        message = "The dream is centered and narrow, with one main symbolic message."

    elif len(base_matches) >= 3:
        status = "Layered"
        seal_type = "Complex"
        risk = "Medium"
        message = "The dream is layered and should be read carefully, not rushed."

    return {
        "status": status,
        "type": seal_type,
        "risk": risk,
        "message": message,
        "ending_text": ending_text,
        "ending_hits": ending_hits,
    }


def compute_seal_from_symbol_count(symbol_count: int) -> Dict[str, str]:
    if symbol_count <= 0:
        return {"status": "Delayed", "type": "Unclear", "risk": "High"}
    if symbol_count == 1:
        return {"status": "Live", "type": "Confirmed", "risk": "Low"}
    return {"status": "Delayed", "type": "Processing", "risk": "Medium"}
