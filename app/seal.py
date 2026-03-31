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


def compute_doctrine_seal(
    dream: str,
    base_matches: List[Tuple[Dict[str, Any], int, Dict[str, Any]]],
    behaviors: List[Dict[str, Any]],
    states: List[Dict[str, Any]],
    locations: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
    override_hit: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    ending_text = extract_dream_ending_text(dream)
    ending_norm = normalize_text(ending_text)

    behavior_names = {normalize_text(x["name"]) for x in behaviors}
    state_names = {normalize_text(x["name"]) for x in states}
    location_names = {normalize_text(x["name"]) for x in locations}
    relationship_names = {normalize_text(x["name"]) for x in relationships}

    ending_hits = []
    for row, _score, _hit in base_matches:
        symbol = get_base_symbol_input(row)
        if symbol and contains_phrase(ending_norm, symbol):
            ending_hits.append(symbol)

    status = "Processing"
    seal_type = "Layered"
    risk = "Medium"
    message = "The ending suggests the dream should be taken seriously in prayer."

    override_name = normalize_text((override_hit or {}).get("override_name", ""))
    override_spiritual = normalize_text((override_hit or {}).get("spiritual", ""))

    if override_hit and any(x in f"{override_name} {override_spiritual}" for x in ["death", "grave", "death omen"]):
        status = "Sealed"
        seal_type = "Death Omen"
        risk = "High"
        message = "The ending and override logic point to a sealed warning."
    elif {"escaping", "crossing", "finding"} & behavior_names:
        status = "Open"
        seal_type = "Breakthrough"
        risk = "Low"
        message = "The ending shows movement toward escape, transition, or release."
    elif {"being attacked", "being bitten", "being chased", "fighting"} & behavior_names:
        status = "Contested"
        seal_type = "Warfare"
        risk = "High"
        message = "The ending shows active contention around the message."
    elif {"dirty", "murky", "broken", "bleeding", "dark"} & state_names:
        status = "Warning"
        seal_type = "Corrupted"
        risk = "High"
        message = "The ending carries warning signs of contamination, confusion, or damage."
    elif {"graveyard", "prison", "darkness"} & location_names:
        status = "Warning"
        seal_type = "Bound"
        risk = "High"
        message = "The ending places the message in a heavy or restrictive spiritual environment."
    elif {"mother", "father", "child", "spouse", "friend"} & relationship_names:
        status = "Focused"
        seal_type = "Relational"
        risk = "Medium"
        message = "The dream ending points toward a relationship-centered message."
    elif ending_hits:
        status = "Sealed"
        seal_type = "Symbol Confirmed"
        risk = "Medium"
        message = "The ending repeated or confirmed a major symbol from the dream."
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
