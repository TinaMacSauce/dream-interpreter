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


def dream_has_caught_cue(dream: str) -> bool:
    cues = [
        "caught",
        "grabbed",
        "captured",
        "trapped",
        "could not escape",
        "couldn't escape",
        "they caught me",
        "it caught me",
        "held me down",
    ]
    return contains_any_phrase(dream, cues)


def dream_has_continued_cue(dream: str) -> bool:
    cues = [
        "kept chasing",
        "never ended",
        "still running",
        "continued chasing",
        "no ending",
        "i was still running",
        "it kept going",
    ]
    return contains_any_phrase(dream, cues)


def dream_has_helped_cue(dream: str) -> bool:
    cues = [
        "someone helped me",
        "someone rescued me",
        "they helped me",
        "he helped me",
        "she helped me",
        "pulled me out",
        "saved me",
        "rescued me",
    ]
    return contains_any_phrase(dream, cues)


def dream_has_fightback_cue(dream: str) -> bool:
    cues = [
        "fought back",
        "fight back",
        "i fought",
        "i hit back",
        "i resisted",
        "i defeated",
        "i won",
        "overcame",
    ]
    return contains_any_phrase(dream, cues)


def _normalized_names(items: List[Dict[str, Any]]) -> set:
    return {normalize_text(x.get("name", "")) for x in items if x.get("name")}


def _base_symbol_names(base_matches: List[Tuple[Dict[str, Any], int, Dict[str, Any]]]) -> set:
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


def _has_death_omen_signal(base_symbol_names: set, override_text: str) -> bool:
    death_terms = {
        "teeth",
        "teeth falling out",
        "dead person",
        "dead people",
        "deceased",
        "corpse",
    }

    if death_terms & base_symbol_names:
        return True

    return any(term in override_text for term in ["death omen", "death", "grave"])


def _has_dead_person_signal(base_symbol_names: set, override_text: str) -> bool:
    if {"dead person", "dead people", "deceased", "corpse"} & base_symbol_names:
        return True

    return any(term in override_text for term in ["dead person", "deceased", "familiar spirit"])


def _has_attack_signal(behavior_names: set, override_text: str) -> bool:
    attack_terms = {
        "being attacked",
        "attacking",
        "being bitten",
        "biting",
        "being chased",
        "chasing",
        "chased",
        "fighting",
        "stabbing",
        "shooting",
        "strangling",
        "threatening",
    }

    if attack_terms & behavior_names:
        return True

    return any(term in override_text for term in ["attack", "warfare", "spiritual warfare"])


def _has_old_place_signal(location_names: set) -> bool:
    old_place_terms = {
        "old_place",
        "old place",
        "old school",
        "old house",
        "old neighborhood",
        "old job",
        "childhood home",
        "former workplace",
        "old classroom",
        "primary school",
    }
    return bool(old_place_terms & location_names)


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


def _ending_outcome(dream: str) -> Dict[str, str]:
    ending_text = extract_dream_ending_text(dream) or dream

    if dream_has_caught_cue(ending_text):
        return {
            "outcome": "caught",
            "status": "Contested",
            "type": "Attack Reaching",
            "risk": "High",
            "message": "The ending suggests the pressure reached the dreamer and should be handled urgently in prayer.",
        }

    if dream_has_escape_cue(ending_text):
        return {
            "outcome": "escaped",
            "status": "Open",
            "type": "Deliverance",
            "risk": "Low",
            "message": "The ending suggests protection, escape, or movement away from what was pursuing the dreamer.",
        }

    if dream_has_fightback_cue(ending_text):
        return {
            "outcome": "fought_back",
            "status": "Resisting",
            "type": "Overcoming",
            "risk": "Medium",
            "message": "The ending suggests resistance, spiritual authority, or movement toward overcoming the issue.",
        }

    if dream_has_helped_cue(ending_text):
        return {
            "outcome": "helped",
            "status": "Assisted",
            "type": "Intervention",
            "risk": "Medium",
            "message": "The ending suggests help, covering, or intervention around the matter.",
        }

    if dream_has_continued_cue(ending_text):
        return {
            "outcome": "continued",
            "status": "Unresolved",
            "type": "Ongoing Pressure",
            "risk": "High",
            "message": "The ending suggests the matter is ongoing, unresolved, or still applying pressure.",
        }

    return {
        "outcome": "",
        "status": "",
        "type": "",
        "risk": "",
        "message": "",
    }


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
    Seal hierarchy:

    1. hard override / death / dead person
    2. ending outcome if attack/chase is present
    3. attack / warfare
    4. escape / breakthrough
    5. old place backwardness/stagnation
    6. state corruption
    7. restrictive location
    8. relationship focus
    9. ending symbol confirmation
    10. generic fallback
    """
    ending_text = extract_dream_ending_text(dream)
    ending_norm = normalize_text(ending_text)

    behavior_names = _normalized_names(behaviors)
    state_names = _normalized_names(states)
    location_names = _normalized_names(locations)
    relationship_names = _normalized_names(relationships)
    base_symbol_names = _base_symbol_names(base_matches)
    override_text = _override_text(override_hit)

    ending_hits: List[str] = []
    for row, _score, _hit in base_matches:
        symbol = get_base_symbol_input(row)
        if symbol and contains_phrase(ending_norm, symbol):
            ending_hits.append(symbol)

    outcome = _ending_outcome(dream)

    status = "Processing"
    seal_type = "Layered"
    risk = "Medium"
    message = "The ending suggests the dream should be taken seriously in prayer."

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

    elif _has_attack_signal(behavior_names, override_text) and outcome.get("type"):
        status = outcome["status"]
        seal_type = outcome["type"]
        risk = outcome["risk"]
        message = outcome["message"]

    elif _has_attack_signal(behavior_names, override_text):
        status = "Contested"
        seal_type = "Warfare"
        risk = "High"
        message = "The action shows active contention around the message and calls for prayer."

    elif {"escaping", "crossing", "finding"} & behavior_names or dream_has_escape_cue(dream):
        status = "Open"
        seal_type = "Breakthrough"
        risk = "Low"
        message = "The ending shows movement toward escape, transition, or release."

    elif _has_old_place_signal(location_names):
        status = "Warning"
        seal_type = "Backwardness"
        risk = "Medium"
        message = "The place connects the dream to backwardness, stagnation, regression, or old cycles."

    elif {"dirty", "murky", "broken", "bleeding", "dark", "stuck", "heavy"} & state_names:
        status = "Warning"
        seal_type = "Corrupted"
        risk = "High"
        message = "The ending carries warning signs of confusion, damage, or obstruction."

    elif {"graveyard", "prison", "darkness"} & location_names:
        status = "Warning"
        seal_type = "Bound"
        risk = "High"
        message = "The ending places the message in a heavy or restrictive spiritual environment."

    elif _has_relationship_focus(relationship_names):
        status = "Focused"
        seal_type = "Relational"
        risk = "Medium"
        message = "The dream ending keeps the message centered on a specific person or relationship."

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
        "outcome": outcome.get("outcome", ""),
    }


def compute_seal_from_symbol_count(symbol_count: int) -> Dict[str, str]:
    if symbol_count <= 0:
        return {"status": "Delayed", "type": "Unclear", "risk": "High"}
    if symbol_count == 1:
        return {"status": "Live", "type": "Confirmed", "risk": "Low"}
    return {"status": "Delayed", "type": "Processing", "risk": "Medium"}
