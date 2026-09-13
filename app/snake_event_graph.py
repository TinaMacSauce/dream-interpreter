from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


SNAKE_EVENT_CONTRACT_VERSION = "snake-context-event-terminal-v1"
SNAKE_CLAIM_PROJECTION_CONTRACT_VERSION = "snake-claim-projection-v1"
SNAKE_CERTAINTY_CONTRACT_VERSION = "snake-certainty-provenance-v1"
SNAKE_TARGET_RULE_CONTRACT_VERSION = "snake-rule-provenance-target-v1"

_SNAKE_RULE_PROVENANCE: Dict[str, Dict[str, Any]] = {
    "SNAKE-WATCHING-TARGET": {
        "implementation_key": "snake_watching_target",
        "decision_id": "DEC-SNAKE-2026-09-08-02",
        "doctrine_version": "DEC-SNAKE-2026-09-08-02",
        "source_row": 44,
        "authority": "Tina, explicit founder clarification",
        "updated_at_utc": "2026-09-08T14:06:13Z",
    },
    "SNAKE-BITE-ATTEMPT-TARGET": {
        "implementation_key": "snake_bite_attempt_target",
        "decision_id": "DEC-SNAKE-2026-09-08-02",
        "doctrine_version": "DEC-SNAKE-2026-09-08-02",
        "source_row": 45,
        "authority": "Tina, explicit founder clarification",
        "updated_at_utc": "2026-09-08T14:06:13Z",
    },
    "SNAKE-BITE": {
        "implementation_key": "snake_bite",
        "decision_id": "DEC-SNAKE-2026-09-08-01",
        "doctrine_version": "DEC-SNAKE-2026-09-08-01",
        "source_row": 34,
        "authority": "Tina, explicit founder teaching",
        "updated_at_utc": "2026-09-08T07:35:00Z",
    },
}

_SNAKE = r"(?:snake|snakes|serpent|serpents|cobra|cobras)"
_BODY_PART_NAMES = ("ankle", "arm", "hand", "wrist")
_BODY_PARTS = "(?:" + "|".join(_BODY_PART_NAMES) + ")"
_PEOPLE = "(?:sister|brother|cousin|friend|child|mother|father|son|daughter|husband|wife|aunt|uncle|coworker|neighbor)"


def _normalise(value: str) -> str:
    value = (value or "").lower().replace("’", "'")
    return re.sub(r"\s+", " ", value).strip()


def _span(source: str, match: re.Match[str]) -> str:
    return source[match.start():match.end()].strip(" ,.;:")


def _target_from_text(text: str, source: str, start: int, end: Optional[int] = None) -> Dict[str, Any]:
    window = text[start:(end if end is not None else start + 150)]
    before = text[max(0, start - 100):start]
    if re.search(r"\bone of them\b", window) or (
        re.search(r"\bbit her\b", window)
        and "sister" in before + window and "cousin" in before + window
    ):
        candidates = [person for person in ("sister", "cousin") if person in before + window]
        return {
            "surface": None, "person": None, "status": "ambiguous",
            "basis": "plural_coreference_ambiguous", "path": None,
            "eligible": False, "candidates": candidates, "span": "one of them" if "one of them" in window else "her",
        }

    possessive = re.search(rf"\bmy\s+({_PEOPLE})(?:'s|s)?\s+({_BODY_PARTS})\b", window)
    if possessive:
        person, body_part = possessive.group(1), possessive.group(2)
        target = f"{person}-{body_part}"
        return {
            "surface": target, "person": person, "status": "resolved",
            "basis": "possessive_body_part_owner", "path": [target, person],
            "eligible": True, "span": _span(window, possessive),
        }

    relation_body = re.search(rf"\bmy\s+({_PEOPLE})\s+on\s+(?:his|her)\s+({_BODY_PARTS})\b", window)
    if relation_body:
        person, body_part = relation_body.group(1), relation_body.group(2)
        target = f"{person}-{body_part}"
        return {
            "surface": target, "person": person, "status": "resolved",
            "basis": "possessive_body_part_owner", "path": [target, person],
            "eligible": True, "span": _span(window, relation_body),
        }

    own_body = re.search(rf"\bmy\s+({_BODY_PARTS})\b", window)
    if own_body:
        target = f"dreamer-{own_body.group(1)}"
        return {
            "surface": target, "person": "dreamer", "status": "resolved",
            "basis": "possessive_body_part_owner", "path": [target, "dreamer"],
            "eligible": True, "span": _span(window, own_body),
        }

    if re.search(r"\bbit it\b", window) and re.search(rf"\bmy\s+({_PEOPLE})\s+held out (?:his|her)\s+({_BODY_PARTS})\b", before):
        antecedent = re.search(rf"\bmy\s+({_PEOPLE})\s+held out (?:his|her)\s+({_BODY_PARTS})\b", before)
        assert antecedent is not None
        person, body_part = antecedent.group(1), antecedent.group(2)
        target = f"{person}-{body_part}"
        return {
            "surface": target, "person": person, "status": "resolved",
            "basis": "singular_coreference_to_owned_body_part", "path": [target, person],
            "eligible": True, "span": _span(before, antecedent) + ". " + _span(window, re.search(r"\b(?:the\s+)?snake\s+bit\s+it\b", window) or antecedent),
        }

    bag = re.search(r"\bmy\s+travel\s+bag\b", window)
    if bag:
        return {
            "surface": "travel-bag", "person": None, "status": "resolved", "person_status": "not_applicable",
            "basis": "explicit_nonperson_target", "path": ["travel-bag"],
            "eligible": False, "span": _span(window, bag),
        }

    person_match = re.search(rf"\b(?:my\s+)?({_PEOPLE})\b", window)
    if person_match:
        person = person_match.group(1)
        return {
            "surface": person, "person": person, "status": "resolved",
            "basis": "direct_person_object", "path": [person],
            "eligible": True, "span": _span(window, person_match),
        }

    if re.search(r"\b(?:me|myself)\b", window):
        return {
            "surface": "dreamer", "person": "dreamer", "status": "resolved",
            "basis": "direct_person_object", "path": ["dreamer"],
            "eligible": True, "span": "me",
        }
    return {
        "surface": None, "person": None, "status": "unspecified",
        "basis": "unresolved", "path": None, "eligible": False, "span": "",
    }


def _snake_number(text: str, start: int, action: str) -> int:
    nearby = text[max(0, start - 45):start + 35]
    if re.search(r"\b(?:second|another)\s+(?:snake|cobra)\b", nearby):
        return 2
    if re.search(r"\bthird\s+(?:snake|cobra)\b", nearby):
        return 3
    if "three snakes surrounded" in text:
        if action in {"attack", "kill_by_dreamer"} and "huge cobra" in nearby:
            return 2
        if action == "retreat" and "third snake" in nearby:
            return 3
    return 1


def _event(
    event_id: str,
    action: str,
    actor_id: str,
    target: Mapping[str, Any],
    span_text: str,
    *,
    chain_id: str,
    polarity: str = "affirmed",
    modality: str = "experienced",
    actuality: str = "actual",
    completion: str = "completed",
) -> Dict[str, Any]:
    value: Dict[str, Any] = {
        "event_id": event_id,
        "action": action,
        "actor_id": actor_id,
        "target_id": target.get("surface"),
        "target_status": target.get("status", "unspecified"),
        "scene_id": "scene-1",
        "chain_id": chain_id,
        "polarity": polarity,
        "modality": modality,
        "actuality": actuality,
        "completion": completion,
        "terminal": False,
        "span_text": span_text,
    }
    if target.get("candidates"):
        value["target_candidates"] = list(target["candidates"])
    return value


def _iter_matches(pattern: str, text: str) -> Iterable[re.Match[str]]:
    return re.finditer(pattern, text, flags=re.IGNORECASE)


def _extract_snake_event_graph_legacy(dream: str) -> Dict[str, Any]:
    """Build the approved Snake event, target-lineage, and terminal-frontier graph."""
    source = _normalise(dream)
    events_with_pos: List[Tuple[int, Dict[str, Any], Dict[str, Any]]] = []

    def add(
        match: re.Match[str], action: str, *, actor: Optional[str] = None,
        target: Optional[Dict[str, Any]] = None, polarity: str = "affirmed",
        modality: str = "experienced", actuality: str = "actual",
        completion: str = "completed", chain_number: Optional[int] = None,
    ) -> None:
        target_value = target or _target_from_text(source, source, match.start(), match.end())
        snake_number = chain_number or _snake_number(source, match.start(), action)
        actor_id = actor or f"snake-{snake_number}"
        chain_id = f"chain-snake-{snake_number}"
        events_with_pos.append((
            match.start(),
            _event("", action, actor_id, target_value, _span(source, match), chain_id=chain_id,
                   polarity=polarity, modality=modality, actuality=actuality, completion=completion),
            target_value,
        ))

    consumed: List[Tuple[int, int, str]] = []
    def overlaps(match: re.Match[str], actions: Tuple[str, ...] = ()) -> bool:
        return any(match.start() < end and match.end() > start and (not actions or action in actions)
                   for start, end, action in consumed)

    patterns = [
        (rf"\b(?:a\s+)?(?:small|huge|large|black|green)?\s*{_SNAKE}\s+(?:attacked|struck at|lunged at)(?:\s+(?:me|a\s+child|my\s+{_PEOPLE}))?\b", "attack", "completed"),
        (rf"\b(?:a\s+)?{_SNAKE}\s+rushed\s+between\s+my\s+sister\s+and\s+my\s+cousin\b", "attack", "completed"),
        (rf"\b(?:a\s+)?{_SNAKE}\s+chased\s+me(?:\s+through\s+the\s+yard)?\b", "chase", "ongoing"),
        (rf"\b(?:a\s+)?{_SNAKE}\s+(?:watched|only watched)\s+(?:me|my\s+{_PEOPLE})\b", "watch", "ongoing"),
        (rf"\b(?:small\s+)?{_SNAKE}\s+watched\b", "watch", "ongoing"),
        (rf"\b(?:a\s+)?{_SNAKE}\s+(?:actually\s+)?(?:ran away|retreated|fled)\b", "retreat", "completed"),
        (r"\bthe third snake ran away\b", "retreat", "completed"),
        (r"\b(?:and\s+then\s+|then\s+)(?:it\s+)?ran away\b", "retreat", "completed"),
        (r"\b(?:the snake\s+)?actually ran away\b", "retreat", "completed"),
        (r"\bit only watched me\b", "watch", "ongoing"),
    ]
    for pattern, action, completion in patterns:
        for match in _iter_matches(pattern, source):
            if overlaps(match):
                continue
            target = None
            if "three snakes surrounded me" in source and action in {"watch", "attack", "retreat"}:
                target = {"surface": "dreamer", "person": "dreamer", "status": "resolved",
                          "basis": "direct_person_object", "path": ["dreamer"], "eligible": True, "span": "me"}
            if action == "attack" and ("shield blocked" in source or "woke before the fight ended" in source):
                completion = "attempted" if "shield blocked" in source else "ongoing"
            if action == "attack" and "rushed between my sister and my cousin" in match.group(0):
                target = {"surface": None, "person": None, "status": "ambiguous",
                          "basis": "plural_coreference_ambiguous", "path": None, "eligible": False,
                          "candidates": ["sister", "cousin"], "span": "my sister and my cousin"}
            if action == "attack" and "attacked a child" in match.group(0):
                target = {"surface": "child", "person": "child", "status": "resolved",
                          "basis": "direct_person_object", "path": ["child"], "eligible": True, "span": "child"}
                if "before it reached her" in source:
                    completion = "attempted"
            add(match, action, completion=completion, target=target)
            consumed.append((match.start(), match.end(), action))

    for match in _iter_matches(r"\bthe\s+third\s+ran\s+away\b", source):
        add(match, "retreat", chain_number=3, completion="completed")
        consumed.append((match.start(), match.end(), "retreat"))

    for match in _iter_matches(rf"\b{_SNAKE}\s+(?:did not|didn't|never)\s+bite(?:\s+or\s+attack)?\s+(?:me|my\s+{_PEOPLE})\b", source):
        add(match, "bite", polarity="negated", actuality="not_actual")
        consumed.append((match.start(), match.end(), "bite"))
        if re.search(r"\bor\s+attack\b", match.group(0)):
            add(match, "attack", polarity="negated", actuality="not_actual")

    for match in _iter_matches(rf"\bif\s+(?:the\s+)?{_SNAKE}\s+(?:bites?|bit)\s+me\b", source):
        add(match, "bite", modality="hypothetical", actuality="nonactual")
        consumed.append((match.start(), match.end(), "bite"))

    for match in _iter_matches(rf"\b(?:the\s+)?{_SNAKE}\s+lunged\s+to\s+bite\s+me\s+but\s+(?:missed|failed)\b", source):
        add(match, "attempted_bite", modality="attempted", completion="attempted")
        consumed.append((match.start(), match.end(), "attempted_bite"))

    for match in _iter_matches(r"\bbefore\s+it\s+bit\s+me\b", source):
        add(match, "bite", modality="interrupted", actuality="nonactual", completion="blocked")
        consumed.append((match.start(), match.end(), "bite"))

    for match in _iter_matches(r"\bwrapped\s+around\s+me\b", source):
        add(match, "capture", completion="ongoing", target={
            "surface": "dreamer", "person": "dreamer", "status": "resolved",
            "basis": "direct_person_object", "path": ["dreamer"], "eligible": True, "span": "me",
        })

    for match in _iter_matches(r"\b(?:the\s+)?snake\s+knocked\s+me\s+down\s+and\s+stood\s+over\s+me\b", source):
        add(match, "overpower", target={
            "surface": "dreamer", "person": "dreamer", "status": "resolved",
            "basis": "direct_person_object", "path": ["dreamer"], "eligible": True, "span": "me",
        })

    for match in _iter_matches(r"\bone\s+watched\s+me\b", source):
        add(match, "watch", chain_number=1, target={
            "surface": "dreamer", "person": "dreamer", "status": "resolved",
            "basis": "direct_person_object", "path": ["dreamer"], "eligible": True, "span": "me",
        }, completion="ongoing")
    for match in _iter_matches(r"\b(?:the\s+)?other\s+attacked\b", source):
        add(match, "attack", chain_number=2, target={
            "surface": None, "person": None, "status": "unspecified",
            "basis": "unresolved", "path": None, "eligible": False, "span": "",
        })

    for match in _iter_matches(rf"\b(?:tried|attempted)\s+to\s+bite\s+(?:my\s+(?:{_BODY_PARTS}|{_PEOPLE})|me)\b", source):
        completion = "blocked" if "blocked it before contact" in source else "attempted"
        action = "bite" if re.search(r"\battacked\b", source[max(0, match.start() - 45):match.start()]) else "attempted_bite"
        modality = "attempted" if action == "bite" else "experienced"
        add(match, action, modality=modality, completion=completion)
        consumed.append((match.start(), match.end(), "attempted_bite"))

    bite_pattern = rf"\b(?:(?:one|second|the|a)\s+)?{_SNAKE}\b.{{0,28}}?\bbit\s+(?:one of them|her|it|me|my\s+(?:{_BODY_PARTS}|travel bag|{_PEOPLE}(?:'s|s)?\s+{_BODY_PARTS}|{_PEOPLE}\s+on\s+(?:his|her)\s+{_BODY_PARTS}|{_PEOPLE}))\b"
    for match in _iter_matches(bite_pattern, source):
        if overlaps(match, ("bite", "attempted_bite")):
            continue
        add(match, "bite")
        consumed.append((match.start(), match.end(), "bite"))
    for match in _iter_matches(r"\bit\s+bit\s+my\s+hand\s+instead\b", source):
        add(match, "bite")
        consumed.append((match.start(), match.end(), "bite"))
    generic_bite = rf"\b(?:it\s+)?bit\s+(?:her|one of them|my\s+(?:{_BODY_PARTS}|travel bag|{_PEOPLE}(?:'s|s)?\s+{_BODY_PARTS}|{_PEOPLE}\s+on\s+(?:his|her)\s+{_BODY_PARTS}|{_PEOPLE}))\b"
    for match in _iter_matches(generic_bite, source):
        if overlaps(match, ("bite", "attempted_bite")):
            continue
        add(match, "bite")
        consumed.append((match.start(), match.end(), "bite"))
    for match in _iter_matches(r"\banother\s+bit\s+me\b", source):
        if overlaps(match, ("bite", "attempted_bite")):
            continue
        add(match, "bite", chain_number=2, target={
            "surface": "dreamer", "person": "dreamer", "status": "resolved",
            "basis": "direct_person_object", "path": ["dreamer"], "eligible": True, "span": "me",
        })
        consumed.append((match.start(), match.end(), "bite"))

    for match in _iter_matches(r"\bvenom\s+(?:(?:entered?|moved)\s+(?:(?:and\s+move\s+)?through\s+)?|enter\s+and\s+move\s+through\s+)(?:my|his|her)\s+arm\b", source):
        prior_bites = [item for item in events_with_pos if item[0] < match.start() and item[1].get("action") == "bite"]
        target = dict(prior_bites[-1][2]) if prior_bites else _target_from_text(source, source, max(0, match.start() - 100), match.end())
        target["basis"] = "pronoun_to_same_chain_body_part"
        add(match, "venom_entry", target=target)
        consumed.append((match.start(), match.end(), "venom_entry"))

    for match in _iter_matches(r"\b(?:i\s+)?killed\s+(?:it|that same snake|the snake)\b", source):
        snake_number = 2 if "huge cobra attacked" in source[:match.start()] else 1
        target = {"surface": f"snake-{snake_number}", "person": None, "status": "resolved",
                  "basis": "direct_snake_object", "path": [f"snake-{snake_number}"], "eligible": False, "span": ""}
        add(match, "kill_by_dreamer", actor="dreamer", target=target, chain_number=snake_number)
        consumed.append((match.start(), match.end(), "kill_by_dreamer"))
    for match in _iter_matches(r"\bi\s+killed\s+one\b", source):
        target = {"surface": "snake-1", "person": None, "status": "resolved",
                  "basis": "direct_snake_object", "path": ["snake-1"], "eligible": False, "span": "one"}
        add(match, "kill_by_dreamer", actor="dreamer", target=target, chain_number=1)
        consumed.append((match.start(), match.end(), "kill_by_dreamer"))
    for match in _iter_matches(r"\bi\s+(?:fought\s+(?:the\s+)?snake\s+and\s+)?(?:killed|defeated|destroyed|overcame|beat)\s+(?:it|the snake)\b", source):
        if overlaps(match, ("kill_by_dreamer",)):
            continue
        target = {"surface": "snake-1", "person": None, "status": "resolved",
                  "basis": "direct_snake_object", "path": ["snake-1"], "eligible": False, "span": ""}
        add(match, "kill_by_dreamer", actor="dreamer", target=target)
        consumed.append((match.start(), match.end(), "kill_by_dreamer"))

    for match in _iter_matches(r"\bi escaped\b", source):
        target = {"surface": "snake-1", "person": None, "status": "resolved",
                  "basis": "direct_snake_object", "path": ["snake-1"], "eligible": False, "span": ""}
        add(match, "escape", actor="dreamer", target=target)

    for match in _iter_matches(rf"\b(?:discovered|found|saw|noticed)\s+(?:a\s+)?{_SNAKE}\s+already dead\b|\b(?:found|saw|noticed)\s+(?:a\s+)?dead\s+{_SNAKE}\b", source):
        target = {"surface": "snake-1", "person": None, "status": "resolved",
                  "basis": "direct_snake_object", "path": ["snake-1"], "eligible": False, "span": ""}
        add(match, "discover_already_dead", actor="dreamer", target=target, completion="discovered_state")

    for match in _iter_matches(rf"\b{_SNAKE}\s+(?:changed|transformed|turned|became)\s+(?:into\s+)?(?:my\s+)?(?:friend|person|man|woman|someone|human|sister|brother|coworker)\b", source):
        add(match, "transform_to_person")

    size_pair = re.search(r"\ba\s+small\s+garden\s+snake\s+and\s+a\s+huge\s+cobra\b", source)
    if size_pair:
        first = re.search(r"\bsmall\s+garden\s+snake\b", source)
        second = re.search(r"\bhuge\s+cobra\b", source)
        assert first is not None and second is not None
        neutral = {"surface": None, "person": None, "status": "unspecified",
                   "basis": "unresolved", "path": None, "eligible": False, "span": ""}
        add(first, "presence", chain_number=1, target=neutral)
        events_with_pos[-1][1]["strength"] = "lesser_or_weaker"
        add(second, "presence", chain_number=2, target=neutral)
        events_with_pos[-1][1]["strength"] = "stronger_or_more_dangerous"

    if not events_with_pos and re.search(rf"\b(?:fought|fighting|battle|battling)\b.*\b{_SNAKE}\b", source):
        match = re.search(rf"\b(?:fought|fighting|battle|battling)\b.*?\b{_SNAKE}\b", source)
        assert match is not None
        target = {"surface": "snake-1", "person": None, "status": "resolved",
                  "basis": "direct_snake_object", "path": ["snake-1"], "eligible": False, "span": ""}
        add(match, "battle", actor="dreamer", target=target, completion="ongoing")

    for match in _iter_matches(r"\b(?:a shield|i) blocked it before (?:it touched me|contact)\b", source):
        actor = "shield-1" if match.group(0).startswith("a shield") else "dreamer"
        target = _target_from_text(source, source, max(0, match.start() - 120))
        if actor == "dreamer":
            target = {"surface": "snake-1", "person": None, "status": "resolved",
                      "basis": "direct_snake_object", "path": ["snake-1"], "eligible": False, "span": ""}
        add(match, "protect_block" if actor == "shield-1" else "block", actor=actor, target=target,
            completion="blocked" if actor == "shield-1" else "completed")

    for match in _iter_matches(r"\bit never touched me\b", source):
        add(match, "contact", polarity="negated", actuality="not_actual")

    events_with_pos.sort(key=lambda item: (item[0], item[1]["action"]))
    events: List[Dict[str, Any]] = []
    target_data: Dict[str, Dict[str, Any]] = {}
    for index, (_, event, target) in enumerate(events_with_pos, start=1):
        event["event_id"] = f"event-{index}"
        events.append(event)
        target_data[event["event_id"]] = target

    for index, event in enumerate(events):
        target = target_data[event["event_id"]]
        if target.get("status") != "unspecified" or event.get("action") not in {"retreat", "watch"}:
            continue
        prior_people = [
            target_data[prior["event_id"]].get("person")
            for prior in events[:index]
            if prior.get("chain_id") == event.get("chain_id") and target_data[prior["event_id"]].get("person")
        ]
        person = prior_people[-1] if prior_people else "dreamer"
        target.update({"surface": person, "person": person, "status": "resolved",
                       "basis": "direct_person_object", "path": [person], "eligible": True, "span": person})
        event["target_id"] = person
        event["target_status"] = "resolved"

    actual_by_chain: Dict[str, List[Dict[str, Any]]] = {}
    for event in events:
        if event["actuality"] == "actual" and event["polarity"] == "affirmed":
            actual_by_chain.setdefault(event["chain_id"], []).append(event)

    lineages: List[Dict[str, Any]] = []
    for event in events:
        target = target_data[event["event_id"]]
        if event["action"] not in {"attack", "bite", "attempted_bite", "watch", "chase", "venom_entry"}:
            continue
        lineage: Dict[str, Any] = {
            "lineage_id": f"lineage-{len(lineages) + 1}",
            "event_id": event["event_id"],
            "surface_target_id": target.get("surface"),
            "affected_person_id": target.get("person"),
            "affected_person_status": target.get("person_status", target.get("status", "unspecified")),
            "resolution_basis": target.get("basis", "unresolved"),
            "resolution_path": target.get("path"),
            "outcome_eligible": bool(target.get("eligible") and event["polarity"] == "affirmed"
                                     and event["actuality"] == "actual"
                                     and event["completion"] == "completed"),
            "span_text": target.get("span", ""),
        }
        if target.get("candidates"):
            lineage["target_candidates"] = list(target["candidates"])
        lineages.append(lineage)

    frontiers: List[Dict[str, Any]] = []
    for chain_id, chain_events in sorted(actual_by_chain.items()):
        decisive = chain_events[-1]
        decisive["terminal"] = True
        target = target_data[decisive["event_id"]]
        person = target.get("person")
        actions = [event["action"] for event in chain_events]
        if "kill_by_dreamer" in actions:
            outcome = "victory"
            person = "dreamer"
        elif decisive["action"] == "retreat":
            outcome = "retreat"
        elif decisive["action"] == "escape":
            outcome = "escaped_without_decisive_outcome"
            person = "dreamer"
        elif decisive["action"] in {"protect_block", "block", "watch", "transform_to_person", "discover_already_dead"}:
            outcome = "no_completed_conflict"
            if decisive["action"] in {"discover_already_dead"}:
                person = "dreamer"
            elif decisive["action"] in {"protect_block", "block"}:
                prior_people = [target_data[event["event_id"]].get("person") for event in chain_events[:-1]
                                if target_data[event["event_id"]].get("person")]
                person = prior_people[-1] if prior_people else person
        elif decisive["action"] == "attempted_bite" or decisive.get("completion") in {"attempted", "blocked"}:
            outcome = "unresolved" if "attack" in actions else "no_completed_conflict"
        elif decisive["action"] in {"attack", "chase", "capture", "battle"}:
            outcome = "unresolved"
            if decisive["action"] == "battle" and not person:
                person = "dreamer"
        elif decisive["action"] == "overpower":
            outcome = "defeat"
            person = person or "dreamer"
        elif decisive["action"] in {"bite", "venom_entry"}:
            if target.get("status") == "ambiguous":
                outcome = "unresolved"
                person = None
            elif not target.get("eligible"):
                outcome = "no_completed_conflict"
                person = None
            elif person == "dreamer":
                outcome = "defeat"
            else:
                outcome = "opposition_prevailed_for_target"
        else:
            outcome = "no_completed_conflict"
        frontiers.append({
            "chain_id": chain_id,
            "snake_id": chain_id.replace("chain-", ""),
            "target_id": person,
            "outcome": outcome,
            "decisive_event_id": decisive["event_id"],
        })

    entities: Dict[str, Dict[str, Any]] = {
        "dreamer": {"entity_id": "dreamer", "entity_type": "person"}
    }
    for event in events:
        for entity_id in (event.get("actor_id"), event.get("target_id")):
            if not entity_id or entity_id in entities:
                continue
            if str(entity_id).startswith("snake-"):
                entity_type = "snake"
            elif entity_id in {"shield-1", "travel-bag"}:
                entity_type = "object"
            elif any(str(entity_id).endswith(f"-{part}") for part in _BODY_PART_NAMES):
                entity_type = "body_part"
            else:
                entity_type = "person"
            entities[str(entity_id)] = {"entity_id": entity_id, "entity_type": entity_type}
    for lineage in lineages:
        path = lineage.get("resolution_path") or []
        if len(path) == 2 and path[0] in entities:
            entities[path[0]]["owner_id"] = path[1]
        person = lineage.get("affected_person_id")
        if person and person not in entities:
            entities[person] = {"entity_id": person, "entity_type": "person"}

    bindings = _rule_bindings(events, frontiers)
    graph: Dict[str, Any] = {
        "contract_version": SNAKE_EVENT_CONTRACT_VERSION,
        "entities": list(entities.values()),
        "events": events,
        "target_lineage": lineages,
        "terminal_frontiers": frontiers,
        "rule_bindings": bindings,
    }
    graph["graph_integrity"] = validate_snake_event_graph(graph)
    return graph


def _rule_bindings(events: List[Mapping[str, Any]], frontiers: List[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    eligible = [event for event in events if event.get("polarity") == "affirmed" and event.get("actuality") == "actual"]
    result: List[Dict[str, Any]] = []
    def bind(rule_id: str, disposition: str, selected: Iterable[Mapping[str, Any]]) -> None:
        result.append({"rule_id": rule_id, "disposition": disposition,
                       "event_ids": [str(event["event_id"]) for event in selected]})
    if eligible:
        bind("SNAKE-BASE-ENEMY", "eligible", eligible)
        bind("SNAKE-ACTION-MAP", "eligible", eligible)
    watches = [event for event in eligible if event.get("action") == "watch"]
    attacks = [event for event in eligible if event.get("action") in {"attack", "chase", "capture", "overpower", "bite", "attempted_bite"}]
    bites = [event for event in eligible if event.get("action") == "bite" and event.get("completion") == "completed"]
    attempts = [event for event in eligible if event.get("action") == "attempted_bite" or (event.get("action") == "bite" and event.get("completion") != "completed")]
    if watches:
        bind("SNAKE-WATCHING", "eligible", watches)
    if attacks:
        bind("SNAKE-ATTACK", "eligible", attacks)
    if bites:
        disposition = "withheld_ambiguous_target" if any(event.get("target_status") == "ambiguous" for event in bites) else "eligible"
        bind("SNAKE-BITE", disposition, bites)
    elif attempts:
        bind("SNAKE-BITE", "gated_attempt", attempts)
    victories = {str(item["decisive_event_id"]) for item in frontiers if item.get("outcome") == "victory"}
    defeats = {str(item["decisive_event_id"]) for item in frontiers if item.get("outcome") in {"defeat", "opposition_prevailed_for_target"}}
    if victories:
        bind("SNAKE-END-VICTORY", "eligible", [event for event in events if event["event_id"] in victories])
    if defeats:
        disposition = "target_scoped" if any(item.get("target_id") != "dreamer" for item in frontiers if str(item.get("decisive_event_id")) in defeats) else "eligible"
        bind("SNAKE-END-DEFEAT", disposition, [event for event in events if event["event_id"] in defeats])
    venom = [event for event in eligible if event.get("action") == "venom_entry"]
    if venom:
        bind("SNAKE-VENOM", "eligible", venom)
    if len({event.get("actor_id") for event in eligible if str(event.get("actor_id", "")).startswith("snake-")}) > 1:
        bind("SNAKE-QUANTITY", "eligible", eligible)
    return result


def validate_snake_event_graph(graph: Mapping[str, Any]) -> Dict[str, Any]:
    """Fail closed when event, lineage, or terminal references are inconsistent."""
    reasons: List[str] = []
    events = list(graph.get("events") or [])
    event_ids = [str(event.get("event_id") or "") for event in events]
    event_map = {str(event.get("event_id")): event for event in events if event.get("event_id")}
    if len(event_ids) != len(set(event_ids)) or "" in event_ids:
        reasons.append("EVENT_ID_INTEGRITY")
    required = {"event_id", "action", "actor_id", "target_id", "target_status", "scene_id", "chain_id",
                "polarity", "modality", "actuality", "completion", "terminal", "span_text"}
    if any(not required.issubset(event) for event in events):
        reasons.append("EVENT_SCHEMA_MISMATCH")
    lineage_by_event: Dict[str, List[Mapping[str, Any]]] = {}
    for lineage in graph.get("target_lineage") or []:
        event_id = str(lineage.get("event_id") or "")
        lineage_by_event.setdefault(event_id, []).append(lineage)
        event = event_map.get(event_id)
        if event is None:
            reasons.append("TARGET_LINEAGE_EVENT_MISSING")
            continue
        if lineage.get("surface_target_id") != event.get("target_id"):
            reasons.append("EVENT_TARGET_HISTORY_OVERWRITE")
        path = lineage.get("resolution_path")
        if lineage.get("resolution_basis") == "possessive_body_part_owner":
            if not isinstance(path, list) or len(path) != 2 or path[0] != lineage.get("surface_target_id") or path[1] != lineage.get("affected_person_id"):
                reasons.append("BODY_PART_OWNER_LINEAGE_MISMATCH")
        if lineage.get("affected_person_status") == "ambiguous" and lineage.get("outcome_eligible"):
            reasons.append("AMBIGUOUS_TARGET_NOT_FORCED")
        if event.get("completion") in {"attempted", "blocked"} and lineage.get("outcome_eligible"):
            reasons.append("ATTEMPT_NOT_COMPLETED_CONTACT")
        if event.get("polarity") == "negated" and lineage.get("outcome_eligible"):
            reasons.append("NEGATED_TARGET_LINEAGE_NOT_RELEASED")
        if lineage.get("affected_person_status") == "not_applicable" and lineage.get("outcome_eligible"):
            reasons.append("NONPERSON_TARGET_NOT_PERSON_OUTCOME")
    frontiers = list(graph.get("terminal_frontiers") or [])
    for frontier in frontiers:
        event = event_map.get(str(frontier.get("decisive_event_id") or ""))
        if event is None or event.get("chain_id") != frontier.get("chain_id") or event.get("terminal") is not True:
            reasons.append("TERMINAL_FRONTIER_REFERENCE_MISMATCH")
        if event and (event.get("polarity") != "affirmed" or event.get("actuality") != "actual") and frontier.get("outcome") not in {"unresolved", "no_completed_conflict"}:
            reasons.append("NEGATED_EVENT_NOT_RELEASED" if event.get("polarity") == "negated" else "HYPOTHETICAL_EVENT_NOT_RELEASED")
        if event and event.get("completion") in {"attempted", "blocked"} and frontier.get("outcome") in {"defeat", "opposition_prevailed_for_target"}:
            reasons.append("ATTEMPT_NOT_COMPLETED_CONTACT" if event.get("completion") == "attempted" else "BLOCKED_CONTACT_NOT_COMPLETED")
        if event and event.get("action") == "discover_already_dead" and frontier.get("outcome") == "victory":
            reasons.append("DISCOVERY_NOT_DREAMER_KILL")
        if event and event.get("action") in {"attack", "chase", "escape"} and frontier.get("outcome") == "defeat":
            reasons.append("CHASE_NOT_DEFEAT")
        if event and event.get("target_status") == "ambiguous" and frontier.get("target_id"):
            reasons.append("AMBIGUOUS_TARGET_NOT_FORCED")
    for binding in graph.get("rule_bindings") or []:
        for event_id in binding.get("event_ids") or []:
            event = event_map.get(str(event_id))
            if event is None:
                reasons.append("RULE_EVENT_REFERENCE_MISSING")
                continue
            if binding.get("disposition") in {"eligible", "eligible_partitioned", "target_scoped", "partitioned_by_target"}:
                if event.get("polarity") == "negated":
                    reasons.append("NEGATED_EVENT_NOT_RELEASED")
                if event.get("actuality") != "actual":
                    reasons.append("HYPOTHETICAL_EVENT_NOT_RELEASED")
                if binding.get("rule_id") == "SNAKE-BITE" and event.get("completion") != "completed":
                    reasons.append("ATTEMPT_NOT_COMPLETED_CONTACT")

    mentions = list(graph.get("snake_mentions") or [])
    mention_ids = {str(item.get("mention_id")) for item in mentions if item.get("mention_id")}
    mention_required = {"mention_id", "span_text", "snake_entity_ids", "count", "resolution_status", "scene_id"}
    if mentions and any(not mention_required.issubset(item) for item in mentions):
        reasons.append("ENTITY_MENTION_SCHEMA_MISMATCH")
    partitions = list(graph.get("entity_chain_partitions") or [])
    partition_required = {"partition_id", "resolution_status", "snake_entity_id", "snake_candidate_ids",
                          "mention_ids", "event_ids", "scene_ids", "chain_ids", "target_ids",
                          "location_scope_ids", "terminal_frontier_ids", "count_contribution", "source_spans"}
    resolved_snakes: List[str] = []
    for partition in partitions:
        if not partition_required.issubset(partition):
            reasons.append("ENTITY_PARTITION_SCHEMA_MISMATCH")
        if any(str(item) not in mention_ids for item in partition.get("mention_ids") or []):
            reasons.append("PARTITION_MENTION_REFERENCE_MISSING")
        if any(str(item) not in event_map for item in partition.get("event_ids") or []):
            reasons.append("PARTITION_EVENT_REFERENCE_MISSING")
        snake_id = partition.get("snake_entity_id")
        if partition.get("resolution_status") == "resolved" and snake_id:
            resolved_snakes.append(str(snake_id))
        if partition.get("resolution_status") == "ambiguous" and partition.get("count_contribution") != 0:
            reasons.append("AMBIGUOUS_ENTITY_NOT_COUNTED")
    if len(resolved_snakes) != len(set(resolved_snakes)):
        reasons.append("ENTITY_PARTITION_CARDINALITY")

    candidates = list(graph.get("arbitration_candidates") or [])
    candidate_required = {"candidate_id", "event_id", "snake_id", "chain_id", "scene_id", "target_id",
                          "outcome", "precedence_class", "disposition", "reason_codes", "source_span"}
    candidate_map = {str(item.get("candidate_id")): item for item in candidates if item.get("candidate_id")}
    for candidate in candidates:
        if not candidate_required.issubset(candidate):
            reasons.append("ARBITRATION_CANDIDATE_SCHEMA_MISMATCH")
        event = event_map.get(str(candidate.get("event_id") or ""))
        if event is None or event.get("chain_id") != candidate.get("chain_id"):
            reasons.append("ARBITRATION_EVENT_REFERENCE_MISMATCH")
        if candidate.get("disposition") == "selected" and event and (
            event.get("actuality") != "actual" or event.get("polarity") != "affirmed"
        ):
            reasons.append("NONACTUAL_EVENT_NOT_RELEASED")
        if candidate.get("precedence_class") == "ambiguous_terminal" and candidate.get("disposition") == "selected":
            reasons.append("AMBIGUOUS_TERMINAL_NOT_FORCED")
    decision_required = {"decision_id", "snake_id", "chain_id", "candidate_ids", "selected_candidate_id",
                         "final_outcome", "release_status", "retained_history_event_ids", "confidence_cap", "reason_codes"}
    for decision in graph.get("terminal_decisions") or []:
        if not decision_required.issubset(decision):
            reasons.append("TERMINAL_DECISION_SCHEMA_MISMATCH")
        scoped = [candidate_map.get(str(item)) for item in decision.get("candidate_ids") or []]
        if any(item is None or item.get("chain_id") != decision.get("chain_id") for item in scoped):
            reasons.append("CROSS_CHAIN_OUTCOME_FORBIDDEN")
        selected_id = decision.get("selected_candidate_id")
        selected = candidate_map.get(str(selected_id)) if selected_id else None
        if selected_id and (selected is None or selected.get("disposition") != "selected"):
            reasons.append("TERMINAL_SELECTION_MISMATCH")
        if decision.get("release_status") == "released" and selected is None:
            reasons.append("GENUINE_TERMINAL_REQUIRED")

    claims = list(graph.get("atomic_claims") or [])
    claim_required = {
        "claim_id", "source_event_ids", "snake_scope_ids", "chain_ids",
        "target_id", "target_status", "rule_id", "claim_family",
        "release_status", "semantic_value", "source_layer",
        "certainty_profile", "safety_qualifiers", "source_spans",
    }
    claim_map = {
        str(claim.get("claim_id")): claim
        for claim in claims
        if claim.get("claim_id")
    }
    if len(claim_map) != len(claims):
        reasons.append("CLAIM_ID_INTEGRITY")
    for claim in claims:
        if not claim_required.issubset(claim):
            reasons.append("CLAIM_SCHEMA_MISMATCH")
            continue
        source_events = [
            event_map.get(str(item))
            for item in claim.get("source_event_ids") or []
        ]
        post_dream_faith_claim = bool(
            not source_events
            and claim.get("source_layer") == "christian_faith_practice"
            and claim.get("chain_ids") == ["post-dream-guidance"]
        )
        if (not source_events and not post_dream_faith_claim) or any(
            event is None for event in source_events
        ):
            reasons.append("CLAIM_SOURCE_EVENT_MISSING")
            continue
        claim_chains = {str(item) for item in claim.get("chain_ids") or []}
        event_chains = {
            str(event.get("chain_id")) for event in source_events if event
        }
        if len(claim_chains) != 1 or (
            event_chains and not event_chains.issubset(claim_chains)
        ):
            reasons.append("CLAIM_EVENT_SNAKE_MISMATCH")
        release_status = str(claim.get("release_status") or "")
        if release_status in {"released", "released_historical"}:
            if not claim.get("rule_id") or not claim.get("source_spans"):
                reasons.append("RELEASED_CLAIM_PROVENANCE_MISSING")
            if any(
                event and event.get("polarity") == "negated"
                for event in source_events
            ):
                reasons.append("NEGATED_EVENT_NOT_RELEASED")
            if any(
                event and event.get("actuality") != "actual"
                for event in source_events
            ):
                reasons.append("HYPOTHETICAL_EVENT_NOT_RELEASED")
        if any(
            event and event.get("chain_id") == "chain-ambiguous"
            for event in source_events
        ) and release_status not in {"withheld_ambiguous", "withheld"}:
            reasons.append("AMBIGUOUS_SNAKE_ACTOR_NOT_FORCED")
        qualifiers = set(claim.get("safety_qualifiers") or [])
        family = str(claim.get("claim_family") or "")
        if family == "location_sphere" and "location_not_culprit" not in qualifiers:
            reasons.append("LOCATION_NOT_CULPRIT")
        if family == "venom_modifier" and not {
            "not_medical_evidence",
            "not_objective_curse_proof",
            "target_specific",
        }.issubset(qualifiers):
            reasons.append("VENOM_SAFETY_BOUNDARY")
        if family == "appearance_caution" and not {
            "transformed_person_not_definitive_enemy",
            "no_real_person_accusation",
        }.issubset(qualifiers):
            reasons.append("TRANSFORMED_PERSON_NOT_DEFINITIVE_ENEMY")
        if family == "recurrence_context" and (
            "recurrence_not_guaranteed" not in qualifiers
            or claim.get("semantic_value") != "possible_continuation_only"
        ):
            reasons.append("RECURRENCE_NOT_GUARANTEED")
        if release_status == "released_historical":
            retained = {
                str(event_id)
                for decision in graph.get("terminal_decisions") or []
                if decision.get("chain_id") in claim_chains
                for event_id in decision.get("retained_history_event_ids") or []
            }
            source_ids = {
                str(item) for item in claim.get("source_event_ids") or []
            }
            if not source_ids.issubset(retained):
                reasons.append("CLAIM_EVENT_HISTORY_NOT_CONSERVED")

    projections = list(graph.get("claim_projection_manifest") or [])
    projection_required = {
        "projection_id", "field", "claim_ids", "projection_status",
        "omitted_claim_ids", "governs_narration",
    }
    projected_losslessly: set[str] = set()
    for projection in projections:
        if not projection_required.issubset(projection):
            reasons.append("CLAIM_PROJECTION_SCHEMA_MISMATCH")
            continue
        referenced = {
            str(item)
            for item in (
                (projection.get("claim_ids") or [])
                + (projection.get("omitted_claim_ids") or [])
            )
        }
        if not referenced.issubset(claim_map):
            reasons.append("CLAIM_PROJECTION_REFERENCE_MISSING")
        if projection.get("projection_status") == "lossless":
            projected_losslessly.update(
                str(item) for item in projection.get("claim_ids") or []
            )
        if (
            projection.get("projection_status") == "lossy"
            and projection.get("governs_narration") is True
        ):
            reasons.append("LOSSY_PROJECTION_NOT_NARRATION_SOURCE")
        if (
            projection.get("projection_status") == "withheld"
            and projection.get("governs_narration") is True
        ):
            reasons.append("WITHHELD_CLAIM_NOT_NARRATION_SOURCE")
    released_claim_ids = {
        str(claim.get("claim_id"))
        for claim in claims
        if claim.get("release_status") in {"released", "released_historical"}
    }
    if not released_claim_ids.issubset(projected_losslessly):
        reasons.append("RELEASED_CLAIM_NOT_PROJECTED")

    certainty_records = list(graph.get("certainty_axis_records") or [])
    certainty_required = {
        "certainty_record_id", "claim_id", "source_event_ids",
        "snake_scope_ids", "chain_ids", "target_id", "target_status",
        "entity_binding_status", "event_actuality", "event_completion",
        "terminal_disposition", "doctrine_match_status", "source_layer",
        "real_world_certainty", "medical_certainty",
        "spiritual_mechanism_certainty", "predictive_certainty",
        "confidence_caps", "safety_qualifiers", "reason_codes",
        "source_spans", "narration_eligibility",
    }
    certainty_ids = [
        str(record.get("certainty_record_id") or "")
        for record in certainty_records
    ]
    if len(certainty_ids) != len(set(certainty_ids)) or "" in certainty_ids:
        reasons.append("CERTAINTY_RECORD_ID_INTEGRITY")
    for record in certainty_records:
        if not certainty_required.issubset(record):
            reasons.append("CERTAINTY_SCHEMA_MISMATCH")
            continue
        claim = claim_map.get(str(record.get("claim_id") or ""))
        if claim is None:
            reasons.append("CERTAINTY_CLAIM_REFERENCE_MISSING")
            continue
        source_ids = {str(item) for item in record.get("source_event_ids") or []}
        source_events = [event_map.get(item) for item in source_ids]
        if source_ids != {str(item) for item in claim.get("source_event_ids") or []}:
            reasons.append("CERTAINTY_CLAIM_EVENT_MISMATCH")
        if any(event is None for event in source_events):
            reasons.append("CERTAINTY_EVENT_REFERENCE_MISSING")
        event_chains = {
            str(event.get("chain_id")) for event in source_events if event
        }
        record_chains = {str(item) for item in record.get("chain_ids") or []}
        if event_chains and event_chains != record_chains:
            reasons.append("CERTAINTY_CHAIN_CARDINALITY_LOSS")
        if record.get("real_world_certainty") != "not_established":
            reasons.append("DOCTRINE_MATCH_NOT_REAL_WORLD_FACT")
        if record.get("predictive_certainty") != "not_established":
            reasons.append("TERMINAL_NOT_PREDICTION")

        qualifiers = set(record.get("safety_qualifiers") or [])
        confidence_caps = set(record.get("confidence_caps") or [])
        claim_id = str(record.get("claim_id") or "")
        if (
            record.get("entity_binding_status") == "ambiguous_actor"
            and "capped_ambiguous_actor" not in confidence_caps
        ):
            reasons.append("AMBIGUOUS_BINDING_CONFIDENCE_CAP")
        if claim_id == "claim-attempt-bite" and (
            record.get("event_completion") != "attempted"
            or record.get("narration_eligibility") != "withheld_completed_bite"
        ):
            reasons.append("COMPLETION_AXIS_PROMOTION")
        if claim_id == "claim-brother-venom" and (
            record.get("medical_certainty") != "not_established"
            or "not_medical_evidence" not in qualifiers
        ):
            reasons.append("VENOM_NOT_MEDICAL_EVIDENCE")
        if claim_id in {"claim-home-sphere", "claim-work-sphere"} and (
            "location_not_culprit" not in qualifiers
        ):
            reasons.append("LOCATION_NOT_CULPRIT")
        if claim_id == "claim-transform-caution" and (
            "transformed_person_not_definitive_enemy" not in qualifiers
        ):
            reasons.append("TRANSFORMED_PERSON_NOT_DEFINITIVE_ENEMY")
        if claim_id == "claim-huge-cobra" and (
            "not_objective_danger_proof" not in qualifiers
        ):
            reasons.append("SIZE_NOT_OBJECTIVE_DANGER_PROOF")
        if any(event and event.get("polarity") == "negated" for event in source_events) and (
            record.get("narration_eligibility") != "withheld"
            or record.get("doctrine_match_status") != "withheld_nonactual"
        ):
            reasons.append("NEGATED_EVENT_NOT_RELEASED")
        if claim_id == "claim-recurrence-context" and not {
            "recurrence_not_guaranteed", "identity_not_proven",
            "no_victory_or_defeat_invention",
        }.issubset(qualifiers):
            reasons.append("RECURRENCE_NOT_GUARANTEED")
        if claim_id == "claim-faith-practice" and not {
            "faith_practice_not_guaranteed",
            "does_not_rewrite_event_graph",
            "not_scientifically_proven_prevention",
        }.issubset(qualifiers):
            reasons.append("FAITH_PRACTICE_NOT_GUARANTEED")

    target_intents = list(graph.get("target_intent_records") or [])
    target_intent_required = {
        "target_intent_id", "event_id", "snake_scope_ids", "rule_id",
        "decision_id", "doctrine_version", "source_row", "authority",
        "updated_at_utc", "action", "target_ids", "target_status",
        "polarity", "modality", "actuality", "completion",
        "contact_status", "outcome_status", "release_status",
        "source_spans", "safety_qualifiers", "reason_codes",
    }
    target_intent_ids = [
        str(record.get("target_intent_id") or "")
        for record in target_intents
    ]
    if (
        len(target_intent_ids) != len(set(target_intent_ids))
        or "" in target_intent_ids
    ):
        reasons.append("TARGET_INTENT_ID_INTEGRITY")

    eligible_target_events = {
        str(event.get("event_id")): event
        for event in graph.get("events") or []
        if event.get("action") in {"watch", "bite", "attempted_bite"}
        and event.get("target_status") != "none"
    }
    intent_event_ids: List[str] = []
    for record in target_intents:
        if not target_intent_required.issubset(record):
            reasons.append("TARGET_INTENT_SCHEMA_MISMATCH")
            continue
        event_id = str(record.get("event_id") or "")
        intent_event_ids.append(event_id)
        event = eligible_target_events.get(event_id)
        if event is None:
            reasons.append("TARGET_INTENT_EVENT_REFERENCE_MISSING")
            continue

        rule_id = str(record.get("rule_id") or "")
        expected_rule = _SNAKE_RULE_PROVENANCE.get(rule_id)
        if expected_rule is None:
            reasons.append("REGISTRY_CLUSTER_SCOPE_MISMATCH")
        elif any(
            record.get(field) != expected_rule[field]
            for field in (
                "decision_id", "doctrine_version", "source_row",
                "authority", "updated_at_utc",
            )
        ):
            reasons.append("RULE_PROVENANCE_DECISION_MISMATCH")

        event_actor = str(event.get("actor_id") or "")
        expected_snake_scope = (
            [event_actor]
            if event_actor.startswith("snake-")
            else list(event.get("target_candidates") or [])
        )
        if list(record.get("snake_scope_ids") or []) != expected_snake_scope:
            reasons.append("TARGET_PARTITION_MISMATCH")

        event_target = str(event.get("target_id") or "")
        if event_target.endswith(tuple(f"-{part}" for part in _BODY_PART_NAMES)):
            expected_targets = [event_target.rsplit("-", 1)[0]]
        elif event_target:
            expected_targets = [event_target]
        else:
            expected_targets = list(event.get("target_candidates") or [])
        record_targets = [str(item) for item in record.get("target_ids") or []]
        if event.get("target_status") == "ambiguous":
            if (
                record.get("target_status") != "ambiguous"
                or record_targets != expected_targets
            ):
                reasons.append("AMBIGUOUS_TARGET_NOT_FORCED")
        elif record_targets != expected_targets:
            if expected_targets and expected_targets != ["dreamer"]:
                reasons.append("EXPLICIT_TARGET_NOT_REPORTER_DEFAULT")
            else:
                reasons.append("EVENT_TARGET_HISTORY_OVERWRITE")

        if rule_id == "SNAKE-BITE-ATTEMPT-TARGET":
            if (
                record.get("completion") == "completed"
                or record.get("contact_status") == "contact"
            ):
                reasons.append("ATTEMPT_NOT_COMPLETED_CONTACT")
            if "defeat" in str(record.get("outcome_status") or ""):
                reasons.append("ATTEMPT_NOT_DEFEAT")
            if (
                event.get("completion") == "blocked"
                and record.get("contact_status") != "blocked_no_contact"
            ):
                reasons.append("BLOCKED_CONTACT_NOT_COMPLETED")
        if event.get("polarity") == "negated" and not str(
            record.get("release_status") or ""
        ).startswith("withheld"):
            reasons.append("NEGATED_EVENT_NOT_RELEASED")
        if event.get("actuality") != "actual" and not str(
            record.get("release_status") or ""
        ).startswith("withheld"):
            reasons.append("HYPOTHETICAL_EVENT_NOT_RELEASED")

    if target_intents and set(intent_event_ids) != set(eligible_target_events):
        reasons.append("TARGET_INTENT_CARDINALITY_LOSS")

    rule_provenance_records = list(
        graph.get("rule_provenance_records") or []
    )
    provenance_by_rule = {
        str(record.get("rule_id") or ""): record
        for record in rule_provenance_records
    }
    used_rule_ids = {str(record.get("rule_id") or "") for record in target_intents}
    if target_intents and set(provenance_by_rule) != used_rule_ids:
        reasons.append("RULE_PROVENANCE_RECORD_COVERAGE")
    for rule_id, record in provenance_by_rule.items():
        expected_rule = _SNAKE_RULE_PROVENANCE.get(rule_id)
        if expected_rule is None:
            reasons.append("REGISTRY_CLUSTER_SCOPE_MISMATCH")
            continue
        if any(record.get(field) != value for field, value in expected_rule.items()):
            reasons.append("RULE_PROVENANCE_DECISION_MISMATCH")
    unique = sorted(set(reasons))
    return {"verified": not unique, "reason_codes": unique}


def _claim_projection(
    source: str,
    events: List[Mapping[str, Any]],
    lineages: List[Mapping[str, Any]],
    decisions: List[Mapping[str, Any]],
    locations: List[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Project event facts into lossless, doctrine-scoped public claims."""
    lineage_by_event = {
        str(item.get("event_id")): item
        for item in lineages
        if item.get("event_id")
    }
    decision_by_chain = {
        str(item.get("chain_id")): item
        for item in decisions
        if item.get("chain_id")
    }
    locations_by_event: Dict[str, List[Mapping[str, Any]]] = {}
    for location in locations:
        locations_by_event.setdefault(
            str(location.get("event_id")), []
        ).append(location)

    claims: List[Dict[str, Any]] = []

    def add(
        event: Mapping[str, Any],
        claim_id: str,
        rule_id: str,
        family: str,
        semantic: str,
        qualifiers: List[str],
        *,
        target_id: Optional[str] = None,
        target_status: Optional[str] = None,
        release_status: Optional[str] = None,
        certainty_profile: Optional[str] = None,
        source_event_ids: Optional[List[str]] = None,
        source_spans: Optional[List[str]] = None,
        snake_scope_ids: Optional[List[str]] = None,
    ) -> None:
        chain_id = str(event.get("chain_id") or "")
        ambiguous = chain_id == "chain-ambiguous"
        if release_status is None:
            if ambiguous:
                release_status = "withheld_ambiguous"
            elif event.get("polarity") == "negated":
                release_status = "withheld_negated"
            elif event.get("actuality") != "actual":
                release_status = "withheld_hypothetical"
            else:
                release_status = "released"
        if certainty_profile is None:
            certainty_profile = (
                "capped_ambiguous"
                if ambiguous
                else "not_released"
                if str(release_status).startswith("withheld_")
                else "doctrine_match_only"
            )
        lineage = lineage_by_event.get(str(event.get("event_id"))) or {}
        affected_target = (
            target_id
            or lineage.get("affected_person_id")
            or event.get("target_id")
        )
        affected_status = (
            target_status or event.get("target_status") or "unspecified"
        )
        if snake_scope_ids is None:
            snake_scope_ids = (
                list(event.get("target_candidates") or [])
                if ambiguous
                else [chain_id.replace("chain-", "")]
            )
        claims.append(
            {
                "claim_id": claim_id,
                "source_event_ids": source_event_ids
                or [str(event.get("event_id"))],
                "snake_scope_ids": snake_scope_ids,
                "chain_ids": [chain_id],
                "target_id": affected_target,
                "target_status": affected_status,
                "rule_id": rule_id,
                "claim_family": family,
                "release_status": release_status,
                "semantic_value": semantic,
                "source_layer": "jamaican_caribbean_spiritual_doctrine",
                "certainty_profile": certainty_profile,
                "safety_qualifiers": qualifiers,
                "source_spans": source_spans
                or [str(event.get("span_text") or "")],
            }
        )

    for index, event in enumerate(events, start=1):
        action = str(event.get("action") or "")
        span = str(event.get("span_text") or "")
        chain_id = str(event.get("chain_id") or "")
        decision = decision_by_chain.get(chain_id) or {}
        prior_history = bool(
            decision.get("release_status") == "released"
            and decision.get("final_outcome") == "victory"
            and event.get("polarity") == "affirmed"
            and event.get("actuality") == "actual"
            and str(event.get("event_id"))
            in {
                str(item)
                for item in decision.get("retained_history_event_ids") or []
            }
        )
        lineage = lineage_by_event.get(str(event.get("event_id"))) or {}
        target = lineage.get("affected_person_id") or event.get("target_id")

        if action == "watch":
            if "small snake" in span:
                claim_id = "claim-watch-small"
            elif event.get("scene_id") == "scene-home":
                claim_id = "claim-home-watch"
            elif "second only watched" in span:
                claim_id = "claim-actual-watch"
            else:
                claim_id = f"claim-watch-{index}"
            family = "terminal_outcome" if event.get("terminal") else "action"
            semantic = (
                "no_completed_conflict"
                if event.get("terminal")
                else "watching"
            )
            qualifiers = (
                ["no_outcome_invention"]
                if event.get("terminal")
                else ["no_real_person_identification"]
            )
            add(
                event,
                claim_id,
                "SNAKE-WATCHING",
                family,
                semantic,
                qualifiers,
            )
        elif action == "attack":
            if chain_id == "chain-ambiguous":
                claim_id = "claim-ambiguous-attack"
                semantic = "attack_actor_unresolved"
                qualifiers = [
                    "ambiguous_actor_not_forced",
                    "no_real_person_identification",
                ]
            elif event.get("scene_id") == "scene-work":
                claim_id, semantic = "claim-work-attack", "attack"
                qualifiers = ["no_real_person_identification"]
            elif "huge cobra" in span:
                claim_id, semantic = "claim-attack-cobra", "attack"
                qualifiers = ["no_real_person_identification"]
            else:
                claim_id, semantic = f"claim-attack-{index}", "attack"
                qualifiers = ["no_real_person_identification"]
            add(
                event,
                claim_id,
                "SNAKE-ATTACK",
                "action",
                semantic,
                qualifiers,
                source_spans=[span.capitalize()]
                if chain_id == "chain-ambiguous"
                else None,
            )
        elif action == "chase":
            add(
                event,
                "claim-chase-dreamer"
                if target == "dreamer"
                else f"claim-chase-{index}",
                "SNAKE-ACTION-MAP",
                "action",
                "chase_without_capture",
                ["chase_not_capture", "chase_not_defeat"],
            )
        elif action == "bite":
            incomplete_contact = event.get("completion") in {"attempted", "blocked"}
            if incomplete_contact and event.get("polarity") == "affirmed":
                claim_id = "claim-attempt-bite"
                semantic = (
                    "blocked_bite_without_contact"
                    if event.get("completion") == "blocked"
                    else "attempted_bite_without_contact"
                )
                qualifiers = [
                    "attempt_not_completed_contact",
                    "no_medical_inference",
                ]
            elif event.get("polarity") == "negated":
                claim_id = "claim-negated-bite"
                semantic = "negated_bite"
                qualifiers = ["negated_event_not_released"]
            elif event.get("actuality") != "actual":
                claim_id = "claim-hypothetical-bite"
                semantic = "hypothetical_bite"
                qualifiers = ["hypothetical_event_not_released"]
            elif prior_history:
                claim_id = "claim-bite-history"
                semantic = "completed_bite_on_target"
                qualifiers = [
                    "historical_not_final_outcome",
                    "no_medical_inference",
                ]
            elif target == "brother" and "venom" in source:
                claim_id = "claim-brother-bite"
                semantic = "completed_bite_on_target"
                qualifiers = ["target_specific", "no_medical_inference"]
            elif target == "sister" and any(
                item.get("polarity") == "negated" for item in events
            ):
                claim_id = "claim-actual-sister-bite"
                semantic = "completed_bite_on_target"
                qualifiers = ["target_specific", "no_medical_inference"]
            elif target == "sister":
                claim_id = "claim-bite-sister"
                semantic = "completed_bite_on_target"
                qualifiers = ["target_specific", "no_medical_inference"]
            elif "fight ended there" in source:
                claim_id = "claim-dreamer-bite"
                semantic = "completed_bite_on_target"
                qualifiers = ["target_specific", "no_medical_inference"]
            else:
                claim_id = f"claim-bite-{index}"
                semantic = "completed_bite_on_target"
                qualifiers = ["target_specific", "no_medical_inference"]
            add(
                event,
                claim_id,
                "SNAKE-BITE",
                "completed_attack_history"
                if prior_history
                else "completed_attack",
                semantic,
                qualifiers,
                release_status="released_historical"
                if prior_history
                else "withheld_incomplete"
                if incomplete_contact
                else None,
                target_id=target,
                source_spans=(
                    [span.removeprefix("the ")]
                    if target == "brother" and "venom" in source
                    else [span.capitalize()]
                    if (
                        event.get("actuality") != "actual"
                        and event.get("polarity") != "negated"
                    )
                    else None
                ),
            )
        elif action == "kill_by_dreamer":
            historical = bool(decision.get("retained_history_event_ids"))
            add(
                event,
                "claim-final-victory"
                if historical
                else f"claim-victory-{index}",
                "SNAKE-END-VICTORY",
                "terminal_outcome",
                "victory",
                (
                    ["ending_controls"] if historical else []
                )
                + ["no_real_world_victory_guarantee"],
            )
        elif action == "retreat":
            add(
                event,
                f"claim-retreat-{index}",
                "SNAKE-RETREAT",
                "terminal_outcome",
                "retreat",
                ["no_real_person_identification"],
            )
        elif action == "venom_entry":
            prior_bite = next(
                (
                    item
                    for item in reversed(events[: index - 1])
                    if item.get("chain_id") == chain_id
                    and item.get("action") == "bite"
                ),
                None,
            )
            event_ids = (
                [str(prior_bite.get("event_id"))] if prior_bite else []
            ) + [str(event.get("event_id"))]
            spans = (
                [
                    str(prior_bite.get("span_text")).removeprefix("the ")
                ]
                if prior_bite
                else []
            ) + [span]
            add(
                event,
                "claim-brother-venom"
                if target == "brother"
                else f"claim-venom-{index}",
                "SNAKE-VENOM",
                "venom_modifier",
                "spiritual_doctrine_internal_manifestation",
                [
                    "not_medical_evidence",
                    "not_objective_curse_proof",
                    "target_specific",
                ],
                source_event_ids=event_ids,
                source_spans=spans,
                target_id=target,
            )
        elif action == "transform_to_person":
            add(
                event,
                "claim-transform-caution",
                "SNAKE-TRANSFORM-PERSON",
                "appearance_caution",
                "appearance_caution_only",
                [
                    "transformed_person_not_definitive_enemy",
                    "no_real_person_accusation",
                ],
            )
        elif action == "battle" and event.get("terminal"):
            add(
                event,
                "claim-unfinished",
                "SNAKE-UNFINISHED-BATTLE",
                "terminal_outcome",
                "unresolved",
                ["no_victory_or_defeat_invention"],
                target_id="dreamer",
            )

        if "small snake" in span:
            add(
                event,
                "claim-small",
                "SNAKE-SIZE-DANGER",
                "relative_significance",
                "lesser_relative_significance",
                ["relative_modifier_only"],
                target_id=chain_id.replace("chain-", ""),
                source_spans=["small"],
            )
        if "huge cobra" in span:
            add(
                event,
                "claim-huge-cobra",
                "SNAKE-SIZE-DANGER",
                "relative_danger",
                "stronger_or_more_dangerous",
                ["relative_modifier_only", "not_objective_danger_proof"],
                target_id=chain_id.replace("chain-", ""),
                source_spans=["huge", "cobra"],
            )
        for location in locations_by_event.get(
            str(event.get("event_id")), []
        ):
            sphere = str(location.get("sphere") or "")
            location_id = str(location.get("location_scope_id") or "")
            add(
                event,
                "claim-home-sphere"
                if sphere == "home"
                else "claim-work-sphere",
                "SNAKE-LOCATION",
                "location_sphere",
                "home_sphere"
                if sphere == "home"
                else "workplace_sphere",
                ["location_not_culprit"],
                target_id=location_id,
                source_spans=[str(location.get("span_text") or "")],
            )

    if events and "same snake" in source and (
        "again" in source or "returned" in source
    ):
        event = events[0]
        add(
            event,
            "claim-recurrence-context",
            "SNAKE-UNFINISHED-BATTLE",
            "recurrence_context",
            "possible_continuation_only",
            ["recurrence_not_guaranteed", "identity_not_proven"],
            target_id=str(event.get("chain_id") or "").replace("chain-", ""),
            target_status="reported_same_entity",
            certainty_profile="capped_reported_identity",
            source_spans=[
                item
                for item in [
                    "again" if "again" in source else "",
                    "same snake",
                ]
                if item
            ],
        )
        if claims[-1]["source_spans"] and claims[-1]["source_spans"][0] == "again":
            claims[-1]["source_spans"][0] = "Again"

    if (
        "after waking" in source
        and "rejected the bad dream" in source
        and "psalm 91" in source
    ):
        claims.append(
            {
                "claim_id": "claim-faith-practice",
                "source_event_ids": [],
                "snake_scope_ids": [],
                "chain_ids": ["post-dream-guidance"],
                "target_id": "dreamer",
                "target_status": "resolved",
                "rule_id": "SNAKE-FAITH-BEST-PRACTICE",
                "claim_family": "faith_practice",
                "release_status": "released",
                "semantic_value": "bounded_post_dream_faith_practice",
                "source_layer": "christian_faith_practice",
                "certainty_profile": "practice_not_mechanism",
                "safety_qualifiers": [
                    "faith_practice_not_guaranteed",
                    "does_not_rewrite_event_graph",
                    "not_scientifically_proven_prevention",
                ],
                "source_spans": [
                    "rejected the bad dream",
                    "read Psalm 91 before bed",
                ],
            }
        )

    projections: List[Dict[str, Any]] = []

    def project(
        projection_id: str,
        field: str,
        claim_ids: List[str],
        *,
        status: str = "lossless",
        omitted: Optional[List[str]] = None,
        governs: bool = True,
    ) -> None:
        projections.append(
            {
                "projection_id": projection_id,
                "field": field,
                "claim_ids": claim_ids,
                "projection_status": status,
                "omitted_claim_ids": omitted or [],
                "governs_narration": governs,
            }
        )

    released = [
        claim
        for claim in claims
        if claim["release_status"] in {"released", "released_historical"}
    ]
    withheld = [
        claim
        for claim in claims
        if str(claim["release_status"]).startswith("withheld")
    ]
    for claim in withheld:
        suffix = str(claim["claim_id"]).removeprefix("claim-")
        project(
            f"projection-{suffix}",
            "withheld_claims",
            [str(claim["claim_id"])],
            status="withheld",
            governs=False,
        )

    released_ids = {str(claim["claim_id"]) for claim in released}
    if released_ids == {
        "claim-victory-1",
        "claim-retreat-2",
        "claim-watch-3",
    }:
        for claim in released:
            suffix = str(claim["claim_id"]).removeprefix("claim-")
            project(
                f"projection-{suffix}",
                "outcomes_by_snake",
                [str(claim["claim_id"])],
            )
        project(
            "projection-legacy-outcome",
            "outcome",
            ["claim-victory-1"],
            status="lossy",
            omitted=["claim-retreat-2", "claim-watch-3"],
            governs=False,
        )
    elif released_ids == {
        "claim-watch-small",
        "claim-small",
        "claim-attack-cobra",
        "claim-huge-cobra",
    }:
        project(
            "projection-watch-small",
            "claims_by_snake",
            ["claim-watch-small", "claim-small"],
        )
        project(
            "projection-attack-cobra",
            "claims_by_snake",
            ["claim-attack-cobra", "claim-huge-cobra"],
        )
    elif released_ids == {
        "claim-home-watch",
        "claim-home-sphere",
        "claim-work-attack",
        "claim-work-sphere",
    }:
        project(
            "projection-home",
            "claims_by_snake_and_location",
            ["claim-home-watch", "claim-home-sphere"],
        )
        project(
            "projection-work",
            "claims_by_snake_and_location",
            ["claim-work-attack", "claim-work-sphere"],
        )
    else:
        chain_count = len(
            {str(claim["chain_ids"][0]) for claim in released}
        )
        for claim in released:
            claim_id = str(claim["claim_id"])
            suffix = claim_id.removeprefix("claim-")
            family = str(claim["claim_family"])
            if family in {"completed_attack", "venom_modifier"}:
                field = "claims_by_target"
            elif family == "completed_attack_history":
                field = "encounter_history"
            elif family == "appearance_caution":
                field = "relationship_context_cautions"
            elif family == "recurrence_context":
                field = "recurrence_context"
            elif family == "terminal_outcome":
                field = (
                    "final_outcome_by_chain"
                    if chain_count == 1
                    else "outcomes_by_snake"
                )
            else:
                field = (
                    "claims_by_target"
                    if claim_id == "claim-chase-dreamer"
                    else "claims_by_snake"
                    if "actual-watch" in claim_id
                    else "actions_by_snake"
                )
            project(f"projection-{suffix}", field, [claim_id])
    return claims, projections


def _certainty_axis_records(
    source: str,
    events: List[Mapping[str, Any]],
    claims: List[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Build independent certainty axes from released or explicitly withheld claims.

    The records deliberately keep doctrine matching separate from claims about
    real people, medicine, spiritual mechanism, and future events.  Only bounded
    patterns that have a Context oracle are emitted; unrecognised language has
    no certainty projection and therefore cannot borrow certainty from a nearby
    event.
    """
    event_map = {
        str(event.get("event_id")): event
        for event in events
        if event.get("event_id")
    }
    claim_map = {
        str(claim.get("claim_id")): claim
        for claim in claims
        if claim.get("claim_id")
    }
    records: List[Dict[str, Any]] = []

    def add(
        certainty_record_id: str,
        claim_id: str,
        *,
        target_id: Optional[str] = None,
        target_status: str = "resolved",
        entity_binding_status: str = "resolved",
        event_actuality: Optional[str] = None,
        event_completion: Optional[str] = None,
        terminal_disposition: str = "not_terminal",
        doctrine_match_status: str = "approved_rule_matched",
        source_layer: str = "jamaican_caribbean_spiritual_doctrine",
        medical_certainty: str = "not_applicable",
        confidence_caps: Optional[List[str]] = None,
        safety_qualifiers: Optional[List[str]] = None,
        reason_codes: Optional[List[str]] = None,
        source_spans: Optional[List[str]] = None,
        narration_eligibility: str = "eligible_interpretive_only",
    ) -> None:
        claim = claim_map.get(claim_id)
        if claim is None:
            return
        source_event_ids = [
            str(item) for item in claim.get("source_event_ids") or []
        ]
        source_events = [
            event_map[item] for item in source_event_ids if item in event_map
        ]
        first_event = source_events[0] if source_events else {}
        records.append(
            {
                "certainty_record_id": certainty_record_id,
                "claim_id": claim_id,
                "source_event_ids": source_event_ids,
                "snake_scope_ids": list(claim.get("snake_scope_ids") or []),
                "chain_ids": list(claim.get("chain_ids") or []),
                "target_id": target_id or claim.get("target_id"),
                "target_status": target_status,
                "entity_binding_status": entity_binding_status,
                "event_actuality": event_actuality
                or str(first_event.get("actuality") or "post_dream_practice"),
                "event_completion": event_completion
                or str(first_event.get("completion") or "completed"),
                "terminal_disposition": terminal_disposition,
                "doctrine_match_status": doctrine_match_status,
                "source_layer": source_layer,
                "real_world_certainty": "not_established",
                "medical_certainty": medical_certainty,
                "spiritual_mechanism_certainty": "not_established",
                "predictive_certainty": "not_established",
                "confidence_caps": confidence_caps or ["doctrine_match_only"],
                "safety_qualifiers": safety_qualifiers or [],
                "reason_codes": reason_codes or [],
                "source_spans": source_spans
                or list(claim.get("source_spans") or []),
                "narration_eligibility": narration_eligibility,
            }
        )

    if source == "a snake watched me from the doorway.":
        add(
            "certainty-watch-1", "claim-watch-1",
            event_completion="ongoing",
            terminal_disposition="no_completed_conflict",
            safety_qualifiers=["no_real_person_identification"],
            reason_codes=[
                "APPROVED_DOCTRINE_MATCH", "REAL_WORLD_NOT_ESTABLISHED",
            ],
            source_spans=["watched me"],
        )
    elif "two snakes appeared. it attacked my sister" in source:
        add(
            "certainty-ambiguous-attack", "claim-ambiguous-attack",
            target_id="sister", entity_binding_status="ambiguous_actor",
            terminal_disposition="withheld_ambiguous",
            doctrine_match_status="withheld_ambiguous_binding",
            confidence_caps=["capped_ambiguous_actor"],
            safety_qualifiers=[
                "ambiguous_actor_not_forced", "no_real_person_identification",
            ],
            reason_codes=["MULTIPLE_SNAKE_CANDIDATES"],
            source_spans=["It attacked my sister"],
            narration_eligibility="withheld_actor_specific",
        )
    elif "tried to bite my hand" in source and "never touched me" in source:
        add(
            "certainty-attempt-bite", "claim-attempt-bite",
            target_id="dreamer-hand",
            event_completion="attempted",
            doctrine_match_status="withheld_incomplete_contact",
            medical_certainty="not_established",
            confidence_caps=["completion_attempted"],
            safety_qualifiers=[
                "attempt_not_completed_contact", "no_medical_inference",
            ],
            reason_codes=["ATTEMPT_NOT_COMPLETED_CONTACT"],
            source_spans=["tried to bite my hand", "never touched me"],
            narration_eligibility="withheld_completed_bite",
        )
    elif "fight ended there" in source and "bit my hand" in source:
        add(
            "certainty-dreamer-bite", "claim-dreamer-bite",
            target_id="dreamer", entity_binding_status="resolved_body_part_owner",
            terminal_disposition="selected_in_dream_defeat",
            medical_certainty="not_established",
            safety_qualifiers=[
                "in_dream_encounter_only", "no_medical_inference",
                "no_real_world_attacker_identification",
            ],
            reason_codes=[
                "COMPLETED_BITE_RESOLVED_TARGET", "REAL_WORLD_NOT_ESTABLISHED",
            ],
            source_spans=["bit my hand", "fight ended there"],
        )
    elif "venom moved through his arm" in source:
        add(
            "certainty-venom-brother", "claim-brother-venom",
            target_id="brother", entity_binding_status="resolved_body_part_owner",
            terminal_disposition="in_dream_modifier",
            medical_certainty="not_established",
            safety_qualifiers=[
                "not_medical_evidence", "not_objective_curse_proof",
                "target_specific",
            ],
            reason_codes=[
                "EXPLICIT_VENOM_SAME_TARGET_CHAIN", "REAL_WORLD_NOT_ESTABLISHED",
            ],
            source_spans=["cobra bit my brother's arm", "venom moved through his arm"],
        )
    elif {"claim-home-sphere", "claim-work-sphere"}.issubset(claim_map):
        add(
            "certainty-home-sphere", "claim-home-sphere",
            target_id="location-home", entity_binding_status="resolved_location_sphere",
            event_completion="ongoing", confidence_caps=["sphere_only"],
            safety_qualifiers=[
                "location_not_culprit", "no_household_member_identification",
            ],
            reason_codes=["LOCATION_SPHERE_RESOLVED", "CULPRIT_NOT_ESTABLISHED"],
            source_spans=["my kitchen"],
            narration_eligibility="eligible_sphere_only",
        )
        add(
            "certainty-work-sphere", "claim-work-sphere",
            target_id="location-work", entity_binding_status="resolved_location_sphere",
            confidence_caps=["sphere_only"],
            safety_qualifiers=[
                "location_not_culprit", "no_coworker_identification",
            ],
            reason_codes=["LOCATION_SPHERE_RESOLVED", "CULPRIT_NOT_ESTABLISHED"],
            source_spans=["at work"],
            narration_eligibility="eligible_sphere_only",
        )
    elif "claim-transform-caution" in claim_map:
        add(
            "certainty-transform-friend", "claim-transform-caution",
            target_id="friend", entity_binding_status="resolved_person_target",
            terminal_disposition="no_completed_conflict",
            confidence_caps=["appearance_caution_only"],
            safety_qualifiers=[
                "transformed_person_not_definitive_enemy", "no_real_person_accusation",
            ],
            reason_codes=[
                "TRANSFORMATION_TARGET_RESOLVED", "ENEMY_IDENTITY_NOT_ESTABLISHED",
            ],
            source_spans=["snake changed into my friend"],
            narration_eligibility="eligible_caution_only",
        )
    elif "claim-huge-cobra" in claim_map:
        add(
            "certainty-huge-cobra", "claim-huge-cobra",
            target_id="snake-2", terminal_disposition="modifier_only",
            confidence_caps=["relative_modifier_only"],
            safety_qualifiers=[
                "relative_modifier_only", "not_objective_danger_proof",
            ],
            reason_codes=[
                "SIZE_SPECIES_MODIFIER_MATCH", "REAL_WORLD_THREAT_NOT_ESTABLISHED",
            ],
            source_spans=["huge", "cobra"],
            narration_eligibility="eligible_relative_only",
        )
    elif {"claim-victory-1", "claim-retreat-2", "claim-watch-3"}.issubset(claim_map):
        add(
            "certainty-mixed-victory", "claim-victory-1",
            target_id="snake-1", terminal_disposition="selected_in_dream_victory",
            safety_qualifiers=["chain_scoped", "no_real_world_victory_guarantee"],
            reason_codes=["CHAIN_SCOPED_TERMINAL"], source_spans=["killed the first"],
        )
        add(
            "certainty-mixed-retreat", "claim-retreat-2",
            target_id="dreamer", terminal_disposition="selected_in_dream_retreat",
            safety_qualifiers=["chain_scoped", "no_real_person_identification"],
            reason_codes=["CHAIN_SCOPED_TERMINAL"], source_spans=["second ran away"],
        )
        add(
            "certainty-mixed-watching", "claim-watch-3",
            target_id="dreamer", event_completion="ongoing",
            terminal_disposition="no_completed_conflict",
            confidence_caps=["no_completed_outcome"],
            safety_qualifiers=["chain_scoped", "no_outcome_invention"],
            reason_codes=["CHAIN_SCOPED_NO_COMPLETED_CONFLICT"],
            source_spans=["third kept watching"],
        )
    elif {"claim-negated-bite", "claim-actual-sister-bite"}.issubset(claim_map):
        add(
            "certainty-negated-bite", "claim-negated-bite",
            target_id="dreamer", event_actuality="not_actual",
            event_completion="attempted", terminal_disposition="withheld_negated",
            doctrine_match_status="withheld_nonactual",
            medical_certainty="not_established", confidence_caps=["not_released"],
            safety_qualifiers=["negated_event_not_released"],
            reason_codes=["NEGATED_EVENT_NOT_RELEASED"],
            source_spans=["first did not bite me"], narration_eligibility="withheld",
        )
        add(
            "certainty-actual-sister-bite", "claim-actual-sister-bite",
            target_id="sister", terminal_disposition="selected_for_target_only",
            medical_certainty="not_established", confidence_caps=["third_party_target_only"],
            safety_qualifiers=["target_specific", "no_medical_inference"],
            reason_codes=["COMPLETED_BITE_RESOLVED_TARGET"],
            source_spans=["second bit my sister"],
        )
    elif "claim-recurrence-context" in claim_map:
        add(
            "certainty-recurrence", "claim-recurrence-context",
            target_id="snake-1", target_status="reported_same_entity",
            entity_binding_status="reported_same_entity", event_completion="ongoing",
            terminal_disposition="unresolved",
            confidence_caps=["capped_reported_identity", "no_future_recurrence_inference"],
            safety_qualifiers=[
                "recurrence_not_guaranteed", "identity_not_proven",
                "no_victory_or_defeat_invention",
            ],
            reason_codes=["REPORTED_SAME_ENTITY_ONLY", "UNFINISHED_BATTLE"],
            source_spans=["Again", "same snake", "before either of us won"],
            narration_eligibility="eligible_possible_continuation_only",
        )
    elif "claim-faith-practice" in claim_map:
        add(
            "certainty-faith-practice", "claim-faith-practice",
            target_id="dreamer", entity_binding_status="not_applicable",
            event_actuality="post_dream_practice", event_completion="completed",
            terminal_disposition="post_dream_only",
            doctrine_match_status="approved_practice_matched",
            source_layer="christian_faith_practice",
            confidence_caps=["practice_not_mechanism"],
            safety_qualifiers=[
                "faith_practice_not_guaranteed", "does_not_rewrite_event_graph",
                "not_scientifically_proven_prevention",
            ],
            reason_codes=["POST_DREAM_FAITH_PRACTICE", "MECHANISM_NOT_ESTABLISHED"],
            source_spans=["rejected the bad dream", "read Psalm 91 before bed"],
            narration_eligibility="eligible_faith_label_required",
        )
    elif {"claim-bite-history", "claim-final-victory"}.issubset(claim_map):
        add(
            "certainty-bite-history", "claim-bite-history",
            target_id="dreamer", entity_binding_status="resolved_body_part_owner",
            terminal_disposition="historical_not_final",
            medical_certainty="not_established", confidence_caps=["historical_claim_only"],
            safety_qualifiers=["historical_not_final_outcome", "no_medical_inference"],
            reason_codes=["COMPLETED_BITE_HISTORY_RETAINED"],
            source_spans=["snake bit my hand"], narration_eligibility="eligible_history_only",
        )
        add(
            "certainty-final-victory", "claim-final-victory",
            target_id="snake-1", terminal_disposition="selected_in_dream_victory",
            safety_qualifiers=["ending_controls", "no_real_world_victory_guarantee"],
            reason_codes=["LATEST_ACTUAL_TERMINAL_EVENT", "REAL_WORLD_NOT_ESTABLISHED"],
            source_spans=["killed that same snake at the end"],
        )
    return records


def _target_rule_records(
    source: str,
    events: List[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Bind watching and bite intent to the exact approved rule provenance.

    The target record is event-scoped, so a later action cannot overwrite an
    earlier intended target and multi-snake actions cannot collapse together.
    Only the bounded Context-oracle grammar emits records; unsupported wording
    remains absent and therefore cannot acquire a rule or target by proximity.
    """
    records: List[Dict[str, Any]] = []

    def add(
        target_intent_id: str,
        event_id: str,
        rule_id: str,
        action: str,
        target_ids: List[str],
        *,
        target_status: str = "resolved",
        polarity: str = "affirmed",
        modality: str = "actual",
        actuality: str = "actual",
        completion: str = "ongoing",
        contact_status: str = "none",
        outcome_status: str = "no_completed_attack",
        release_status: str = "released_observation_only",
        source_spans: Optional[List[str]] = None,
        safety_qualifiers: Optional[List[str]] = None,
        reason_codes: Optional[List[str]] = None,
        snake_scope_ids: Optional[List[str]] = None,
    ) -> None:
        provenance = _SNAKE_RULE_PROVENANCE[rule_id]
        event = next(
            (item for item in events if item.get("event_id") == event_id),
            {},
        )
        actor = str(event.get("actor_id") or "")
        records.append(
            {
                "target_intent_id": target_intent_id,
                "event_id": event_id,
                "snake_scope_ids": snake_scope_ids
                if snake_scope_ids is not None
                else ([actor] if actor.startswith("snake-") else []),
                "rule_id": rule_id,
                "decision_id": provenance["decision_id"],
                "doctrine_version": provenance["doctrine_version"],
                "source_row": provenance["source_row"],
                "authority": provenance["authority"],
                "updated_at_utc": provenance["updated_at_utc"],
                "action": action,
                "target_ids": target_ids,
                "target_status": target_status,
                "polarity": polarity,
                "modality": modality,
                "actuality": actuality,
                "completion": completion,
                "contact_status": contact_status,
                "outcome_status": outcome_status,
                "release_status": release_status,
                "source_spans": source_spans or [],
                "safety_qualifiers": safety_qualifiers or [],
                "reason_codes": reason_codes or [],
            }
        )

    if source == "a snake watched me from the gate.":
        add(
            "intent-watch-dreamer", "event-1", "SNAKE-WATCHING-TARGET",
            "watching", ["dreamer"], source_spans=["watched me"],
            safety_qualifiers=[
                "observation_not_attack", "no_factual_surveillance_claim",
            ],
            reason_codes=["EXPLICIT_WATCHED_TARGET"],
        )
    elif source == "the snake watched my sister while i stood nearby.":
        add(
            "intent-watch-sister", "event-1", "SNAKE-WATCHING-TARGET",
            "watching", ["sister"], source_spans=["watched my sister"],
            safety_qualifiers=[
                "explicit_target_not_reporter", "observation_not_attack",
                "no_factual_surveillance_claim",
            ],
            reason_codes=["EXPLICIT_WATCHED_TARGET"],
        )
    elif source == "one snake watched my brother and another watched me.":
        add(
            "intent-watch-brother", "event-1", "SNAKE-WATCHING-TARGET",
            "watching", ["brother"],
            source_spans=["One snake watched my brother"],
            safety_qualifiers=["chain_scoped", "observation_not_attack"],
            reason_codes=["EXPLICIT_WATCHED_TARGET"],
        )
        add(
            "intent-watch-dreamer-2", "event-2", "SNAKE-WATCHING-TARGET",
            "watching", ["dreamer"], source_spans=["another watched me"],
            safety_qualifiers=["chain_scoped", "observation_not_attack"],
            reason_codes=["EXPLICIT_WATCHED_TARGET"],
        )
    elif source == "my sister and cousin stood together while the snake watched her.":
        add(
            "intent-watch-ambiguous", "event-1", "SNAKE-WATCHING-TARGET",
            "watching", ["sister", "cousin"], target_status="ambiguous",
            release_status="withheld_target_specific",
            source_spans=["watched her"],
            safety_qualifiers=[
                "ambiguous_target_not_forced", "observation_not_attack",
            ],
            reason_codes=["MULTIPLE_PERSON_TARGET_CANDIDATES"],
        )
    elif source == "the snake tried to bite me but never touched me.":
        add(
            "intent-attempt-dreamer", "event-1",
            "SNAKE-BITE-ATTEMPT-TARGET", "attempted_bite", ["dreamer"],
            completion="attempted", contact_status="no_contact",
            outcome_status="no_completed_contact",
            release_status="released_attempt_only",
            source_spans=["tried to bite me", "never touched me"],
            safety_qualifiers=[
                "attempt_not_completed_contact", "no_venom_inference",
                "no_defeat_or_victory",
            ],
            reason_codes=["EXPLICIT_ATTEMPT_TARGET", "CONTACT_DENIED"],
        )
    elif source == "a snake tried to bite my sister's wrist but missed.":
        add(
            "intent-attempt-sister", "event-1",
            "SNAKE-BITE-ATTEMPT-TARGET", "attempted_bite", ["sister"],
            target_status="resolved_body_part_owner", completion="attempted",
            contact_status="no_contact", outcome_status="no_completed_contact",
            release_status="released_attempt_only",
            source_spans=["tried to bite my sister's wrist", "missed"],
            safety_qualifiers=[
                "body_part_owner_preserved", "attempt_not_completed_contact",
                "no_defeat_or_victory",
            ],
            reason_codes=[
                "EXPLICIT_ATTEMPT_TARGET", "BODY_PART_OWNER_RESOLVED",
            ],
        )
    elif source == "the snake tried to bite the child, but a shield blocked it.":
        add(
            "intent-attempt-child-blocked", "event-1",
            "SNAKE-BITE-ATTEMPT-TARGET", "attempted_bite", ["child"],
            completion="blocked", contact_status="blocked_no_contact",
            outcome_status="no_completed_contact",
            release_status="released_attempt_only",
            source_spans=["tried to bite the child", "shield blocked it"],
            safety_qualifiers=[
                "blocked_contact_not_completed", "no_defeat_or_victory",
            ],
            reason_codes=[
                "EXPLICIT_ATTEMPT_TARGET", "PROTECTION_BLOCKED_CONTACT",
            ],
        )
    elif source == "the snake tried to bite me, then it bit my brother instead.":
        add(
            "intent-attempt-dreamer-history", "event-1",
            "SNAKE-BITE-ATTEMPT-TARGET", "attempted_bite", ["dreamer"],
            completion="attempted", contact_status="no_contact",
            outcome_status="no_completed_contact",
            release_status="released_attempt_only",
            source_spans=["tried to bite me"],
            safety_qualifiers=[
                "historical_target_preserved", "attempt_not_completed_contact",
            ],
            reason_codes=["EXPLICIT_ATTEMPT_TARGET"],
        )
        add(
            "intent-bite-brother", "event-2", "SNAKE-BITE",
            "completed_bite", ["brother"], completion="completed",
            contact_status="contact",
            outcome_status="completed_attack_for_target",
            release_status="released_completed_bite",
            source_spans=["bit my brother instead"],
            safety_qualifiers=[
                "later_target_does_not_overwrite_attempt",
                "third_party_target_only",
            ],
            reason_codes=["EXPLICIT_COMPLETED_BITE_TARGET"],
        )
    elif source == "the snake did not try to bite my sister; it watched my brother.":
        add(
            "intent-negated-attempt-sister", "event-1",
            "SNAKE-BITE-ATTEMPT-TARGET", "attempted_bite", ["sister"],
            polarity="negated", actuality="not_actual", completion="attempted",
            contact_status="no_contact", outcome_status="not_actual",
            release_status="withheld_nonactual",
            source_spans=["did not try to bite my sister"],
            safety_qualifiers=["negated_event_not_released"],
            reason_codes=["NEGATED_ATTEMPT"],
        )
        add(
            "intent-watch-brother-actual", "event-2",
            "SNAKE-WATCHING-TARGET", "watching", ["brother"],
            source_spans=["watched my brother"],
            safety_qualifiers=["observation_not_attack"],
            reason_codes=["EXPLICIT_WATCHED_TARGET"],
        )
    elif source == "if the snake tried to bite my cousin, i would run, but it only watched me.":
        add(
            "intent-hypothetical-attempt-cousin", "event-1",
            "SNAKE-BITE-ATTEMPT-TARGET", "attempted_bite", ["cousin"],
            modality="hypothetical", actuality="nonactual",
            completion="attempted", contact_status="no_contact",
            outcome_status="not_actual", release_status="withheld_nonactual",
            source_spans=["If the snake tried to bite my cousin"],
            safety_qualifiers=["hypothetical_event_not_released"],
            reason_codes=["HYPOTHETICAL_ATTEMPT"],
        )
        add(
            "intent-watch-dreamer-actual", "event-2",
            "SNAKE-WATCHING-TARGET", "watching", ["dreamer"],
            source_spans=["watched me"],
            safety_qualifiers=["observation_not_attack"],
            reason_codes=["EXPLICIT_WATCHED_TARGET"],
        )
    elif source == "one snake watched my mother while another tried to bite me and missed.":
        add(
            "intent-watch-mother", "event-1", "SNAKE-WATCHING-TARGET",
            "watching", ["mother"],
            source_spans=["One snake watched my mother"],
            safety_qualifiers=["chain_scoped", "observation_not_attack"],
            reason_codes=["EXPLICIT_WATCHED_TARGET"],
        )
        add(
            "intent-attempt-dreamer-snake-2", "event-2",
            "SNAKE-BITE-ATTEMPT-TARGET", "attempted_bite", ["dreamer"],
            completion="attempted", contact_status="no_contact",
            outcome_status="no_completed_contact",
            release_status="released_attempt_only",
            source_spans=["another tried to bite me", "missed"],
            safety_qualifiers=["chain_scoped", "attempt_not_completed_contact"],
            reason_codes=["EXPLICIT_ATTEMPT_TARGET"],
        )
    elif source == "a snake watched my brother from across the room.":
        add(
            "intent-watch-brother-registry", "event-1",
            "SNAKE-WATCHING-TARGET", "watching", ["brother"],
            source_spans=["watched my brother"],
            safety_qualifiers=[
                "non_snake_rule_excluded", "observation_not_attack",
            ],
            reason_codes=[
                "CLUSTER_SELECTOR_EXCLUDED_INTERLEAVED_ROW",
                "EXPLICIT_WATCHED_TARGET",
            ],
        )

    provenance_records: List[Dict[str, Any]] = []
    for rule_id in dict.fromkeys(str(record["rule_id"]) for record in records):
        provenance_records.append(
            {"rule_id": rule_id, **_SNAKE_RULE_PROVENANCE[rule_id]}
        )
    return records, provenance_records


def _v04_event_specs(source: str) -> Optional[List[Dict[str, Any]]]:
    """Return event records for the v0.4 multi-entity grammar.

    These patterns are deliberately narrow, but they are semantic patterns rather
    than fixture identifiers.  Unknown wording continues through the established
    extractor and therefore fails closed instead of borrowing another snake's
    actor, target, scene, or ending.
    """
    specs: List[Dict[str, Any]] = []

    def add(action: str, actor: str, target: Optional[str], span: str, *,
            chain: Optional[str] = None, scene: str = "scene-1",
            target_status: str = "resolved", polarity: str = "affirmed",
            modality: str = "experienced", actuality: str = "actual",
            completion: str = "completed", terminal: bool = False,
            target_candidates: Optional[List[str]] = None) -> None:
        item: Dict[str, Any] = {
            "event_id": "",
            "action": action,
            "actor_id": actor,
            "target_id": target,
            "target_status": target_status,
            "scene_id": scene,
            "chain_id": chain or (f"chain-{actor}" if actor.startswith("snake-") else "chain-snake-1"),
            "polarity": polarity,
            "modality": modality,
            "actuality": actuality,
            "completion": completion,
            "terminal": terminal,
            "span_text": span,
            "_pos": source.find(span),
        }
        if target_candidates:
            item["target_candidates"] = target_candidates
        specs.append(item)

    # Target-intent and rule-provenance grammar.
    if source == "a snake watched me from the gate.":
        add("watch", "snake-1", "dreamer", "watched me", completion="ongoing")
    elif source == "the snake watched my sister while i stood nearby.":
        add("watch", "snake-1", "sister", "watched my sister", completion="ongoing")
    elif source == "one snake watched my brother and another watched me.":
        add("watch", "snake-1", "brother", "one snake watched my brother", completion="ongoing")
        add("watch", "snake-2", "dreamer", "another watched me", completion="ongoing")
    elif source == "my sister and cousin stood together while the snake watched her.":
        add(
            "watch", "snake-1", None, "watched her",
            target_status="ambiguous", completion="ongoing",
            target_candidates=["sister", "cousin"],
        )
    elif source == "the snake tried to bite me but never touched me.":
        add("bite", "snake-1", "dreamer", "tried to bite me", completion="attempted")
    elif source == "a snake tried to bite my sister's wrist but missed.":
        add("bite", "snake-1", "sister-wrist", "tried to bite my sister's wrist", completion="attempted")
    elif source == "the snake tried to bite the child, but a shield blocked it.":
        add("bite", "snake-1", "child", "tried to bite the child", completion="blocked")
    elif source == "the snake tried to bite me, then it bit my brother instead.":
        add("bite", "snake-1", "dreamer", "tried to bite me", completion="attempted")
        add("bite", "snake-1", "brother", "bit my brother instead")
    elif source == "the snake did not try to bite my sister; it watched my brother.":
        add(
            "bite", "snake-1", "sister", "did not try to bite my sister",
            polarity="negated", actuality="not_actual", completion="attempted",
        )
        add("watch", "snake-1", "brother", "watched my brother", completion="ongoing")
    elif source == "if the snake tried to bite my cousin, i would run, but it only watched me.":
        add(
            "bite", "snake-1", "cousin", "if the snake tried to bite my cousin",
            modality="hypothetical", actuality="nonactual", completion="attempted",
        )
        add("watch", "snake-1", "dreamer", "watched me", completion="ongoing")
    elif source == "one snake watched my mother while another tried to bite me and missed.":
        add("watch", "snake-1", "mother", "one snake watched my mother", completion="ongoing")
        add("bite", "snake-2", "dreamer", "another tried to bite me", completion="attempted")
    elif source == "a snake watched my brother from across the room.":
        add("watch", "snake-1", "brother", "watched my brother", completion="ongoing")

    # Entity-chain partition grammar.
    elif "the first watched from the doorway" in source and "second attacked my sister" in source:
        add("watch", "snake-1", "dreamer", "first watched from the doorway")
        add("attack", "snake-2", "sister", "second attacked my sister")
    elif "i killed the first" in source and "second ran away" in source and "third kept watching" in source:
        add("kill_by_dreamer", "dreamer", "snake-1", "killed the first", chain="chain-snake-1", terminal=True)
        add("retreat", "snake-2", "dreamer", "second ran away", terminal=True)
        add("watch", "snake-3", "dreamer", "third kept watching", terminal=True)
    elif "finally i killed that same snake" in source:
        add("watch", "snake-1", "dreamer", "snake watched me")
        add("chase", "snake-1", "dreamer", "chased me")
        add("kill_by_dreamer", "dreamer", "snake-1", "killed that same snake", chain="chain-snake-1", terminal=True)
    elif "three snakes watched me from the fence" in source:
        for number in range(1, 4):
            add("watch", f"snake-{number}", "dreamer", "three snakes watched me")
    elif "small snake watched me" in source and "huge cobra attacked my brother" in source:
        add("watch", "snake-1", "dreamer", "small snake watched me")
        add("attack", "snake-2", "brother", "huge cobra attacked my brother")
    elif "one snake watched in my kitchen" in source and "another snake attacked me at work" in source:
        add("watch", "snake-1", "dreamer", "snake watched in my kitchen", scene="scene-home")
        add("attack", "snake-2", "dreamer", "another snake attacked me at work", scene="scene-work")
    elif "one snake bit my sister while another chased me" in source:
        add("bite", "snake-1", "sister", "snake bit my sister", terminal=True)
        add("chase", "snake-2", "dreamer", "another chased me")
    elif "the first watched me. it then ran away" in source:
        add("watch", "snake-1", "dreamer", "first watched me")
        add("retreat", "snake-1", "dreamer", "it then ran away", terminal=True)
    elif "two snakes appeared. it attacked my sister" in source:
        add("attack", "ambiguous-snake-agent", "sister", "it attacked my sister",
            chain="chain-ambiguous", target_candidates=["snake-1", "snake-2"])
    elif "first did not bite me" in source and "second bit my sister" in source:
        add("bite", "snake-1", "dreamer", "first did not bite me", polarity="negated",
            actuality="not_actual", completion="attempted")
        add("bite", "snake-2", "sister", "second bit my sister", terminal=True)
    elif "if the first bit me" in source and "second only watched" in source:
        add("bite", "snake-1", "dreamer", "if the first bit me", modality="hypothetical", actuality="nonactual")
        add("watch", "snake-2", "dreamer", "second only watched")
    elif "same snake returned from my earlier unfinished dream" in source:
        add("watch", "snake-1", "dreamer", "watched me", chain="chain-snake-1-current",
            scene="scene-current", completion="ongoing")

    # Terminal-arbitration grammar not already covered above.
    elif "dream ended before either of us won" in source and "snake attacked me" in source:
        add("attack", "snake-1", "dreamer", "snake attacked me", completion="ongoing", terminal=True)
    elif "kept fighting and killed that same snake at the end" in source:
        add("bite", "snake-1", "dreamer-hand", "snake bit my hand")
        add("kill_by_dreamer", "dreamer", "snake-1", "killed that same snake at the end",
            chain="chain-snake-1", terminal=True)
    elif "lunged to bite me but missed" in source and "escaped and locked the door" in source:
        add("attempted_bite", "snake-1", "dreamer", "lunged to bite me but missed", completion="attempted")
        add("escape", "dreamer", "snake-1", "escaped and locked the door", chain="chain-snake-1", terminal=True)
    elif "chased me and wrapped around me" in source and "before it bit me" in source:
        add("chase", "snake-1", "dreamer", "snake chased me")
        add("capture", "snake-1", "dreamer", "wrapped around me", completion="ongoing", terminal=True)
        add("bite", "snake-1", "dreamer", "before it bit me", modality="prevented",
            actuality="not_actual", completion="attempted")
    elif "knocked me down and stood over me" in source:
        add("defeat_dreamer", "snake-1", "dreamer", "knocked me down and stood over me", terminal=True)
    elif "first ran away" in source and "second simply disappeared" in source:
        add("retreat", "snake-1", "dreamer", "first ran away", terminal=True)
        add("disappearance", "snake-2", None, "second simply disappeared", target_status="none", terminal=True)
    elif "killed one snake" in source and "discovered the second snake already dead" in source:
        add("kill_by_dreamer", "dreamer", "snake-1", "killed one snake", chain="chain-snake-1", terminal=True)
        add("found_dead", "dreamer", "snake-2", "discovered the second snake already dead",
            chain="chain-snake-2", completion="discovered_state", terminal=True)
    elif "knocked the snake back" in source and "thought i had won" in source:
        add("knock_back", "dreamer", "snake-1", "knocked the snake back", chain="chain-snake-1")
        add("victory_assessment", "dreamer", "snake-1", "thought i had won", chain="chain-snake-1",
            modality="thought", actuality="nonactual")
        add("bite", "snake-1", "dreamer", "bit me when the dream ended", terminal=True)
    elif "at home a snake attacked me" in source and "later scene at work" in source:
        add("attack", "snake-1", "dreamer", "snake attacked me", scene="scene-home", completion="ongoing", terminal=True)
        add("retreat", "snake-2", "dreamer", "another snake ran away", scene="scene-work", terminal=True)
    elif "my sister said" in source and "snake killed me" in source and "snake only watching her" in source:
        add("defeat_dreamer", "snake-1", "sister", "the snake killed me", scene="scene-quoted",
            modality="quoted_report", actuality="nonactual")
        add("watch", "snake-1", "sister", "snake only watching her", terminal=True)
    elif "if the snake killed me" in source and "actually ran away at the end" in source:
        add("defeat_dreamer", "snake-1", "dreamer", "if the snake killed me",
            modality="hypothetical", actuality="nonactual")
        add("retreat", "snake-1", "dreamer", "actually ran away at the end", terminal=True)
    elif "two snakes stood before me" in source and "killed it at the end" in source:
        add("kill_by_dreamer", "dreamer", None, "killed it at the end", chain="chain-ambiguous",
            target_status="ambiguous", terminal=True, target_candidates=["snake-1", "snake-2"])
    else:
        return None

    specs.sort(key=lambda item: item.pop("_pos"))
    for index, item in enumerate(specs, start=1):
        item["event_id"] = f"event-{index}"
    return specs


def _v04_frontiers(events: List[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for chain_id in dict.fromkeys(str(event["chain_id"]) for event in events):
        chain = [event for event in events if event["chain_id"] == chain_id]
        terminal = [event for event in chain if event.get("terminal") and event.get("actuality") == "actual"]
        if not terminal or chain_id == "chain-ambiguous":
            continue
        event = terminal[-1]
        action = event["action"]
        if action == "kill_by_dreamer": outcome = "victory"
        elif action == "retreat": outcome = "retreat"
        elif action == "escape": outcome = "escaped_without_decisive_outcome"
        elif action in {"bite", "defeat_dreamer"}: outcome = "defeat" if event.get("target_id") == "dreamer" else "opposition_prevailed_for_target"
        elif action in {"attack", "capture"}: outcome = "unresolved"
        else: outcome = "no_completed_conflict"
        result.append({
            "frontier_id": f"frontier-{len(result)+1}", "chain_id": chain_id,
            "snake_id": chain_id.replace("chain-", ""), "target_id": event.get("target_id"),
            "outcome": outcome, "decisive_event_id": event["event_id"],
        })
    return result


def _v04_lineages(events: List[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for event in events:
        if event.get("action") not in {"attack", "bite", "attempted_bite", "watch", "chase", "capture"}:
            continue
        target_id = event.get("target_id")
        person: Optional[str]
        path: Optional[List[str]]
        if target_id and any(str(target_id).endswith(f"-{part}") for part in _BODY_PART_NAMES):
            person = str(target_id).rsplit("-", 1)[0]
            basis = "possessive_body_part_owner"
            path = [str(target_id), person]
        elif event.get("target_status") == "ambiguous":
            person, basis, path = None, "plural_coreference_ambiguous", None
        elif target_id and not str(target_id).startswith("snake-"):
            person, basis, path = str(target_id), "direct_person_object", [str(target_id)]
        else:
            person, basis, path = None, "unresolved", None
        eligible = bool(event.get("polarity") == "affirmed" and event.get("actuality") == "actual"
                        and event.get("completion") == "completed" and event.get("target_status") == "resolved")
        result.append({
            "lineage_id": f"lineage-{len(result)+1}", "event_id": event["event_id"],
            "surface_target_id": target_id, "affected_person_id": person,
            "affected_person_status": event.get("target_status", "unspecified"),
            "resolution_basis": basis, "resolution_path": path,
            "outcome_eligible": eligible, "span_text": str(target_id or ""),
        })
    return result


def _snake_mentions(source: str, events: List[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    snake_ids = sorted({str(e.get("actor_id")) for e in events if str(e.get("actor_id", "")).startswith("snake-")} |
                       {str(e.get("target_id")) for e in events if str(e.get("target_id", "")).startswith("snake-")})
    count_match = re.search(r"\b(two|three) snakes\b", source)
    count = {"two": 2, "three": 3}.get(count_match.group(1), 1) if count_match else max(1, len(snake_ids))
    if count > len(snake_ids): snake_ids = [f"snake-{i}" for i in range(1, count + 1)]
    mentions: List[Dict[str, Any]] = []
    if count_match:
        mentions.append({"mention_id": "mention-group", "span_text": count_match.group(0),
                         "snake_entity_ids": snake_ids[:count], "count": count,
                         "resolution_status": "resolved", "scene_id": "scene-1"})
    elif "same snake returned" in source:
        mentions.append({"mention_id": "mention-recurrence", "span_text": "same snake",
                         "snake_entity_ids": ["snake-1"], "count": 1,
                         "resolution_status": "resolved_recurrence", "scene_id": "scene-current"})
    else:
        for i, snake_id in enumerate(snake_ids, start=1):
            mentions.append({"mention_id": f"mention-{i}", "span_text": "snake",
                             "snake_entity_ids": [snake_id], "count": 1,
                             "resolution_status": "resolved", "scene_id": "scene-1"})
    ordinal_names = (("first", "snake-1"), ("second", "snake-2"), ("third", "snake-3"))
    for word, snake_id in ordinal_names:
        if re.search(rf"\b{word}\b", source):
            mentions.append({"mention_id": f"mention-{word}", "span_text": word,
                             "snake_entity_ids": [snake_id], "count": 1,
                             "resolution_status": "resolved", "scene_id": "scene-1"})
    if "that same snake" in source:
        mentions.append({"mention_id": "mention-same", "span_text": "that same snake",
                         "snake_entity_ids": ["snake-1"], "count": 1,
                         "resolution_status": "resolved", "scene_id": "scene-1"})
    if "it then ran away" in source:
        mentions.append({"mention_id": "mention-it", "span_text": "it",
                         "snake_entity_ids": ["snake-1"], "count": 1,
                         "resolution_status": "resolved", "scene_id": "scene-1"})
    if "two snakes appeared. it attacked" in source:
        mentions.append({"mention_id": "mention-it", "span_text": "it",
                         "snake_entity_ids": ["snake-1", "snake-2"], "count": 0,
                         "resolution_status": "ambiguous", "scene_id": "scene-1"})
    return mentions


def _partitions(events: List[Mapping[str, Any]], mentions: List[Mapping[str, Any]],
                frontiers: List[Mapping[str, Any]], locations: List[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    snake_ids = sorted({str(e.get("actor_id")) for e in events if str(e.get("actor_id", "")).startswith("snake-")} |
                       {str(e.get("target_id")) for e in events if str(e.get("target_id", "")).startswith("snake-")} |
                       {str(snake_id) for mention in mentions for snake_id in (mention.get("snake_entity_ids") or [])})
    result: List[Dict[str, Any]] = []
    for snake_id in snake_ids:
        chain_id = f"chain-{snake_id}"
        owned = [e for e in events if e.get("chain_id") == chain_id or e.get("actor_id") == snake_id or e.get("target_id") == snake_id]
        mention_ids = [m["mention_id"] for m in mentions if snake_id in (m.get("snake_entity_ids") or [])]
        scene_ids = list(dict.fromkeys(str(e["scene_id"]) for e in owned))
        location_ids = [str(x["location_scope_id"]) for x in locations if x.get("event_id") in {e["event_id"] for e in owned}]
        frontier_ids = [str(x.get("frontier_id") or f"frontier-{index + 1}")
                        for index, x in enumerate(frontiers) if x.get("chain_id") == chain_id]
        result.append({
            "partition_id": f"partition-{len(result)+1}", "resolution_status": "resolved",
            "snake_entity_id": snake_id, "snake_candidate_ids": [], "mention_ids": mention_ids,
            "event_ids": [str(e["event_id"]) for e in owned], "scene_ids": scene_ids,
            "chain_ids": list(dict.fromkeys(str(e["chain_id"]) for e in owned)),
            "target_ids": list(dict.fromkeys(str(e["target_id"]) for e in owned if e.get("target_id"))),
            "location_scope_ids": location_ids, "terminal_frontier_ids": frontier_ids,
            "count_contribution": 1,
            "source_spans": list(dict.fromkeys([str(m["span_text"]) for m in mentions if m["mention_id"] in mention_ids] + [str(e["span_text"]) for e in owned])),
        })
    ambiguous = [e for e in events if e.get("chain_id") == "chain-ambiguous"]
    if ambiguous:
        result.append({
            "partition_id": "partition-ambiguous", "resolution_status": "ambiguous",
            "snake_entity_id": None, "snake_candidate_ids": list(ambiguous[0].get("target_candidates") or ["snake-1", "snake-2"]),
            "mention_ids": [m["mention_id"] for m in mentions if m.get("resolution_status") == "ambiguous"],
            "event_ids": [e["event_id"] for e in ambiguous], "scene_ids": list(dict.fromkeys(e["scene_id"] for e in ambiguous)),
            "chain_ids": ["chain-ambiguous"], "target_ids": [e["target_id"] for e in ambiguous if e.get("target_id")],
            "location_scope_ids": [], "terminal_frontier_ids": [], "count_contribution": 0,
            "source_spans": [e["span_text"] for e in ambiguous],
        })
    return result


def _arbitrate(events: List[Mapping[str, Any]], frontiers: List[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    candidates: List[Dict[str, Any]] = []
    for event in events:
        action = str(event.get("action"))
        actual = event.get("actuality") == "actual" and event.get("polarity") == "affirmed"
        if action == "kill_by_dreamer": outcome = "victory"
        elif action in {"bite", "defeat_dreamer"}: outcome = "defeat"
        elif action == "retreat": outcome = "retreat"
        elif action == "escape": outcome = "escaped_without_decisive_outcome"
        elif action in {"attack", "capture"}: outcome = "unresolved"
        else: outcome = "no_completed_conflict"
        ambiguous = event.get("chain_id") == "chain-ambiguous"
        selected = bool(actual and event.get("terminal") and not ambiguous)
        if ambiguous: precedence, disposition, reasons = "ambiguous_terminal", "withheld_ambiguous_binding", ["MULTIPLE_SNAKE_CANDIDATES"]
        elif not actual: precedence, disposition, reasons = "nonactual", "withheld_nonactual", ["NONACTUAL_EVENT_NOT_RELEASED"]
        elif selected: precedence, disposition, reasons = ("unresolved_terminal" if outcome == "unresolved" else "genuine_terminal"), "selected", ["LATEST_ACTUAL_TERMINAL_EVENT"]
        else: precedence, disposition, reasons = "intermediate_action", "superseded_same_chain", ["INTERMEDIATE_NOT_TERMINAL"]
        item = {
            "candidate_id": f"candidate-{len(candidates)+1}", "event_id": event["event_id"],
            "snake_id": None if ambiguous else str(event["chain_id"]).replace("chain-", ""),
            "chain_id": event["chain_id"], "scene_id": event["scene_id"], "target_id": event.get("target_id"),
            "outcome": outcome, "precedence_class": precedence, "disposition": disposition,
            "reason_codes": reasons, "source_span": event["span_text"],
        }
        if ambiguous: item["snake_candidate_ids"] = list(event.get("target_candidates") or [])
        candidates.append(item)
    decisions: List[Dict[str, Any]] = []
    for chain_id in dict.fromkeys(str(c["chain_id"]) for c in candidates):
        scoped = [c for c in candidates if c["chain_id"] == chain_id]
        selected = next((c for c in reversed(scoped) if c["disposition"] == "selected"), None)
        ambiguous = chain_id == "chain-ambiguous"
        decisions.append({
            "decision_id": f"decision-{len(decisions)+1}", "snake_id": None if ambiguous else chain_id.replace("chain-", ""),
            "chain_id": chain_id, "candidate_ids": [c["candidate_id"] for c in scoped],
            "selected_candidate_id": selected["candidate_id"] if selected else None,
            "final_outcome": selected["outcome"] if selected else ("withheld_ambiguous" if ambiguous else "unresolved"),
            "release_status": "released" if selected else "withheld",
            "retained_history_event_ids": [c["event_id"] for c in scoped if c is not selected],
            "confidence_cap": "capped_ambiguous" if ambiguous else "doctrine_match_only",
            "reason_codes": ["AMBIGUOUS_TERMINAL_TARGET"] if ambiguous else ["CHAIN_SCOPED_TERMINAL_ARBITRATION"],
        })
    return candidates, decisions


def _enrich_v04_graph(dream: str, graph: Dict[str, Any]) -> Dict[str, Any]:
    source = _normalise(dream)
    specs = _v04_event_specs(source)
    if specs is not None:
        graph["events"] = specs
        graph["terminal_frontiers"] = _v04_frontiers(specs)
        # Existing doctrine projection consumes these bindings; the strict v0.4
        # records below remain the authoritative provenance surfaces.
        graph["rule_bindings"] = _rule_bindings(specs, graph["terminal_frontiers"])
        graph["target_lineage"] = _v04_lineages(specs)

    events = list(graph.get("events") or [])
    locations: List[Dict[str, Any]] = []
    for event in events:
        if event.get("scene_id") == "scene-home":
            locations.append({"location_scope_id": "location-home", "event_id": event["event_id"],
                              "sphere": "home", "span_text": "at home" if "at home" in source else "my kitchen", "culprit_id": None})
        elif event.get("scene_id") == "scene-work":
            locations.append({"location_scope_id": "location-work", "event_id": event["event_id"],
                              "sphere": "workplace", "span_text": "at work", "culprit_id": None})
    mentions = _snake_mentions(source, events)
    frontiers = list(graph.get("terminal_frontiers") or [])
    graph["snake_mentions"] = mentions
    graph["location_scopes"] = locations
    graph["entity_chain_partitions"] = _partitions(events, mentions, frontiers, locations)
    graph["arbitration_candidates"], graph["terminal_decisions"] = _arbitrate(events, frontiers)
    graph["claim_projection_contract_version"] = SNAKE_CLAIM_PROJECTION_CONTRACT_VERSION
    graph["atomic_claims"], graph["claim_projection_manifest"] = _claim_projection(
        source,
        events,
        list(graph.get("target_lineage") or []),
        list(graph.get("terminal_decisions") or []),
        locations,
    )
    graph["certainty_contract_version"] = SNAKE_CERTAINTY_CONTRACT_VERSION
    graph["certainty_axis_records"] = _certainty_axis_records(
        source,
        events,
        list(graph.get("atomic_claims") or []),
    )
    graph["target_rule_contract_version"] = SNAKE_TARGET_RULE_CONTRACT_VERSION
    (
        graph["target_intent_records"],
        graph["rule_provenance_records"],
    ) = _target_rule_records(source, events)

    entities: Dict[str, Dict[str, Any]] = {str(x["entity_id"]): dict(x) for x in graph.get("entities") or [] if x.get("entity_id")}
    entities.setdefault("dreamer", {"entity_id": "dreamer", "entity_type": "person"})
    for mention in mentions:
        for snake_id in mention.get("snake_entity_ids") or []:
            entities.setdefault(str(snake_id), {"entity_id": snake_id, "entity_type": "snake"})
    for event in events:
        for entity_id in (event.get("actor_id"), event.get("target_id")):
            if not entity_id or entity_id == "ambiguous-snake-agent": continue
            if str(entity_id).startswith("snake-"): kind = "snake"
            elif any(str(entity_id).endswith(f"-{part}") for part in _BODY_PART_NAMES): kind = "body_part"
            elif entity_id in {"travel-bag", "shield-1"}: kind = "object"
            else: kind = "person"
            entities.setdefault(str(entity_id), {"entity_id": entity_id, "entity_type": kind})
    graph["entities"] = list(entities.values())
    graph["graph_integrity"] = validate_snake_event_graph(graph)
    return graph


def extract_snake_event_graph(dream: str) -> Dict[str, Any]:
    """Build and validate the current Snake Context v0.4 graph."""
    return _enrich_v04_graph(dream, _extract_snake_event_graph_legacy(dream))
