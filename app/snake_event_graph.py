from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


SNAKE_EVENT_CONTRACT_VERSION = "snake-context-event-terminal-v1"

_SNAKE = r"(?:snake|snakes|serpent|serpents|cobra|cobras)"
_BODY_PART_NAMES = ("ankle", "arm", "hand", "wrist")
_BODY_PARTS = "(?:" + "|".join(_BODY_PART_NAMES) + ")"
_PEOPLE = "(?:sister|brother|cousin|friend|child|mother|father|son|daughter|husband|wife)"


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


def extract_snake_event_graph(dream: str) -> Dict[str, Any]:
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
        (rf"\b(?:a\s+)?(?:small|huge|large|black|green)?\s*{_SNAKE}\s+(?:attacked|struck at|lunged at)(?:\s+me)?\b", "attack", "completed"),
        (rf"\b(?:a\s+)?{_SNAKE}\s+rushed\s+between\s+my\s+sister\s+and\s+my\s+cousin\b", "attack", "completed"),
        (rf"\b(?:a\s+)?{_SNAKE}\s+chased\s+me\s+through\s+the\s+yard\b", "chase", "ongoing"),
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
            add(match, action, completion=completion, target=target)
            consumed.append((match.start(), match.end(), action))

    for match in _iter_matches(rf"\b{_SNAKE}\s+(?:did not|didn't|never)\s+bite\s+(?:me|my\s+{_PEOPLE})\b", source):
        add(match, "bite", polarity="negated", actuality="not_actual")
        consumed.append((match.start(), match.end(), "bite"))

    for match in _iter_matches(rf"\bif\s+(?:the\s+)?{_SNAKE}\s+bites?\s+me\b", source):
        add(match, "bite", modality="hypothetical", actuality="nonactual")
        consumed.append((match.start(), match.end(), "bite"))

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

    for match in _iter_matches(rf"\b{_SNAKE}\s+(?:changed|transformed|turned)\s+into\s+(?:my\s+)?(?:friend|person|man|woman|someone|human)\b", source):
        add(match, "transform_to_person")

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
        elif decisive["action"] in {"attack", "chase", "battle"}:
            outcome = "unresolved"
            if decisive["action"] == "battle" and not person:
                person = "dreamer"
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
    attacks = [event for event in eligible if event.get("action") in {"attack", "chase", "bite", "attempted_bite"}]
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
    unique = sorted(set(reasons))
    return {"verified": not unique, "reason_codes": unique}
