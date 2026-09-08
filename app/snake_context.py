from __future__ import annotations

import re
from typing import Any, Dict, List

from app.utils import normalize_text
from app.snake_event_graph import (
    SNAKE_EVENT_CONTRACT_VERSION,
    extract_snake_event_graph,
)


SNAKE_CONTEXT_VERSION = "snake-context-event-terminal-v1"

_SNAKE = r"(?:snake|snakes|serpent|serpents|cobra|cobras)"
_COLORS = (
    "black", "white", "red", "green", "yellow", "blue", "brown", "gold",
    "golden", "orange", "purple", "pink", "grey", "gray", "striped",
)


def _has(pattern: str, text: str) -> bool:
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def _target(text: str, action: str) -> str:
    patterns = {
        "bite": rf"{_SNAKE}.{{0,45}}(?:bit|bite|bitten)",
        "attack": rf"{_SNAKE}.{{0,45}}(?:attack|attacked|strike|struck|lunged|chased)",
        "watch": rf"{_SNAKE}.{{0,45}}(?:watch|watched|stare|stared|observed)",
    }
    start = re.search(patterns.get(action, r"$^"), text, flags=re.IGNORECASE)
    window = text[start.start(): start.start() + 100] if start else text
    if _has(r"\b(?:me|myself|at me|toward me|towards me)\b", window):
        return "dreamer"
    relationships = (
        "mother", "father", "sister", "brother", "husband", "wife", "son",
        "daughter", "friend", "aunt", "uncle", "cousin", "coworker",
    )
    for relationship in relationships:
        if _has(rf"\b{relationship}\b", window):
            return relationship
    return "unspecified"


def extract_snake_context(dream: str) -> Dict[str, Any]:
    """Extract only explicit Snake facts needed by approved Decisions 01 and 02."""
    text = normalize_text(dream)
    has_snake_token = _has(rf"\b{_SNAKE}\b", text)
    representation_type = "carving" if has_snake_token and _has(
        r"\b(?:carving|carved image|carved figure)\b",
        text,
    ) else ""
    presence_negated = _has(
        rf"\b(?:no|without)\s+{_SNAKE}\b|\b(?:did not|didn't|never)\s+(?:see|saw|notice)\s+(?:a\s+|any\s+)?{_SNAKE}\b",
        text,
    )
    live_snake_present = bool(has_snake_token and not presence_negated and not representation_type)
    has_snake = bool(live_snake_present or representation_type == "carving")

    nonactual_bite = _has(
        rf"\b(?:wondered|asked|imagined)\b.{{0,55}}\bif\b.{{0,35}}\b{_SNAKE}\b.{{0,20}}\b(?:bit|bite|bites)\b|"
        rf"\b(?:woke|awoke)\b.{{0,25}}\bbefore\b.{{0,20}}\b(?:it|the\s+{_SNAKE})\b.{{0,12}}\bbit\b|"
        rf"\bbefore\b.{{0,20}}\b(?:it|the\s+{_SNAKE})\b.{{0,12}}\bbit\b",
        text,
    )
    explicit_nonoccurrence = _has(
        rf"\b{_SNAKE}\b.{{0,30}}\b(?:did not|didn't|never)\s+bite(?:\s+or\s+attack)?\b",
        text,
    )
    explicit_nonattack = _has(
        rf"\b{_SNAKE}\b.{{0,45}}\b(?:did not|didn't|never)\s+(?:approach|attack|chase|strike|lunge)\b|"
        rf"\b(?:no|without)\b.{{0,20}}\b(?:attack|chase|strike|lunge)\b",
        text,
    )
    attempted_bite = bool(not explicit_nonoccurrence and _has(
        rf"\b{_SNAKE}\b.{{0,55}}\b(?:tried|attempted|almost|nearly)\b.{{0,20}}\b(?:bite|bit)\b|"
        rf"\b{_SNAKE}\b.{{0,55}}\blunged\s+to\s+bite\b.{{0,25}}\b(?:missed|failed|could not|couldn't)\b|"
        rf"\b{_SNAKE}\b.{{0,55}}\b(?:could not|couldn't|failed to)\b.{{0,15}}\bbite\b",
        text,
    ))
    completed_bite = bool(
        _has(rf"\b{_SNAKE}\b.{{0,55}}\b(?:bit|bitten)\b|\b(?:bit|bitten)\b.{{0,55}}\bby\s+(?:a\s+)?{_SNAKE}\b", text)
        and not attempted_bite and not nonactual_bite and not explicit_nonoccurrence
    )
    attack = _has(
        rf"\b{_SNAKE}\b.{{0,60}}\b(?:attack(?:ed|ing)?|struck|lunged|chased)\b|"
        rf"\b(?:attacked|chased)\b.{{0,50}}\bby\s+(?:a\s+)?{_SNAKE}\b",
        text,
    )
    watching = _has(rf"\b{_SNAKE}\b.{{0,55}}\b(?:watch(?:ed|ing)?|observ(?:ed|ing)|star(?:ed|ing))\b", text)
    retreat = _has(
        rf"\b{_SNAKE}\b.{{0,55}}\b(?:ran away|run away|retreat(?:ed|ing)?|fled|slithered away|backed away)\b",
        text,
    )

    found_dead = _has(rf"\b(?:found|saw|noticed)\b.{{0,30}}\b(?:dead|lifeless)\b.{{0,15}}\b{_SNAKE}\b", text)
    dreamer_victory = bool(
        not found_dead
        and _has(
            rf"\b(?:i|we)\b.{{0,35}}\b(?:killed|defeated|destroyed|overcame|beat)\b.{{0,35}}\b(?:the\s+|a\s+)?{_SNAKE}\b|"
            rf"\b(?:i|we)\b.{{0,35}}\b{_SNAKE}\b.{{0,35}}\b(?:killed|defeated|destroyed|overcame|beat)\b(?:\s+it|\s+them)?\b",
            text,
        )
    )
    explicit_defeat = _has(
        rf"\b{_SNAKE}\b.{{0,45}}\b(?:defeated|killed|overcame|beat)\b.{{0,20}}\b(?:me|us)\b|"
        rf"\b(?:i|we)\b.{{0,25}}\b(?:was|were|felt)\b.{{0,15}}\bdefeated\b.{{0,35}}\b(?:by\s+)?(?:the\s+)?{_SNAKE}\b",
        text,
    )
    bite_target = _target(text, "bite") if (completed_bite or attempted_bite) else ""
    defeat = bool(explicit_defeat or (completed_bite and bite_target == "dreamer"))

    battle = _has(rf"\b(?:fight|fighting|fought|battle|battling|struggle|struggling)\b.{{0,60}}\b{_SNAKE}\b|\b{_SNAKE}\b.{{0,60}}\b(?:fight|fighting|battle|struggle)\b", text)
    unfinished = bool(
        battle
        and _has(r"\b(?:dream ended|woke up|awoke|before (?:it|the fight|the battle) ended|still fighting)\b", text)
        and not dreamer_victory
        and not defeat
    )

    snake_mentions = re.findall(rf"\b{_SNAKE}\b", text, flags=re.IGNORECASE)
    quantity = "multiple" if (
        len(snake_mentions) > 1
        or _has(r"\b(?:snakes|serpents|cobras|two|three|four|five|many|several|multiple)\b", text)
    ) else "one_or_unspecified"
    strength = ""
    if _has(rf"\b(?:large|big|huge|giant|massive|fierce|powerful|dangerous|venomous)\b.{{0,25}}\b{_SNAKE}\b|\b(?:cobra|cobras)\b", text):
        strength = "stronger_or_more_dangerous"
    elif _has(rf"\b(?:small|tiny|little|weak)\b.{{0,25}}\b{_SNAKE}\b", text):
        strength = "lesser_or_weaker"

    location = ""
    location_observed = ""
    if _has(rf"\b{_SNAKE}\b.{{0,60}}\bbedroom\b|\bbedroom\b.{{0,60}}\b{_SNAKE}\b", text):
        location = "intimate_life_sphere"
        location_observed = "bedroom"
    elif _has(rf"\b{_SNAKE}\b.{{0,60}}\bkitchen\b|\bkitchen\b.{{0,60}}\b{_SNAKE}\b", text):
        location = "productivity_healing_replenishment_sphere"
        location_observed = "kitchen"
    elif _has(rf"\b{_SNAKE}\b.{{0,60}}\b(?:home|house|yard)\b|\b(?:home|house|yard)\b.{{0,60}}\b{_SNAKE}\b", text):
        location = "life_sphere"
        location_observed = "house"
    elif _has(rf"\b{_SNAKE}\b.{{0,60}}\bbathroom\b|\bbathroom\b.{{0,60}}\b{_SNAKE}\b", text):
        location_observed = "bathroom"
    elif _has(rf"\b{_SNAKE}\b.{{0,60}}\b(?:work|workplace|office|job)\b|\b(?:work|workplace|office|job)\b.{{0,60}}\b{_SNAKE}\b", text):
        location = "work_sphere"
        location_observed = "work"

    transform_person = _has(
        rf"\b{_SNAKE}\b.{{0,45}}\b(?:turned|transformed|changed|became)\b"
        rf".{{0,20}}\b(?:into|to)?\s*(?:my\s+)?(?:person|man|woman|someone|human|sister|brother|coworker|friend)\b",
        text,
    )
    ownership_mentioned = _has(rf"\b(?:owned|owns|owner of|belonged to|pet)\b.{{0,35}}\b{_SNAKE}\b|\b{_SNAKE}\b.{{0,35}}\b(?:belonged to|was .* pet)\b", text)
    colors: List[str] = [color for color in _COLORS if _has(rf"\b{color}\b.{{0,20}}\b{_SNAKE}\b|\b{_SNAKE}\b.{{0,20}}\b{color}\b", text)]
    venom_absent = _has(
        r"\b(?:no|without)\s+(?:venom|poison)\b|"
        r"\b(?:never|did not|didn't)\b.{0,35}\b(?:show(?:ed)?|mention(?:ed)?)\b.{0,15}\b(?:venom|poison)\b|"
        r"\b(?:venom|poison)\b.{0,20}\b(?:absent|not present)\b",
        text,
    )
    venom = bool(completed_bite and not venom_absent and _has(r"\b(?:venom|venomous|poison|poisonous)\b", text))

    if dreamer_victory:
        outcome = "dreamer_victory"
    elif defeat:
        outcome = "opposition_victory_in_encounter"
    elif unfinished:
        outcome = "unresolved"
    else:
        outcome = "not_established"

    action = ""
    if completed_bite:
        action = "completed_bite"
    elif attempted_bite:
        action = "attempted_bite"
    elif attack:
        action = "attack"
    elif watching:
        action = "watching"
    elif retreat:
        action = "retreat"

    target = _target(text, "bite" if (completed_bite or attempted_bite) else "attack" if attack else "watch" if watching else "") if action else ""
    event_graph = extract_snake_event_graph(dream) if live_snake_present else {
        "contract_version": SNAKE_EVENT_CONTRACT_VERSION,
        "entities": [], "events": [], "target_lineage": [],
        "terminal_frontiers": [], "rule_bindings": [],
        "graph_integrity": {"verified": True, "reason_codes": []},
    }
    graph_frontiers = event_graph.get("terminal_frontiers") or []
    graph_events = event_graph.get("events") or []
    graph_outcomes = [frontier.get("outcome") for frontier in graph_frontiers]
    decisive_outcomes = {
        value for value in graph_outcomes
        if value not in {None, "no_completed_conflict"}
    }
    if len(decisive_outcomes) > 1:
        outcome = "mixed"
    elif "victory" in graph_outcomes:
        outcome = "dreamer_victory"
    elif "defeat" in graph_outcomes:
        outcome = "opposition_victory_in_encounter"
    elif graph_outcomes and all(value == "opposition_prevailed_for_target" for value in graph_outcomes):
        outcome = "opposition_prevailed_for_target"
    elif any(value == "unresolved" for value in graph_outcomes):
        outcome = "unresolved"
    elif graph_outcomes:
        outcome = "not_established"

    eligible_lineage_event_ids = {
        lineage.get("event_id")
        for lineage in event_graph.get("target_lineage") or []
        if lineage.get("outcome_eligible") is True
    }
    completed_bite_events = [
        event for event in graph_events
        if event.get("action") == "bite"
        and event.get("polarity") == "affirmed"
        and event.get("actuality") == "actual"
        and event.get("completion") == "completed"
        and event.get("event_id") in eligible_lineage_event_ids
    ]
    attempted_bite_events = [
        event for event in graph_events
        if event.get("action") == "attempted_bite"
        or (event.get("action") == "bite" and event.get("completion") == "attempted")
    ]
    graph_bite_events = [
        event for event in graph_events
        if event.get("action") in {"bite", "attempted_bite"}
    ]
    if completed_bite_events:
        completed_bite = True
        attempted_bite = False
        first_bite = completed_bite_events[0]
        bite_target = next(
            (
                lineage.get("affected_person_id") or lineage.get("surface_target_id")
                for lineage in event_graph.get("target_lineage") or []
                if lineage.get("event_id") == first_bite.get("event_id")
            ),
            first_bite.get("target_id") or "",
        )
    elif attempted_bite_events:
        attempted_bite = True
        completed_bite = False
    elif graph_bite_events:
        attempted_bite = False
        completed_bite = False
    action_events = [event for event in graph_events if event.get("polarity") == "affirmed" and event.get("actuality") == "actual"]
    if graph_events and not action_events:
        action = ""
        target = ""
        attack = False
        watching = False
        retreat = False
    if action_events:
        action_map = {
            "watch": "watching", "attack": "attack", "chase": "attack",
            "capture": "attack", "overpower": "attack",
            "bite": "completed_bite", "attempted_bite": "attempted_bite",
            "retreat": "retreat",
        }
        first_action = next((event for event in action_events if event.get("action") in action_map), None)
        if first_action:
            action = action_map[str(first_action.get("action"))]
            first_lineage = next(
                (lineage for lineage in event_graph.get("target_lineage") or [] if lineage.get("event_id") == first_action.get("event_id")),
                {},
            )
            target = first_lineage.get("affected_person_id") or first_action.get("target_id") or target
        attack = any(event.get("action") in {"attack", "chase", "capture", "overpower", "bite", "attempted_bite"} for event in action_events)
        watching = any(event.get("action") == "watch" for event in action_events)
        retreat = any(event.get("action") == "retreat" for event in action_events)
        transform_person = bool(transform_person or any(event.get("action") == "transform_to_person" for event in action_events))

    if explicit_nonattack and not completed_bite and not attempted_bite:
        attack = False
        if action == "attack":
            action = ""
            target = ""
    if representation_type:
        action = ""
        target = ""
        attack = watching = retreat = completed_bite = attempted_bite = venom = False
        outcome = "not_established"
        unfinished = transform_person = False

    unfinished = bool(
        unfinished
        or (
            outcome == "unresolved"
            and _has(r"\b(?:dream ended|when the dream ended|woke|awoke)\b.{0,35}\b(?:before|neither|either)\b|\bbefore either of us won\b", text)
        )
    )

    troubling = bool(
        attack or completed_bite or attempted_bite
        or outcome in {"opposition_victory_in_encounter", "unresolved", "mixed"}
        or _has(r"\b(?:bad|frightening|terrifying|troubling)\s+dream\b", text)
    )
    waking_supported = _has(r"\b(?:woke|woke up|awoke|upon waking|when i woke)\b", text)
    faith_response_eligible = bool(has_snake and troubling and waking_supported)
    faith_best_practice_eligible = bool(has_snake and troubling)

    if not has_snake:
        action = ""
        target = ""
        attack = watching = retreat = completed_bite = attempted_bite = venom = False
        outcome = "not_established"
        unfinished = transform_person = False
        quantity = "one_or_unspecified"

    return {
        "context_version": SNAKE_CONTEXT_VERSION,
        "has_snake": has_snake,
        "live_snake_present": live_snake_present,
        "representation_type": representation_type,
        "entity_form": "representation" if representation_type else ("live_snake" if live_snake_present else ""),
        "presence_negated": presence_negated,
        "quantity": quantity,
        "action": action,
        "action_target": target,
        "attack": attack,
        "watching": watching,
        "retreat": retreat,
        "completed_bite": completed_bite,
        "attempted_bite": attempted_bite,
        "venom": venom,
        "outcome": outcome,
        "found_dead": found_dead,
        "unfinished_battle": unfinished,
        "strength": strength,
        "location_scope": location,
        "location_observed": location_observed,
        "transformed_into_person": transform_person,
        "ownership_mentioned": ownership_mentioned,
        "colors_ignored": colors,
        "faith_response_eligible": faith_response_eligible,
        "faith_best_practice_eligible": faith_best_practice_eligible,
        "event_graph": event_graph,
        "event_inventory": graph_events,
        "target_lineage": event_graph.get("target_lineage") or [],
        "snake_mentions": event_graph.get("snake_mentions") or [],
        "entity_chain_partitions": event_graph.get("entity_chain_partitions") or [],
        "terminal_frontiers": graph_frontiers,
        "arbitration_candidates": event_graph.get("arbitration_candidates") or [],
        "terminal_decisions": event_graph.get("terminal_decisions") or [],
        "location_scopes": event_graph.get("location_scopes") or [],
        "rule_bindings": event_graph.get("rule_bindings") or [],
        "graph_integrity": event_graph.get("graph_integrity") or {"verified": False, "reason_codes": ["GRAPH_MISSING"]},
    }
