from __future__ import annotations

import re
from typing import Any, Dict, List

from app.utils import normalize_text


SNAKE_CONTEXT_VERSION = "snake-context-v1"

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
    """Extract only explicit Snake facts needed by approved Decision 01 rules."""
    text = normalize_text(dream)
    has_snake_token = _has(rf"\b{_SNAKE}\b", text)
    presence_negated = _has(
        rf"\b(?:no|without)\s+{_SNAKE}\b|\b(?:did not|didn't|never)\s+(?:see|saw|notice)\s+(?:a\s+|any\s+)?{_SNAKE}\b",
        text,
    )
    has_snake = bool(has_snake_token and not presence_negated)

    attempted_bite = _has(
        rf"\b{_SNAKE}\b.{{0,55}}\b(?:tried|attempted|almost|nearly)\b.{{0,20}}\b(?:bite|bit)\b|"
        rf"\b{_SNAKE}\b.{{0,55}}\b(?:did not|didn't|could not|couldn't|failed to)\b.{{0,15}}\bbite\b",
        text,
    )
    completed_bite = bool(
        _has(rf"\b{_SNAKE}\b.{{0,55}}\b(?:bit|bitten)\b|\b(?:bit|bitten)\b.{{0,55}}\bby\s+(?:a\s+)?{_SNAKE}\b", text)
        and not attempted_bite
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
    bite_target = _target(text, "bite") if completed_bite else ""
    defeat = bool(explicit_defeat or (completed_bite and bite_target == "dreamer"))

    battle = _has(rf"\b(?:fight|fighting|fought|battle|battling|struggle|struggling)\b.{{0,60}}\b{_SNAKE}\b|\b{_SNAKE}\b.{{0,60}}\b(?:fight|fighting|battle|struggle)\b", text)
    unfinished = bool(
        battle
        and _has(r"\b(?:dream ended|woke up|awoke|before (?:it|the fight|the battle) ended|still fighting)\b", text)
        and not dreamer_victory
        and not defeat
    )

    quantity = "multiple" if _has(r"\b(?:snakes|serpents|cobras|two|three|four|five|many|several|multiple)\b", text) else "one_or_unspecified"
    strength = ""
    if _has(rf"\b(?:large|big|huge|giant|massive|fierce|powerful|dangerous|venomous)\b.{{0,25}}\b{_SNAKE}\b|\b(?:cobra|cobras)\b", text):
        strength = "stronger_or_more_dangerous"
    elif _has(rf"\b(?:small|tiny|little|weak)\b.{{0,25}}\b{_SNAKE}\b", text):
        strength = "lesser_or_weaker"

    location = ""
    if _has(rf"\b{_SNAKE}\b.{{0,60}}\b(?:home|house|bedroom|kitchen|yard)\b|\b(?:home|house|bedroom|kitchen|yard)\b.{{0,60}}\b{_SNAKE}\b", text):
        location = "home_or_family_sphere"
    elif _has(rf"\b{_SNAKE}\b.{{0,60}}\b(?:work|workplace|office|job)\b|\b(?:work|workplace|office|job)\b.{{0,60}}\b{_SNAKE}\b", text):
        location = "work_sphere"

    transform_person = _has(rf"\b{_SNAKE}\b.{{0,45}}\b(?:turned|transformed|changed)\b.{{0,20}}\b(?:into|to)\b.{{0,10}}\b(?:a\s+)?(?:person|man|woman|someone|human)\b", text)
    ownership_mentioned = _has(rf"\b(?:owned|owns|owner of|belonged to|pet)\b.{{0,35}}\b{_SNAKE}\b|\b{_SNAKE}\b.{{0,35}}\b(?:belonged to|was .* pet)\b", text)
    colors: List[str] = [color for color in _COLORS if _has(rf"\b{color}\b.{{0,20}}\b{_SNAKE}\b|\b{_SNAKE}\b.{{0,20}}\b{color}\b", text)]
    venom = bool(completed_bite and _has(r"\b(?:venom|venomous|poison|poisonous)\b", text))

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

    target = _target(text, "bite" if completed_bite else "attack" if attack else "watch" if watching else "") if action else ""
    return {
        "context_version": SNAKE_CONTEXT_VERSION,
        "has_snake": has_snake,
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
        "transformed_into_person": transform_person,
        "ownership_mentioned": ownership_mentioned,
        "colors_ignored": colors,
    }
