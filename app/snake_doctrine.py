from __future__ import annotations

from typing import Any, Dict, List

from app.release_info import DOCTRINE_REGISTRY
from app.snake_context import SNAKE_CONTEXT_VERSION, extract_snake_context
from app.snake_registry import (
    get_snake_registry_snapshot,
    public_snake_registry_metadata,
    snake_rule_id_for,
)


def build_snake_doctrine_context(dream: str) -> Dict[str, Any]:
    registry = get_snake_registry_snapshot()
    context = extract_snake_context(dream)
    active = bool(
        registry.get("verified") is True
        and context.get("has_snake")
        and (context.get("graph_integrity") or {}).get("verified") is True
    )
    rules: List[str] = []
    eligible_binding_ids = {
        str(binding.get("rule_id"))
        for binding in context.get("rule_bindings") or []
        if binding.get("disposition") in {"eligible", "eligible_partitioned", "target_scoped", "partitioned_by_target"}
    }

    def apply(key: str, condition: bool = True) -> None:
        if active and condition:
            rule_id = snake_rule_id_for(registry, key)
            if rule_id and rule_id not in rules:
                rules.append(rule_id)

    apply("snake_base_enemy")
    apply("snake_action_map", bool(context.get("action")))
    apply(
        "snake_attack",
        bool(context.get("attack") or context.get("completed_bite") or context.get("attempted_bite")),
    )
    apply("snake_watching", bool(context.get("watching")))
    apply("snake_retreat", bool(context.get("retreat")))
    apply("snake_victory", context.get("outcome") == "dreamer_victory" or "SNAKE-END-VICTORY" in eligible_binding_ids)
    apply("snake_defeat", context.get("outcome") == "opposition_victory_in_encounter" or "SNAKE-END-DEFEAT" in eligible_binding_ids)
    apply("snake_quantity", context.get("quantity") == "multiple")
    apply("snake_size_danger", bool(context.get("strength")))
    apply("snake_bite", bool(context.get("completed_bite")))
    apply("snake_venom", bool(context.get("venom")))
    apply("snake_location", bool(context.get("location_scope")))
    apply("snake_transform_person", bool(context.get("transformed_into_person")))
    apply("snake_ownership_low", bool(context.get("ownership_mentioned")))
    apply("snake_unfinished_battle", bool(context.get("unfinished_battle")))
    apply("snake_color_excluded", bool(context.get("colors_ignored")))
    # Tina's response/practice teaching follows an active troubling Snake reading,
    # and remains a separate guidance axis rather than changing the outcome.
    apply("snake_faith_response", bool(context.get("faith_response_eligible")))
    apply("snake_faith_best_practice", bool(context.get("faith_best_practice_eligible")))

    return {
        "active_doctrine": active,
        "symbol": "Snake" if active else "",
        "base_meaning": "enemy_or_opposition" if active else "",
        "action": context.get("action", "") if active else "",
        "action_target": context.get("action_target", "") if active else "",
        "outcome": context.get("outcome", "") if active else "",
        "quantity": context.get("quantity", "") if active else "",
        "strength": context.get("strength", "") if active else "",
        "location_scope": context.get("location_scope", "") if active else "",
        "completed_bite": bool(active and context.get("completed_bite")),
        "attempted_bite": bool(active and context.get("attempted_bite")),
        "venom": bool(active and context.get("venom")),
        "transformed_into_person": bool(active and context.get("transformed_into_person")),
        "ownership_weight": "low" if active and context.get("ownership_mentioned") else "",
        "unfinished_battle": bool(active and context.get("unfinished_battle")),
        "colors_ignored": list(context.get("colors_ignored") or []) if active else [],
        "response_guidance": (
            "Immediately cancel or reject the bad dream upon waking within Tina's Christian faith practice."
            if active and context.get("faith_response_eligible") else ""
        ),
        "best_practice_guidance": (
            "Repentance, reading Psalms, and reading Psalm 91 before bed are recommended Christian spiritual practices within the Jamaican True Stories framework."
            if active and context.get("faith_best_practice_eligible") else ""
        ),
        "predictive_certainty": "none",
        "applied_rule_ids": rules,
        "doctrine_version": registry.get("doctrine_version") or "DEC-SNAKE-2026-09-08-01",
        "context_version": SNAKE_CONTEXT_VERSION,
        "event_graph": context.get("event_graph") or {},
        "event_inventory": list(context.get("event_inventory") or []),
        "target_lineage": list(context.get("target_lineage") or []),
        "terminal_frontiers": list(context.get("terminal_frontiers") or []),
        "rule_bindings": list(context.get("rule_bindings") or []),
        "graph_integrity": dict(context.get("graph_integrity") or {}),
        "doctrine_source": DOCTRINE_REGISTRY,
        "doctrine_registry": public_snake_registry_metadata(registry),
        "context": context,
    }


def build_snake_narration_facts(dream: str) -> Dict[str, Any]:
    doctrine = build_snake_doctrine_context(dream)
    if not doctrine.get("active_doctrine"):
        return {**doctrine, "active": False, "narration_text": ""}

    parts = [
        "Within Jamaican and Caribbean spiritual tradition, the snake represents an enemy or opposition."
    ]
    action = doctrine.get("action")
    raw_target = doctrine.get("action_target") or ""
    target = "you" if raw_target == "dreamer" else (raw_target or "the described target")
    if action == "watching":
        parts.append(f"Its watching points to monitoring directed toward {target}, not a completed attack.")
    elif action == "attack":
        parts.append(f"Its attack represents conflict directed toward {target}; the attack alone does not settle the outcome.")
    elif action == "attempted_bite":
        parts.append("The attempted bite shows an attempted attack, not completed harm.")
    elif action == "completed_bite":
        parts.append(f"The completed bite represents an attack completed against {target} in this spiritual reading.")
    elif action == "retreat":
        parts.append("The snake's retreat is favorable directionally, without inventing a total defeat.")

    if doctrine.get("quantity") == "multiple":
        parts.append("Multiple snakes indicate multiple opponents; their separate actions and outcomes should remain distinct.")
    if doctrine.get("strength") == "lesser_or_weaker":
        parts.append("Its smaller or weaker form indicates lesser opposition.")
    elif doctrine.get("strength") == "stronger_or_more_dangerous":
        parts.append("Its larger, fiercer, or dangerous form indicates stronger opposition.")
    if doctrine.get("location_scope") == "home_or_family_sphere":
        parts.append("The location associates the issue with the home or family sphere without identifying a culprit.")
    elif doctrine.get("location_scope") == "work_sphere":
        parts.append("The location associates the issue with the work sphere without identifying a culprit.")
    if doctrine.get("venom"):
        parts.append("Explicit venom signifies, within Tina's doctrine, an attack understood as entering or moving internally; this is not medical evidence or objective proof of a curse.")
    if doctrine.get("transformed_into_person"):
        parts.append("Transformation into a person raises a relationship-context warning, but does not prove that person is hostile.")
    if doctrine.get("ownership_weight"):
        parts.append("Ownership carries little weight here and does not identify the owner as a culprit.")
    if doctrine.get("colors_ignored"):
        parts.append("The snake's color is deliberately excluded from the meaning.")

    outcome = doctrine.get("outcome")
    if outcome == "dreamer_victory":
        parts.append("The genuine ending seals victory over the opposition in this spiritual battle, without guaranteeing a real-world result.")
    elif outcome == "opposition_victory_in_encounter":
        parts.append("The ending indicates that the opposition prevailed in this encounter within the spiritual interpretation; it does not establish supernatural causation or a physical event as fact.")
    elif outcome == "unresolved":
        parts.append("The battle ended unresolved, so neither victory nor defeat is declared and recurrence is not guaranteed.")

    return {
        **doctrine,
        "active": True,
        "narration_text": " ".join(parts),
    }
