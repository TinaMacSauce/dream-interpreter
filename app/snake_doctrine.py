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

    apply("snake_base_enemy", bool(context.get("live_snake_present")))
    apply("snake_representation_carving", context.get("representation_type") == "carving")
    apply("snake_action_map", bool(context.get("action")))
    apply(
        "snake_attack",
        bool(context.get("attack") or context.get("completed_bite") or context.get("attempted_bite")),
    )
    apply("snake_watching", bool(context.get("watching")))
    apply("snake_watching_target", bool(context.get("watching") and context.get("action_target") not in {"", "dreamer", "unspecified"}))
    apply("snake_retreat", bool(context.get("retreat")))
    apply("snake_victory", context.get("outcome") == "dreamer_victory" or "SNAKE-END-VICTORY" in eligible_binding_ids)
    apply("snake_defeat", context.get("outcome") == "opposition_victory_in_encounter" or "SNAKE-END-DEFEAT" in eligible_binding_ids)
    apply("snake_quantity", context.get("quantity") == "multiple")
    apply("snake_size_danger", bool(context.get("strength")))
    apply("snake_bite", bool(context.get("completed_bite")))
    apply("snake_bite_attempt_target", bool(context.get("attempted_bite") and context.get("action_target") not in {"", "dreamer", "unspecified"}))
    apply("snake_venom", bool(context.get("venom")))
    apply("snake_location", bool(context.get("location_scope")))
    apply("snake_location_house_life", context.get("location_scope") == "life_sphere")
    apply("snake_location_bedroom_intimacy", context.get("location_scope") == "intimate_life_sphere")
    apply(
        "snake_location_kitchen_productivity_healing_replenishment",
        context.get("location_scope") == "productivity_healing_replenishment_sphere",
    )
    apply("snake_transform_person", bool(context.get("transformed_into_person")))
    apply("snake_ownership_low", bool(context.get("ownership_mentioned")))
    apply("snake_unfinished_battle", bool(context.get("unfinished_battle")))
    apply("snake_color_excluded", bool(context.get("colors_ignored")))
    # Tina's response/practice teaching follows an active troubling Snake reading,
    # and remains a separate guidance axis rather than changing the outcome.
    apply("snake_faith_response", bool(context.get("faith_response_eligible")))
    apply("snake_faith_best_practice", bool(context.get("faith_best_practice_eligible")))

    # Stable keys route context; approved wording comes from the private source.
    location_key = {
        "life_sphere": "snake_location_house_life",
        "intimate_life_sphere": "snake_location_bedroom_intimacy",
        "productivity_healing_replenishment_sphere":
            "snake_location_kitchen_productivity_healing_replenishment",
    }.get(context.get("location_scope"))
    location_rule = (
        (registry.get("rules") or {}).get(location_key) or {}
        if active and registry.get("canonical_location_text") is True
        and snake_rule_id_for(registry, location_key) in rules
        else {}
    )

    return {
        "active_doctrine": active,
        "symbol": "Snake" if active else "",
        "base_meaning": (
            "lurking_opposition_warning"
            if active and context.get("representation_type") == "carving"
            else "enemy_or_opposition" if active else ""
        ),
        "representation_type": context.get("representation_type", "") if active else "",
        "entity_form": context.get("entity_form", "") if active else "",
        "action": context.get("action", "") if active else "",
        "action_target": context.get("action_target", "") if active else "",
        "outcome": context.get("outcome", "") if active else "",
        "quantity": context.get("quantity", "") if active else "",
        "strength": context.get("strength", "") if active else "",
        "location_scope": context.get("location_scope", "") if active else "",
        "location_observed": context.get("location_observed", "") if active else "",
        "location_governing_meaning": location_rule.get("governing_meaning", ""),
        "location_rule_provenance": {
            field: location_rule.get(field, "")
            for field in ("rule_id", "doctrine_version", "decision_id", "updated_at_utc")
        } if location_rule else {},
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
        "doctrine_version": registry.get("doctrine_version") or "DEC-SNAKE-2026-09-08-02",
        "context_version": SNAKE_CONTEXT_VERSION,
        "event_graph": context.get("event_graph") or {},
        "event_inventory": list(context.get("event_inventory") or []),
        "target_lineage": list(context.get("target_lineage") or []),
        "snake_mentions": list(context.get("snake_mentions") or []),
        "entity_chain_partitions": list(context.get("entity_chain_partitions") or []),
        "terminal_frontiers": list(context.get("terminal_frontiers") or []),
        "arbitration_candidates": list(context.get("arbitration_candidates") or []),
        "terminal_decisions": list(context.get("terminal_decisions") or []),
        "claim_projection_contract_version": context.get("claim_projection_contract_version") or "",
        "atomic_claims": list(context.get("atomic_claims") or []),
        "claim_projection_manifest": list(context.get("claim_projection_manifest") or []),
        "certainty_contract_version": context.get("certainty_contract_version") or "",
        "certainty_axis_records": list(context.get("certainty_axis_records") or []),
        "target_rule_contract_version": context.get("target_rule_contract_version") or "",
        "target_intent_records": list(context.get("target_intent_records") or []),
        "rule_provenance_records": list(context.get("rule_provenance_records") or []),
        "snake_registry_decision_ids": list(registry.get("decision_ids") or [
            "DEC-SNAKE-2026-09-08-01",
            "DEC-SNAKE-2026-09-08-02",
        ]),
        "snake_registry_content_revision": registry.get("content_revision") or "",
        "location_scopes": list(context.get("location_scopes") or []),
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

    if doctrine.get("representation_type") == "carving":
        parts = [
            "Within Jamaican and Caribbean spiritual tradition, a snake carving carries a bounded warning of lurking opposition; it is not a live-snake event or proof of a hidden person, surveillance, or supernatural cause."
        ]
    else:
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
        parts.append(f"The attempted bite shows an attempted attack directed toward {target}, not completed contact or harm.")
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
    if doctrine.get("location_scope") == "life_sphere":
        parts.append(
            doctrine["location_governing_meaning"] + " This does not identify a culprit."
            if doctrine.get("location_governing_meaning") else
            "The house location associates the warning with the dreamer's life generally, without identifying a culprit."
        )
    elif doctrine.get("location_scope") == "intimate_life_sphere":
        parts.append(
            doctrine["location_governing_meaning"] + " This is without identifying a partner or culprit."
            if doctrine.get("location_governing_meaning") else
            "The bedroom location associates the warning with the intimate sphere, without identifying a partner or culprit."
        )
    elif doctrine.get("location_scope") == "productivity_healing_replenishment_sphere":
        parts.append(
            doctrine["location_governing_meaning"] + " This is without implying contamination, illness, or a culprit."
            if doctrine.get("location_governing_meaning") else
            "The kitchen location associates the warning with productivity, healing, and replenishment, without implying contamination, illness, or a culprit."
        )
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
