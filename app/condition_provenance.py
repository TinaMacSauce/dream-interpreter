from __future__ import annotations

from typing import Any, Dict, List, Mapping


CONDITION_PROVENANCE_CONTRACT_VERSION = "condition-state-provenance/1.0"
CONDITION_RULE_TYPES = {
    "TEETH-OMEN-GUM-BLOOD": "gum_bleeding_condition",
    "TEETH-STATE-LOOSE": "loose_tooth_condition",
}
FALLOUT_COUNT_RULE_IDS = {"TEETH-FALLOUT-ONE", "TEETH-FALLOUT-MULTIPLE"}
CONDITION_EVENT_TYPES = set(CONDITION_RULE_TYPES.values())


def validate_condition_provenance(graph: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate condition-event, rule, and released-claim reachability."""
    reasons: List[str] = []

    def reject(reason: str) -> None:
        if reason not in reasons:
            reasons.append(reason)

    events = {
        str(event.get("event_id")): event
        for event in graph.get("event_inventory", [])
        if event.get("event_id")
    }
    chains = {
        str(chain.get("event_chain_id")): chain
        for chain in graph.get("event_chain_inventory", [])
        if chain.get("event_chain_id")
    }
    condition_events = {
        event_id: event
        for event_id, event in events.items()
        if event.get("event_type") in CONDITION_EVENT_TYPES
    }
    public_records = list(graph.get("rule_sets", {}).get("public_applied", []))
    warning_records = list(graph.get("rule_sets", {}).get("warning_active", []))
    active_condition_records = [
        record for record in public_records
        if record.get("rule_id") in CONDITION_RULE_TYPES
    ]

    for record in active_condition_records:
        source_ids = list(record.get("source_event_ids", []))
        if not source_ids or any(event_id not in condition_events for event_id in source_ids):
            reject("CONDITION_EVENT_MISSING")
        expected_type = CONDITION_RULE_TYPES[str(record.get("rule_id"))]
        if any(
            events.get(event_id, {}).get("event_type") != expected_type
            for event_id in source_ids
        ):
            reject("CONDITION_RULE_EVENT_MISMATCH")

    for event_id, event in condition_events.items():
        owner = event.get("owner_id_or_ambiguous")
        if owner in {None, ""}:
            reject("CONDITION_OWNER_MISSING")
        targets = event.get("target_entity_ids_or_ambiguous")
        expected_entity = "gums" if event.get("event_type") == "gum_bleeding_condition" else "tooth"
        entities = {
            str(entity.get("entity_id")): entity
            for entity in graph.get("entity_inventory", [])
            if entity.get("entity_id")
        }
        if (
            not isinstance(targets, list)
            or not targets
            or any(
                target not in entities
                or entities[target].get("entity_type") != expected_entity
                for target in targets
            )
        ):
            reject("CONDITION_TARGET_MISSING")

        eligible = event.get("doctrine_eligible") is True
        chain_id = event.get("event_chain_id_or_null")
        if eligible and (
            not chain_id
            or chain_id not in chains
            or event_id not in chains[chain_id].get("event_ids", [])
            or any(target not in chains[chain_id].get("entity_ids", []) for target in (targets or []))
        ):
            reject("CONDITION_CHAIN_MISSING")

        span = event.get("source_span") or {}
        if (
            not span.get("span_id")
            or not isinstance(span.get("start"), int)
            or not isinstance(span.get("end"), int)
            or span.get("end") <= span.get("start")
            or not span.get("text")
        ):
            reject("CONDITION_SPAN_MISSING")

    for record in active_condition_records:
        source_ids = list(record.get("source_event_ids", []))
        source_spans = list(record.get("source_span_ids", []))
        expected_spans = {
            events[event_id].get("source_span", {}).get("span_id")
            for event_id in source_ids if event_id in events
        }
        if not source_spans or set(source_spans) != expected_spans:
            reject("CONDITION_SPAN_MISSING")

    active_rule_ids = {str(record.get("rule_id")) for record in warning_records}
    active_source_ids = {
        event_id
        for record in warning_records
        for event_id in record.get("source_event_ids", [])
    }
    nonactual_condition_ids = {
        event_id for event_id, event in condition_events.items()
        if event.get("doctrine_eligible") is not True
        or event.get("actuality") != "actual"
        or event.get("phase") != "dream"
        or event.get("polarity") != "affirmed"
    }
    if active_source_ids & nonactual_condition_ids:
        if any(events[event_id].get("polarity") == "negated" for event_id in active_source_ids & nonactual_condition_ids):
            reject("NEGATED_CONDITION_RULE_LEAK")
        else:
            reject("NONACTUAL_CONDITION_RULE_LEAK")

    eligible_losses = {
        event_id: event for event_id, event in events.items()
        if event.get("event_type") == "tooth_loss" and event.get("doctrine_eligible") is True
    }
    if any(
        event.get("event_type") == "tooth_loss"
        and event.get("polarity") == "negated"
        and event_id in active_source_ids
        for event_id, event in events.items()
    ):
        reject("NEGATED_CONDITION_RULE_LEAK")

    public_rule_ids = {str(record.get("rule_id")) for record in public_records}
    if public_rule_ids & FALLOUT_COUNT_RULE_IDS and not eligible_losses:
        reject("CONDITION_QUANTITY_RULE_LEAK")

    transition_edges = list(graph.get("condition_transition_edges", []))
    for condition_id, condition in condition_events.items():
        if condition.get("event_type") != "loose_tooth_condition" or condition.get("doctrine_eligible") is not True:
            continue
        later_losses = [
            loss_id for loss_id, loss in eligible_losses.items()
            if loss.get("event_chain_id_or_null") == condition.get("event_chain_id_or_null")
            and loss.get("source_span", {}).get("start", -1) > condition.get("source_span", {}).get("start", -1)
        ]
        if later_losses and not any(
            edge.get("from_event_id") == condition_id
            and edge.get("to_event_id") in later_losses
            for edge in transition_edges
        ):
            reject("CONDITION_LOSS_COLLAPSE")

    for edge in transition_edges:
        source = events.get(str(edge.get("from_event_id")))
        target = events.get(str(edge.get("to_event_id")))
        if (
            not source
            or not target
            or source.get("event_type") != "loose_tooth_condition"
            or target.get("event_type") not in {"tooth_loss", "retained_tooth_state"}
            or source.get("event_chain_id_or_null") != target.get("event_chain_id_or_null")
        ):
            reject("CONDITION_LOSS_COLLAPSE")

    eligible_conditions = [
        event for event in condition_events.values()
        if event.get("doctrine_eligible") is True
    ]
    owners = {event.get("owner_id_or_ambiguous") for event in eligible_conditions}
    if len(owners) > 1:
        for chain in chains.values():
            chain_conditions = [
                events[event_id] for event_id in chain.get("event_ids", [])
                if event_id in condition_events
            ]
            if len({event.get("owner_id_or_ambiguous") for event in chain_conditions}) > 1:
                reject("CROSS_OWNER_CONDITION_COLLAPSE")
        for event in eligible_conditions:
            chain = chains.get(str(event.get("event_chain_id_or_null")))
            if chain and chain.get("owner_id_or_ambiguous") != event.get("owner_id_or_ambiguous"):
                reject("CROSS_OWNER_CONDITION_COLLAPSE")

    active_condition_rule_ids = set(CONDITION_RULE_TYPES) & active_rule_ids
    if active_condition_rule_ids:
        condition_claims = [
            claim for claim in graph.get("claim_manifest", [])
            if claim.get("released")
            and set(claim.get("consumed_rule_ids", [])) & active_condition_rule_ids
        ]
        if not condition_claims:
            reject("CONDITION_CLAIM_PATH_MISSING")
        for claim in condition_claims:
            claim_rules = set(claim.get("consumed_rule_ids", [])) & active_condition_rule_ids
            event_ids = set(claim.get("consumed_event_ids", []))
            span_ids = set(claim.get("consumed_span_ids", []))
            if not claim_rules or not event_ids or not span_ids:
                reject("CONDITION_CLAIM_PATH_MISSING")
                continue
            for rule_id in claim_rules:
                expected_type = CONDITION_RULE_TYPES[rule_id]
                matching_ids = {
                    event_id for event_id in event_ids
                    if events.get(event_id, {}).get("event_type") == expected_type
                }
                expected_span_ids = {
                    events[event_id].get("source_span", {}).get("span_id")
                    for event_id in matching_ids
                }
                if not matching_ids or not expected_span_ids or not expected_span_ids.issubset(span_ids):
                    reject("CONDITION_CLAIM_PATH_MISSING")

    return {"verified": not reasons, "reason_codes": reasons}
