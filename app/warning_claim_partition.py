from __future__ import annotations

from typing import Any, Dict, List, Mapping


WARNING_CLAIM_PARTITION_CONTRACT_VERSION = (
    "warning-claim-partition-conservation/1.0"
)

WARNING_EVENT_FAMILIES = {
    "tooth_loss": "tooth_loss",
    "loose_tooth_condition": "loose_sickness",
    "gum_bleeding_condition": "bleeding_gums",
}

RULE_EVENT_TYPES = {
    "TEETH-FALLOUT-OWN": {"tooth_loss"},
    "TEETH-FALLOUT-OTHER": {"tooth_loss"},
    "TEETH-FALLOUT-ONE": {"tooth_loss"},
    "TEETH-FALLOUT-MULTIPLE": {"tooth_loss"},
    "TEETH-PULL-SELF": {"tooth_loss"},
    "TEETH-PULL-EXTERNAL": {"tooth_loss"},
    "TEETH-STATE-LOOSE": {"loose_tooth_condition"},
    "TEETH-OMEN-GUM-BLOOD": {"gum_bleeding_condition"},
    "TEETH-MOD-PAIN": {"pain_modifier"},
    "TEETH-MOD-PAINLESS": {"tooth_loss"},
    "TEETH-MOD-BLOOD": {"tooth_blood_modifier"},
}


def _inventory(values: List[Mapping[str, Any]], key: str) -> Dict[str, Mapping[str, Any]]:
    return {
        str(value[key]): value
        for value in values
        if value.get(key)
    }


def validate_warning_claim_partition(graph: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate owner-, chain-, and warning-family claim conservation."""
    reasons: List[str] = []

    def reject(reason: str) -> None:
        if reason not in reasons:
            reasons.append(reason)

    events = _inventory(list(graph.get("event_inventory", [])), "event_id")
    records = list(graph.get("rule_sets", {}).get("public_applied", []))
    claims = list(graph.get("claim_manifest", []))
    atomic = [
        claim for claim in claims
        if claim.get("claim_scope") == "atomic_warning"
        and claim.get("released") is True
    ]
    compounds = [
        claim for claim in claims
        if claim.get("claim_scope") == "compound_presentation"
        and claim.get("released") is True
    ]
    historical_ids = {
        str(event_id)
        for frontier in graph.get("terminal_frontiers", [])
        if events.get(str(frontier.get("terminal_event_id")), {}).get("event_type")
        in {"tooth_loss", "firm_same_tooth_return"}
        for event_id in frontier.get("historical_event_ids", [])
        if events.get(str(event_id), {}).get("event_type") in WARNING_EVENT_FAMILIES
    }
    active_warning_ids = {
        event_id
        for event_id, event in events.items()
        if event.get("event_type") in WARNING_EVENT_FAMILIES
        and event.get("doctrine_eligible") is True
        and event_id not in historical_ids
    }

    released_occurrences: Dict[str, int] = {event_id: 0 for event_id in active_warning_ids}
    atomic_ids = {str(claim.get("claim_id")) for claim in atomic}
    for claim in atomic:
        event_ids = [str(value) for value in claim.get("consumed_event_ids", [])]
        consumed = [events[event_id] for event_id in event_ids if event_id in events]
        base_events = [
            event for event in consumed
            if event.get("event_type") in WARNING_EVENT_FAMILIES
        ]
        owners = {
            event.get("owner_id_or_ambiguous")
            for event in consumed
            if event.get("owner_id_or_ambiguous") not in {None, "", "ambiguous", "unknown"}
        }
        chains = {
            event.get("event_chain_id_or_null")
            for event in consumed
            if event.get("event_chain_id_or_null")
        }
        families = {
            WARNING_EVENT_FAMILIES[event.get("event_type")]
            for event in base_events
        }
        if len(owners) != 1 or set(claim.get("owner_ids", [])) != owners:
            reject("ATOMIC_CLAIM_OWNER_MIX")
        if len(chains) != 1 or set(claim.get("event_chain_ids", [])) != chains:
            reject("ATOMIC_CLAIM_CHAIN_MIX")
        if len(families) != 1 or claim.get("warning_family") not in families:
            reject("ATOMIC_CLAIM_FAMILY_MIX")
        if any(event.get("doctrine_eligible") is not True for event in consumed):
            reject("INELIGIBLE_EVENT_CONSUMED")
        if any(event.get("event_id") in historical_ids for event in consumed):
            reject("HISTORICAL_EVENT_REACTIVATED")
        for event in base_events:
            event_id = str(event.get("event_id"))
            if event_id in released_occurrences:
                released_occurrences[event_id] += 1

        expected_spans = {
            events[event_id].get("source_span", {}).get("span_id")
            for event_id in event_ids
            if event_id in events
        }
        if set(claim.get("consumed_span_ids", [])) != expected_spans:
            reject("CLAIM_SPAN_MISMATCH")

        consumed_types = {event.get("event_type") for event in consumed}
        for rule_id in claim.get("consumed_rule_ids", []):
            allowed = RULE_EVENT_TYPES.get(str(rule_id))
            if allowed is not None and not (allowed & consumed_types):
                reject("CLAIM_EVENT_RULE_MISMATCH")

    if any(count == 0 for count in released_occurrences.values()):
        reject("ELIGIBLE_WARNING_EVENT_OMITTED")
    if any(count != 1 for count in released_occurrences.values()):
        reject("CLAIM_CARDINALITY_DRIFT")

    for record in records:
        source_ids = [str(value) for value in record.get("source_event_ids", [])]
        sources = [events[event_id] for event_id in source_ids if event_id in events]
        owners = {event.get("owner_id_or_ambiguous") for event in sources}
        chains = {event.get("event_chain_id_or_null") for event in sources}
        if len(source_ids) != 1 or len(owners) > 1 or len(chains) > 1:
            reject("RULE_SOURCE_PARTITION_MISSING")
        if any(event.get("doctrine_eligible") is not True for event in sources):
            reject("INELIGIBLE_EVENT_CONSUMED")
        expected_types = RULE_EVENT_TYPES.get(str(record.get("rule_id")))
        if expected_types is not None and any(
            event.get("event_type") not in expected_types for event in sources
        ):
            reject("CLAIM_EVENT_RULE_MISMATCH")
        expected_spans = {
            event.get("source_span", {}).get("span_id") for event in sources
        }
        if set(record.get("source_span_ids", [])) != expected_spans:
            reject("CLAIM_SPAN_MISMATCH")

    if len(atomic) > 1:
        if len(compounds) != 1 or set(compounds[0].get("member_claim_ids", [])) != atomic_ids:
            reject("COMPOUND_MEMBER_MISSING")
    elif compounds:
        reject("CLAIM_CARDINALITY_DRIFT")
    for compound in compounds:
        if any(
            compound.get(field)
            for field in ("consumed_event_ids", "consumed_rule_ids", "consumed_span_ids")
        ):
            reject("COMPOUND_RAW_SOURCE_CONSUMPTION")
        if any(member not in atomic_ids for member in compound.get("member_claim_ids", [])):
            reject("COMPOUND_MEMBER_MISSING")

    dispositions = list(graph.get("warning_claim_dispositions", []))
    disposition_by_event = {
        str(item.get("event_id")): item.get("disposition")
        for item in dispositions
        if item.get("event_id")
    }
    for event_id in active_warning_ids:
        if disposition_by_event.get(event_id) != "released":
            reject("ELIGIBLE_WARNING_EVENT_OMITTED")
    for event_id in historical_ids:
        if (
            events.get(event_id, {}).get("event_type") in WARNING_EVENT_FAMILIES
            and disposition_by_event.get(event_id) == "released"
        ):
            reject("HISTORICAL_EVENT_REACTIVATED")

    summary = graph.get("warning_claim_partition_summary", {})
    expected_summary = {
        "atomic_claim_count": len(atomic),
        "compound_claim_count": len(compounds),
        "released_warning_event_count": len(active_warning_ids),
        "owner_count": len({
            owner
            for claim in atomic
            for owner in claim.get("owner_ids", [])
        }),
        "chain_count": len({
            chain
            for claim in atomic
            for chain in claim.get("event_chain_ids", [])
        }),
        "gated_or_withheld_event_count": sum(
            item.get("disposition") in {"gated", "historical", "terminal", "withheld"}
            for item in dispositions
        ),
    }
    if any(summary.get(key) != value for key, value in expected_summary.items()):
        reject("CLAIM_CARDINALITY_DRIFT")

    return {
        "verified": not reasons,
        "reason_codes": reasons,
        **expected_summary,
    }
