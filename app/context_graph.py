from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence

from app.teeth_registry import rule_id_for


GRAPH_CONTRACT_VERSION = "context-graph-referential-integrity/1.0"
RULE_SETS_CONTRACT_VERSION = "teeth-rule-sets-v1"

RELATIONSHIPS = (
    "mother", "father", "mom", "mum", "dad", "sister", "brother",
    "son", "daughter", "child", "husband", "wife", "spouse", "friend",
    "aunt", "uncle", "grandmother", "grandfather", "grandma", "grandpa",
    "cousin", "niece", "nephew",
)

_RELATIONSHIP_PATTERN = "|".join(RELATIONSHIPS)
_JOINT_OWNER_LOSS = re.compile(
    rf"\b(?P<dreamer>my\s+(?:left\s+|right\s+)?tooth)\s+and\s+"
    rf"(?P<other>my\s+(?P<relationship>{_RELATIONSHIP_PATTERN})(?:'s|’s)\s+tooth)\s+"
    r"(?P<verb>fell\s+out|came\s+out)\b",
    re.IGNORECASE,
)
_OTHER_OWNER_LOSS = re.compile(
    rf"\bmy\s+(?P<relationship>{_RELATIONSHIP_PATTERN})(?:'s|’s)\s+"
    r"(?P<tooth>(?:left\s+|right\s+)?(?:tooth|teeth))\s+"
    r"(?P<verb>fell\s+out|came\s+out)\b",
    re.IGNORECASE,
)
_DREAMER_LOSS = re.compile(
    r"\b(?P<subject>(?:(?:one|two|three|four|five|six|seven|eight|nine|ten)\s+of\s+)?"
    r"my\s+(?:(?:left|right)\s+)?(?:tooth|teeth))\s+"
    r"(?P<verb>fell\s+out|came\s+out)\b",
    re.IGNORECASE,
)
_ANOTHER_LOSS = re.compile(
    r"\b(?P<subject>another\s+tooth)\s+(?P<verb>fell\s+out|came\s+out)\b",
    re.IGNORECASE,
)
_FIRM_RETURN = re.compile(
    r"\b(?P<return>(?:the\s+)?same\s+tooth\s+"
    r"(?:fitted|fit|returned|went|came|was\s+put)\s+(?:firmly\s+)?back"
    r"(?:\s+into\s+the\s+same\s+socket)?)\b",
    re.IGNORECASE,
)
_SPEECH = re.compile(
    rf"\bmy\s+(?P<speaker>{_RELATIONSHIP_PATTERN})\s+"
    r"(?P<verb>said|told\s+me(?:\s+that)?)\b",
    re.IGNORECASE,
)


def _span(text: str, start: int, end: int, event_id: str) -> Dict[str, Any]:
    return {
        "span_id": f"span-{event_id}",
        "start": start,
        "end": end,
        "text": text[start:end],
    }


def _overlaps(start: int, end: int, occupied: Sequence[tuple[int, int]]) -> bool:
    return any(start < used_end and used_start < end for used_start, used_end in occupied)


def _loss_actuality(text: str, start: int) -> tuple[str, str, str, bool]:
    prefix = text[max(0, start - 90):start].lower()
    if re.search(r"(?:^|[.!?]\s*)if\s*$", prefix):
        return "nonactual", "conditional_hypothetical", "hypothetical", False
    if re.search(r"\b(?:no|never|did\s+not|didn't)\s*$", prefix):
        return "not_actual", "negated", "negated", False
    return "actual", "experienced", "affirmed", True


def _quantity(subject: str) -> str:
    lowered = subject.lower()
    if re.search(r"\bone\s+of\b", lowered):
        return "one"
    if "teeth" in lowered or re.search(
        r"\b(?:two|three|four|five|six|seven|eight|nine|ten)\b", lowered
    ):
        return "multiple"
    return "one"


def _entity(entity_id: str, entity_type: str, **values: Any) -> Dict[str, Any]:
    return {"entity_id": entity_id, "entity_type": entity_type, **values}


def _put_entity(inventory: MutableMapping[str, Dict[str, Any]], record: Dict[str, Any]) -> None:
    inventory.setdefault(record["entity_id"], record)


def _tooth_chain_id(tooth_id: str) -> str:
    return f"chain-{tooth_id}"


def _extract_losses(dream: str, attempt_records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    raw: List[Dict[str, Any]] = []
    occupied: List[tuple[int, int]] = []

    for match in _JOINT_OWNER_LOSS.finditer(dream):
        relationship = match.group("relationship").lower()
        verb_end = match.end("verb")
        raw.extend(
            [
                {
                    "start": match.start("dreamer"),
                    "end": match.end("dreamer"),
                    "owner": "dreamer",
                    "relationship": "",
                    "subject": match.group("dreamer"),
                    "position": "left" if "left" in match.group("dreamer").lower() else "",
                },
                {
                    "start": match.start("other"),
                    "end": verb_end,
                    "owner": relationship,
                    "relationship": relationship,
                    "subject": match.group("other"),
                    "position": "",
                },
            ]
        )
        occupied.append((match.start(), match.end()))

    for pattern, owner_kind in (
        (_OTHER_OWNER_LOSS, "other"),
        (_DREAMER_LOSS, "dreamer"),
        (_ANOTHER_LOSS, "another"),
    ):
        for match in pattern.finditer(dream):
            if _overlaps(match.start(), match.end(), occupied):
                continue
            subject = match.group("subject") if "subject" in match.groupdict() else match.group("tooth")
            relationship = (
                match.group("relationship").lower()
                if owner_kind == "other"
                else ""
            )
            raw.append(
                {
                    "start": match.start(),
                    "end": match.end(),
                    "owner": relationship if owner_kind == "other" else "dreamer",
                    "relationship": relationship,
                    "subject": subject,
                    "position": "left" if "left" in subject.lower() else "",
                    "another": owner_kind == "another",
                }
            )
            occupied.append((match.start(), match.end()))

    raw.sort(key=lambda item: (item["start"], item["end"]))
    referenced_teeth = {
        target
        for record in attempt_records
        for target in (
            record.get("target_tooth_ids_or_ambiguous")
            if isinstance(record.get("target_tooth_ids_or_ambiguous"), list)
            else []
        )
    }
    explicit_other_loss = any(item["owner"] != "dreamer" for item in raw)
    explicit_other_tooth = explicit_other_loss or any(
        not tooth.startswith(("tooth-", "left-tooth", "right-tooth"))
        for tooth in referenced_teeth
    )
    multiple_losses = len(raw) > 1

    events: List[Dict[str, Any]] = []
    for index, item in enumerate(raw, start=1):
        owner = item["owner"]
        if item.get("another"):
            tooth_id = "tooth-2"
        elif item.get("position") == "left":
            tooth_id = "left-tooth-1"
        elif owner != "dreamer":
            tooth_id = f"{owner}-tooth-1"
        elif explicit_other_loss:
            tooth_id = "dreamer-tooth-1"
        else:
            tooth_id = "tooth-1"

        if item.get("another"):
            event_id = "second-loss-1"
        elif tooth_id == "left-tooth-1" and multiple_losses:
            event_id = "left-loss-1"
        elif (multiple_losses or explicit_other_tooth) and owner == "dreamer":
            event_id = "dreamer-loss-1"
        elif multiple_losses and owner != "dreamer":
            event_id = f"{owner}-loss-1"
        else:
            event_id = "loss-1"

        actuality, modality, polarity, eligible = _loss_actuality(dream, item["start"])
        if not eligible and modality == "conditional_hypothetical":
            event_id = "hypothetical-loss-1"
        quantity = _quantity(item["subject"])
        events.append(
            {
                "event_id": event_id,
                "event_type": "tooth_loss",
                "actor_id_or_null": None,
                "owner_id_or_ambiguous": owner,
                "target_entity_ids_or_ambiguous": [tooth_id],
                "event_chain_id_or_null": _tooth_chain_id(tooth_id) if eligible else None,
                "scene_id": "scene-1",
                "phase": "dream" if eligible else "dream_speech_or_thought",
                "channel": "narrative" if eligible else "thought_or_hypothetical",
                "polarity": polarity,
                "modality": modality,
                "actuality": actuality,
                "completion": "completed" if eligible else "not_completed",
                "quantity": quantity,
                "doctrine_eligible": eligible,
                "source_span": _span(dream, item["start"], item["end"], event_id),
            }
        )
    return events


def _event_from_attempt(dream: str, record: Mapping[str, Any]) -> Dict[str, Any]:
    span = record.get("source_span") or {}
    event_id = str(record.get("attempt_id") or "attempt-unknown")
    return {
        "event_id": event_id,
        "event_type": "restoration_attempt",
        "actor_id_or_null": record.get("actor_id_or_ambiguous"),
        "owner_id_or_ambiguous": record.get("owner_id_or_ambiguous"),
        "target_entity_ids_or_ambiguous": record.get("target_tooth_ids_or_ambiguous"),
        "event_chain_id_or_null": record.get("event_chain_id_or_null"),
        "scene_id": record.get("scene_id"),
        "phase": record.get("phase"),
        "channel": record.get("channel"),
        "polarity": record.get("polarity"),
        "modality": record.get("modality"),
        "actuality": record.get("actuality"),
        "completion": record.get("completion"),
        "quantity": "one",
        "doctrine_eligible": bool(record.get("narration_eligibility")),
        "source_span": _span(
            dream,
            int(span.get("start") or 0),
            int(span.get("end") or 0),
            event_id,
        ),
    }


def build_context_graph(dream: str, context: Mapping[str, Any]) -> Dict[str, Any]:
    attempts = [dict(record) for record in context.get("restoration_attempt_records", [])]
    losses = _extract_losses(dream, attempts)

    if not losses and context.get("explicit_pull_removal"):
        lowered = dream.lower()
        tooth_match = re.search(r"\b(?:my\s+)?(?:tooth|teeth|molar|molars)\b", lowered)
        start, end = tooth_match.span() if tooth_match else (0, len(dream))
        losses.append(
            {
                "event_id": "loss-1",
                "event_type": "tooth_loss",
                "actor_id_or_null": "dreamer" if context.get("removal_actor") == "self" else "ambiguous",
                "owner_id_or_ambiguous": "dreamer",
                "target_entity_ids_or_ambiguous": ["tooth-1"],
                "event_chain_id_or_null": "chain-tooth-1",
                "scene_id": "scene-1",
                "phase": "dream",
                "channel": "narrative",
                "polarity": "affirmed",
                "modality": "experienced",
                "actuality": "actual",
                "completion": "completed",
                "quantity": context.get("count", "one"),
                "doctrine_eligible": True,
                "source_span": _span(dream, start, end, "loss-1"),
            }
        )

    events = list(losses)
    for match_index, match in enumerate(_SPEECH.finditer(dream), start=1):
        event_id = f"speech-{match_index}"
        speaker = match.group("speaker").lower()
        events.append(
            {
                "event_id": event_id,
                "event_type": "speech",
                "actor_id_or_null": speaker,
                "owner_id_or_ambiguous": "ambiguous",
                "target_entity_ids_or_ambiguous": [],
                "event_chain_id_or_null": None,
                "scene_id": "scene-1",
                "phase": "dream_speech_or_thought",
                "channel": "reported_speech" if "told" in match.group("verb").lower() else "quoted_speech",
                "polarity": "affirmed",
                "modality": "reported",
                "actuality": "speech_actual",
                "completion": "completed",
                "quantity": "none",
                "doctrine_eligible": False,
                "source_span": _span(dream, match.start(), match.end(), event_id),
            }
        )

    events.extend(_event_from_attempt(dream, record) for record in attempts)

    firm_match = _FIRM_RETURN.search(dream)
    if firm_match and attempts:
        target = attempts[0].get("target_tooth_ids_or_ambiguous")
        target_ids = target if isinstance(target, list) else []
        if target_ids:
            event_id = "firm-return-1"
            tooth_id = target_ids[0]
            events.append(
                {
                    "event_id": event_id,
                    "event_type": "firm_same_tooth_return",
                    "actor_id_or_null": attempts[0].get("actor_id_or_ambiguous"),
                    "owner_id_or_ambiguous": attempts[0].get("owner_id_or_ambiguous"),
                    "target_entity_ids_or_ambiguous": [tooth_id],
                    "event_chain_id_or_null": _tooth_chain_id(tooth_id),
                    "scene_id": "scene-1",
                    "phase": "dream",
                    "channel": "narrative",
                    "polarity": "affirmed",
                    "modality": "experienced",
                    "actuality": "actual",
                    "completion": "completed",
                    "quantity": "one",
                    "doctrine_eligible": True,
                    "source_span": _span(dream, firm_match.start(), firm_match.end(), event_id),
                }
            )

    events.sort(key=lambda event: (event["source_span"]["start"], event["source_span"]["end"]))

    entities: Dict[str, Dict[str, Any]] = {}
    for event in events:
        for person_id in (event.get("actor_id_or_null"), event.get("owner_id_or_ambiguous")):
            if person_id and person_id not in {"ambiguous", "unknown"}:
                _put_entity(entities, _entity(str(person_id), "person"))
        targets = event.get("target_entity_ids_or_ambiguous")
        if isinstance(targets, list):
            owner_id = event.get("owner_id_or_ambiguous")
            for tooth_id in targets:
                _put_entity(
                    entities,
                    _entity(
                        str(tooth_id),
                        "tooth",
                        owner_id_or_ambiguous=owner_id,
                        natural_status="human_natural",
                    ),
                )

    chains: Dict[str, Dict[str, Any]] = {}
    for event in events:
        chain_id = event.get("event_chain_id_or_null")
        if not chain_id:
            continue
        chain = chains.setdefault(
            chain_id,
            {
                "event_chain_id": chain_id,
                "owner_id_or_ambiguous": event.get("owner_id_or_ambiguous"),
                "entity_ids": [],
                "event_ids": [],
                "scene_ids": [],
                "phase": event.get("phase"),
            },
        )
        for target in event.get("target_entity_ids_or_ambiguous") or []:
            if target not in chain["entity_ids"]:
                chain["entity_ids"].append(target)
        chain["event_ids"].append(event["event_id"])
        if event.get("scene_id") not in chain["scene_ids"]:
            chain["scene_ids"].append(event.get("scene_id"))

    eligible_losses = [
        event for event in events
        if event["event_type"] == "tooth_loss" and event["doctrine_eligible"]
    ]
    omitted_losses = [
        event["event_id"] for event in events
        if event["event_type"] == "tooth_loss" and not event["doctrine_eligible"]
    ]
    total_loss_scope = sum(2 if event.get("quantity") == "multiple" else 1 for event in eligible_losses)
    warning_count = "multiple_people" if total_loss_scope > 1 else (
        "one_person" if total_loss_scope == 1 else ""
    )
    owners = sorted({event["owner_id_or_ambiguous"] for event in eligible_losses})
    owner_value = owners[0] if len(owners) == 1 else ("mixed" if owners else "unknown")
    contributors = [event["event_id"] for event in eligible_losses]
    aggregate_derivations = [
        {
            "aggregate_id": f"aggregate-{field}",
            "field": field,
            "value": value,
            "contributing_event_ids": contributors,
            "omitted_event_ids": omitted_losses,
            "derivation_policy": "eligible_actual_loss_events",
        }
        for field, value in (
            ("active_fallout", bool(eligible_losses)),
            ("owner", owner_value),
            ("count", "multiple" if total_loss_scope > 1 else ("one" if total_loss_scope == 1 else "unknown")),
            ("warning_count", warning_count),
        )
    ]

    terminal_frontiers: List[Dict[str, Any]] = []
    state_types = {"tooth_loss", "firm_same_tooth_return"}
    for chain in chains.values():
        candidates = [
            event for event in events
            if event["event_id"] in chain["event_ids"]
            and event["event_type"] in state_types
            and event["doctrine_eligible"]
        ]
        if candidates:
            terminal = max(candidates, key=lambda event: event["source_span"]["start"])
            terminal_frontiers.append(
                {
                    "frontier_id": f"frontier-{chain['event_chain_id']}",
                    "event_chain_id": chain["event_chain_id"],
                    "terminal_event_id": terminal["event_id"],
                    "historical_event_ids": [
                        event_id for event_id in chain["event_ids"]
                        if event_id != terminal["event_id"]
                    ],
                }
            )

    return {
        "contract_version": GRAPH_CONTRACT_VERSION,
        "entity_inventory": list(entities.values()),
        "event_inventory": events,
        "event_chain_inventory": list(chains.values()),
        "restoration_attempt_records": attempts,
        "aggregate_derivations": aggregate_derivations,
        "rule_sets": {
            "contract_version": RULE_SETS_CONTRACT_VERSION,
            "public_applied": [],
            "warning_active": [],
            "matched_historical": [],
            "structural": [],
            "withheld_unresolved": [],
        },
        "claim_manifest": [],
        "terminal_frontiers": terminal_frontiers,
        "integrity": {"verified": False, "reason_codes": ["NOT_FINALIZED"]},
    }


def apply_loss_projection(context: MutableMapping[str, Any], graph: Mapping[str, Any]) -> None:
    aggregate = {
        item["field"]: item["value"]
        for item in graph.get("aggregate_derivations", [])
    }
    if not aggregate.get("active_fallout"):
        return
    context["count"] = aggregate["count"]
    owner = aggregate["owner"]
    if owner == "dreamer":
        context["owner"] = "dreamer"
        context["owner_relationship"] = ""
    elif owner == "mixed":
        context["owner"] = "mixed"
        context["owner_relationship"] = ""
    else:
        context["owner"] = "other"
        context["owner_relationship"] = owner


def _rule_record(rule_id: str, event_ids: Iterable[str], events: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    source_ids = [event_id for event_id in event_ids if event_id in events]
    return {
        "rule_id": rule_id,
        "source_event_ids": source_ids,
        "source_span_ids": [events[event_id]["source_span"]["span_id"] for event_id in source_ids],
    }


def finalize_context_graph(
    graph: MutableMapping[str, Any],
    result: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> Dict[str, Any]:
    events = {event["event_id"]: event for event in graph["event_inventory"]}
    eligible_losses = [
        event for event in events.values()
        if event["event_type"] == "tooth_loss" and event["doctrine_eligible"]
    ]
    loss_ids = [event["event_id"] for event in eligible_losses]
    dreamer_loss_ids = [
        event["event_id"] for event in eligible_losses
        if event.get("owner_id_or_ambiguous") == "dreamer"
    ]
    other_loss_ids = [
        event["event_id"] for event in eligible_losses
        if event.get("owner_id_or_ambiguous") != "dreamer"
    ]
    terminal_ids = [
        frontier["terminal_event_id"] for frontier in graph["terminal_frontiers"]
        if events.get(frontier["terminal_event_id"], {}).get("event_type") == "firm_same_tooth_return"
    ]

    public_records: List[Dict[str, Any]] = []
    for rule_id in result.get("applied_rule_ids", []):
        if rule_id == rule_id_for(registry, "terminal_ending"):
            source_ids = terminal_ids
        elif rule_id == rule_id_for(registry, "own_fallout"):
            source_ids = dreamer_loss_ids
        elif rule_id == rule_id_for(registry, "other_fallout"):
            source_ids = other_loss_ids
        else:
            source_ids = loss_ids
        if not source_ids:
            source_ids = [
                event["event_id"] for event in events.values()
                if event.get("doctrine_eligible")
            ][:1]
        public_records.append(_rule_record(rule_id, source_ids, events))

    historical: List[Dict[str, Any]] = []
    if terminal_ids and loss_ids:
        historical_ids: List[str] = []
        if any(event["owner_id_or_ambiguous"] == "dreamer" for event in eligible_losses):
            historical_ids.append(rule_id_for(registry, "own_fallout"))
        if any(event["owner_id_or_ambiguous"] != "dreamer" for event in eligible_losses):
            historical_ids.append(rule_id_for(registry, "other_fallout"))
        historical_ids.append(
            rule_id_for(
                registry,
                "multiple_fallout" if result.get("count") == "multiple" else "one_fallout",
            )
        )
        historical = [
            _rule_record(rule_id, loss_ids, events)
            for rule_id in historical_ids if rule_id
        ]

    unresolved = [
        _rule_record(rule_id, terminal_ids or loss_ids, events)
        for rule_id in result.get("unresolved_rule_ids", [])
    ]
    graph["rule_sets"] = {
        "contract_version": RULE_SETS_CONTRACT_VERSION,
        "public_applied": public_records,
        "warning_active": public_records if result.get("active_warning") else [],
        "matched_historical": historical,
        "structural": [record for record in public_records if record["rule_id"] == "TEETH-END-TERMINAL"],
        "withheld_unresolved": unresolved,
    }

    claims: List[Dict[str, Any]] = []
    public_rule_ids = [record["rule_id"] for record in public_records]
    if result.get("active_warning") and loss_ids:
        claims.append(
            {
                "claim_id": "claim-warning-1",
                "claim_type": "tradition_scoped_warning",
                "released": True,
                "consumed_event_ids": loss_ids,
                "consumed_attempt_ids": [],
                "consumed_rule_ids": public_rule_ids,
                "consumed_span_ids": [events[event_id]["source_span"]["span_id"] for event_id in loss_ids],
                "comparison_scope": (
                    "owner_and_event_bound_aggregate"
                    if (
                        len({event["owner_id_or_ambiguous"] for event in eligible_losses}) > 1
                        or len({event["event_chain_id_or_null"] for event in eligible_losses}) > 1
                    )
                    else ""
                ),
            }
        )
    if terminal_ids:
        claims.append(
            {
                "claim_id": "claim-terminal-1",
                "claim_type": "terminal_state_without_consequence",
                "released": True,
                "consumed_event_ids": terminal_ids,
                "consumed_attempt_ids": [],
                "consumed_rule_ids": public_rule_ids,
                "consumed_span_ids": [events[event_id]["source_span"]["span_id"] for event_id in terminal_ids],
            }
        )
    consumed_attempts = list(result.get("narration_consumed_attempt_ids", []))
    if consumed_attempts:
        claims.append(
            {
                "claim_id": "claim-attempt-1",
                "claim_type": "structural_attempt_narration",
                "released": True,
                "consumed_event_ids": consumed_attempts,
                "consumed_attempt_ids": consumed_attempts,
                "consumed_rule_ids": public_rule_ids,
                "consumed_span_ids": [events[event_id]["source_span"]["span_id"] for event_id in consumed_attempts],
            }
        )
    graph["claim_manifest"] = claims
    graph["integrity"] = validate_context_graph(graph)
    return dict(graph)


def validate_context_graph(graph: Mapping[str, Any]) -> Dict[str, Any]:
    reasons: List[str] = []

    def inventory(values: Sequence[Mapping[str, Any]], key: str) -> Dict[str, Mapping[str, Any]]:
        result: Dict[str, Mapping[str, Any]] = {}
        for value in values:
            identifier = value.get(key)
            if not identifier or identifier in result:
                reasons.append("DUPLICATE_TYPED_ID")
            elif identifier:
                result[str(identifier)] = value
        return result

    entities = inventory(graph.get("entity_inventory", []), "entity_id")
    events = inventory(graph.get("event_inventory", []), "event_id")
    chains = inventory(graph.get("event_chain_inventory", []), "event_chain_id")
    attempts = inventory(graph.get("restoration_attempt_records", []), "attempt_id")
    spans = {
        event.get("source_span", {}).get("span_id")
        for event in events.values()
        if event.get("source_span", {}).get("span_id")
    }

    for event in events.values():
        for entity_id in (event.get("actor_id_or_null"), event.get("owner_id_or_ambiguous")):
            if entity_id and entity_id not in {"ambiguous", "unknown"} and entity_id not in entities:
                reasons.append("DANGLING_ENTITY_REFERENCE")
        targets = event.get("target_entity_ids_or_ambiguous")
        if isinstance(targets, list) and any(target not in entities for target in targets):
            reasons.append("DANGLING_ENTITY_REFERENCE")
        chain_id = event.get("event_chain_id_or_null")
        if chain_id and (
            chain_id not in chains or event["event_id"] not in chains[chain_id].get("event_ids", [])
        ):
            reasons.append("DANGLING_CHAIN_REFERENCE")

    for chain in chains.values():
        event_ids = chain.get("event_ids", [])
        entity_ids = chain.get("entity_ids", [])
        if any(event_id not in events for event_id in event_ids):
            reasons.append("DANGLING_EVENT_REFERENCE")
        if any(entity_id not in entities for entity_id in entity_ids):
            reasons.append("DANGLING_ENTITY_REFERENCE")
        for event_id in event_ids:
            event = events.get(event_id)
            if not event:
                continue
            if event.get("event_chain_id_or_null") != chain.get("event_chain_id"):
                reasons.append("DANGLING_CHAIN_REFERENCE")
            event_owner = event.get("owner_id_or_ambiguous")
            chain_owner = chain.get("owner_id_or_ambiguous")
            if (
                event_owner not in {None, "ambiguous", "unknown"}
                and chain_owner not in {None, "ambiguous", "unknown"}
                and event_owner != chain_owner
            ):
                reasons.append("CROSS_BOUNDARY_REFERENCE")
            if event.get("phase") != chain.get("phase"):
                reasons.append("CROSS_BOUNDARY_REFERENCE")
            targets = event.get("target_entity_ids_or_ambiguous")
            if isinstance(targets, list) and any(target not in entity_ids for target in targets):
                reasons.append("CROSS_BOUNDARY_REFERENCE")

    for attempt in attempts.values():
        target = attempt.get("target_tooth_ids_or_ambiguous")
        if isinstance(target, list) and any(entity_id not in entities for entity_id in target):
            reasons.append("DANGLING_ENTITY_REFERENCE")
        chain_id = attempt.get("event_chain_id_or_null")
        if chain_id and chain_id not in chains:
            reasons.append("DANGLING_CHAIN_REFERENCE")
        if "ambiguous_binding" in attempt.get("ineligibility_reasons", []) and (
            attempt.get("owner_id_or_ambiguous") != "ambiguous"
            or attempt.get("target_tooth_ids_or_ambiguous") != "ambiguous"
            or attempt.get("event_chain_id_or_null") is not None
        ):
            reasons.append("AMBIGUITY_FABRICATED")

    eligible_loss_events = {
        event_id: event for event_id, event in events.items()
        if event.get("event_type") == "tooth_loss" and event.get("doctrine_eligible")
    }
    for aggregate in graph.get("aggregate_derivations", []):
        referenced = list(aggregate.get("contributing_event_ids", [])) + list(aggregate.get("omitted_event_ids", []))
        if any(event_id not in events for event_id in referenced):
            reasons.append("DANGLING_EVENT_REFERENCE")
        explicit_loss_ids = {
            event_id for event_id, event in events.items() if event.get("event_type") == "tooth_loss"
        }
        if explicit_loss_ids - set(referenced):
            reasons.append("AGGREGATE_EVENT_OMITTED")
        contributors = [
            eligible_loss_events[event_id]
            for event_id in aggregate.get("contributing_event_ids", [])
            if event_id in eligible_loss_events
        ]
        cardinality = sum(
            2 if event.get("quantity") == "multiple" else 1
            for event in contributors
        )
        expected_by_field = {
            "active_fallout": bool(cardinality),
            "count": "multiple" if cardinality > 1 else ("one" if cardinality == 1 else "unknown"),
            "warning_count": "multiple_people" if cardinality > 1 else ("one_person" if cardinality == 1 else ""),
        }
        field = aggregate.get("field")
        if field in expected_by_field and aggregate.get("value") != expected_by_field[field]:
            reasons.append("AGGREGATE_CARDINALITY_MISMATCH")

    known_rule_ids = set()
    for partition, records in graph.get("rule_sets", {}).items():
        if partition == "contract_version":
            continue
        for record in records:
            known_rule_ids.add(record.get("rule_id"))
            if not record.get("source_event_ids") or any(
                event_id not in events for event_id in record.get("source_event_ids", [])
            ):
                reasons.append("RULE_EVENT_PROVENANCE_MISSING")
            if any(span_id not in spans for span_id in record.get("source_span_ids", [])):
                reasons.append("RULE_EVENT_PROVENANCE_MISSING")
            if any(
                not events[event_id].get("doctrine_eligible")
                for event_id in record.get("source_event_ids", [])
                if event_id in events
            ):
                reasons.append("RULE_EVENT_PROVENANCE_MISSING")

    for claim in graph.get("claim_manifest", []):
        event_ids = claim.get("consumed_event_ids", [])
        attempt_ids = claim.get("consumed_attempt_ids", [])
        rule_ids = claim.get("consumed_rule_ids", [])
        span_ids = claim.get("consumed_span_ids", [])
        if (
            any(event_id not in events for event_id in event_ids)
            or any(attempt_id not in attempts for attempt_id in attempt_ids)
            or any(rule_id not in known_rule_ids for rule_id in rule_ids)
            or any(span_id not in spans for span_id in span_ids)
        ):
            reasons.append("CLAIM_REFERENCE_MISSING")
        if claim.get("released") and claim.get("claim_type") == "structural_attempt_narration" and (
            not event_ids or not attempt_ids or not rule_ids or not span_ids
        ):
            reasons.append("CLAIM_REFERENCE_MISSING")
        if any(not attempts[attempt_id].get("narration_eligibility") for attempt_id in attempt_ids if attempt_id in attempts):
            reasons.append("INELIGIBLE_ATTEMPT_CONSUMED")
        referenced_owners = {
            events[event_id].get("owner_id_or_ambiguous")
            for event_id in event_ids if event_id in events
        } - {None, "ambiguous", "unknown"}
        referenced_chains = {
            events[event_id].get("event_chain_id_or_null")
            for event_id in event_ids if event_id in events
        } - {None}
        if (
            (len(referenced_owners) > 1 or len(referenced_chains) > 1)
            and not claim.get("comparison_scope")
        ):
            reasons.append("CROSS_BOUNDARY_REFERENCE")

    for frontier in graph.get("terminal_frontiers", []):
        chain = chains.get(frontier.get("event_chain_id"))
        event_id = frontier.get("terminal_event_id")
        if not chain or event_id not in events or event_id not in chain.get("event_ids", []):
            reasons.append("TERMINAL_FRONTIER_DANGLING")
            continue
        if any(
            historical_id not in events or historical_id not in chain.get("event_ids", [])
            for historical_id in frontier.get("historical_event_ids", [])
        ):
            reasons.append("TERMINAL_FRONTIER_DANGLING")
        eligible_state_events = [
            events[candidate]
            for candidate in chain.get("event_ids", [])
            if candidate in events
            and events[candidate].get("doctrine_eligible")
            and events[candidate].get("event_type") in {"tooth_loss", "firm_same_tooth_return"}
        ]
        if eligible_state_events:
            latest = max(eligible_state_events, key=lambda item: item["source_span"]["start"])
            if latest["event_id"] != event_id:
                reasons.append("TERMINAL_FRONTIER_DANGLING")

    unique_reasons = list(dict.fromkeys(reasons))
    return {
        "verified": not unique_reasons,
        "reason_codes": unique_reasons,
        "entity_count": len(entities),
        "event_count": len(events),
        "chain_count": len(chains),
        "claim_count": len(graph.get("claim_manifest", [])),
    }
