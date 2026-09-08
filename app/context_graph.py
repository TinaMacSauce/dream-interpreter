from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence

from app.teeth_registry import rule_id_for
from app.claim_provenance import finalize_claim_provenance, validate_claim_provenance
from app.condition_provenance import (
    CONDITION_PROVENANCE_CONTRACT_VERSION,
    validate_condition_provenance,
)
from app.warning_claim_partition import (
    WARNING_CLAIM_PARTITION_CONTRACT_VERSION,
    WARNING_EVENT_FAMILIES,
    validate_warning_claim_partition,
)


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
_NEGATED_LOSS = re.compile(
    r"\b(?P<subject>my\s+(?:tooth|teeth)|no\s+tooth|none|neither)\s+"
    r"(?P<negation>did\s+not|didn't|never)?\s*(?P<verb>fall\s+out|fell\s+out)\b",
    re.IGNORECASE,
)
_IMPLICIT_NEGATED_LOSS = re.compile(r"\b(?:did\s+not|didn't|never)\s+fall\s+out\b", re.IGNORECASE)
_LOOSE = re.compile(
    rf"\b(?P<subject>(?:(?P<number>two)\s+of\s+)?my\s+(?:(?P<relationship>{_RELATIONSHIP_PATTERN})(?:'s|’s)\s+)?"
    r"(?P<tooth>tooth|teeth))\s+(?P<verb>was|were|is|became)\s+"
    r"(?P<negation>not\s+)?(?P<state>loose|wobbly)\b",
    re.IGNORECASE,
)
_NO_TOOTH_LOOSE = re.compile(
    r"\b(?P<subject>no\s+tooth)\s+(?:was|is)\s+(?P<state>loose|wobbly)\b",
    re.IGNORECASE,
)
_GUM_BLOOD = re.compile(r"\b(?P<subject>my\s+gums)\s+were\s+bleeding\b", re.IGNORECASE)
_PRONOUN_LOSS = re.compile(r"\b(?P<subject>it|the\s+tooth)\s+(?P<verb>fell\s+out|came\s+out)\b", re.IGNORECASE)
_RETAINED_STATE = re.compile(
    r"\b(?P<state>(?:every\s+tooth|my\s+own\s+tooth)\s+stayed\s+firm|stayed\s+in\s+my\s+mouth)\b",
    re.IGNORECASE,
)
_PAIN = re.compile(r"\b(?:it\s+)?(?P<pain>hurt(?:\s+badly)?|was\s+painful)\b", re.IGNORECASE)
_TOOTH_BLOOD = re.compile(
    r"\b(?P<blood>blood\s+on\s+(?:the\s+)?(?:fallen\s+)?tooth)\b",
    re.IGNORECASE,
)
_EXTERNAL_PULL = re.compile(
    rf"\bmy\s+(?P<actor>{_RELATIONSHIP_PATTERN})\s+pulled\s+my\s+(?:tooth|teeth)\s+out\b",
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


def _quoted_ranges(text: str) -> List[tuple[int, int]]:
    ranges: List[tuple[int, int]] = []
    opened: int | None = None
    for index, character in enumerate(text):
        if character not in {'"', "“", "”", "'", "‘", "’"}:
            continue
        if character in {"'", "’"} and index > 0 and index + 1 < len(text):
            if text[index - 1].isalnum() and text[index + 1].isalnum():
                continue
        if opened is None:
            opened = index
        else:
            ranges.append((opened, index + 1))
            opened = None
    return ranges


def _inside_ranges(start: int, end: int, ranges: Sequence[tuple[int, int]]) -> bool:
    return any(range_start <= start and end <= range_end for range_start, range_end in ranges)


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


def _quantity_cardinality(subject: str) -> int:
    words = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    }
    lowered = subject.lower()
    for word, value in words.items():
        if re.search(rf"\b{word}\b", lowered):
            return value
    return 2 if "teeth" in lowered else 1


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
                "quantity_cardinality": _quantity_cardinality(item["subject"]),
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
        "quantity_cardinality": 1,
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
        pull_match = _EXTERNAL_PULL.search(dream)
        tooth_match = re.search(r"\b(?:my\s+)?(?:tooth|teeth|molar|molars)\b", lowered)
        start, end = pull_match.span() if pull_match else (tooth_match.span() if tooth_match else (0, len(dream)))
        actor = pull_match.group("actor").lower() if pull_match else (
            "dreamer" if context.get("removal_actor") == "self" else "ambiguous"
        )
        losses.append(
            {
                "event_id": "loss-1",
                "event_type": "tooth_loss",
                "actor_id_or_null": actor,
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
                "quantity_cardinality": 2 if context.get("count") == "multiple" else 1,
                "doctrine_eligible": True,
                "source_span": _span(dream, start, end, "loss-1"),
            }
        )

    events = list(losses)

    def add_fact_event(
        match: re.Match[str], event_id: str, event_type: str, *, eligible: bool,
        target_id: str = "tooth-1", chain_id: str | None = "chain-tooth-1",
        owner: str = "dreamer", polarity: str = "affirmed",
        modality: str = "experienced", completion: str = "observed",
        actuality: str | None = None, phase: str = "dream",
        channel: str = "narrative", quantity: str = "one",
        quantity_cardinality: int = 1,
    ) -> None:
        events.append({
            "event_id": event_id,
            "event_type": event_type,
            "actor_id_or_null": None,
            "owner_id_or_ambiguous": owner,
            "target_entity_ids_or_ambiguous": [target_id],
            "event_chain_id_or_null": chain_id if eligible else None,
            "scene_id": "scene-1",
            "phase": phase,
            "channel": channel,
            "polarity": polarity,
            "modality": modality,
            "actuality": actuality or ("actual" if eligible else "not_actual"),
            "completion": completion,
            "quantity": quantity,
            "quantity_cardinality": quantity_cardinality,
            "doctrine_eligible": eligible,
            "source_span": _span(dream, match.start(), match.end(), event_id),
        })

    quoted_ranges = _quoted_ranges(dream)

    def condition_scope(match: re.Match[str], negated: bool = False) -> Dict[str, Any]:
        prefix = dream[max(0, match.start() - 80):match.start()].lower()
        if _inside_ranges(match.start(), match.end(), quoted_ranges):
            return {
                "eligible": False, "polarity": "affirmed", "modality": "quoted",
                "actuality": "nonactual", "phase": "dream_speech_or_thought",
                "channel": "quoted_speech",
            }
        if re.search(r"(?:^|[.!?]\s*)if\s*$", prefix):
            return {
                "eligible": False, "polarity": "affirmed",
                "modality": "conditional_hypothetical", "actuality": "nonactual",
                "phase": "dream_speech_or_thought", "channel": "hypothetical",
            }
        if negated:
            return {
                "eligible": False, "polarity": "negated", "modality": "negated",
                "actuality": "not_actual", "phase": "dream", "channel": "narrative",
            }
        return {
            "eligible": True, "polarity": "affirmed", "modality": "experienced",
            "actuality": "actual", "phase": "dream", "channel": "narrative",
        }

    condition_events: List[Dict[str, Any]] = []

    def add_loose_condition(match: re.Match[str], *, forced_negated: bool = False) -> None:
        subject = match.group("subject")
        groups = match.groupdict()
        relationship = str(groups.get("relationship") or "").lower()
        owner = relationship or "dreamer"
        negated = forced_negated or bool(groups.get("negation")) or subject.lower().startswith("no ")
        scope = condition_scope(match, negated)
        quoted = scope["modality"] == "quoted"
        hypothetical = scope["modality"] == "conditional_hypothetical"
        if quoted:
            prior_speech = [speech for speech in _SPEECH.finditer(dream) if speech.start() < match.start()]
            owner = prior_speech[-1].group("speaker").lower() if prior_speech else "ambiguous"
        cardinality = 2 if groups.get("number") or "teeth" in subject.lower() else 1
        if owner != "dreamer" and owner != "ambiguous":
            targets = [f"{owner}-tooth-{index}" for index in range(1, cardinality + 1)]
        else:
            targets = [f"tooth-{index}" for index in range(1, cardinality + 1)]
        if negated:
            event_id = f"{owner}-negated-loose-1" if owner != "dreamer" else "negated-loose-1"
        elif hypothetical:
            event_id = "hypothetical-loose-1"
        elif quoted:
            event_id = "quoted-loose-1"
        elif owner != "dreamer":
            event_id = f"{owner}-loose-1"
        else:
            event_id = "loose-1"
        chain_id = _tooth_chain_id(targets[0]) if scope["eligible"] else None
        span_start = match.start()
        if hypothetical:
            prefix_start = max(0, match.start() - 12)
            if_prefix = re.search(r"\bIf\s+$", dream[prefix_start:match.start()], re.IGNORECASE)
            if if_prefix:
                span_start = prefix_start + if_prefix.start()
        events.append({
            "event_id": event_id,
            "event_type": "loose_tooth_condition",
            "actor_id_or_null": None,
            "owner_id_or_ambiguous": owner,
            "target_entity_ids_or_ambiguous": targets,
            "event_chain_id_or_null": chain_id,
            "scene_id": "scene-1",
            "phase": scope["phase"],
            "channel": scope["channel"],
            "polarity": scope["polarity"],
            "modality": scope["modality"],
            "actuality": scope["actuality"],
            "completion": "observed" if scope["eligible"] else "not_observed",
            "quantity": "multiple" if cardinality > 1 else "one",
            "quantity_cardinality": cardinality,
            "doctrine_eligible": scope["eligible"],
            "state_alias": str(groups.get("state") or "loose").lower(),
            "source_span": _span(dream, span_start, match.end(), event_id),
        })
        condition_events.append(events[-1])

    for loose_match in _LOOSE.finditer(dream):
        add_loose_condition(loose_match)
    for loose_match in _NO_TOOTH_LOOSE.finditer(dream):
        if not _overlaps(
            loose_match.start(),
            loose_match.end(),
            [(e["source_span"]["start"], e["source_span"]["end"]) for e in condition_events],
        ):
            add_loose_condition(loose_match, forced_negated=True)

    for gum_index, gum_match in enumerate(_GUM_BLOOD.finditer(dream), start=1):
        scope = condition_scope(gum_match)
        event_id = "gum-bleeding-1" if gum_index == 1 else f"gum-bleeding-{gum_index}"
        add_fact_event(
            gum_match, event_id, "gum_bleeding_condition", eligible=scope["eligible"],
            target_id="gums-1", chain_id="chain-gums-1", polarity=scope["polarity"],
            modality=scope["modality"], completion="observed" if scope["eligible"] else "not_observed",
            actuality=scope["actuality"], phase=scope["phase"], channel=scope["channel"],
        )
        condition_events.append(events[-1])

    retained_match = _RETAINED_STATE.search(dream)
    if retained_match:
        prior_conditions = [
            event for event in condition_events
            if event["source_span"]["start"] < retained_match.start()
            and event.get("owner_id_or_ambiguous") == "dreamer"
            and event.get("event_type") == "loose_tooth_condition"
        ]
        bound = prior_conditions[-1] if prior_conditions else None
        target_id = (
            bound["target_entity_ids_or_ambiguous"][0]
            if bound and isinstance(bound.get("target_entity_ids_or_ambiguous"), list)
            else "tooth-1"
        )
        terminal_state = "retained_loose" if bound and bound.get("event_type") == "loose_tooth_condition" else "retained_firm"
        add_fact_event(
            retained_match, "retained-state-1", "retained_tooth_state", eligible=True,
            target_id=target_id, chain_id=_tooth_chain_id(target_id), completion="retained",
        )
        events[-1]["terminal_state"] = terminal_state

    # Bind a later pronoun loss to the nearest preceding tooth condition.
    for pronoun_loss in _PRONOUN_LOSS.finditer(dream):
        if any(_overlaps(pronoun_loss.start(), pronoun_loss.end(), [(e["source_span"]["start"], e["source_span"]["end"])]) for e in losses):
            continue
        prior = [
            event for event in condition_events
            if event.get("event_type") == "loose_tooth_condition"
            and event["source_span"]["start"] < pronoun_loss.start()
        ]
        if not prior:
            continue
        bound = prior[-1]
        target_values = bound.get("target_entity_ids_or_ambiguous")
        target_id = target_values[0] if isinstance(target_values, list) and target_values else "tooth-1"
        event = {
            "event_id": "loss-1",
            "event_type": "tooth_loss",
            "actor_id_or_null": None,
            "owner_id_or_ambiguous": bound.get("owner_id_or_ambiguous"),
            "target_entity_ids_or_ambiguous": [target_id],
            "event_chain_id_or_null": _tooth_chain_id(target_id),
            "scene_id": "scene-1",
            "phase": "dream",
            "channel": "narrative",
            "polarity": "affirmed",
            "modality": "experienced",
            "actuality": "actual",
            "completion": "completed",
            "quantity": "one",
            "quantity_cardinality": 1,
            "doctrine_eligible": True,
            "source_span": _span(dream, pronoun_loss.start(), pronoun_loss.end(), "loss-1"),
        }
        losses.append(event)
        events.append(event)
    for negated_loss in _NEGATED_LOSS.finditer(dream):
        if not negated_loss.group("negation") and not negated_loss.group("subject").lower().startswith(("no ", "none")):
            if negated_loss.group("subject").lower() != "neither":
                continue
        prior_conditions = [
            event for event in condition_events
            if event.get("event_type") == "loose_tooth_condition"
            and event["source_span"]["start"] < negated_loss.start()
        ]
        bound = prior_conditions[-1] if prior_conditions else None
        targets = bound.get("target_entity_ids_or_ambiguous") if bound else None
        target_id = targets[0] if isinstance(targets, list) and targets else "tooth-1"
        owner = str(bound.get("owner_id_or_ambiguous")) if bound else "dreamer"
        negated_event_id = f"{owner}-negated-loss-1" if owner != "dreamer" else "negated-loss-1"
        add_fact_event(
            negated_loss, negated_event_id, "tooth_loss", eligible=False,
            target_id=target_id, owner=owner,
            polarity="negated", modality="negated", completion="not_completed",
        )
    affirmed_loose = next((
        event for event in condition_events
        if event.get("event_type") == "loose_tooth_condition"
        and event.get("doctrine_eligible")
    ), None)
    if affirmed_loose and not any(event["event_id"] == "negated-loss-1" for event in events):
        implicit_negated_loss = _IMPLICIT_NEGATED_LOSS.search(dream, affirmed_loose["source_span"]["end"])
        if implicit_negated_loss:
            targets = affirmed_loose.get("target_entity_ids_or_ambiguous")
            target_id = targets[0] if isinstance(targets, list) and targets else "tooth-1"
            owner = str(affirmed_loose.get("owner_id_or_ambiguous") or "dreamer")
            event_id = f"{owner}-negated-loss-1" if owner != "dreamer" else "negated-loss-1"
            add_fact_event(
                implicit_negated_loss, event_id, "tooth_loss", eligible=False,
                target_id=target_id, owner=str(affirmed_loose.get("owner_id_or_ambiguous") or "dreamer"),
                polarity="negated", modality="negated", completion="not_completed",
            )

    primary_loss = next((event for event in losses if event.get("doctrine_eligible")), None)
    if primary_loss:
        pain_match = _PAIN.search(dream, primary_loss["source_span"]["end"])
        if pain_match:
            add_fact_event(
                pain_match, "pain-1", "pain_modifier", eligible=True,
                target_id=primary_loss["target_entity_ids_or_ambiguous"][0],
                chain_id=primary_loss["event_chain_id_or_null"],
                owner=primary_loss["owner_id_or_ambiguous"],
            )
        blood_match = _TOOTH_BLOOD.search(dream, primary_loss["source_span"]["end"])
        if blood_match:
            add_fact_event(
                blood_match, "blood-1", "tooth_blood_modifier", eligible=True,
                target_id=primary_loss["target_entity_ids_or_ambiguous"][0],
                chain_id=primary_loss["event_chain_id_or_null"],
                owner=primary_loss["owner_id_or_ambiguous"],
            )
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
    if firm_match and (attempts or losses):
        target = attempts[0].get("target_tooth_ids_or_ambiguous") if attempts else (
            primary_loss.get("target_entity_ids_or_ambiguous") if primary_loss else []
        )
        target_ids = target if isinstance(target, list) else []
        if target_ids:
            event_id = "firm-return-1"
            tooth_id = target_ids[0]
            events.append(
                {
                    "event_id": event_id,
                    "event_type": "firm_same_tooth_return",
                    "actor_id_or_null": attempts[0].get("actor_id_or_ambiguous") if attempts else None,
                    "owner_id_or_ambiguous": attempts[0].get("owner_id_or_ambiguous") if attempts else primary_loss.get("owner_id_or_ambiguous"),
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
                    "quantity_cardinality": 1,
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
                entity_type = "gums" if event.get("event_type") == "gum_bleeding_condition" else "tooth"
                _put_entity(
                    entities,
                    _entity(
                        str(tooth_id),
                        entity_type,
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
    total_loss_scope = sum(int(event.get("quantity_cardinality") or 1) for event in eligible_losses)
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
    state_types = {
        "tooth_loss", "firm_same_tooth_return", "loose_tooth_condition",
        "retained_tooth_state",
    }
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

    condition_transition_edges: List[Dict[str, Any]] = []
    for chain in chains.values():
        ordered = sorted(
            (event for event in events if event["event_id"] in chain["event_ids"]),
            key=lambda event: event["source_span"]["start"],
        )
        for source in ordered:
            if source.get("event_type") != "loose_tooth_condition":
                continue
            later_states = [
                target for target in ordered
                if target["source_span"]["start"] > source["source_span"]["start"]
                and target.get("event_type") in {"tooth_loss", "retained_tooth_state"}
            ]
            if later_states:
                target = later_states[0]
                condition_transition_edges.append({
                    "edge_id": f"{source['event_id']}_before_{target['event_id']}",
                    "event_chain_id": chain["event_chain_id"],
                    "from_event_id": source["event_id"],
                    "to_event_id": target["event_id"],
                    "relation": "before",
                })

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
        "condition_provenance_contract_version": CONDITION_PROVENANCE_CONTRACT_VERSION,
        "condition_transition_edges": condition_transition_edges,
        "condition_provenance_integrity": {"verified": False, "reason_codes": ["NOT_FINALIZED"]},
        "warning_claim_partition_contract_version": WARNING_CLAIM_PARTITION_CONTRACT_VERSION,
        "warning_claim_dispositions": [],
        "warning_claim_partition_summary": {},
        "warning_claim_partition_integrity": {"verified": False, "reason_codes": ["NOT_FINALIZED"]},
        "provenance_contract_version": "claim-provenance-reachability/1.0",
        "provenance_paths": [],
        "provenance_edges": [],
        "provenance_nodes": [],
        "provenance_summary": {},
        "provenance_digest": "",
        "provenance_integrity": {"verified": False, "reason_codes": ["NOT_FINALIZED"]},
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
    record_suffix = "-".join(source_ids) or "none"
    return {
        "rule_record_id": f"rule-record-{rule_id.lower()}-{record_suffix}",
        "rule_id": rule_id,
        "source_event_ids": source_ids,
        "source_span_ids": [events[event_id]["source_span"]["span_id"] for event_id in source_ids],
        "owner_ids": list(dict.fromkeys(
            events[event_id].get("owner_id_or_ambiguous")
            for event_id in source_ids
            if events[event_id].get("owner_id_or_ambiguous")
        )),
        "event_chain_ids": list(dict.fromkeys(
            events[event_id].get("event_chain_id_or_null")
            for event_id in source_ids
            if events[event_id].get("event_chain_id_or_null")
        )),
    }


def finalize_context_graph(
    graph: MutableMapping[str, Any],
    result: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> Dict[str, Any]:
    events = {event["event_id"]: event for event in graph["event_inventory"]}
    suppressing_frontiers = {
        frontier.get("event_chain_id")
        for frontier in graph["terminal_frontiers"]
        if events.get(frontier.get("terminal_event_id"), {}).get("event_type")
        in {"tooth_loss", "firm_same_tooth_return"}
    }
    historical_warning_event_ids = {
        event_id
        for frontier in graph["terminal_frontiers"]
        if frontier.get("event_chain_id") in suppressing_frontiers
        for event_id in frontier.get("historical_event_ids", [])
        if events.get(event_id, {}).get("event_type") in WARNING_EVENT_FAMILIES
    }
    eligible_losses = [
        event for event in events.values()
        if event["event_type"] == "tooth_loss"
        and event["doctrine_eligible"]
        and event["event_id"] not in historical_warning_event_ids
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
    events_by_type: Dict[str, List[str]] = {}
    for event in events.values():
        events_by_type.setdefault(str(event.get("event_type")), []).append(event["event_id"])

    rule_sources = {
        rule_id_for(registry, "terminal_ending"): terminal_ids,
        rule_id_for(registry, "own_fallout"): dreamer_loss_ids,
        rule_id_for(registry, "other_fallout"): other_loss_ids,
        rule_id_for(registry, "one_fallout"): loss_ids,
        rule_id_for(registry, "multiple_fallout"): loss_ids,
        rule_id_for(registry, "pain"): events_by_type.get("pain_modifier", []),
        rule_id_for(registry, "painless"): loss_ids,
        rule_id_for(registry, "tooth_blood"): events_by_type.get("tooth_blood_modifier", []),
        rule_id_for(registry, "gum_blood"): events_by_type.get("gum_bleeding_condition", []),
        rule_id_for(registry, "loose"): events_by_type.get("loose_tooth_condition", []),
        rule_id_for(registry, "external_pull"): [
            event["event_id"] for event in eligible_losses
            if event.get("actor_id_or_null") not in {None, "dreamer", "ambiguous", "unknown"}
        ],
        rule_id_for(registry, "self_pull"): [
            event["event_id"] for event in eligible_losses
            if event.get("actor_id_or_null") == "dreamer"
        ],
    }

    public_records: List[Dict[str, Any]] = []
    warning_records: List[Dict[str, Any]] = []
    seen_public: set[tuple[str, str]] = set()

    def add_public(rule_id: str, event_id: str, *, warning: bool = True) -> None:
        if not rule_id or event_id not in events:
            return
        key = (rule_id, event_id)
        if key in seen_public:
            return
        seen_public.add(key)
        record = _rule_record(rule_id, [event_id], events)
        public_records.append(record)
        if warning:
            warning_records.append(record)

    active_base_events = [
        event for event in events.values()
        if event.get("event_type") in WARNING_EVENT_FAMILIES
        and event.get("doctrine_eligible") is True
        and event.get("event_id") not in historical_warning_event_ids
    ]
    active_base_ids = {event["event_id"] for event in active_base_events}
    count_rule_id = rule_id_for(
        registry,
        "multiple_fallout" if result.get("count") == "multiple" else "one_fallout",
    )
    for event in active_base_events:
        event_id = event["event_id"]
        event_type = event.get("event_type")
        if event_type == "tooth_loss":
            owner_key = (
                "own_fallout"
                if event.get("owner_id_or_ambiguous") == "dreamer"
                else "other_fallout"
            )
            add_public(rule_id_for(registry, owner_key), event_id)
            add_public(count_rule_id, event_id)
            actor = event.get("actor_id_or_null")
            if actor == "dreamer":
                add_public(rule_id_for(registry, "self_pull"), event_id)
            elif actor not in {None, "", "ambiguous", "unknown"}:
                add_public(rule_id_for(registry, "external_pull"), event_id)
        elif event_type == "loose_tooth_condition":
            add_public(rule_id_for(registry, "loose"), event_id)
        elif event_type == "gum_bleeding_condition":
            add_public(rule_id_for(registry, "gum_blood"), event_id)

    # Preserve approved same-chain modifiers and legacy approved structural
    # rules, but split every source into its own event-scoped record.
    for rule_id in result.get("applied_rule_ids", []):
        if rule_id == rule_id_for(registry, "terminal_ending"):
            for event_id in terminal_ids:
                add_public(rule_id, event_id, warning=False)
            continue
        source_ids = list(rule_sources.get(rule_id, []))
        if not source_ids:
            source_ids = [
                event["event_id"] for event in active_base_events
            ][:1]
        for event_id in source_ids:
            event = events.get(event_id)
            if (
                event
                and event.get("doctrine_eligible") is True
                and event_id not in historical_warning_event_ids
            ):
                add_public(rule_id, event_id)

    historical: List[Dict[str, Any]] = []
    for event_id in historical_warning_event_ids:
        event = events.get(event_id)
        if not event or event.get("doctrine_eligible") is not True:
            continue
        rule_ids: List[str] = []
        if event.get("event_type") == "loose_tooth_condition":
            rule_ids.append(rule_id_for(registry, "loose"))
        elif event.get("event_type") == "gum_bleeding_condition":
            rule_ids.append(rule_id_for(registry, "gum_blood"))
        elif event.get("event_type") == "tooth_loss":
            rule_ids.append(rule_id_for(
                registry,
                "own_fallout"
                if event.get("owner_id_or_ambiguous") == "dreamer"
                else "other_fallout",
            ))
            rule_ids.append(rule_id_for(
                registry,
                "multiple_fallout" if result.get("count") == "multiple" else "one_fallout",
            ))
        historical.extend(
            _rule_record(rule_id, [event_id], events)
            for rule_id in rule_ids if rule_id
        )

    unresolved = [
        _rule_record(rule_id, terminal_ids or loss_ids, events)
        for rule_id in result.get("unresolved_rule_ids", [])
    ]
    graph["rule_sets"] = {
        "contract_version": RULE_SETS_CONTRACT_VERSION,
        "public_applied": public_records,
        "warning_active": warning_records,
        "matched_historical": historical,
        "structural": [record for record in public_records if record["rule_id"] == "TEETH-END-TERMINAL"],
        "withheld_unresolved": unresolved,
    }

    claims: List[Dict[str, Any]] = []
    public_rule_ids = [record["rule_id"] for record in public_records]
    for index, base_event in enumerate(
        sorted(active_base_events, key=lambda item: item["source_span"]["start"]),
        start=1,
    ):
        base_event_id = base_event["event_id"]
        chain_id = base_event.get("event_chain_id_or_null")
        matching_records = [
            record for record in warning_records
            if base_event_id in record.get("source_event_ids", [])
            or (
                base_event.get("event_type") == "tooth_loss"
                and any(
                    events[event_id].get("event_chain_id_or_null") == chain_id
                    and events[event_id].get("event_type") in {
                        "pain_modifier", "tooth_blood_modifier"
                    }
                    for event_id in record.get("source_event_ids", [])
                    if event_id in events
                )
            )
        ]
        if not matching_records:
            continue
        warning_event_ids = list(dict.fromkeys(
            [base_event_id]
            + [
                event_id
                for record in matching_records
                for event_id in record.get("source_event_ids", [])
            ]
        ))
        claim_rule_ids = list(dict.fromkeys(
            record["rule_id"] for record in matching_records
        ))
        claims.append(
            {
                "claim_id": f"claim-warning-{index}",
                "claim_type": "tradition_scoped_warning",
                "claim_scope": "atomic_warning",
                "warning_family": WARNING_EVENT_FAMILIES[base_event["event_type"]],
                "owner_ids": [base_event.get("owner_id_or_ambiguous")],
                "event_chain_ids": [chain_id],
                "release_status": "released",
                "released": True,
                "consumed_event_ids": warning_event_ids,
                "consumed_attempt_ids": [],
                "consumed_rule_ids": claim_rule_ids,
                "consumed_span_ids": [events[event_id]["source_span"]["span_id"] for event_id in warning_event_ids],
                "comparison_scope": "",
            }
        )
    atomic_claim_ids = [
        claim["claim_id"] for claim in claims
        if claim.get("claim_scope") == "atomic_warning"
    ]
    if len(atomic_claim_ids) > 1:
        claims.append({
            "claim_id": "claim-warning-compound-1",
            "claim_type": "compound_warning_presentation",
            "claim_scope": "compound_presentation",
            "member_claim_ids": atomic_claim_ids,
            "release_status": "released",
            "released": True,
            "consumed_event_ids": [],
            "consumed_attempt_ids": [],
            "consumed_rule_ids": [],
            "consumed_span_ids": [],
            "comparison_scope": "member_claims_only",
        })
    if terminal_ids:
        claims.append(
            {
                "claim_id": "claim-terminal-1",
                "claim_type": "terminal_state_without_consequence",
                "claim_scope": "structural_terminal",
                "release_status": "released",
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
                "claim_scope": "structural_narration",
                "release_status": "released",
                "released": True,
                "consumed_event_ids": consumed_attempts,
                "consumed_attempt_ids": consumed_attempts,
                "consumed_rule_ids": public_rule_ids,
                "consumed_span_ids": [events[event_id]["source_span"]["span_id"] for event_id in consumed_attempts],
            }
        )
    graph["claim_manifest"] = claims
    graph["warning_claim_partition_contract_version"] = WARNING_CLAIM_PARTITION_CONTRACT_VERSION
    graph["warning_claim_dispositions"] = [
        {
            "event_id": event_id,
            "disposition": (
                "gated"
                if event.get("doctrine_eligible") is not True
                else "historical"
                if event_id in historical_warning_event_ids
                else "released"
                if event_id in active_base_ids and any(
                    event_id in claim.get("consumed_event_ids", [])
                    for claim in claims
                    if claim.get("claim_scope") == "atomic_warning"
                )
                else "withheld"
            ),
        }
        for event_id, event in events.items()
        if event.get("event_type") in WARNING_EVENT_FAMILIES
    ]
    atomic_claims = [
        claim for claim in claims if claim.get("claim_scope") == "atomic_warning"
    ]
    graph["warning_claim_partition_summary"] = {
        "atomic_claim_count": len(atomic_claims),
        "compound_claim_count": sum(
            claim.get("claim_scope") == "compound_presentation" for claim in claims
        ),
        "released_warning_event_count": len(active_base_ids),
        "owner_count": len({
            owner for claim in atomic_claims for owner in claim.get("owner_ids", [])
        }),
        "chain_count": len({
            chain for claim in atomic_claims for chain in claim.get("event_chain_ids", [])
        }),
        "gated_or_withheld_event_count": sum(
            item["disposition"] in {"gated", "historical", "terminal", "withheld"}
            for item in graph["warning_claim_dispositions"]
        ),
    }
    graph["provenance_rule_registry"] = {
        rule.get("rule_id"): {
            "rule_id": rule.get("rule_id"),
            "status": rule.get("status"),
            "active": rule.get("active"),
        }
        for rule in registry.get("rules", {}).values()
        if rule.get("rule_id")
    }
    finalize_claim_provenance(graph, result, registry)
    graph["condition_provenance_integrity"] = validate_condition_provenance(graph)
    graph["warning_claim_partition_integrity"] = validate_warning_claim_partition(graph)
    graph["integrity"] = validate_context_graph(graph)
    if not graph["provenance_integrity"]["verified"]:
        graph["integrity"]["verified"] = False
        graph["integrity"]["reason_codes"] = list(dict.fromkeys(
            graph["integrity"]["reason_codes"] + graph["provenance_integrity"]["reason_codes"]
        ))
    if not graph["condition_provenance_integrity"]["verified"]:
        graph["integrity"]["verified"] = False
        graph["integrity"]["reason_codes"] = list(dict.fromkeys(
            graph["integrity"]["reason_codes"]
            + graph["condition_provenance_integrity"]["reason_codes"]
        ))
    if not graph["warning_claim_partition_integrity"]["verified"]:
        graph["integrity"]["verified"] = False
        graph["integrity"]["reason_codes"] = list(dict.fromkeys(
            graph["integrity"]["reason_codes"]
            + graph["warning_claim_partition_integrity"]["reason_codes"]
        ))
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
            int(event.get("quantity_cardinality") or 1)
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
            and events[candidate].get("event_type") in {
                "tooth_loss", "firm_same_tooth_return", "loose_tooth_condition",
                "retained_tooth_state",
            }
        ]
        if eligible_state_events:
            latest = max(eligible_state_events, key=lambda item: item["source_span"]["start"])
            if latest["event_id"] != event_id:
                reasons.append("TERMINAL_FRONTIER_DANGLING")

    for edge in graph.get("condition_transition_edges", []):
        source = events.get(edge.get("from_event_id"))
        target = events.get(edge.get("to_event_id"))
        chain = chains.get(edge.get("event_chain_id"))
        if (
            not source
            or not target
            or not chain
            or source.get("event_id") not in chain.get("event_ids", [])
            or target.get("event_id") not in chain.get("event_ids", [])
        ):
            reasons.append("CONDITION_LOSS_COLLAPSE")

    unique_reasons = list(dict.fromkeys(reasons))
    if graph.get("provenance_paths"):
        provenance = validate_claim_provenance(graph)
        unique_reasons = list(dict.fromkeys(unique_reasons + provenance["reason_codes"]))
    if graph.get("warning_claim_partition_contract_version"):
        warning_partition = validate_warning_claim_partition(graph)
        unique_reasons = list(dict.fromkeys(
            unique_reasons + warning_partition["reason_codes"]
        ))
    return {
        "verified": not unique_reasons,
        "reason_codes": unique_reasons,
        "entity_count": len(entities),
        "event_count": len(events),
        "chain_count": len(chains),
        "claim_count": len(graph.get("claim_manifest", [])),
    }
