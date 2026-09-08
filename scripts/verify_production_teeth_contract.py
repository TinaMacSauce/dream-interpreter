#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from typing import Any, Dict, Iterable, List, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


EXPECTED_CONTRACT_VERSION = "teeth-qa-contract-v2"
EXPECTED_REGISTRY_CONTRACT_VERSION = "teeth-doctrine-registry-v1"
EXPECTED_REGISTRY_CONTENT_REVISION = "fnv1a64:c51447de5d35bd59"
EXPECTED_REGISTRY_SHEET_REVISION = "6134"

EXPECTED: Dict[str, Dict[str, Any]] = {
    "quantity_one": {
        "owner": "dreamer",
        "warning_kind": "tooth_loss",
        "warning_count": "one_person",
        "subject_scope": "relative_close_friend_or_relationship_circle",
        "include": ["TEETH-FALLOUT-OWN", "TEETH-FALLOUT-ONE"],
        "exclude": ["TEETH-FALLOUT-MULTIPLE"],
        "narration_contains": ["because this was your own tooth"],
        "narration_excludes": ["because these were your own teeth"],
    },
    "quantity_multiple": {
        "owner": "dreamer",
        "warning_kind": "tooth_loss",
        "warning_count": "multiple_people",
        "subject_scope": "relative_close_friend_or_relationship_circle",
        "include": ["TEETH-FALLOUT-OWN", "TEETH-FALLOUT-MULTIPLE"],
        "exclude": ["TEETH-FALLOUT-ONE"],
        "narration_contains": ["because these were your own teeth"],
        "narration_excludes": ["because this was your own tooth"],
    },
    "ownership_other": {
        "owner": "other",
        "owner_relationship": "sister",
        "subject_scope": "sister",
        "include": ["TEETH-FALLOUT-OTHER", "TEETH-FALLOUT-ONE"],
        "exclude": ["TEETH-FALLOUT-OWN"],
    },
    "ownership_external_actor": {
        "owner": "dreamer",
        "removal_actor": "other",
        "pull_modifier": "external_interference",
        "include": ["TEETH-FALLOUT-OWN", "TEETH-PULL-EXTERNAL"],
        "exclude": ["TEETH-FALLOUT-OTHER"],
    },
    "painful_loss": {
        "pain": "painful",
        "proximity": "very_close_or_close_relative",
        "emotional_intensity": "heightened",
        "include": ["TEETH-MOD-PAIN"],
        "exclude": ["TEETH-MOD-PAINLESS"],
    },
    "painless_loss": {
        "pain": "painless",
        "proximity": "friend_acquaintance_or_more_distant",
        "emotional_intensity": "",
        "include": ["TEETH-MOD-PAINLESS"],
        "exclude": ["TEETH-MOD-PAIN"],
    },
    "blood_after_loss": {
        "blood_on_fallen_tooth": True,
        "severity_modifier": "increased",
        "include": ["TEETH-MOD-BLOOD"],
        "narration_contains": ["emotional depth only", "does not determine"],
    },
    "bleeding_gums_with_negations": {
        "active_fallout": False,
        "loose_warning": False,
        "bleeding_gums_warning": True,
        "warning_kind": "bleeding_gums",
        "include": ["TEETH-OMEN-GUM-BLOOD"],
        "exclude": ["TEETH-STATE-LOOSE", "TEETH-FALLOUT-OWN"],
    },
    "loose_without_loss": {
        "active_fallout": False,
        "loose_warning": True,
        "warning_kind": "loose_sickness",
        "include": ["TEETH-STATE-LOOSE"],
        "exclude": ["TEETH-FALLOUT-OWN"],
    },
    "negated_loss": {
        "active_doctrine": False,
        "active_fallout": False,
        "include": [],
        "exact_rules": [],
    },
    "hypothetical_loss": {
        "active_doctrine": False,
        "active_fallout": False,
        "include": [],
        "exact_rules": [],
    },
    "genuine_terminal_ending": {
        "ending_precedence": True,
        "terminal_ending": "same_tooth_returned_firm",
        "restoration_attempted": False,
        "outcome_resolution": "unresolved",
        "exact_rules": ["TEETH-END-TERMINAL"],
        "unresolved_include": ["TEETH-END-RETURNED-SAME"],
        "narration_contains": ["no outcome is asserted"],
        "narration_excludes": ["attempt to put the tooth back"],
    },
    "attempted_ending": {
        "ending_precedence": False,
        "terminal_ending": "",
        "restoration_attempted": True,
        "include": ["TEETH-FALLOUT-OWN", "TEETH-FALLOUT-ONE"],
        "exclude": ["TEETH-END-TERMINAL"],
        "narration_contains": [
            "attempt to put the tooth back",
            "not treated as a completed restoration or terminal ending",
        ],
    },
}

ATTEMPT_BINDING_EXPECTED: Dict[str, Dict[str, Any]] = {
    "CTX-001-ATTEMPT-BIND-DREAMER-001": {
        "actor_id_or_ambiguous": "dreamer",
        "owner_id_or_ambiguous": "dreamer",
        "target_tooth_ids_or_ambiguous": ["tooth-1"],
        "phase": "dream",
        "channel": "narrative",
        "actuality": "actual_attempt",
        "narration_eligibility": True,
    },
    "CTX-001-ATTEMPT-BIND-NEGATED-001": {
        "polarity": "negated",
        "actuality": "not_actual",
        "narration_eligibility": False,
    },
    "CTX-001-ATTEMPT-BIND-HYPOTHETICAL-001": {
        "phase": "dream_speech_or_thought",
        "channel": "thought_or_hypothetical",
        "modality": "conditional_hypothetical",
        "actuality": "nonactual",
        "narration_eligibility": False,
    },
    "CTX-001-ATTEMPT-BIND-QUOTED-001": {
        "actor_id_or_ambiguous": "aunt",
        "owner_id_or_ambiguous": "ambiguous",
        "target_tooth_ids_or_ambiguous": "ambiguous",
        "channel": "quoted_speech",
        "actuality": "reported_nonactual",
        "narration_eligibility": False,
    },
    "CTX-001-ATTEMPT-BIND-WAKING-001": {
        "phase": "waking",
        "modality": "imagined",
        "actuality": "nonactual",
        "narration_eligibility": False,
    },
    "CTX-001-ATTEMPT-BIND-OTHER-OWNER-001": {
        "actor_id_or_ambiguous": "sister",
        "owner_id_or_ambiguous": "sister",
        "target_tooth_ids_or_ambiguous": ["sister-tooth-1"],
        "actuality": "actual_attempt",
        "narration_eligibility": True,
    },
    "CTX-001-ATTEMPT-BIND-EXTERNAL-ACTOR-001": {
        "actor_id_or_ambiguous": "sister",
        "owner_id_or_ambiguous": "dreamer",
        "target_tooth_ids_or_ambiguous": ["tooth-1"],
        "actuality": "actual_attempt",
        "narration_eligibility": True,
    },
    "CTX-001-ATTEMPT-BIND-MULTI-OWNER-001": {
        "actor_id_or_ambiguous": "dreamer",
        "owner_id_or_ambiguous": "dreamer",
        "target_tooth_ids_or_ambiguous": ["dreamer-tooth-1"],
        "actuality": "actual_attempt",
        "narration_eligibility": True,
    },
    "CTX-001-ATTEMPT-BIND-AMBIGUOUS-TARGET-001": {
        "owner_id_or_ambiguous": "ambiguous",
        "target_tooth_ids_or_ambiguous": "ambiguous",
        "binding_confidence": "low",
        "narration_eligibility": False,
    },
    "CTX-001-ATTEMPT-BIND-THEN-FIRM-001": {
        "actuality": "actual_attempt",
        "completion": "attempted_only",
        "narration_eligibility": True,
    },
    "CTX-001-ATTEMPT-BIND-THEN-SECOND-LOSS-001": {
        "target_tooth_ids_or_ambiguous": ["left-tooth-1"],
        "actuality": "actual_attempt",
        "narration_eligibility": True,
    },
    "CTX-001-ATTEMPT-BIND-REPORTED-001": {
        "actor_id_or_ambiguous": "sister",
        "owner_id_or_ambiguous": "sister",
        "target_tooth_ids_or_ambiguous": ["sister-tooth-1"],
        "event_chain_id_or_null": None,
        "channel": "reported_speech",
        "actuality": "reported_nonactual",
        "narration_eligibility": False,
    },
}

GRAPH_CONTRACT_VERSION = "context-graph-referential-integrity/1.0"
GRAPH_EXPECTED: Dict[str, Dict[str, Any]] = {
    "CTX-001-ATTEMPT-BIND-DREAMER-001": {
        "entities": {"dreamer", "tooth-1"},
        "events": {"loss-1", "attempt-1"},
        "chains": {"chain-tooth-1"},
        "consumed_attempts": {"attempt-1"},
    },
    "CTX-001-ATTEMPT-BIND-NEGATED-001": {
        "events": {"loss-1", "attempt-1"}, "consumed_attempts": set(),
    },
    "CTX-001-ATTEMPT-BIND-HYPOTHETICAL-001": {
        "events": {"hypothetical-loss-1", "attempt-1"},
        "eligible_events": set(), "claims": set(), "consumed_attempts": set(),
    },
    "CTX-001-ATTEMPT-BIND-QUOTED-001": {
        "entities": {"dreamer", "tooth-1", "aunt"},
        "events": {"loss-1", "speech-1", "attempt-1"},
        "attempt_chain": None, "consumed_attempts": set(),
    },
    "CTX-001-ATTEMPT-BIND-WAKING-001": {
        "events": {"loss-1", "waking-imagined-attempt-1"},
        "chains": {"chain-tooth-1"}, "attempt_chain": None,
        "consumed_attempts": set(),
    },
    "CTX-001-ATTEMPT-BIND-OTHER-OWNER-001": {
        "entities": {"sister", "sister-tooth-1"},
        "events": {"loss-1", "attempt-1"},
        "chains": {"chain-sister-tooth-1"},
    },
    "CTX-001-ATTEMPT-BIND-EXTERNAL-ACTOR-001": {
        "entities": {"dreamer", "sister", "tooth-1"},
        "events": {"loss-1", "attempt-1"},
        "attempt_actor": "sister", "attempt_owner": "dreamer",
    },
    "CTX-001-ATTEMPT-BIND-MULTI-OWNER-001": {
        "entities": {"dreamer", "dreamer-tooth-1", "sister", "sister-tooth-1"},
        "events": {"dreamer-loss-1", "attempt-1", "sister-loss-1"},
        "loss_count": 2, "chain_count": 2,
        "aggregate_losses": {"dreamer-loss-1", "sister-loss-1"},
    },
    "CTX-001-ATTEMPT-BIND-AMBIGUOUS-TARGET-001": {
        "loss_count": 2, "attempt_target": "ambiguous",
        "attempt_owner": "ambiguous", "attempt_chain": None,
        "consumed_attempts": set(),
    },
    "CTX-001-ATTEMPT-BIND-THEN-FIRM-001": {
        "entities": {"dreamer", "tooth-1"},
        "events": {"loss-1", "attempt-1", "firm-return-1"},
        "frontier": "firm-return-1", "historical": {"loss-1", "attempt-1"},
        "public_rules": {"TEETH-END-TERMINAL"},
    },
    "CTX-001-ATTEMPT-BIND-THEN-SECOND-LOSS-001": {
        "entities": {"dreamer", "left-tooth-1", "tooth-2"},
        "events": {"left-loss-1", "attempt-1", "second-loss-1"},
        "loss_count": 2, "attempt_target": ["left-tooth-1"],
        "warning_count": "multiple_people",
    },
    "CTX-001-ATTEMPT-BIND-REPORTED-001": {
        "entities": {"dreamer", "tooth-1", "sister", "sister-tooth-1"},
        "events": {"dreamer-loss-1", "speech-1", "attempt-1"},
        "attempt_chain": None, "consumed_attempts": set(),
    },
}

for _case_id, _record in ATTEMPT_BINDING_EXPECTED.items():
    EXPECTED[_case_id] = {"attempt_record": _record}


def fetch_json(url: str, timeout: float) -> Tuple[int, Dict[str, Any]]:
    request = Request(url, headers={"Accept": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
        return response.status, payload


def _check_members(
    *,
    case_id: str,
    label: str,
    actual: Iterable[str],
    expected: Iterable[str],
    should_exist: bool,
) -> List[str]:
    values = set(actual or [])
    errors = []
    for rule_id in expected:
        present = rule_id in values
        if present != should_exist:
            verb = "include" if should_exist else "exclude"
            errors.append(f"{case_id}: {label} must {verb} {rule_id}")
    return errors


def _validate_graph(case_id: str, dream: str, doctrine: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    expected = GRAPH_EXPECTED[case_id]
    graph = doctrine.get("context_graph") or {}
    if graph.get("contract_version") != GRAPH_CONTRACT_VERSION:
        errors.append(f"{case_id}: missing graph contract {GRAPH_CONTRACT_VERSION}")
        return errors

    required = {
        "entity_inventory", "event_inventory", "event_chain_inventory",
        "restoration_attempt_records", "aggregate_derivations", "rule_sets",
        "claim_manifest", "terminal_frontiers",
    }
    if not required.issubset(graph):
        errors.append(f"{case_id}: graph collections are incomplete")
        return errors

    entities = {item.get("entity_id"): item for item in graph["entity_inventory"]}
    events = {item.get("event_id"): item for item in graph["event_inventory"]}
    chains = {item.get("event_chain_id"): item for item in graph["event_chain_inventory"]}
    attempts = {item.get("attempt_id"): item for item in graph["restoration_attempt_records"]}
    spans = {
        item.get("source_span", {}).get("span_id")
        for item in events.values()
    }
    for event in events.values():
        span = event.get("source_span") or {}
        if dream[span.get("start", 0):span.get("end", 0)] != span.get("text"):
            errors.append(f"{case_id}: event source span does not round-trip")
        for entity_id in (event.get("actor_id_or_null"), event.get("owner_id_or_ambiguous")):
            if entity_id not in {None, "ambiguous", "unknown"} and entity_id not in entities:
                errors.append(f"{case_id}: event has dangling entity {entity_id}")
        targets = event.get("target_entity_ids_or_ambiguous")
        if isinstance(targets, list) and any(target not in entities for target in targets):
            errors.append(f"{case_id}: event has dangling target")
        chain_id = event.get("event_chain_id_or_null")
        if chain_id and (chain_id not in chains or event.get("event_id") not in chains[chain_id].get("event_ids", [])):
            errors.append(f"{case_id}: event has dangling chain {chain_id}")
    for chain in chains.values():
        if any(event_id not in events for event_id in chain.get("event_ids", [])):
            errors.append(f"{case_id}: chain has dangling event")
        if any(entity_id not in entities for entity_id in chain.get("entity_ids", [])):
            errors.append(f"{case_id}: chain has dangling entity")
    for attempt in attempts.values():
        target = attempt.get("target_tooth_ids_or_ambiguous")
        if isinstance(target, list) and any(entity_id not in entities for entity_id in target):
            errors.append(f"{case_id}: attempt has dangling target")
        chain_id = attempt.get("event_chain_id_or_null")
        if chain_id and chain_id not in chains:
            errors.append(f"{case_id}: attempt has dangling chain")
    rule_ids = set()
    for partition, records in graph["rule_sets"].items():
        if partition == "contract_version":
            continue
        for record in records:
            rule_ids.add(record.get("rule_id"))
            if not record.get("source_event_ids") or any(event_id not in events for event_id in record.get("source_event_ids", [])):
                errors.append(f"{case_id}: rule has missing event provenance")
            if any(span_id not in spans for span_id in record.get("source_span_ids", [])):
                errors.append(f"{case_id}: rule has missing span provenance")
    for claim in graph["claim_manifest"]:
        if any(event_id not in events for event_id in claim.get("consumed_event_ids", [])):
            errors.append(f"{case_id}: claim has dangling event")
        if any(attempt_id not in attempts for attempt_id in claim.get("consumed_attempt_ids", [])):
            errors.append(f"{case_id}: claim has dangling attempt")
        if any(rule_id not in rule_ids for rule_id in claim.get("consumed_rule_ids", [])):
            errors.append(f"{case_id}: claim has dangling rule")
        if any(span_id not in spans for span_id in claim.get("consumed_span_ids", [])):
            errors.append(f"{case_id}: claim has dangling span")

    actual = {
        "entities": set(entities), "events": set(events), "chains": set(chains),
        "eligible_events": {event_id for event_id, event in events.items() if event.get("doctrine_eligible")},
        "claims": {claim.get("claim_id") for claim in graph["claim_manifest"]},
        "consumed_attempts": {
            attempt_id for claim in graph["claim_manifest"]
            for attempt_id in claim.get("consumed_attempt_ids", [])
        },
        "loss_count": sum(event.get("event_type") == "tooth_loss" for event in events.values()),
        "chain_count": len(chains),
        "public_rules": {item.get("rule_id") for item in graph["rule_sets"]["public_applied"]},
    }
    attempt = next(iter(attempts.values()))
    actual.update({
        "attempt_chain": attempt.get("event_chain_id_or_null"),
        "attempt_actor": attempt.get("actor_id_or_ambiguous"),
        "attempt_owner": attempt.get("owner_id_or_ambiguous"),
        "attempt_target": attempt.get("target_tooth_ids_or_ambiguous"),
        "warning_count": doctrine.get("warning_count"),
    })
    if graph["terminal_frontiers"]:
        actual["frontier"] = graph["terminal_frontiers"][0].get("terminal_event_id")
        actual["historical"] = set(graph["terminal_frontiers"][0].get("historical_event_ids", []))
    aggregate_losses = {
        event_id for aggregate in graph["aggregate_derivations"]
        for event_id in aggregate.get("contributing_event_ids", [])
    }
    actual["aggregate_losses"] = aggregate_losses
    for field, value in expected.items():
        if actual.get(field) != value:
            errors.append(f"{case_id}: graph {field} expected {value!r}, got {actual.get(field)!r}")
    integrity = graph.get("integrity") or {}
    if integrity.get("verified") is not True or integrity.get("reason_codes") != []:
        errors.append(f"{case_id}: graph integrity did not pass: {integrity!r}")
    return errors


def validate(payload: Any, *, expected_commit: str) -> Dict[str, Any]:
    errors: List[str] = []
    case_evidence: Dict[str, Any] = {}
    if not isinstance(payload, dict):
        return {"verified": False, "errors": ["payload is not an object"], "cases": {}}

    if payload.get("contract_version") != EXPECTED_CONTRACT_VERSION:
        errors.append(
            "contract_version expected "
            f"{EXPECTED_CONTRACT_VERSION!r}, got {payload.get('contract_version')!r}"
        )
    release = payload.get("release") or {}
    if release.get("build_commit") != expected_commit:
        errors.append(
            f"release.build_commit expected {expected_commit!r}, "
            f"got {release.get('build_commit')!r}"
        )

    registry = payload.get("doctrine_registry")
    if not isinstance(registry, dict):
        errors.append("doctrine_registry is missing or is not an object")
    else:
        registry_expected = {
            "verified": True,
            "contract_version": EXPECTED_REGISTRY_CONTRACT_VERSION,
            "sheet_revision": EXPECTED_REGISTRY_SHEET_REVISION,
            "content_revision": EXPECTED_REGISTRY_CONTENT_REVISION,
            "doctrine_version": "DEC-TEETH-2026-09-03-05",
            "rule_count": 23,
            "active_rule_count": 17,
            "unresolved_rule_count": 6,
            "loaded_from": "canonical_sheet",
        }
        for field, value in registry_expected.items():
            if registry.get(field) != value:
                errors.append(
                    f"doctrine_registry.{field} expected {value!r}, "
                    f"got {registry.get(field)!r}"
                )

    raw_cases = payload.get("cases") or []
    cases = {
        item.get("case_id"): item
        for item in raw_cases
        if isinstance(item, dict) and item.get("case_id")
    }
    if set(cases) != set(EXPECTED):
        errors.append(
            f"case IDs expected {sorted(EXPECTED)}, got {sorted(cases)}"
        )

    forbidden = ("will die", "is going to die", "will get sick", "definitely")
    for case_id, expected in EXPECTED.items():
        item = cases.get(case_id)
        case_errors: List[str] = []
        if not item:
            case_errors.append(f"{case_id}: case is missing")
            errors.extend(case_errors)
            case_evidence[case_id] = {"passed": False, "errors": case_errors}
            continue

        doctrine = item.get("doctrine") or {}
        narration = item.get("narration") or {}
        rules = doctrine.get("applied_rule_ids") or []
        for field, value in expected.items():
            if field in {
                "include", "exclude", "exact_rules", "unresolved_include",
                "narration_contains", "narration_excludes", "attempt_record",
            }:
                continue
            if doctrine.get(field) != value:
                case_errors.append(
                    f"{case_id}: doctrine.{field} expected {value!r}, "
                    f"got {doctrine.get(field)!r}"
                )

        expected_attempt = expected.get("attempt_record")
        if expected_attempt is not None:
            if doctrine.get("restoration_attempt_contract_version") != "restoration-attempt-binding/1.0":
                case_errors.append(
                    f"{case_id}: restoration attempt contract version is missing or stale"
                )
            records = doctrine.get("restoration_attempt_records") or []
            if len(records) != 1:
                case_errors.append(
                    f"{case_id}: expected exactly one restoration attempt record, got {len(records)}"
                )
            else:
                record = records[0]
                for field, value in expected_attempt.items():
                    if record.get(field) != value:
                        case_errors.append(
                            f"{case_id}: attempt.{field} expected {value!r}, "
                            f"got {record.get(field)!r}"
                        )
                required = {
                    "attempt_id", "action", "actor_id_or_ambiguous",
                    "target_tooth_ids_or_ambiguous", "owner_id_or_ambiguous",
                    "event_chain_id_or_null", "scene_id", "phase", "channel",
                    "polarity", "modality", "actuality", "completion",
                    "source_span", "binding_confidence", "narration_eligibility",
                    "ineligibility_reasons",
                }
                if set(record) != required:
                    case_errors.append(
                        f"{case_id}: attempt record fields do not match the v1 contract"
                    )
                span = record.get("source_span") or {}
                dream = item.get("dream", "")
                if dream[span.get("start", 0):span.get("end", 0)] != span.get("text"):
                    case_errors.append(
                        f"{case_id}: attempt source span does not round-trip to dream text"
                    )
            case_errors.extend(_validate_graph(case_id, item.get("dream", ""), doctrine))

        case_errors.extend(
            _check_members(
                case_id=case_id,
                label="applied_rule_ids",
                actual=rules,
                expected=expected.get("include", []),
                should_exist=True,
            )
        )
        case_errors.extend(
            _check_members(
                case_id=case_id,
                label="applied_rule_ids",
                actual=rules,
                expected=expected.get("exclude", []),
                should_exist=False,
            )
        )
        if "exact_rules" in expected and rules != expected["exact_rules"]:
            case_errors.append(
                f"{case_id}: applied_rule_ids expected exactly "
                f"{expected['exact_rules']!r}, got {rules!r}"
            )
        case_errors.extend(
            _check_members(
                case_id=case_id,
                label="unresolved_rule_ids",
                actual=doctrine.get("unresolved_rule_ids") or [],
                expected=expected.get("unresolved_include", []),
                should_exist=True,
            )
        )

        narration_text = " ".join(
            [narration.get("lead", ""), *(narration.get("details") or [])]
        ).lower()
        for phrase in expected.get("narration_contains", []):
            if phrase not in narration_text:
                case_errors.append(
                    f"{case_id}: narration must contain {phrase!r}"
                )
        for phrase in expected.get("narration_excludes", []):
            if phrase in narration_text:
                case_errors.append(
                    f"{case_id}: narration must exclude {phrase!r}"
                )
        for phrase in forbidden:
            if phrase in narration_text:
                case_errors.append(
                    f"{case_id}: narration contains forbidden certainty {phrase!r}"
                )

        case_evidence[case_id] = {
            "passed": not case_errors,
            "errors": case_errors,
            "applied_rule_ids": rules,
            "warning_kind": doctrine.get("warning_kind"),
        }
        errors.extend(case_errors)

    return {"verified": not errors, "errors": errors, "cases": case_evidence}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Independently verify the deployed Teeth QA contract."
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--delay", type=float, default=15.0)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    url = f"{args.base_url.rstrip('/')}/qa/teeth-regression"
    last_evidence: Dict[str, Any] = {}
    for attempt in range(1, args.attempts + 1):
        try:
            status, payload = fetch_json(url, args.timeout)
            evidence = validate(payload, expected_commit=args.expected_commit)
            if status != 200:
                evidence["errors"].insert(0, f"HTTP status expected 200, got {status}")
                evidence["verified"] = False
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            evidence = {
                "verified": False,
                "errors": [f"{type(exc).__name__}: {exc}"],
                "cases": {},
            }

        evidence.update(
            {
                "attempt": attempt,
                "attempts_allowed": args.attempts,
                "expected_commit": args.expected_commit,
                "url": url,
            }
        )
        print(json.dumps(evidence, sort_keys=True), flush=True)
        last_evidence = evidence
        if evidence["verified"]:
            return 0
        if attempt < args.attempts:
            time.sleep(args.delay)

    print(
        "Deployed Teeth QA contract failed: "
        + json.dumps(last_evidence.get("errors", [])),
        flush=True,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
