#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from typing import Any, Dict, Iterable, List, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


EXPECTED_CONTRACT_VERSION = "teeth-qa-contract-v1"
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
    },
    "quantity_multiple": {
        "owner": "dreamer",
        "warning_kind": "tooth_loss",
        "warning_count": "multiple_people",
        "subject_scope": "relative_close_friend_or_relationship_circle",
        "include": ["TEETH-FALLOUT-OWN", "TEETH-FALLOUT-MULTIPLE"],
        "exclude": ["TEETH-FALLOUT-ONE"],
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
        "outcome_resolution": "unresolved",
        "exact_rules": ["TEETH-END-TERMINAL"],
        "unresolved_include": ["TEETH-END-RETURNED-SAME"],
        "narration_contains": ["no outcome is asserted"],
    },
    "attempted_ending": {
        "ending_precedence": False,
        "terminal_ending": "",
        "include": ["TEETH-FALLOUT-OWN", "TEETH-FALLOUT-ONE"],
        "exclude": ["TEETH-END-TERMINAL"],
    },
}


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
                "narration_contains",
            }:
                continue
            if doctrine.get(field) != value:
                case_errors.append(
                    f"{case_id}: doctrine.{field} expected {value!r}, "
                    f"got {doctrine.get(field)!r}"
                )

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
