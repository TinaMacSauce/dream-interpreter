#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


EXPECTED_CASES: Dict[str, Dict[str, Any]] = {
    "base_enemy": {"active_doctrine": True, "include": ["SNAKE-BASE-ENEMY"]},
    "presence_negated": {"active_doctrine": False, "exact": []},
    "watching": {"action": "watching", "include": ["SNAKE-WATCHING"], "exclude": ["SNAKE-ATTACK"]},
    "attack_not_outcome": {"outcome": "unresolved", "include": ["SNAKE-ATTACK", "SNAKE-UNFINISHED-BATTLE"], "exclude": ["SNAKE-END-DEFEAT"]},
    "retreat": {"action": "retreat", "include": ["SNAKE-RETREAT"]},
    "victory": {"outcome": "dreamer_victory", "include": ["SNAKE-END-VICTORY"]},
    "found_dead_not_victory": {"outcome": "not_established", "exclude": ["SNAKE-END-VICTORY"]},
    "defeat": {"outcome": "opposition_victory_in_encounter", "include": ["SNAKE-END-DEFEAT"]},
    "multiple": {"quantity": "multiple", "include": ["SNAKE-QUANTITY"]},
    "small": {"strength": "lesser_or_weaker", "include": ["SNAKE-SIZE-DANGER"]},
    "large_cobra": {"strength": "stronger_or_more_dangerous", "include": ["SNAKE-SIZE-DANGER"]},
    "bite": {"completed_bite": True, "include": ["SNAKE-BITE", "SNAKE-END-DEFEAT"]},
    "attempted_bite": {"attempted_bite": True, "completed_bite": False, "exclude": ["SNAKE-BITE"]},
    "venom": {"venom": True, "include": ["SNAKE-VENOM"]},
    "home": {"location_scope": "home_or_family_sphere", "include": ["SNAKE-LOCATION"]},
    "work": {"location_scope": "work_sphere", "include": ["SNAKE-LOCATION"]},
    "transformation": {"transformed_into_person": True, "include": ["SNAKE-TRANSFORM-PERSON"]},
    "ownership": {"ownership_weight": "low", "include": ["SNAKE-OWNERSHIP-LOW"]},
    "unfinished": {"outcome": "unresolved", "include": ["SNAKE-UNFINISHED-BATTLE"]},
    "color_excluded": {"colors_ignored": ["red"], "include": ["SNAKE-COLOR-EXCLUDED"]},
}


def fetch_json(url: str, timeout: float) -> tuple[int, Dict[str, Any]]:
    request = Request(url, headers={"Accept": "application/json"}, method="GET")
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        return error.code, json.loads(error.read().decode("utf-8"))


def validate(payload: Any, *, expected_commit: str) -> List[str]:
    if not isinstance(payload, dict):
        return ["payload is not an object"]
    errors: List[str] = []
    if payload.get("contract_version") != "snake-qa-contract-v2":
        errors.append("contract_version mismatch")
    if not isinstance(payload.get("case_count"), int) or payload.get("case_count") < len(EXPECTED_CASES) + 31:
        errors.append(f"case_count expected at least {len(EXPECTED_CASES) + 31}, got {payload.get('case_count')!r}")
    if payload.get("non_billable") is not True or payload.get("customer_credits_consumed") is not False:
        errors.append("bounded QA billing contract mismatch")
    release = payload.get("release") or {}
    if release.get("build_commit") != expected_commit:
        errors.append(f"release.build_commit expected {expected_commit!r}, got {release.get('build_commit')!r}")
    registry = payload.get("doctrine_registry") or {}
    expected_registry = {
        "verified": True,
        "contract_version": "snake-doctrine-registry-v1",
        "sheet_range": "DoctrineRegistry!A25:M42",
        "sheet_revision": "6138",
        "content_revision": "fnv1a64:ae0190f42f79b9c8",
        "rule_count": 18,
        "active_rule_count": 18,
        "unresolved_rule_count": 0,
        "loaded_from": "canonical_sheet",
    }
    for field, value in expected_registry.items():
        if registry.get(field) != value:
            errors.append(f"registry.{field} expected {value!r}, got {registry.get(field)!r}")

    cases = {case.get("case_id"): case for case in payload.get("cases") or []}
    if not set(EXPECTED_CASES).issubset(cases):
        errors.append("baseline case identifiers missing")
    for case_id, expected in EXPECTED_CASES.items():
        doctrine = (cases.get(case_id) or {}).get("doctrine") or {}
        rules = doctrine.get("applied_rule_ids") or []
        for field, value in expected.items():
            if field in {"include", "exclude", "exact"}:
                continue
            if doctrine.get(field) != value:
                errors.append(f"{case_id}.{field} expected {value!r}, got {doctrine.get(field)!r}")
        for rule_id in expected.get("include", []):
            if rule_id not in rules:
                errors.append(f"{case_id} missing rule {rule_id}")
        for rule_id in expected.get("exclude", []):
            if rule_id in rules:
                errors.append(f"{case_id} unexpectedly included rule {rule_id}")
        if "exact" in expected and rules != expected["exact"]:
            errors.append(f"{case_id} exact rules expected {expected['exact']!r}, got {rules!r}")
        narration = str(((cases.get(case_id) or {}).get("narration") or {}).get("narration_text") or "").lower()
        if doctrine.get("active_doctrine") is True and "spiritual tradition" not in narration:
            errors.append(f"{case_id} narration lacks cultural scope")
        for forbidden in ("definitely", "will happen", "is the enemy", "will get sick"):
            if forbidden in narration:
                errors.append(f"{case_id} narration contains forbidden phrase {forbidden!r}")
    context_cases = [case for case_id, case in cases.items() if case_id.startswith("SNAKE-00")]
    if len(context_cases) < 31:
        errors.append(f"context case count expected at least 31, got {len(context_cases)}")
    for case in context_cases:
        doctrine = case.get("doctrine") or {}
        graph = doctrine.get("event_graph") or {}
        integrity = graph.get("graph_integrity") or {}
        if graph.get("contract_version") != "snake-context-event-terminal-v1":
            errors.append(f"{case.get('case_id')} event graph contract mismatch")
        if integrity.get("verified") is not True or integrity.get("reason_codes"):
            errors.append(f"{case.get('case_id')} event graph integrity failed")
        if not graph.get("events") or not graph.get("terminal_frontiers"):
            errors.append(f"{case.get('case_id')} event graph inventory missing")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify bounded Snake production QA contract.")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    url = f"{args.base_url.rstrip('/')}/qa/snake-regression"
    try:
        status, payload = fetch_json(url, args.timeout)
    except (URLError, TimeoutError, json.JSONDecodeError) as error:
        print(json.dumps({"verified": False, "url": url, "errors": [f"{type(error).__name__}: {error}"]}))
        return 1
    errors = validate(payload, expected_commit=args.expected_commit)
    if status != 200:
        errors.insert(0, f"HTTP status expected 200, got {status}")
    print(json.dumps({"verified": not errors, "url": url, "http_status": status, "case_count": payload.get("case_count"), "errors": errors}, sort_keys=True))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
