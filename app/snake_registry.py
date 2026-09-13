from __future__ import annotations

import time
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from app.cache import SNAKE_REGISTRY_CACHE
from app.config import Config
from app.registry_approvals import (
    load_approval_manifest,
    registry_snapshot_sha256,
    validate_approval_manifest,
)
from app.sheets import get_spreadsheet
from app.teeth_registry import (
    REQUIRED_HEADERS,
    cluster_registry_values,
    registry_content_revision,
)
from app.utils import normalize_header


REGISTRY_CONTRACT_VERSION = "snake-doctrine-registry-v1"
EXPECTED_DOCTRINE_VERSION = "DEC-SNAKE-2026-09-08-02"
EXPECTED_SHEET_REVISION = "6144"
EXPECTED_CONTENT_REVISION = "fnv1a64:1c8decb0fe7b56da"
EXPECTED_UPDATED_AT_UTC = "2026-09-08T14:06:13Z"
EXPECTED_AUTHORITY = "Tina, explicit founder clarification"
LEGACY_DOCTRINE_VERSION = "DEC-SNAKE-2026-09-08-01"
LEGACY_UPDATED_AT_UTC = "2026-09-08T07:35:00Z"
LEGACY_AUTHORITY = "Tina, explicit founder teaching"

EXPECTED_RULES: Mapping[str, Tuple[str, str, bool]] = {
    "snake_base_enemy": ("SNAKE-BASE-ENEMY", "APPROVED", True),
    "snake_action_map": ("SNAKE-ACTION-MAP", "APPROVED", True),
    "snake_attack": ("SNAKE-ATTACK", "APPROVED", True),
    "snake_watching": ("SNAKE-WATCHING", "APPROVED", True),
    "snake_retreat": ("SNAKE-RETREAT", "APPROVED", True),
    "snake_victory": ("SNAKE-END-VICTORY", "APPROVED", True),
    "snake_defeat": ("SNAKE-END-DEFEAT", "APPROVED", True),
    "snake_quantity": ("SNAKE-QUANTITY", "APPROVED", True),
    "snake_size_danger": ("SNAKE-SIZE-DANGER", "APPROVED", True),
    "snake_bite": ("SNAKE-BITE", "APPROVED", True),
    "snake_venom": ("SNAKE-VENOM", "APPROVED", True),
    "snake_location": ("SNAKE-LOCATION", "APPROVED", True),
    "snake_transform_person": ("SNAKE-TRANSFORM-PERSON", "APPROVED", True),
    "snake_ownership_low": ("SNAKE-OWNERSHIP-LOW", "APPROVED", True),
    "snake_unfinished_battle": ("SNAKE-UNFINISHED-BATTLE", "APPROVED", True),
    "snake_color_excluded": ("SNAKE-COLOR-EXCLUDED", "APPROVED", True),
    "snake_faith_response": ("SNAKE-FAITH-RESPONSE", "APPROVED", True),
    "snake_faith_best_practice": (
        "SNAKE-FAITH-BEST-PRACTICE",
        "APPROVED",
        True,
    ),
    "snake_watching_target": ("SNAKE-WATCHING-TARGET", "APPROVED", True),
    "snake_bite_attempt_target": ("SNAKE-BITE-ATTEMPT-TARGET", "APPROVED", True),
    "snake_representation_carving": ("SNAKE-REP-CARVING", "APPROVED", True),
    "snake_location_house_life": ("SNAKE-LOC-HOUSE", "APPROVED", True),
    "snake_location_bedroom_intimacy": ("SNAKE-LOC-BEDROOM", "APPROVED", True),
    "snake_location_kitchen_productivity_healing_replenishment": (
        "SNAKE-LOC-KITCHEN",
        "APPROVED",
        True,
    ),
    "pending_snake_location_bathroom": ("SNAKE-LOC-BATHROOM", "UNRESOLVED", False),
    "pending_snake_location_living_area": ("SNAKE-LOC-LIVING-AREA", "UNRESOLVED", False),
    "pending_snake_noncarving_representation": ("SNAKE-REP-NONCARVING", "UNRESOLVED", False),
    "pending_snake_unusual_control_ownership": ("SNAKE-OWNERSHIP-CONTROL", "UNRESOLVED", False),
    "pending_snake_faith_eligibility_extended": ("SNAKE-FAITH-ELIGIBILITY-EXTENDED", "UNRESOLVED", False),
}

LEGACY_IMPLEMENTATION_KEYS = frozenset(list(EXPECTED_RULES)[:18])


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _rows_from_values(
    values: Sequence[Sequence[Any]],
) -> Tuple[List[str], List[Dict[str, str]]]:
    if not values:
        raise RuntimeError("snake_registry_empty")
    headers = [normalize_header(str(value or "")) for value in values[0]]
    if tuple(headers) != REQUIRED_HEADERS:
        raise RuntimeError("snake_registry_schema_mismatch")
    rows: List[Dict[str, str]] = []
    for raw in values[1:]:
        if len(raw) != len(headers):
            raise RuntimeError("snake_registry_row_width_mismatch")
        padded = list(raw) + [""] * max(0, len(headers) - len(raw))
        row = {
            headers[index]: str(padded[index] or "").strip()
            for index in range(len(headers))
        }
        if any(row.values()):
            rows.append(row)
    return headers, rows


def validate_snake_registry_values(
    values: Sequence[Sequence[Any]],
    *,
    expected_content_revision: str | None = None,
    approval_manifest: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    scoped_values = cluster_registry_values(values, "Snake")
    _headers, rows = _rows_from_values(scoped_values)
    content_revision = registry_content_revision(scoped_values)
    profile = None
    if approval_manifest is not None:
        if expected_content_revision is not None:
            raise RuntimeError("snake_registry_approval_override_conflict")
        profiles = validate_approval_manifest(approval_manifest, EXPECTED_RULES)
        digest = registry_snapshot_sha256(scoped_values)
        profile = next((item for item in profiles if item["snapshot_sha256"] == digest), None)
    if profile is None and content_revision != (expected_content_revision or EXPECTED_CONTENT_REVISION):
        raise RuntimeError("snake_registry_content_revision_mismatch")
    if len(rows) != len(EXPECTED_RULES):
        raise RuntimeError("snake_registry_rule_count_mismatch")

    rules: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        key = row["implementation_key"]
        if not key or key in rules or key not in EXPECTED_RULES:
            raise RuntimeError("snake_registry_implementation_key_mismatch")
        expected_rule_id, expected_status, expected_active = EXPECTED_RULES[key]
        if row["active"].lower() not in {"1", "0", "true", "false", "yes", "no", "on", "off"}:
            raise RuntimeError("snake_registry_activation_mismatch")
        active = _truthy(row["active"])
        if row["rule_id"] != expected_rule_id:
            raise RuntimeError("snake_registry_rule_id_mismatch")
        if row["status"] != expected_status or active is not expected_active:
            raise RuntimeError("snake_registry_activation_mismatch")
        legacy = key in LEGACY_IMPLEMENTATION_KEYS
        expected_version = LEGACY_DOCTRINE_VERSION if legacy else EXPECTED_DOCTRINE_VERSION
        expected_authority = LEGACY_AUTHORITY if legacy else EXPECTED_AUTHORITY
        expected_timestamp = LEGACY_UPDATED_AT_UTC if legacy else EXPECTED_UPDATED_AT_UTC
        if profile is not None:
            metadata = profile["rule_metadata"][key]
            expected_version = metadata["doctrine_version"]
            expected_authority = metadata["authority"]
            expected_timestamp = metadata["updated_at_utc"]
        if row["doctrine_version"] != expected_version:
            raise RuntimeError("snake_registry_doctrine_version_mismatch")
        if row["decision_id"] != expected_version:
            raise RuntimeError("snake_registry_decision_id_mismatch")
        if row["cluster"] != "Snake":
            raise RuntimeError("snake_registry_cluster_mismatch")
        if row["authority"] != expected_authority:
            raise RuntimeError("snake_registry_authority_mismatch")
        if row["updated_at_utc"] != expected_timestamp:
            raise RuntimeError("snake_registry_timestamp_mismatch")
        if any(not row[field] for field in (
            "trigger_context", "governing_meaning", "precedence", "safety_boundary",
        )):
            raise RuntimeError("snake_registry_required_field_missing")
        rules[key] = {
            **row,
            "active": active,
        }

    active_rule_ids = sorted(
        rule["rule_id"] for rule in rules.values() if rule["active"]
    )
    unresolved_rule_ids = sorted(
        rule["rule_id"] for rule in rules.values() if not rule["active"]
    )
    return {
        "verified": True,
        "contract_version": REGISTRY_CONTRACT_VERSION,
        "sheet_name": Config.SHEET_DOCTRINE_REGISTRY,
        "sheet_range": "DoctrineRegistry!A25:M54",
        "sheet_revision": profile["sheet_revision"] if profile is not None else EXPECTED_SHEET_REVISION,
        "content_revision": content_revision,
        "doctrine_version": profile["doctrine_version"] if profile is not None else EXPECTED_DOCTRINE_VERSION,
        "decision_id": profile["doctrine_version"] if profile is not None else EXPECTED_DOCTRINE_VERSION,
        "decision_ids": sorted({row["decision_id"] for row in rows}),
        "canonical_location_text": profile is not None,
        "rule_count": len(rules),
        "active_rule_count": len(active_rule_ids),
        "unresolved_rule_count": len(unresolved_rule_ids),
        "active_rule_ids": active_rule_ids,
        "unresolved_rule_ids": unresolved_rule_ids,
        "rules": rules,
        "loaded_from": "canonical_sheet",
        "error": "",
    }


def _test_manifest_snapshot() -> Dict[str, Any]:
    rules = {
        key: {"rule_id": value[0], "status": value[1], "active": value[2]}
        for key, value in EXPECTED_RULES.items()
    }
    return {
        "verified": True,
        "contract_version": REGISTRY_CONTRACT_VERSION,
        "sheet_name": Config.SHEET_DOCTRINE_REGISTRY,
        "sheet_range": "DoctrineRegistry!A25:M54",
        "sheet_revision": EXPECTED_SHEET_REVISION,
        "content_revision": EXPECTED_CONTENT_REVISION,
        "doctrine_version": EXPECTED_DOCTRINE_VERSION,
        "decision_id": EXPECTED_DOCTRINE_VERSION,
        "rule_count": len(rules),
        "active_rule_count": sum(1 for rule in rules.values() if rule["active"]),
        "unresolved_rule_count": sum(1 for rule in rules.values() if not rule["active"]),
        "active_rule_ids": sorted(rule["rule_id"] for rule in rules.values() if rule["active"]),
        "unresolved_rule_ids": sorted(rule["rule_id"] for rule in rules.values() if not rule["active"]),
        "rules": rules,
        "loaded_from": "verified_test_manifest",
        "error": "",
    }


def _failed_snapshot(error: Exception) -> Dict[str, Any]:
    return {
        "verified": False,
        "contract_version": REGISTRY_CONTRACT_VERSION,
        "sheet_name": Config.SHEET_DOCTRINE_REGISTRY,
        "sheet_range": "DoctrineRegistry!A25:M54",
        "sheet_revision": EXPECTED_SHEET_REVISION,
        "content_revision": "",
        "doctrine_version": EXPECTED_DOCTRINE_VERSION,
        "decision_id": EXPECTED_DOCTRINE_VERSION,
        "rule_count": 0,
        "active_rule_count": 0,
        "unresolved_rule_count": 0,
        "active_rule_ids": [],
        "unresolved_rule_ids": [],
        "rules": {},
        "loaded_from": "canonical_sheet",
        "error": str(error) or type(error).__name__,
    }


def get_snake_registry_snapshot(*, force: bool = False) -> Dict[str, Any]:
    if Config.APP_ENV == "test":
        return _test_manifest_snapshot()
    now = time.time()
    cached = SNAKE_REGISTRY_CACHE.get("snapshot")
    if (
        not force
        and isinstance(cached, dict)
        and now - float(SNAKE_REGISTRY_CACHE.get("loaded_at") or 0)
        < Config.CACHE_TTL_SECONDS
    ):
        return cached
    try:
        approval_manifest = load_approval_manifest(Config.SNAKE_REGISTRY_APPROVALS_FILE)
        worksheet = get_spreadsheet().worksheet(Config.SHEET_DOCTRINE_REGISTRY)
        snapshot = validate_snake_registry_values(
            worksheet.get_all_values(), approval_manifest=approval_manifest,
        )
    except Exception as error:
        snapshot = _failed_snapshot(error)
    SNAKE_REGISTRY_CACHE["snapshot"] = snapshot
    SNAKE_REGISTRY_CACHE["loaded_at"] = now
    return snapshot


def snake_rule_id_for(snapshot: Mapping[str, Any], implementation_key: str) -> str:
    rule = (snapshot.get("rules") or {}).get(implementation_key) or {}
    if snapshot.get("verified") is not True or rule.get("active") is not True:
        return ""
    return str(rule.get("rule_id") or "")


def public_snake_registry_metadata(
    snapshot: Mapping[str, Any],
    *,
    include_rule_ids: bool = False,
) -> Dict[str, Any]:
    keys = (
        "verified",
        "contract_version",
        "sheet_name",
        "sheet_range",
        "sheet_revision",
        "content_revision",
        "doctrine_version",
        "decision_id",
        "decision_ids",
        "rule_count",
        "active_rule_count",
        "unresolved_rule_count",
        "loaded_from",
        "error",
    )
    metadata = {key: snapshot.get(key) for key in keys}
    if include_rule_ids:
        metadata["active_rule_ids"] = list(snapshot.get("active_rule_ids") or [])
        metadata["unresolved_rule_ids"] = list(
            snapshot.get("unresolved_rule_ids") or []
        )
    return metadata
