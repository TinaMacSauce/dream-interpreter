from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from app.cache import TEETH_REGISTRY_CACHE
from app.config import Config
from app.sheets import get_spreadsheet
from app.utils import normalize_header


REGISTRY_CONTRACT_VERSION = "teeth-doctrine-registry-v1"
EXPECTED_DOCTRINE_VERSION = "DEC-TEETH-2026-09-03-05"
EXPECTED_SHEET_REVISION = "6134"
EXPECTED_CONTENT_REVISION = "fnv1a64:c51447de5d35bd59"
EXPECTED_UPDATED_AT_UTC = "2026-09-03T18:43:07Z"

REQUIRED_HEADERS = (
    "rule_id",
    "doctrine_version",
    "cluster",
    "trigger_context",
    "governing_meaning",
    "precedence",
    "safety_boundary",
    "status",
    "authority",
    "decision_id",
    "implementation_key",
    "updated_at_utc",
    "active",
)

# This is a governance manifest, not a second doctrine source. Runtime rule IDs,
# statuses, and active flags come from the canonical Sheet only after the exact
# registry content has been verified against revision 6134.
EXPECTED_RULES: Mapping[str, Tuple[str, str, bool]] = {
    "own_fallout": ("TEETH-FALLOUT-OWN", "APPROVED", True),
    "one_fallout": ("TEETH-FALLOUT-ONE", "APPROVED", True),
    "multiple_fallout": ("TEETH-FALLOUT-MULTIPLE", "APPROVED", True),
    "other_fallout": ("TEETH-FALLOUT-OTHER", "APPROVED", True),
    "self_pull": ("TEETH-PULL-SELF", "APPROVED", True),
    "external_pull": ("TEETH-PULL-EXTERNAL", "APPROVED", True),
    "loose": ("TEETH-STATE-LOOSE", "APPROVED", True),
    "broken": ("TEETH-STATE-BROKEN", "APPROVED", True),
    "rotten": ("TEETH-STATE-ROTTEN", "APPROVED", True),
    "pain": ("TEETH-MOD-PAIN", "APPROVED", True),
    "painless": ("TEETH-MOD-PAINLESS", "APPROVED", True),
    "gum_blood": ("TEETH-OMEN-GUM-BLOOD", "APPROVED", True),
    "tooth_blood": ("TEETH-MOD-BLOOD", "APPROVED", True),
    "gold": ("TEETH-MOD-GOLD", "APPROVED", True),
    "repetition": ("TEETH-MOD-REPETITION", "APPROVED", True),
    "subject_scope": ("TEETH-SUBJECT-EXPLICIT", "APPROVED", True),
    "terminal_ending": ("TEETH-END-TERMINAL", "APPROVED", True),
    "pending_returned_same_tooth": ("TEETH-END-RETURNED-SAME", "UNRESOLVED", False),
    "pending_position_mapping": ("TEETH-POSITION-MAP", "UNRESOLVED", False),
    "pending_state_fallout_precedence": ("TEETH-COMBO-STATE-FALLOUT", "UNRESOLVED", False),
    "pending_non_human": ("TEETH-NONHUMAN", "UNRESOLVED", False),
    "pending_replacement_growth": ("TEETH-REPLACEMENT-GROWTH", "UNRESOLVED", False),
    "pending_spitting": ("TEETH-SPITTING", "UNRESOLVED", False),
}


def _fnv1a64(text: str) -> str:
    value = 0xCBF29CE484222325
    for byte in text.encode("utf-8"):
        value ^= byte
        value = (value * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"fnv1a64:{value:016x}"


def registry_content_revision(values: Sequence[Sequence[Any]]) -> str:
    canonical = json.dumps(
        [[str(cell) for cell in row] for row in values],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return _fnv1a64(canonical)


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _rows_from_values(values: Sequence[Sequence[Any]]) -> Tuple[List[str], List[Dict[str, str]]]:
    if not values:
        raise RuntimeError("registry_empty")

    headers = [normalize_header(str(value or "")) for value in values[0]]
    if tuple(headers) != REQUIRED_HEADERS:
        raise RuntimeError("registry_schema_mismatch")

    rows: List[Dict[str, str]] = []
    for raw in values[1:]:
        padded = list(raw) + [""] * max(0, len(headers) - len(raw))
        row = {
            headers[index]: str(padded[index] or "").strip()
            for index in range(len(headers))
        }
        if any(row.values()):
            rows.append(row)
    return headers, rows


def validate_registry_values(
    values: Sequence[Sequence[Any]],
    *,
    expected_content_revision: str | None = None,
) -> Dict[str, Any]:
    """Validate the exact approved Sheet registry and return a safe snapshot."""
    _headers, rows = _rows_from_values(values)
    content_revision = registry_content_revision(values)
    if content_revision != (expected_content_revision or EXPECTED_CONTENT_REVISION):
        raise RuntimeError("registry_content_revision_mismatch")
    if len(rows) != len(EXPECTED_RULES):
        raise RuntimeError("registry_rule_count_mismatch")

    rules: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        key = row["implementation_key"]
        if not key or key in rules or key not in EXPECTED_RULES:
            raise RuntimeError("registry_implementation_key_mismatch")

        expected_rule_id, expected_status, expected_active = EXPECTED_RULES[key]
        active = _truthy(row["active"])
        if row["rule_id"] != expected_rule_id:
            raise RuntimeError("registry_rule_id_mismatch")
        if row["status"] != expected_status or active is not expected_active:
            raise RuntimeError("registry_activation_mismatch")
        if row["doctrine_version"] != EXPECTED_DOCTRINE_VERSION:
            raise RuntimeError("registry_doctrine_version_mismatch")
        if row["decision_id"] != EXPECTED_DOCTRINE_VERSION:
            raise RuntimeError("registry_decision_id_mismatch")
        if row["cluster"] != "Teeth":
            raise RuntimeError("registry_cluster_mismatch")
        if row["authority"] != "Tina, latest explicit decision":
            raise RuntimeError("registry_authority_mismatch")
        if row["updated_at_utc"] != EXPECTED_UPDATED_AT_UTC:
            raise RuntimeError("registry_timestamp_mismatch")

        rules[key] = {
            "rule_id": row["rule_id"],
            "status": row["status"],
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
        "sheet_revision": EXPECTED_SHEET_REVISION,
        "content_revision": content_revision,
        "doctrine_version": EXPECTED_DOCTRINE_VERSION,
        "decision_id": EXPECTED_DOCTRINE_VERSION,
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


def get_teeth_registry_snapshot(*, force: bool = False) -> Dict[str, Any]:
    """Load and verify the canonical Teeth registry, failing closed on drift."""
    if Config.APP_ENV == "test":
        return _test_manifest_snapshot()

    now = time.time()
    cached = TEETH_REGISTRY_CACHE.get("snapshot")
    if (
        not force
        and isinstance(cached, dict)
        and now - float(TEETH_REGISTRY_CACHE.get("loaded_at") or 0) < Config.CACHE_TTL_SECONDS
    ):
        return cached

    try:
        worksheet = get_spreadsheet().worksheet(Config.SHEET_DOCTRINE_REGISTRY)
        snapshot = validate_registry_values(worksheet.get_all_values())
    except Exception as error:
        snapshot = _failed_snapshot(error)

    TEETH_REGISTRY_CACHE["snapshot"] = snapshot
    TEETH_REGISTRY_CACHE["loaded_at"] = now
    return snapshot


def rule_id_for(snapshot: Mapping[str, Any], implementation_key: str) -> str:
    rule = (snapshot.get("rules") or {}).get(implementation_key) or {}
    if snapshot.get("verified") is not True or rule.get("active") is not True:
        return ""
    return str(rule.get("rule_id") or "")


def unresolved_rule_id_for(snapshot: Mapping[str, Any], implementation_key: str) -> str:
    rule = (snapshot.get("rules") or {}).get(implementation_key) or {}
    if snapshot.get("verified") is not True or rule.get("active") is not False:
        return ""
    return str(rule.get("rule_id") or "")


def public_registry_metadata(
    snapshot: Mapping[str, Any],
    *,
    include_rule_ids: bool = False,
) -> Dict[str, Any]:
    """Return non-sensitive registry identity and activation evidence."""
    metadata = {
        key: snapshot.get(key)
        for key in (
            "verified",
            "contract_version",
            "sheet_name",
            "sheet_revision",
            "content_revision",
            "doctrine_version",
            "decision_id",
            "rule_count",
            "active_rule_count",
            "unresolved_rule_count",
            "loaded_from",
            "error",
        )
    }
    if include_rule_ids:
        metadata["active_rule_ids"] = list(snapshot.get("active_rule_ids") or [])
        metadata["unresolved_rule_ids"] = list(snapshot.get("unresolved_rule_ids") or [])
    return metadata
