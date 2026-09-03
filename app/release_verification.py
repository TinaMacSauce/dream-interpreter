from __future__ import annotations

from typing import Any, Dict, List


EXPECTED_SERVICE = "dream-interpreter"
EXPECTED_RELEASE_ID = "teeth-registry-v1"
EXPECTED_REPOSITORY = "TinaMacSauce/dream-interpreter"
EXPECTED_REPOSITORY_URL = "https://github.com/TinaMacSauce/dream-interpreter"
EXPECTED_PRODUCTION_URL = "https://interpreter.jamaicantruestories.com"
EXPECTED_TEETH_DOCTRINE_VERSION = "DEC-TEETH-2026-09-03-05"
EXPECTED_TEETH_CONTEXT_VERSION = "teeth-context-v2"
EXPECTED_TEETH_REGISTRY_SHEET_REVISION = "6134"
EXPECTED_TEETH_REGISTRY_CONTENT_REVISION = "fnv1a64:c51447de5d35bd59"
EXPECTED_TEETH_REGISTRY_CONTRACT_VERSION = "teeth-doctrine-registry-v1"


def validate_release_metadata(
    release: Any,
    *,
    expected_commit: str,
) -> List[str]:
    """Return contract violations for public release metadata."""
    if not isinstance(release, dict):
        return ["release metadata is missing or is not an object"]

    expected = {
        "build_commit": expected_commit,
        "release_id": EXPECTED_RELEASE_ID,
        "repository": EXPECTED_REPOSITORY,
        "repository_url": EXPECTED_REPOSITORY_URL,
        "commit_url": f"{EXPECTED_REPOSITORY_URL}/commit/{expected_commit}",
        "production_url": EXPECTED_PRODUCTION_URL,
        "version_endpoint": f"{EXPECTED_PRODUCTION_URL}/version",
        "qa_status_endpoint": f"{EXPECTED_PRODUCTION_URL}/qa/status",
        "teeth_doctrine_version": EXPECTED_TEETH_DOCTRINE_VERSION,
        "teeth_context_version": EXPECTED_TEETH_CONTEXT_VERSION,
        "teeth_registry_sheet_revision": EXPECTED_TEETH_REGISTRY_SHEET_REVISION,
        "teeth_registry_content_revision": EXPECTED_TEETH_REGISTRY_CONTENT_REVISION,
        "teeth_registry_contract_version": EXPECTED_TEETH_REGISTRY_CONTRACT_VERSION,
    }
    return [
        f"release.{field} expected {value!r}, got {release.get(field)!r}"
        for field, value in expected.items()
        if release.get(field) != value
    ]


def validate_live_payload(
    payload: Any,
    *,
    expected_commit: str,
) -> List[str]:
    """Validate the non-billable public liveness and version contract."""
    if not isinstance(payload, dict):
        return ["live payload is not an object"]

    errors: List[str] = []
    if payload.get("alive") is not True:
        errors.append(f"alive expected True, got {payload.get('alive')!r}")
    if payload.get("service") != EXPECTED_SERVICE:
        errors.append(
            f"service expected {EXPECTED_SERVICE!r}, got {payload.get('service')!r}"
        )
    errors.extend(
        validate_release_metadata(
            payload.get("release"),
            expected_commit=expected_commit,
        )
    )
    return errors


def validate_health_payload(
    payload: Any,
    *,
    expected_commit: str,
) -> List[str]:
    """Validate production dependencies and the deployed version together."""
    if not isinstance(payload, dict):
        return ["health payload is not an object"]

    expected: Dict[str, Any] = {
        "service": EXPECTED_SERVICE,
        "status": "healthy",
        "spreadsheet_connected": True,
        "doctrine_sheets_available": True,
    }
    errors = [
        f"{field} expected {value!r}, got {payload.get(field)!r}"
        for field, value in expected.items()
        if payload.get(field) != value
    ]
    errors.extend(
        validate_release_metadata(
            payload.get("release"),
            expected_commit=expected_commit,
        )
    )
    registry = payload.get("teeth_registry")
    if not isinstance(registry, dict):
        errors.append("teeth_registry is missing or is not an object")
    else:
        registry_expected: Dict[str, Any] = {
            "verified": True,
            "contract_version": EXPECTED_TEETH_REGISTRY_CONTRACT_VERSION,
            "sheet_revision": EXPECTED_TEETH_REGISTRY_SHEET_REVISION,
            "content_revision": EXPECTED_TEETH_REGISTRY_CONTENT_REVISION,
            "doctrine_version": EXPECTED_TEETH_DOCTRINE_VERSION,
            "rule_count": 23,
            "active_rule_count": 17,
            "unresolved_rule_count": 6,
            "loaded_from": "canonical_sheet",
        }
        errors.extend(
            f"teeth_registry.{field} expected {value!r}, got {registry.get(field)!r}"
            for field, value in registry_expected.items()
            if registry.get(field) != value
        )
    return errors


def validate_version_payload(
    payload: Any,
    *,
    expected_commit: str,
) -> List[str]:
    """Validate the dependency-light deployment identity endpoint."""
    if not isinstance(payload, dict):
        return ["version payload is not an object"]

    errors: List[str] = []
    if payload.get("service") != EXPECTED_SERVICE:
        errors.append(
            f"service expected {EXPECTED_SERVICE!r}, got {payload.get('service')!r}"
        )
    if payload.get("production_url") != EXPECTED_PRODUCTION_URL:
        errors.append(
            "production_url expected "
            f"{EXPECTED_PRODUCTION_URL!r}, got {payload.get('production_url')!r}"
        )
    errors.extend(
        validate_release_metadata(
            payload.get("release"),
            expected_commit=expected_commit,
        )
    )
    return errors


def validate_qa_status_payload(
    payload: Any,
    *,
    expected_commit: str,
) -> List[str]:
    """Validate protected, isolated QA access without exposing credentials."""
    if not isinstance(payload, dict):
        return ["QA status payload is not an object"]

    errors: List[str] = []
    if payload.get("service") != EXPECTED_SERVICE:
        errors.append(
            f"service expected {EXPECTED_SERVICE!r}, got {payload.get('service')!r}"
        )
    if payload.get("ready") is not True:
        errors.append(f"ready expected True, got {payload.get('ready')!r}")
    errors.extend(
        validate_release_metadata(
            payload.get("release"),
            expected_commit=expected_commit,
        )
    )

    qa_access = payload.get("qa_access")
    if not isinstance(qa_access, dict):
        errors.append("qa_access is missing or is not an object")
    else:
        expected: Dict[str, Any] = {
            "configured": True,
            "storage_ready": True,
            "grant_route": "/admin/qa-grant",
            "revoke_route": "/admin/qa-revoke",
            "interpret_route": "/qa/interpret",
            "application_route": "/interpret",
            "fixed_contract_route": "/qa/teeth-regression",
            "grant_authentication": "X-Admin-Key",
            "interpret_authentication": "X-QA-Token or Authorization Bearer",
            "non_billable": True,
            "customer_credits_consumed": False,
            "customer_entitlement_store_used": False,
            "token_storage": "sha256_hash_only",
            "revocable": True,
        }
        errors.extend(
            f"qa_access.{field} expected {value!r}, got {qa_access.get(field)!r}"
            for field, value in expected.items()
            if qa_access.get(field) != value
        )

    registry = payload.get("doctrine_registry")
    if not isinstance(registry, dict) or registry.get("verified") is not True:
        errors.append("doctrine_registry.verified expected True")
    return errors
