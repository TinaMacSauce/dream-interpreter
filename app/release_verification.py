from __future__ import annotations

from typing import Any, Dict, List


EXPECTED_SERVICE = "dream-interpreter"
EXPECTED_RELEASE_ID = "teeth-dec05-v1"
EXPECTED_TEETH_DOCTRINE_VERSION = "DEC-TEETH-2026-09-03-05"
EXPECTED_TEETH_CONTEXT_VERSION = "teeth-context-v2"


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
        "teeth_doctrine_version": EXPECTED_TEETH_DOCTRINE_VERSION,
        "teeth_context_version": EXPECTED_TEETH_CONTEXT_VERSION,
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
    return errors
